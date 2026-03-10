"""Pipecat TTS processor using the Qwen3-TTS HTTP streaming API.

Strategy: buffer LLM tokens and detect sentence boundaries. Each
sentence is dispatched to TTS as soon as it is complete (chained
sequentially so audio arrives in order).  The first sentence fires
immediately for low TTFB; subsequent sentences begin synthesising as
soon as the previous finishes, eliminating the silence gap that
occurs when all remaining text is held until the LLM is done.

API: POST /v1/tts/stream  (see docs/TTS-API.md)
"""

import asyncio
import time
from typing import Optional

import httpx
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    from multiagent.voice.frames import VoiceSwitchFrame
except ImportError:
    VoiceSwitchFrame = None

WAV_HEADER_SIZE = 44
SAMPLE_RATE = 24000


class Qwen3TTSService(FrameProcessor):
    """Streams LLM text to Qwen3-TTS via HTTP, pushes audio downstream."""

    def __init__(
        self,
        *,
        server: str = "http://192.168.7.163:8100",
        voice: str = "default",
        language: str = "English",
        emit_every_frames: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server = server.rstrip("/")
        self._voice = voice
        self._language = language
        self._emit_every_frames = emit_every_frames

        self._buffer: str = ""
        self._sentence_idx: int = 0
        self._tts_chain: Optional[asyncio.Task] = None
        self._all_tasks: list[asyncio.Task] = []
        self._t0: float = 0
        self._first_audio: bool = True
        self._total_bytes: int = 0
        self._total_chunks: int = 0
        self._text_full: str = ""
        self._client: Optional[httpx.AsyncClient] = None
        self._interrupted = False
        self._active_resp: Optional[httpx.Response] = None

    def set_voice(self, voice: str):
        """Switch the TTS voice for subsequent requests."""
        if voice != self._voice:
            logger.info(f"[Q3-TTS] Voice changed: {self._voice} → {voice}")
            self._voice = voice

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def _stream_tts(self, text: str, label: str, voice: str | None = None):
        """POST text to /v1/tts/stream and push audio frames as they arrive."""
        await self._ensure_client()
        url = f"{self._server}/v1/tts/stream"
        payload = {
            "text": text,
            "voice": voice or self._voice,
            "language": self._language,
            "emit_every_frames": self._emit_every_frames,
        }

        t_req = time.perf_counter()
        logger.info(
            f"[Q3-TTS] {label} request ({len(text)} chars): "
            f"[{text.strip()[:80]}]"
        )

        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                resp.raise_for_status()
                self._active_resp = resp
                header_skipped = False
                leftover = b""

                try:
                    async for chunk in resp.aiter_bytes():
                        if self._interrupted:
                            logger.debug("[Q3-TTS] Interrupted mid-stream")
                            return

                        data = leftover + chunk

                        if not header_skipped:
                            if len(data) < WAV_HEADER_SIZE:
                                leftover = data
                                continue
                            data = data[WAV_HEADER_SIZE:]
                            header_skipped = True

                        usable = len(data) - (len(data) % 2)
                        leftover = data[usable:]

                        if usable > 0:
                            pcm = data[:usable]
                            self._total_bytes += usable
                            self._total_chunks += 1

                            if self._first_audio:
                                self._first_audio = False
                                ttfb = (time.perf_counter() - self._t0) * 1000
                                logger.info(
                                    f"[Q3-TTS] TTFB={ttfb:.0f}ms "
                                    f"first_chunk={len(pcm)}B"
                                )

                            await self.push_frame(TTSAudioRawFrame(
                                audio=pcm,
                                sample_rate=SAMPLE_RATE,
                                num_channels=1,
                            ))
                finally:
                    self._active_resp = None

                if self._interrupted:
                    return

                if leftover:
                    if len(leftover) % 2:
                        leftover += b"\x00"
                    self._total_bytes += len(leftover)
                    await self.push_frame(TTSAudioRawFrame(
                        audio=leftover,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                    ))

            elapsed = (time.perf_counter() - t_req) * 1000
            logger.info(f"[Q3-TTS] {label} done in {elapsed:.0f}ms")

        except httpx.RemoteProtocolError:
            logger.debug(f"[Q3-TTS] {label} connection closed (interrupted)")
        except Exception as e:
            if self._interrupted:
                logger.debug(f"[Q3-TTS] {label} aborted on interrupt")
            else:
                logger.error(f"[Q3-TTS] {label} error: {type(e).__name__}: {e}")

    async def _chained_tts(self, prev: Optional[asyncio.Task], text: str, label: str, voice: str | None = None):
        """Wait for the previous TTS task to finish, then run ours."""
        if prev is not None:
            await prev
        if self._interrupted:
            return
        await self._stream_tts(text, label, voice=voice)

    def _enqueue_sentence(self, text: str):
        """Queue a sentence for TTS, chained after the previous one.

        Captures self._voice at enqueue time so a later voice switch
        (for the next agent) doesn't affect queued sentences.
        """
        self._sentence_idx += 1
        label = f"S{self._sentence_idx}"
        captured_voice = self._voice
        task = asyncio.create_task(
            self._chained_tts(self._tts_chain, text, label, voice=captured_voice)
        )
        self._tts_chain = task
        self._all_tasks.append(task)

    def _check_sentence_boundary(self):
        """Dispatch all complete sentences from the buffer.

        Scans for intermediate sentence boundaries ('. ', '? ', '! ')
        so that large chunks arriving at once get split properly.
        """
        while True:
            buf = self._buffer
            if not buf.strip():
                break

            # Find the first sentence boundary: ./?/! followed by a space
            best = -1
            for i in range(len(buf) - 1):
                if buf[i] in ".!?" and buf[i + 1] == " ":
                    best = i
                    break

            if best >= 0:
                sentence = buf[: best + 1]
                self._buffer = buf[best + 2 :]
                self._enqueue_sentence(sentence)
                continue

            # No intermediate boundary — check trailing markers
            stripped = buf.rstrip()
            if stripped and stripped[-1] in ".!?":
                self._enqueue_sentence(buf)
                self._buffer = ""
            elif stripped and stripped[-1] == "," and len(stripped) >= 40:
                self._enqueue_sentence(buf)
                self._buffer = ""
            break

    def _cancel_all_tts(self, reason: str = ""):
        """Cancel all in-flight TTS tasks and abort the active HTTP stream."""
        self._interrupted = True
        resp = self._active_resp
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass
            self._active_resp = None
        for task in self._all_tasks:
            if not task.done():
                task.cancel()
        cancelled_count = sum(1 for t in self._all_tasks if not t.done())
        self._all_tasks.clear()
        self._tts_chain = None
        self._buffer = ""
        if reason:
            logger.info(f"[Q3-TTS] {reason} (cancelled {cancelled_count} tasks)")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self._cancel_all_tts("InterruptionFrame received")
            await self.push_frame(frame, direction)

        elif VoiceSwitchFrame is not None and isinstance(frame, VoiceSwitchFrame):
            old = self._voice
            self._voice = frame.voice
            if old != frame.voice:
                logger.info(
                    f"[Q3-TTS] Voice changed (in-queue): {old} → {frame.voice}"
                )

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._sentence_idx = 0
            self._tts_chain = None
            self._all_tasks.clear()
            self._interrupted = False
            self._active_resp = None
            self._t0 = time.perf_counter()
            self._first_audio = True
            self._total_bytes = 0
            self._total_chunks = 0
            self._text_full = ""
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame):
            self._buffer += frame.text
            self._text_full += frame.text
            self._check_sentence_boundary()
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            remaining = self._buffer.strip()
            if remaining:
                self._enqueue_sentence(self._buffer)
            elif self._sentence_idx == 0 and self._text_full.strip():
                self._enqueue_sentence(self._text_full)

            await self.push_frame(TTSStartedFrame())

            if self._tts_chain is not None:
                try:
                    await self._tts_chain
                except asyncio.CancelledError:
                    self._cancel_all_tts("Framework cancelled chain wait")
                    raise

            self._tts_chain = None
            self._all_tasks.clear()

            if not self._interrupted:
                elapsed = time.perf_counter() - self._t0
                audio_dur = self._total_bytes / (SAMPLE_RATE * 2)
                logger.info(
                    f"[Q3-TTS] Done: chunks={self._total_chunks} "
                    f"bytes={self._total_bytes} audio={audio_dur:.2f}s "
                    f"elapsed={elapsed * 1000:.0f}ms "
                    f"text=[{self._text_full[:100]}...]"
                )

            self._buffer = ""
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            self._cancel_all_tts("Pipeline ending")
            if self._client:
                await self._client.aclose()
                self._client = None
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
