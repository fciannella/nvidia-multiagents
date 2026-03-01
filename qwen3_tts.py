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
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

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
        self._t0: float = 0
        self._first_audio: bool = True
        self._total_bytes: int = 0
        self._total_chunks: int = 0
        self._text_full: str = ""
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def _stream_tts(self, text: str, label: str):
        """POST text to /v1/tts/stream and push audio frames as they arrive."""
        await self._ensure_client()
        url = f"{self._server}/v1/tts/stream"
        payload = {
            "text": text,
            "voice": self._voice,
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
                header_skipped = False
                leftover = b""

                async for chunk in resp.aiter_bytes():
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

        except Exception as e:
            logger.error(f"[Q3-TTS] {label} error: {type(e).__name__}: {e}")

    async def _chained_tts(self, prev: Optional[asyncio.Task], text: str, label: str):
        """Wait for the previous TTS task to finish, then run ours."""
        if prev is not None:
            await prev
        await self._stream_tts(text, label)

    def _enqueue_sentence(self, text: str):
        """Queue a sentence for TTS, chained after the previous one."""
        self._sentence_idx += 1
        label = f"S{self._sentence_idx}"
        self._tts_chain = asyncio.create_task(
            self._chained_tts(self._tts_chain, text, label)
        )

    def _check_sentence_boundary(self):
        """If the buffer ends at a sentence/clause boundary, dispatch it."""
        stripped = self._buffer.rstrip()
        n = len(self._buffer)
        is_sentence = stripped and stripped[-1] in ".!?"
        is_clause = stripped and stripped[-1] == "," and n >= 40

        if is_sentence or is_clause:
            self._enqueue_sentence(self._buffer)
            self._buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._sentence_idx = 0
            self._tts_chain = None
            self._t0 = time.perf_counter()
            self._first_audio = True
            self._total_bytes = 0
            self._total_chunks = 0
            self._text_full = ""
            await self.push_frame(TTSStartedFrame())
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

            if self._tts_chain is not None:
                await self._tts_chain
                self._tts_chain = None

            elapsed = time.perf_counter() - self._t0
            audio_dur = self._total_bytes / (SAMPLE_RATE * 2)
            logger.info(
                f"[Q3-TTS] Done: chunks={self._total_chunks} "
                f"bytes={self._total_bytes} audio={audio_dur:.2f}s "
                f"elapsed={elapsed*1000:.0f}ms "
                f"text=[{self._text_full[:100]}...]"
            )

            self._buffer = ""
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            if self._tts_chain and not self._tts_chain.done():
                self._tts_chain.cancel()
            if self._client:
                await self._client.aclose()
                self._client = None
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
