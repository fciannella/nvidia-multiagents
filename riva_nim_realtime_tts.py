"""Pipecat TTS processor using Riva NIM's WebSocket realtime API.

Streams LLM tokens directly to Riva via WebSocket and pushes audio frames
back into the pipeline as they arrive.

Commit strategy: commit at every sentence boundary so Riva can start
synthesizing early (low TTFB).  input_text.done is deferred until the
recv loop detects audio has stopped flowing, preventing Riva from
aborting unprocessed commits.

Protocol (Riva NIM /v1/realtime?intent=synthesize):
  1. Open WebSocket, receive conversation.created
  2. Send synthesize_session.update to configure voice/sample_rate
  3. For each LLM token: send input_text.append
  4. At sentence boundaries: send input_text.commit
  5. Recv loop waits for audio to drain, then sends input_text.done
  6. Receive base64 audio in conversation.item.speech.data messages
"""

import asyncio
import base64
import json
import time
import uuid
from typing import Optional

import websockets
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


def _make_event(event_type: str, **kwargs) -> str:
    return json.dumps({
        "event_id": f"evt_{uuid.uuid4().hex[:8]}",
        "type": event_type,
        **kwargs,
    })


class RivaNimRealtimeTTSService(FrameProcessor):
    """Streams LLM text tokens to Riva NIM via WebSocket, pushes audio downstream."""

    def __init__(
        self,
        *,
        server: str = "192.168.7.163:9000",
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        language: str = "en-US",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._server = server.replace("http://", "").replace("https://", "").rstrip("/")
        self._voice_id = voice_id
        self._language = language
        self._sample_rate = sample_rate

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._context_id: Optional[str] = None
        self._t0: float = 0
        self._first_audio: bool = True
        self._total_bytes: int = 0
        self._total_chunks: int = 0
        self._text_buffer: str = ""
        self._uncommitted: str = ""
        self._commits: int = 0
        self._completions: int = 0
        self._all_committed = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def _open_session(self):
        """Open WebSocket and configure the synthesis session."""
        ws_url = f"ws://{self._server}/v1/realtime?intent=synthesize"
        self._ws = await websockets.connect(ws_url, max_size=16 * 1024 * 1024)

        msg = json.loads(await self._ws.recv())
        if msg["type"] != "conversation.created":
            logger.error(f"[RT-TTS] Unexpected first message: {msg['type']}")
            return

        await self._ws.send(_make_event(
            "synthesize_session.update",
            session={
                "input_text_synthesis": {
                    "language_code": self._language,
                    "voice_name": self._voice_id,
                },
                "output_audio_params": {
                    "sample_rate_hz": self._sample_rate,
                    "num_channels": 1,
                    "audio_format": "LINEAR_PCM",
                },
            },
        ))

        msg = json.loads(await self._ws.recv())
        if msg["type"] != "synthesize_session.updated":
            logger.error(f"[RT-TTS] Session update failed: {msg}")
            return

        logger.info(f"[RT-TTS] Session ready: voice={self._voice_id} rate={self._sample_rate}")

    async def _close_session(self):
        """Close the WebSocket session."""
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _recv_audio_loop(self):
        """Background task: receive audio from WebSocket and push downstream.

        Uses an idle timeout to detect when Riva has finished producing audio.
        Once all commits are sent and no data arrives for IDLE_TIMEOUT seconds,
        we send input_text.done and exit.  This prevents Riva from aborting
        unprocessed commits due to a premature done signal.
        """
        IDLE_TIMEOUT = 8.0

        try:
            leftover = b""
            done_sent = False

            while True:
                use_timeout = (
                    IDLE_TIMEOUT if self._all_committed.is_set() and not done_sent
                    else None
                )
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), timeout=use_timeout)
                except asyncio.TimeoutError:
                    if not done_sent:
                        logger.info(
                            f"[RT-TTS] Audio idle for {IDLE_TIMEOUT}s after all commits, "
                            f"sending input_text.done"
                        )
                        try:
                            await self._ws.send(_make_event("input_text.done"))
                        except Exception:
                            pass
                        done_sent = True
                        continue
                    break

                msg = json.loads(raw)
                etype = msg["type"]

                if etype == "conversation.item.speech.data":
                    audio_bytes = base64.b64decode(msg["audio"])

                    if self._first_audio:
                        self._first_audio = False
                        ttfb = (time.perf_counter() - self._t0) * 1000
                        logger.info(
                            f"[RT-TTS] TTFB={ttfb:.0f}ms "
                            f"first_chunk={len(audio_bytes)}B"
                        )

                    pcm = leftover + audio_bytes
                    usable = len(pcm) - (len(pcm) % 2)
                    leftover = pcm[usable:]
                    if usable > 0:
                        self._total_bytes += usable
                        self._total_chunks += 1
                        await self.push_frame(TTSAudioRawFrame(
                            audio=pcm[:usable],
                            sample_rate=self._sample_rate,
                            num_channels=1,
                        ))

                elif etype == "conversation.item.speech.completed":
                    self._completions += 1
                    is_last = msg.get("is_last_result", False)
                    meta = msg.get("synthesis_metadata") or {}
                    logger.info(
                        f"[RT-TTS] Synthesis completed "
                        f"({self._completions}/{self._commits}) "
                        f"is_last={is_last} "
                        f"synth={meta.get('synthesis_time_ms', '?')}ms "
                        f"audio_dur={meta.get('audio_duration_ms', '?')}ms"
                    )
                    if done_sent and is_last:
                        break

                elif etype == "input_text.committed":
                    pass

                elif etype == "error":
                    err = msg.get("error", {})
                    logger.error(f"[RT-TTS] Error: {err.get('message', msg)}")
                    break

            if leftover:
                self._total_bytes += len(leftover)
                await self.push_frame(TTSAudioRawFrame(
                    audio=leftover + b"\x00" * (len(leftover) % 2),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                ))

        except websockets.ConnectionClosed:
            logger.warning("[RT-TTS] WebSocket closed during recv")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[RT-TTS] Recv error: {type(e).__name__}: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._t0 = time.perf_counter()
            self._first_audio = True
            self._total_bytes = 0
            self._total_chunks = 0
            self._text_buffer = ""
            self._uncommitted = ""
            self._commits = 0
            self._completions = 0
            self._all_committed = asyncio.Event()
            self._context_id = uuid.uuid4().hex[:16]

            try:
                await self._open_session()
            except Exception as e:
                logger.error(f"[RT-TTS] Failed to open session: {e}")
                await self.push_frame(frame, direction)
                return

            self._recv_task = asyncio.create_task(self._recv_audio_loop())
            await self.push_frame(TTSStartedFrame())
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame) and self._ws:
            token = frame.text
            self._text_buffer += token
            self._uncommitted += token
            try:
                await self._ws.send(_make_event("input_text.append", text=token))

                if self._commits == 0:
                    stripped = self._uncommitted.rstrip()
                    n = len(self._uncommitted)
                    is_sentence_end = stripped and stripped[-1] in ".!?"
                    is_clause = stripped and stripped[-1] == "," and n >= 40

                    if is_sentence_end or is_clause:
                        await self._ws.send(_make_event("input_text.commit"))
                        self._commits += 1
                        reason = "sentence" if is_sentence_end else "clause"
                        logger.info(
                            f"[RT-TTS] Commit #{self._commits} [{reason}] "
                            f"({n} chars): "
                            f"[{self._uncommitted.strip()[:80]}]"
                        )
                        self._uncommitted = ""
            except Exception as e:
                logger.error(f"[RT-TTS] Send error: {e}")
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame) and self._ws:
            try:
                if self._uncommitted.strip():
                    await self._ws.send(_make_event("input_text.commit"))
                    self._commits += 1
                    logger.info(
                        f"[RT-TTS] Final commit #{self._commits} "
                        f"({len(self._uncommitted)} chars): "
                        f"[{self._uncommitted.strip()[:60]}]"
                    )
                logger.info(
                    f"[RT-TTS] All text sent: {self._commits} commits, "
                    f"recv loop will send done after audio drains"
                )
            except Exception as e:
                logger.error(f"[RT-TTS] Finalize error: {e}")
            self._all_committed.set()

            if self._recv_task:
                try:
                    await asyncio.wait_for(self._recv_task, timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("[RT-TTS] Timed out waiting for final audio")
                    self._recv_task.cancel()

            elapsed = time.perf_counter() - self._t0
            audio_dur = self._total_bytes / (self._sample_rate * 2)
            logger.info(
                f"[RT-TTS] Done: chunks={self._total_chunks} "
                f"bytes={self._total_bytes} audio={audio_dur:.2f}s "
                f"elapsed={elapsed*1000:.0f}ms "
                f"text=[{self._text_buffer[:100]}...]"
            )

            await self.push_frame(TTSStoppedFrame())
            await self._close_session()
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            await self._close_session()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
