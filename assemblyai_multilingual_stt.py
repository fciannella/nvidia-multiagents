"""AssemblyAI multilingual STT service for Pipecat.

Custom STTService that connects directly to AssemblyAI's v3 streaming
WebSocket API with ``language=multi`` support for automatic language
detection — something the built-in pipecat integration does not expose.

See docs/ASSEMBLYAI_MULTILINGUAL_SUPPORT.md for background.
"""

import asyncio
import json
import os
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import STTService


class AssemblyAIMultilingualSTTService(STTService):
    """Real-time STT using AssemblyAI v3 with automatic language detection."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        language: str = "multi",
        sample_rate: int = 16000,
        format_turns: bool = True,
        end_of_turn_confidence_threshold: float = 0.7,
        min_end_of_turn_silence_when_confident: int = 160,
        max_turn_silence: int = 2400,
        keyterms_prompt: Optional[list[str]] = None,
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("ASSEMBLYAI_API_KEY is required")

        self._language = language
        self._sample_rate = sample_rate
        self._format_turns = format_turns
        self._end_of_turn_confidence_threshold = end_of_turn_confidence_threshold
        self._min_end_of_turn_silence_when_confident = min_end_of_turn_silence_when_confident
        self._max_turn_silence = max_turn_silence
        self._keyterms_prompt = keyterms_prompt
        self._api_endpoint_base_url = api_endpoint_base_url

        self._websocket = None
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._termination_event = asyncio.Event()

        self._audio_buffer = bytearray()
        # 50ms chunks at 16kHz, 16-bit mono = 1600 bytes
        self._chunk_size_bytes = int(self._sample_rate * 2 * 0.05)

        self._pending_frame: Optional[Frame] = None

    def _build_ws_url(self) -> str:
        params = {
            "sample_rate": self._sample_rate,
            "language": self._language,
            "format_turns": str(self._format_turns).lower(),
            "end_of_turn_confidence_threshold": self._end_of_turn_confidence_threshold,
            "min_end_of_turn_silence_when_confident": self._min_end_of_turn_silence_when_confident,
            "max_turn_silence": self._max_turn_silence,
        }
        if self._keyterms_prompt:
            params["keyterms_prompt"] = json.dumps(self._keyterms_prompt)

        return f"{self._api_endpoint_base_url}?{urlencode(params)}"

    async def _connect(self):
        ws_url = self._build_ws_url()
        logger.info(f"[AssemblyAI] Connecting: language={self._language}")
        logger.debug(f"[AssemblyAI] URL: {ws_url}")

        self._websocket = await websocket_connect(
            ws_url,
            additional_headers={"Authorization": self._api_key},
        )
        self._connected = True
        self._termination_event.clear()
        self._receive_task = asyncio.create_task(self._receive_messages())
        logger.info("[AssemblyAI] Connected")

    async def _disconnect(self):
        if not self._connected:
            return

        try:
            if self._audio_buffer and self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(bytes(self._audio_buffer))
                self._audio_buffer.clear()

            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "Terminate"}))
                try:
                    await asyncio.wait_for(self._termination_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("[AssemblyAI] Termination ack timed out")
        except Exception as e:
            logger.warning(f"[AssemblyAI] Error during disconnect: {e}")

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        self._receive_task = None

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
        self._websocket = None
        self._connected = False
        logger.info("[AssemblyAI] Disconnected")

    async def _receive_messages(self):
        try:
            async for message in self._websocket:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "Turn":
                    transcript = data.get("transcript", "")
                    end_of_turn = data.get("end_of_turn", False)
                    turn_is_formatted = data.get("turn_is_formatted", False)

                    if not transcript:
                        continue

                    if end_of_turn and turn_is_formatted:
                        logger.debug(f"[AssemblyAI] Final: {transcript[:80]}")
                        self._pending_frame = TranscriptionFrame(
                            text=transcript, user_id="", timestamp=""
                        )
                    else:
                        self._pending_frame = InterimTranscriptionFrame(
                            text=transcript, user_id="", timestamp=""
                        )

                elif msg_type == "Begin":
                    session_id = data.get("id", "?")
                    logger.info(f"[AssemblyAI] Session started: {session_id}")

                elif msg_type == "Termination":
                    duration = data.get("audio_duration_seconds", 0)
                    logger.info(f"[AssemblyAI] Session ended, audio={duration:.1f}s")
                    self._termination_event.set()

                elif msg_type == "Error":
                    logger.error(f"[AssemblyAI] Error: {data.get('error', data)}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._connected:
                logger.error(f"[AssemblyAI] Receive error: {e}")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await self._disconnect()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        await self._disconnect()
        await super().cancel(frame)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._connected or not self._websocket:
            yield None
            return

        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            try:
                await self._websocket.send(chunk)
            except Exception as e:
                logger.error(f"[AssemblyAI] Send error: {e}")
                yield None
                return

        if self._pending_frame:
            frame = self._pending_frame
            self._pending_frame = None
            yield frame
        else:
            yield None
