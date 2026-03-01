"""Pipecat TTS service for self-hosted NVIDIA Riva NIM (HTTP API).

The NIM TTS container exposes an HTTP REST API (not gRPC), so the standard
NvidiaTTSService (which uses the riva-client gRPC library) doesn't work.
This service talks to the NIM HTTP endpoints directly via aiohttp.
"""

import asyncio
import time
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language


class RivaNimTTSService(TTSService):
    """TTS service for self-hosted NVIDIA Riva NIM containers (HTTP API).

    Uses the /v1/audio/synthesize_online endpoint for streaming synthesis
    and /v1/audio/synthesize for non-streaming.
    """

    def __init__(
        self,
        *,
        server: str = "http://192.168.7.163:9000",
        voice_id: str = "Magpie-Multilingual.EN-US.Aria",
        language: str = "en-US",
        sample_rate: int = 22050,
        streaming: bool = True,
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(
                model="magpie-tts-multilingual",
                voice=voice_id,
                language=language,
            ),
            **kwargs,
        )
        self._server = server.rstrip("/")
        self._voice_id = voice_id
        self._language = language
        self._streaming = streaming
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()
        logger.debug(
            f"RivaNimTTSService ready: server={self._server} voice={self._voice_id}"
        )

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"[TTS-REQ] ctx={context_id[:8]} text=[{text}]")
        t0 = time.perf_counter()

        if not self._session:
            self._session = aiohttp.ClientSession()

        endpoint = "synthesize_online" if self._streaming else "synthesize"
        url = f"{self._server}/v1/audio/{endpoint}"

        form = aiohttp.FormData()
        form.add_field("text", text)
        form.add_field("language", self._language)
        form.add_field("voice", self._voice_id)
        form.add_field("sample_rate_hz", str(self.sample_rate))
        form.add_field("encoding", "LINEAR_PCM")

        try:
            await self.start_ttfb_metrics()
            yield TTSStartedFrame(context_id=context_id)

            async with self._session.post(url, data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"[TTS-ERR] ctx={context_id[:8]} HTTP {resp.status}: {body}")
                    yield ErrorFrame(error=f"Riva NIM TTS error {resp.status}: {body}")
                    return

                first_chunk = True
                chunk_count = 0
                total_bytes = 0
                async for chunk in resp.content.iter_chunked(4096):
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        ttfb = time.perf_counter() - t0
                        logger.info(
                            f"[TTS-TTFB] ctx={context_id[:8]} "
                            f"first_chunk={len(chunk)}B ttfb={ttfb*1000:.0f}ms"
                        )
                        first_chunk = False
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )

            elapsed = time.perf_counter() - t0
            audio_duration = total_bytes / (self.sample_rate * 2)
            logger.info(
                f"[TTS-DONE] ctx={context_id[:8]} "
                f"chunks={chunk_count} bytes={total_bytes} "
                f"audio={audio_duration:.2f}s elapsed={elapsed*1000:.0f}ms "
                f"text=[{text}]"
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStoppedFrame(context_id=context_id)

        except asyncio.TimeoutError:
            logger.error(f"[TTS-TIMEOUT] ctx={context_id[:8]} text=[{text}]")
            yield ErrorFrame(error=f"{self} timeout")
        except Exception as e:
            logger.error(f"[TTS-EXCEPTION] ctx={context_id[:8]} {type(e).__name__}: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")
