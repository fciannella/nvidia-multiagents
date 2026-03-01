"""Standalone pipecat script that synthesizes speech using the Riva NIM TTS
server and writes the result to a WAV file.

Usage:
    python run_tts.py
    python run_tts.py --text "Your custom text here"
    python run_tts.py --voice Magpie-Multilingual.EN-US.Ray --output ray.wav
"""

import argparse
import asyncio
import struct
import sys

from loguru import logger

sys.path.insert(0, ".")
sys.path.insert(0, "pipecat/src")

from pipecat.frames.frames import EndFrame, Frame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from riva_nim_tts import RivaNimTTSService


class AudioFileSink(FrameProcessor):
    """Collects raw audio frames and writes them to a WAV file."""

    def __init__(self, output_path: str, sample_rate: int = 22050, **kwargs):
        super().__init__(**kwargs)
        self._output_path = output_path
        self._sample_rate = sample_rate
        self._audio_chunks: list[bytes] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            self._audio_chunks.append(frame.audio)
        elif isinstance(frame, EndFrame):
            self._write_wav()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    def _write_wav(self):
        if not self._audio_chunks:
            logger.warning("No audio data received")
            return

        pcm_data = b"".join(self._audio_chunks)
        num_channels = 1
        sample_width = 2  # 16-bit
        byte_rate = self._sample_rate * num_channels * sample_width
        block_align = num_channels * sample_width

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + len(pcm_data),
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM format
            num_channels,
            self._sample_rate,
            byte_rate,
            block_align,
            sample_width * 8,
            b"data",
            len(pcm_data),
        )

        with open(self._output_path, "wb") as f:
            f.write(header + pcm_data)

        duration = len(pcm_data) / byte_rate
        logger.info(
            f"Wrote {self._output_path}: {len(pcm_data)} bytes, {duration:.2f}s"
        )


async def main(text: str, voice: str, server: str, output: str, sample_rate: int):
    tts = RivaNimTTSService(
        server=server,
        voice_id=voice,
        language="en-US",
        sample_rate=sample_rate,
        streaming=True,
    )

    sink = AudioFileSink(output_path=output, sample_rate=sample_rate)

    pipeline = Pipeline([tts, sink])
    task = PipelineTask(pipeline)

    await task.queue_frames([TTSSpeakFrame(text), EndFrame()])

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Riva NIM TTS with pipecat")
    parser.add_argument(
        "--text",
        default="Hello! This is a test of the NVIDIA Riva text to speech system, running through pipecat.",
    )
    parser.add_argument(
        "--voice", default="Magpie-Multilingual.EN-US.Aria"
    )
    parser.add_argument(
        "--server", default="http://192.168.7.163:9000"
    )
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    asyncio.run(main(args.text, args.voice, args.server, args.output, args.sample_rate))
