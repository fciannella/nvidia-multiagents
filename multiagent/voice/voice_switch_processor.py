"""Processor that intercepts VoiceSwitchFrames and forwards them to TTS.

VoiceSwitchFrames are pushed downstream so the TTS processes voice
changes in queue order alongside LLMFullResponseStart/End/TextFrames.
This prevents race conditions where a later voice switch (e.g. from an
executor delivery) overrides the voice before TTS reaches earlier
queued frames.
"""

from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from multiagent.voice.frames import VoiceSwitchFrame


class VoiceSwitchProcessor(FrameProcessor):
    """Sits before the TTS in the pipeline.

    Passes VoiceSwitchFrames downstream so the TTS handles voice
    changes in the correct queue order.
    """

    def __init__(self, tts, **kwargs):
        super().__init__(**kwargs)
        self._tts = tts

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VoiceSwitchFrame):
            logger.info(f"Voice switch queued → {frame.voice} ({frame.speaker})")

        await self.push_frame(frame, direction)
