"""Hybrid user turn start strategy: VAD when bot silent, min-words when bot speaking.

Combines fast VAD-based turn detection (when the bot is not speaking) with
echo-resilient min-words detection (when the bot is speaking). This avoids
false interruptions from audio echo while keeping turn starts snappy.
"""

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy


class HybridUserTurnStartStrategy(BaseUserTurnStartStrategy):
    """VAD for instant turn start; min-words transcription for barge-in."""

    def __init__(self, *, min_words: int = 2, use_interim: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._min_words = min_words
        self._use_interim = use_interim
        self._bot_speaking = False

    async def reset(self):
        await super().reset()
        self._bot_speaking = False

    async def process_frame(self, frame: Frame):
        await super().process_frame(frame)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                logger.debug(f"{self} VAD trigger (bot silent) -> turn start")
                await self.trigger_user_turn_started()
        elif isinstance(frame, TranscriptionFrame):
            await self._check_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame) and self._use_interim:
            await self._check_transcription(frame)

    async def _check_transcription(self, frame):
        word_count = len(frame.text.split())
        min_words = self._min_words if self._bot_speaking else 1
        is_interim = isinstance(frame, InterimTranscriptionFrame)
        should_trigger = word_count >= min_words

        logger.debug(
            f"{self} should_trigger={should_trigger} words={word_count} "
            f"min={min_words} bot_speaking={self._bot_speaking} interim={is_interim}"
        )

        if should_trigger:
            await self.trigger_user_turn_started()
