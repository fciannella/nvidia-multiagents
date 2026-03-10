"""Custom frames for the multi-agent voice pipeline."""

from dataclasses import dataclass

from pipecat.frames.frames import ControlFrame


@dataclass
class VoiceSwitchFrame(ControlFrame):
    """Instructs the TTS service to switch to a different voice.

    Processed in order with data frames, so voice changes happen
    at the correct point between agent responses.
    """

    voice: str = ""
    speaker: str = ""
