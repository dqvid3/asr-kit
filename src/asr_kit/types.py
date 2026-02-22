"""Shared data types for ASR-Kit."""

from dataclasses import dataclass, field


@dataclass
class WordTimestamp:
    """Word-level timing entry.

    Attributes:
        text: The transcribed word.
        start: Start time in seconds.
        end: End time in seconds.
    """

    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Result of a single audio file transcription.

    Attributes:
        text: Full transcript string.
        audio_path: Absolute path to the source WAV file.
        model: Model identifier string (e.g. "Qwen/Qwen3-ASR").
        language: Detected language code, or None if unavailable.
        timestamps: Word-level timing, or None if not requested.
    """

    text: str
    audio_path: str
    model: str
    language: str | None = field(default=None)
    timestamps: list[WordTimestamp] | None = field(default=None)
