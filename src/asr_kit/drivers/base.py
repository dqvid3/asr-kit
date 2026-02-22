"""Abstract base class for ASR-Kit drivers."""

from abc import ABC, abstractmethod

from asr_kit.types import TranscriptionResult


class BaseDriver(ABC):
    """Interface every ASR driver must implement.

    Call load_model() before transcribe(). Raises ModelNotLoadedError otherwise.
    """

    @property
    def supports_timestamps(self) -> bool:
        """Whether this driver supports word-level timestamps.

        Override in drivers that support timestamps. May depend on runtime state
        (e.g. whether an aligner was loaded).
        """
        return False

    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load and cache the model.

        Args:
            **kwargs: Driver-specific model configuration (device, dtype, etc.).

        Raises:
            ModelLoadError: If the model fails to load.
        """

    @abstractmethod
    def transcribe(self, audio_paths: list[str], **kwargs) -> list[TranscriptionResult]:
        """Transcribe one or more WAV files.

        Args:
            audio_paths: Absolute paths to WAV files.
            **kwargs: Driver-specific transcription options (language, return_timestamps, etc.).

        Returns:
            One TranscriptionResult per input path, in the same order.

        Raises:
            ModelNotLoadedError: If load_model() has not been called.
        """