"""Transcriber: unified entry point for all ASR drivers."""

import importlib
import os

from asr_kit.drivers.base import BaseDriver
from asr_kit.exceptions import UnsupportedModelError
from asr_kit.types import TranscriptionResult

# Maps model key → (module path, class name). Only the requested driver is imported.
_DRIVER_REGISTRY: dict[str, tuple[str, str]] = {
    "qwen": ("asr_kit.drivers.qwen_driver", "QwenDriver"),
    # "whisper": ("asr_kit.drivers.whisper_driver", "WhisperDriver"),
    # "parakeet": ("asr_kit.drivers.parakeet_driver", "ParakeetDriver"),
}


class Transcriber:
    """Unified interface for ASR transcription.

    Wraps a model-specific driver. Call transcribe() with one or more WAV paths.

    Example:
        t = Transcriber(model="qwen")
        result = t.transcribe("audio.wav")
        results = t.transcribe(["a.wav", "b.wav"], language="English", return_timestamps=True)
    """

    def __init__(self, model: str, **load_kwargs) -> None:
        """Initialise and load the requested model driver.

        Args:
            model: Model key. Supported: "qwen".
            **load_kwargs: Passed to the driver's load_model() (device, batch_size, etc.).

        Raises:
            UnsupportedModelError: If the model key is not recognised.
            ModelLoadError: If the driver fails to load the model.
        """
        key = model.lower()
        if key not in _DRIVER_REGISTRY:
            raise UnsupportedModelError(
                f"Unknown model '{model}'. Supported: {sorted(_DRIVER_REGISTRY)}"
            )

        module_path, class_name = _DRIVER_REGISTRY[key]
        module = importlib.import_module(module_path)
        driver_cls: type[BaseDriver] = getattr(module, class_name)

        self._driver: BaseDriver = driver_cls()
        self._driver.load_model(**load_kwargs)

    def transcribe(
        self,
        audio: str | list[str],
        **kwargs,
    ) -> TranscriptionResult | list[TranscriptionResult]:
        """Transcribe one or more WAV files.

        Args:
            audio: A single WAV path or a list of WAV paths (absolute or relative).
            **kwargs: Passed to the driver (language, return_timestamps, etc.).

        Returns:
            A single TranscriptionResult when audio is a str, or a list when audio is a list.

        Raises:
            ValueError: If return_timestamps=True but the driver does not support timestamps.
            FileNotFoundError: If any path does not exist.
        """
        if kwargs.get("return_timestamps") and not self._driver.supports_timestamps:
            raise ValueError(
                "Word-level timestamps are not enabled for this driver. "
                "Re-initialize the Transcriber with timestamp support enabled "
                "(e.g. Transcriber(model='qwen', use_forced_aligner=True) for the Qwen driver)."
            )

        single = isinstance(audio, str)
        paths = [audio] if single else list(audio)
        paths = [os.path.abspath(p) for p in paths]

        results = self._driver.transcribe(paths, **kwargs)
        return results[0] if single else results