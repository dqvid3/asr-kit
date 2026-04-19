"""Transcriber: unified entry point for all ASR drivers."""

import importlib
import os
from typing import Callable

from yaspin import yaspin

from asr_kit.drivers.base import BaseDriver
from asr_kit.exceptions import UnsupportedModelError
from asr_kit.types import TranscriptionResult

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Maps model key → (module path, class name). Only the requested driver is imported.
_DRIVER_REGISTRY: dict[str, tuple[str, str]] = {
    "qwen": ("asr_kit.drivers.qwen_driver", "QwenDriver"),
    "cohere": ("asr_kit.drivers.cohere_driver", "CohereDriver"),
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
            model: Model key. Supported: "qwen", "cohere".
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
        on_result: Callable[[TranscriptionResult], None] | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> TranscriptionResult | list[TranscriptionResult]:
        """Transcribe one or more WAV files.

        Args:
            audio: A single WAV path or a list of WAV paths (absolute or relative).
            on_result: Optional callback triggered after each individual file is transcribed.
            show_progress: Whether to show a progress spinner in the terminal.
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
        if not paths:
            raise ValueError("audio must contain at least one file path.")

        paths = [os.path.abspath(p) for p in paths]
        for path in paths:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Audio file not found: {path}")

        total = len(paths)
        batch_size = self._driver.batch_size
        all_results: list[TranscriptionResult] = []

        # Context manager for the spinner
        if show_progress:
            sp = yaspin(text=f"Transcribing [0/{total}]...")
            sp.start()
        else:
            sp = None

        try:
            for i in range(0, total, batch_size):
                chunk_paths = paths[i : i + batch_size]
                
                # Split any list-based kwargs to match the current chunk
                chunk_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, list) and len(v) == total:
                        chunk_kwargs[k] = v[i : i + batch_size]
                    else:
                        chunk_kwargs[k] = v

                chunk_results = self._driver.transcribe(chunk_paths, **chunk_kwargs)
                
                for res in chunk_results:
                    all_results.append(res)
                    if on_result:
                        on_result(res)
                
                if sp:
                    sp.text = f"Transcribing [{min(i + len(chunk_paths), total)}/{total}]..."
            
            if sp:
                sp.ok("✔")
        except Exception:
            if sp:
                sp.fail("✘")
            raise
        finally:
            if sp:
                sp.stop()

        return all_results[0] if single else all_results