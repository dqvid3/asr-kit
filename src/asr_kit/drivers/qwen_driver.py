"""Qwen3-ASR driver for ASR-Kit."""

import os

from asr_kit.drivers.base import BaseDriver
from asr_kit.exceptions import ModelLoadError, ModelNotLoadedError
from asr_kit.types import TranscriptionResult, WordTimestamp

_DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
_DEFAULT_ALIGNER_ID = "Qwen/Qwen3-ForcedAligner-0.6B"


class QwenDriver(BaseDriver):
    """Driver for the Qwen3-ASR model (transformers backend).

    Supports Qwen/Qwen3-ASR-1.7B and Qwen/Qwen3-ASR-0.6B.
    Word-level timestamps require use_forced_aligner=True at load time and
    a language supported by the aligner (English, Chinese, Cantonese, French, German,
    Italian, Japanese, Korean, Portuguese, Russian, Spanish).

    Example:
        driver = QwenDriver()
        driver.load_model(device="cuda", use_forced_aligner=True)
        results = driver.transcribe(["audio.wav"], language="English", return_timestamps=True)
    """

    def __init__(self) -> None:
        self._model = None
        self._model_id: str = ""
        self._aligner_loaded = False

    @property
    def supports_timestamps(self) -> bool:
        return self._aligner_loaded

    def load_model(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
        use_forced_aligner: bool = False,
        aligner_model_id: str = _DEFAULT_ALIGNER_ID,
        max_inference_batch_size: int = 8,
        **kwargs,
    ) -> None:
        """Load and cache the Qwen3-ASR model.

        Args:
            model_id: HuggingFace model ID. Options: "Qwen/Qwen3-ASR-1.7B" (default),
                "Qwen/Qwen3-ASR-0.6B".
            device: Torch device map string (e.g. "cuda", "cuda:0", "cpu", "mps"). "auto" picks
                cuda → mps → cpu.
            use_forced_aligner: Load the forced aligner for word-level timestamps.
                Aligner supports 11 languages: English, Chinese, Cantonese, French, German,
                Italian, Japanese, Korean, Portuguese, Russian, Spanish.
            aligner_model_id: Forced aligner model ID (default: Qwen/Qwen3-ForcedAligner-0.6B).
            max_inference_batch_size: Batch size for inference.
            **kwargs: Passed through to Qwen3ASRModel.from_pretrained().

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        try:
            import torch  # type: ignore[import]
            from qwen_asr import Qwen3ASRModel  # type: ignore[import]

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            dtype = torch.bfloat16 if device != "cpu" else torch.float32

            load_kwargs: dict = dict(
                dtype=dtype,
                device_map=device,
                max_inference_batch_size=max_inference_batch_size,
                **kwargs,
            )
            if use_forced_aligner:
                load_kwargs["forced_aligner"] = aligner_model_id
                load_kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device)

            self._model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)
            self._model_id = model_id
            self._aligner_loaded = use_forced_aligner
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {model_id}: {exc}") from exc

    def transcribe(
        self,
        audio_paths: list[str],
        *,
        language: str | list[str] | None = None,
        return_timestamps: bool = False,
        **kwargs,
    ) -> list[TranscriptionResult]:
        """Transcribe one or more WAV files with Qwen3-ASR.

        Args:
            audio_paths: Absolute paths to WAV files.
            language: Language name(s) as expected by Qwen (e.g. "English", "Italian").
                Pass a list to set a different language per file. None for auto-detect.
            return_timestamps: Include word-level timestamps. Requires use_forced_aligner=True
                at load_model() time and a supported language.
            **kwargs: Passed through to model.transcribe().

        Returns:
            One TranscriptionResult per input path, in the same order.

        Raises:
            ModelNotLoadedError: If load_model() has not been called.
            FileNotFoundError: If any audio path does not exist.
        """
        if self._model is None:
            raise ModelNotLoadedError("Call load_model() before transcribe().")

        missing = [p for p in audio_paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"Audio file(s) not found: {missing}")

        raw = self._model.transcribe(
            audio=audio_paths,
            language=language,
            return_time_stamps=return_timestamps,
            **kwargs,
        )

        results: list[TranscriptionResult] = []
        for path, item in zip(audio_paths, raw):
            timestamps: list[WordTimestamp] | None = None
            if return_timestamps and item.time_stamps:
                timestamps = [
                    WordTimestamp(text=ts.text, start=ts.start_time, end=ts.end_time)
                    for ts in item.time_stamps
                ]
            results.append(
                TranscriptionResult(
                    text=item.text,
                    audio_path=path,
                    model=self._model_id,
                    language=item.language,
                    timestamps=timestamps,
                )
            )

        return results