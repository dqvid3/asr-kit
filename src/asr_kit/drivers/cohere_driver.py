"""Cohere-Transcribe driver for ASR-Kit."""

import contextlib
import io

from asr_kit.drivers.base import BaseDriver
from asr_kit.drivers.utils import (
    missing_dependency_error,
    normalize_language,
    normalized_result,
    resolve_torch_device,
    resolve_torch_dtype,
)
from asr_kit.exceptions import ModelLoadError, ModelNotLoadedError
from asr_kit.types import TranscriptionResult


_DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


class CohereDriver(BaseDriver):
    """Driver for the Cohere-Transcribe model.

    Supports CohereLabs/cohere-transcribe-03-2026.
    """

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._model_id: str = ""
        self._batch_size: int = 1  # Manual generate works best with batch size 1 for now

    @property
    def batch_size(self) -> int:
        """The batch size used for inference."""
        return self._batch_size

    @property
    def supports_timestamps(self) -> bool:
        """Whether word-level timestamps are supported."""
        return False

    def load_model(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str | None = None,
        batch_size: int = 1,
        token: str | None = None,
        **kwargs,
    ) -> None:
        """Load and cache the Cohere-Transcribe model.

        Args:
            model_id: HuggingFace model ID.
            device: Torch device map string.
            dtype: Torch dtype (e.g. 'float16', 'bfloat16', 'float32'). Defaults to bfloat16 on GPU.
            batch_size: Progress reporting chunk size.
            token: HuggingFace API token for gated models.
            **kwargs: Passed to from_pretrained.
        """
        try:
            import torch
            from transformers import AutoProcessor, CohereAsrForConditionalGeneration

            device = resolve_torch_device(torch, device)
            # Force bfloat16 to avoid -1e9 overflow bug in model's float16 masking code.
            torch_dtype = resolve_torch_dtype(torch, dtype, device)

            self._model = CohereAsrForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map=device,
                token=token,
                **kwargs,
            )
            self._processor = AutoProcessor.from_pretrained(model_id, token=token)
            self._model_id = model_id
            self._batch_size = 1  # Standard generate works best with batch size 1
        except ImportError as exc:
            raise missing_dependency_error("cohere", exc) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {model_id}: {exc}") from exc

    def transcribe(
        self,
        audio_paths: list[str],
        *,
        language: str | list[str] | None = None,
        punctuation: bool = True,
        max_new_tokens: int = 256,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> list[TranscriptionResult]:
        """Transcribe audio files with the recommended 'Native Path' (generate/decode)."""
        if self._model is None:
            raise ModelNotLoadedError("Call load_model() before transcribe().")

        import librosa

        results: list[TranscriptionResult] = []

        # Load all audio files in the batch
        audios = [librosa.load(path, sr=16000)[0] for path in audio_paths]

        normalized_language = normalize_language(language, "code")
        batch_lang = normalized_language[0] if isinstance(normalized_language, list) else normalized_language

        if batch_lang is None:
            raise ValueError(
                "Cohere model requires an explicit language code (e.g. 'en', 'fr'). "
                "Automatic detection is not supported."
            )

        with contextlib.redirect_stderr(io.StringIO()):
            # Process all audio
            inputs = self._processor(
                audios,
                sampling_rate=16000,
                return_tensors="pt",
                language=batch_lang,
                punctuation=punctuation
            ).to(self._model.device, dtype=self._model.dtype)
            
            audio_chunk_index = inputs.get("audio_chunk_index")

            # Generate tokens
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

            # Decode with skip_special_tokens=True (recommended)
            transcriptions = self._processor.decode(
                outputs, 
                skip_special_tokens=skip_special_tokens,
                audio_chunk_index=audio_chunk_index,
                language=batch_lang
            )
            
            if isinstance(transcriptions, str):
                transcriptions = [transcriptions]

        for idx, (path, text) in enumerate(zip(audio_paths, transcriptions)):
            current_lang = (
                normalized_language[idx]
                if isinstance(normalized_language, list)
                else normalized_language
            )
            
            res = normalized_result(
                text,
                audio_path=path,
                model=self._model_id,
                language=current_lang,
            )
            results.append(res)

        return results
