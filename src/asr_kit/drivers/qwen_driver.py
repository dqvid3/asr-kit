"""Qwen3-ASR driver for ASR-Kit."""

import contextlib
import io

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
    """

    def __init__(self) -> None:
        self._model = None
        self._model_id: str = ""
        self._aligner_loaded = False
        self._batch_size: int = 4

    @property
    def batch_size(self) -> int:
        """The batch size used for inference."""
        return self._batch_size

    @property
    def supports_timestamps(self) -> bool:
        """Whether word-level timestamps are supported."""
        return self._aligner_loaded

    def load_model(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str | None = None,
        use_forced_aligner: bool = False,
        aligner_model_id: str = _DEFAULT_ALIGNER_ID,
        batch_size: int = 4,
        max_inference_batch_size: int = 8,
        max_new_tokens: int | None = None,
        use_flash_attention: bool = False,
        token: str | None = None,
        **kwargs,
    ) -> None:
        """Load and cache the Qwen3-ASR model.

        Args:
            model_id: HuggingFace model ID.
            device: Torch device map string.
            dtype: Torch dtype (e.g. 'float16', 'bfloat16', 'float32'). Defaults to bfloat16 on GPU.
            use_forced_aligner: Load aligner for timestamps.
            aligner_model_id: Aligner model ID.
            batch_size: Progress reporting chunk size.
            max_inference_batch_size: GPU batch size.
            max_new_tokens: Max tokens to generate (None uses library default).
            use_flash_attention: Use Flash Attention 2.
            token: HuggingFace API token.
            **kwargs: Passed to from_pretrained.
        """
        try:
            import torch
            import transformers
            from qwen_asr import Qwen3ASRModel

            transformers.logging.set_verbosity_error()

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            if dtype is None:
                torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
            elif isinstance(dtype, str):
                torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else dtype
            else:
                torch_dtype = dtype

            load_kwargs: dict = dict(
                dtype=torch_dtype,
                device_map=device,
                max_inference_batch_size=max_inference_batch_size,
                token=token,
                **kwargs,
            )
            
            if max_new_tokens is not None:
                load_kwargs["max_new_tokens"] = max_new_tokens
            
            if use_flash_attention:
                load_kwargs["attn_implementation"] = "flash_attention_2"

            if use_forced_aligner:
                load_kwargs["forced_aligner"] = aligner_model_id
                load_kwargs["forced_aligner_kwargs"] = dict(dtype=torch_dtype, device_map=device, token=token)

            self._model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)
            self._model_id = model_id
            self._aligner_loaded = use_forced_aligner
            self._batch_size = batch_size
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {model_id}: {exc}") from exc

    def transcribe(
        self,
        audio_paths: list[str],
        *,
        language: str | list[str] | None = None,
        context: str | list[str] = "",
        return_timestamps: bool = False,
        **kwargs,
    ) -> list[TranscriptionResult]:
        """Transcribe a chunk of audio files with Qwen3-ASR."""
        if self._model is None:
            raise ModelNotLoadedError("Call load_model() before transcribe().")

        results: list[TranscriptionResult] = []

        with contextlib.redirect_stderr(io.StringIO()):
            # Qwen's transcribe method internally handles batching and audio loading.
            raw_chunk = self._model.transcribe(
                audio=audio_paths,
                context=context,
                language=language,
                return_time_stamps=return_timestamps,
                **kwargs,
            )
        
        for path, item in zip(audio_paths, raw_chunk):
            timestamps: list[WordTimestamp] | None = None
            if return_timestamps and item.time_stamps:
                timestamps = [
                    WordTimestamp(text=ts.text, start=ts.start_time, end=ts.end_time)
                    for ts in item.time_stamps
                ]
            
            res = TranscriptionResult(
                text=item.text,
                audio_path=path,
                model=self._model_id,
                language=item.language,
                timestamps=timestamps,
            )
            results.append(res)

        return results
