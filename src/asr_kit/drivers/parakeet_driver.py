"""NVIDIA Parakeet driver for ASR-Kit."""

from asr_kit.drivers.base import BaseDriver
from asr_kit.drivers.utils import (
    extract_word_timestamps,
    missing_dependency_error,
    normalized_result,
    resolve_torch_device,
)
from asr_kit.exceptions import ModelLoadError, ModelNotLoadedError
from asr_kit.types import TranscriptionResult

_DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetDriver(BaseDriver):
    """Driver for NVIDIA Parakeet models through NeMo.

    The default model, nvidia/parakeet-tdt-0.6b-v3, is multilingual and supports
    Italian plus other European languages with automatic language detection.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_id: str = ""
        self._batch_size: int = 4

    @property
    def batch_size(self) -> int:
        """The batch size used for inference."""
        return self._batch_size

    @property
    def supports_timestamps(self) -> bool:
        """Whether word-level timestamps are supported."""
        return True

    def load_model(
        self,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "auto",
        batch_size: int = 4,
        local_attention: bool = False,
        att_context_size: list[int] | tuple[int, int] = (256, 256),
        **kwargs,
    ) -> None:
        """Load and cache a Parakeet model.

        Args:
            model_id: Hugging Face model ID or NeMo pretrained model name.
            device: Torch device string. ``"auto"`` selects cuda, mps, then cpu.
            batch_size: Number of audio files sent to NeMo per ASR-Kit chunk.
            local_attention: Switch FastConformer attention to local attention for
                longer audio.
            att_context_size: Left/right context used when ``local_attention`` is enabled.
            **kwargs: Passed to NeMo's ``ASRModel.from_pretrained``.
        """
        try:
            import torch
            import nemo.collections.asr as nemo_asr

            device = resolve_torch_device(torch, device)

            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_id,
                **kwargs,
            )
            self._model.to(device)
            self._model.eval()

            if local_attention:
                self._model.change_attention_model(
                    self_attention_model="rel_pos_local_attn",
                    att_context_size=list(att_context_size),
                )

            self._model_id = model_id
            self._batch_size = batch_size
        except ImportError as exc:
            raise missing_dependency_error("parakeet", exc) from exc
        except Exception as exc:
            raise ModelLoadError(f"Failed to load {model_id}: {exc}") from exc

    def transcribe(
        self,
        audio_paths: list[str],
        *,
        language: str | list[str] | None = None,
        return_timestamps: bool = False,
        batch_size: int | None = None,
        **kwargs,
    ) -> list[TranscriptionResult]:
        """Transcribe audio files with NVIDIA Parakeet.

        Args:
            language: Accepted for API compatibility. Parakeet v3 auto-detects
                language, so this value is not forwarded to NeMo.
        """
        if self._model is None:
            raise ModelNotLoadedError("Call load_model() before transcribe().")

        nemo_kwargs = dict(kwargs)
        if return_timestamps:
            nemo_kwargs["timestamps"] = True
        if batch_size is not None:
            nemo_kwargs["batch_size"] = batch_size

        raw_results = self._model.transcribe(audio_paths, **nemo_kwargs)

        results: list[TranscriptionResult] = []
        for path, item in zip(audio_paths, raw_results):
            results.append(
                normalized_result(
                    item,
                    audio_path=path,
                    model=self._model_id,
                    timestamps=extract_word_timestamps(item) if return_timestamps else None,
                )
            )

        return results
