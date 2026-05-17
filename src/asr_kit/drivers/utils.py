"""Shared helpers for ASR drivers."""

from collections.abc import Mapping
from typing import Any

from asr_kit.exceptions import ModelLoadError
from asr_kit.types import TranscriptionResult, WordTimestamp

_LANGUAGES: dict[str, tuple[str, str]] = {
    "bg": ("bg", "Bulgarian"),
    "ca": ("ca", "Cantonese"),
    "cs": ("cs", "Czech"),
    "da": ("da", "Danish"),
    "de": ("de", "German"),
    "el": ("el", "Greek"),
    "en": ("en", "English"),
    "es": ("es", "Spanish"),
    "et": ("et", "Estonian"),
    "fi": ("fi", "Finnish"),
    "fr": ("fr", "French"),
    "hr": ("hr", "Croatian"),
    "hu": ("hu", "Hungarian"),
    "it": ("it", "Italian"),
    "ja": ("ja", "Japanese"),
    "ko": ("ko", "Korean"),
    "lt": ("lt", "Lithuanian"),
    "lv": ("lv", "Latvian"),
    "mt": ("mt", "Maltese"),
    "nl": ("nl", "Dutch"),
    "pl": ("pl", "Polish"),
    "pt": ("pt", "Portuguese"),
    "ro": ("ro", "Romanian"),
    "ru": ("ru", "Russian"),
    "sk": ("sk", "Slovak"),
    "sl": ("sl", "Slovenian"),
    "sv": ("sv", "Swedish"),
    "uk": ("uk", "Ukrainian"),
    "zh": ("zh", "Chinese"),
}

_LANGUAGE_ALIASES = {
    code: code for code in _LANGUAGES
} | {name.lower(): code for code, (_, name) in _LANGUAGES.items()}


def missing_dependency_error(model_key: str, exc: ImportError) -> ModelLoadError:
    """Build a helpful error for optional backend dependencies."""
    return ModelLoadError(
        f"Missing dependencies for '{model_key}'. Install them with: "
        f'pip install "asr-kit[{model_key}]". Original error: {exc}'
    )


def resolve_torch_device(torch, device: str) -> str:
    """Resolve ``auto`` to the best available torch device."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_dtype(torch, dtype: str | Any | None, device: str, *, gpu_default: str = "bfloat16"):
    """Resolve user dtype strings and backend defaults to torch dtype objects."""
    if dtype is None:
        return getattr(torch, gpu_default) if device != "cpu" else torch.float32
    if isinstance(dtype, str):
        return getattr(torch, dtype) if hasattr(torch, dtype) else dtype
    return dtype


def normalize_language(language: str | list[str] | None, target: str) -> str | list[str] | None:
    """Normalize user language input for a backend.

    ASR-Kit's public convention is ISO 639-1 codes such as ``"it"``. Common
    English names are also accepted for compatibility.
    """
    if language is None:
        return None
    if isinstance(language, list):
        return [normalize_language(item, target) for item in language]

    code = _language_code(language)
    if target == "name":
        return _LANGUAGES.get(code, (code, language))[1]
    if target == "code":
        return code
    raise ValueError(f"Unknown language normalization target: {target}")


def normalized_result(
    item: Any,
    *,
    audio_path: str,
    model: str,
    language: str | None = None,
    timestamps: list[WordTimestamp] | None = None,
) -> TranscriptionResult:
    """Convert a backend item/string into ASR-Kit's result shape."""
    item_text = _get_value(item, "text", item)
    item_language = language if language is not None else _get_value(item, "language")

    return TranscriptionResult(
        text=str(item_text).strip(),
        audio_path=audio_path,
        model=model,
        language=item_language,
        timestamps=timestamps,
    )


def extract_word_timestamps(item: Any) -> list[WordTimestamp] | None:
    """Extract common word timestamp shapes from backend output."""
    raw_words = _get_value(item, "time_stamps") or _get_value(item, "timestamps")
    timestamp = _get_value(item, "timestamp")
    if not raw_words and isinstance(timestamp, Mapping):
        raw_words = timestamp.get("word")
    if not raw_words:
        return None

    parsed: list[WordTimestamp] = []
    for word in raw_words:
        text = _first_present(_get_value(word, "text"), _get_value(word, "word"))
        start = _first_present(_get_value(word, "start"), _get_value(word, "start_time"))
        end = _first_present(_get_value(word, "end"), _get_value(word, "end_time"))
        if text is None or start is None or end is None:
            continue
        parsed.append(WordTimestamp(text=str(text), start=float(start), end=float(end)))

    return parsed or None


def _get_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _language_code(language: str) -> str:
    key = language.strip().lower().replace("_", "-")
    base_key = key.split("-", maxsplit=1)[0]
    return _LANGUAGE_ALIASES.get(key) or _LANGUAGE_ALIASES.get(base_key) or base_key


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
