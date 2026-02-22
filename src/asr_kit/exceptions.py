"""Exceptions for ASR-Kit."""


class ASRKitError(Exception):
    """Base exception for all ASR-Kit errors."""


class ModelLoadError(ASRKitError):
    """Raised when a model fails to load."""


class ModelNotLoadedError(ASRKitError):
    """Raised when transcribe() is called before load_model()."""


class UnsupportedModelError(ASRKitError):
    """Raised when an unknown model key is requested."""
