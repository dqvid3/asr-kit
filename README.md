# ASR-Kit

Modular Python library for Automatic Speech Recognition. Pass WAV(s) to a unified `Transcriber`; get `TranscriptionResult`(s) back. Backed by pluggable model drivers.

**Supported:** Qwen3-ASR | Cohere-Transcribe | NVIDIA Parakeet | Whisper (planned)

## Setup

```bash
# Use one isolated environment per backend model.
# Create one env per model key (e.g. asr-cohere, asr-qwen, asr-parakeet).

MODEL_KEY=cohere
conda create -n asr-${MODEL_KEY} python=3.12
conda activate asr-${MODEL_KEY}

# NVIDIA GPU: install a CUDA-enabled PyTorch build first.
# Use a runtime build published by PyTorch (for example cu124),
# not necessarily your exact driver version.
# Recommended (pip wheel):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install "asr-kit[${MODEL_KEY}] @ git+https://github.com/dqvid3/asr-kit.git"

# Optional for Qwen on Ampere+ GPUs:
pip install -U flash-attn --no-build-isolation
```

Available model keys right now:
- `cohere`
- `parakeet`
- `qwen`

Notes:
- The same install pattern works for new backends as they are added: set `MODEL_KEY` to the backend name.
- Keep backend models in separate environments; their dependency stacks can conflict.
- Use ISO language codes at the ASR-Kit API boundary, such as `it`, `en`, or `fr`.
  Drivers translate that to the backend-specific format when needed.

## Usage

```python
from asr_kit import Transcriber

# Initialize. Use device="mps" on Apple Silicon, or "cuda" on NVIDIA GPUs.
t = Transcriber(model="cohere", device="mps", dtype="bfloat16")

# Single file → TranscriptionResult
result = t.transcribe("audio.wav", language="en")
print(result.text)

# Multiple files → list[TranscriptionResult] with progress spinner
# Cohere uses ISO codes (e.g. "en", "it", "fr") and supports a punctuation toggle
results = t.transcribe(["a.wav", "b.wav"], language="en", punctuation=True, max_new_tokens=256)

# Qwen supports contextual biasing and timestamps when initialized with an aligner.
q = Transcriber(model="qwen", device="cuda", max_new_tokens=256)
results = q.transcribe(
    "audio.wav",
    language="it",
    context="acronyms, product names, speaker names",
    return_timestamps=True,
)

# Parakeet auto-detects language.
p = Transcriber(model="parakeet", device="cuda")
result = p.transcribe("audio.wav", return_timestamps=True)
print(result.text)
print(result.language)

# Disable progress spinner
results = t.transcribe("audio.wav", language="en", show_progress=False)
```

## Gated Models & Authentication

Some models (like `Cohere-Transcribe`) are **gated** on Hugging Face. To use them:

1.  Request access on the [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) page.
2.  Authenticate your local environment:
    *   **CLI:** Run `huggingface-cli login` and enter your token.
    *   **Code:** Pass your token directly: `Transcriber(model="cohere", token="hf_...")`
    *   **Env:** Set the `HF_TOKEN` environment variable in your `.env` file or shell.

## Output: TranscriptionResult

| field | type | description |
|---|---|---|
| `text` | `str` | Full transcript |
| `language` | `str \| None` | Detected language code |
| `timestamps` | `list[WordTimestamp] \| None` | Word-level timing (if requested) |
| `audio_path` | `str` | Absolute source WAV path |
| `model` | `str` | Model identifier |

## Timestamp Support

`return_timestamps=True` is checked before transcription:
- `qwen`: requires `Transcriber(model="qwen", use_forced_aligner=True)`
- `parakeet`: supported by the default v3 model
- `cohere`: not supported

## Adding a Driver

1. Create `src/asr_kit/drivers/my_driver.py` implementing `BaseDriver`.
2. Implement `load_model`, `batch_size`, and `transcribe`.
3. Register the key in `_DRIVER_REGISTRY` in `transcriber.py`.
4. Add optional deps in `pyproject.toml`.

## License

MIT
