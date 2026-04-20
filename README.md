# ASR-Kit

Modular Python library for Automatic Speech Recognition. Pass WAV(s) to a unified `Transcriber`; get `TranscriptionResult`(s) back. Backed by pluggable model drivers.

**Supported:** Qwen3-ASR | Cohere-Transcribe | Whisper (planned) | Parakeet (planned)

## Setup

```bash
# Recommended: use one isolated environment per backend model for reproducibility.
# Create one env per model key (e.g. asr-cohere, asr-qwen).

MODEL_KEY=cohere
conda create -n asr-${MODEL_KEY} python=3.12
conda activate asr-${MODEL_KEY}

# Optional (NVIDIA GPU): install a CUDA-enabled PyTorch build first.
# Use a runtime build published by PyTorch (for example cu124),
# not necessarily your exact driver version.
# Recommended (pip wheel):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install "asr-kit[${MODEL_KEY}] @ git+https://github.com/dqvid3/asr-kit.git"

# Optional (Qwen only, Ampere+ GPUs):
pip install -U flash-attn --no-build-isolation
```

Available model keys right now:
- `cohere`
- `qwen`

Notes:
- The same install pattern works for new backends as they are added: set `MODEL_KEY` to the backend name.
- If you switch backend models, prefer a separate environment per model key.
- Verify GPU setup with: `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.

## Usage

```python
from asr_kit import Transcriber

# Initialize (device="mps" for Mac M1/M2/M3)
# Optional: dtype can be "float16", "bfloat16" (default on GPU), or "float32"
t = Transcriber(model="cohere", device="mps", dtype="bfloat16")

# Single file → TranscriptionResult
result = t.transcribe("audio.wav", language="en")
print(result.text)

# Multiple files → list[TranscriptionResult] with progress spinner
# Cohere uses ISO codes (e.g. "en", "it", "fr") and supports a punctuation toggle
results = t.transcribe(["a.wav", "b.wav"], language="en", punctuation=True, max_new_tokens=256)

# Qwen supports contextual biasing (context) to help with specific jargon/names
# and word-level timestamps (if initialized with use_forced_aligner=True)
q = Transcriber(model="qwen", device="cuda", max_new_tokens=256)
results = q.transcribe(
    "audio.wav",
    context="acronyms, product names, speaker names",
    return_timestamps=True,
)

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

## Adding a Driver

1. Create `src/asr_kit/drivers/my_driver.py` implementing `BaseDriver`.
2. Implement `load_model`, `batch_size`, and `transcribe`.
3. Register the key in `_DRIVER_REGISTRY` in `transcriber.py`.
4. Add optional deps in `pyproject.toml`.

## License

MIT
