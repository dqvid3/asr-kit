# ASR-Kit

Modular Python library for Automatic Speech Recognition. Pass WAV(s) to a unified `Transcriber`; get `TranscriptionResult`(s) back. Backed by pluggable model drivers.

**Supported:** Qwen3-ASR | Whisper (planned) | Parakeet (planned)

## Setup

```bash
# 1. Create and activate a dedicated conda environment (Python 3.12 recommended)
conda create -n asr-kit python=3.12
conda activate asr-kit

# 2. Install directly from GitHub
pip install "asr-kit[qwen] @ git+https://github.com/dqvid3/asr-kit.git"

# 3. (Optional - Qwen only) Install Flash Attention 2 for Ampere+ GPUs (RTX 3090/4090/ADA/A100)
# Significantly reduces VRAM and speeds up long audio processing when using the Qwen model.
# If your machine has less than 96GB of RAM and lots of CPU cores, run:
# MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
pip install -U flash-attn --no-build-isolation

# To remove the environment later
conda env remove -n asr-kit
```

> **Note:** Python 3.13 may have compatibility issues with torch or qwen-asr. Use 3.12 to be safe.

## Usage

```python
from asr_kit import Transcriber

t = Transcriber(model="qwen", device="cuda")

# Single file → TranscriptionResult
result = t.transcribe("audio.wav")
print(result.text)

# Multiple files → list[TranscriptionResult]
results = t.transcribe(["a.wav", "b.wav"], language="English")

# With word-level timestamps (Qwen: requires use_forced_aligner=True)
t = Transcriber(model="qwen", device="cuda", use_forced_aligner=True)
result = t.transcribe("audio.wav", return_timestamps=True)
for word in result.timestamps:
    print(f"{word.start:.2f}s  {word.text}")
```

## Output: TranscriptionResult

| field | type | description |
|---|---|---|
| `text` | `str` | Full transcript |
| `language` | `str \| None` | Detected language code |
| `timestamps` | `list[WordTimestamp] \| None` | Word-level timing (if requested) |
| `audio_path` | `str` | Absolute source WAV path |
| `model` | `str` | Model identifier |

`WordTimestamp`: `text: str`, `start: float`, `end: float` (seconds)

## Adding a Driver

1. Create `src/asr_kit/drivers/my_driver.py` implementing `BaseDriver`
2. Implement `load_model(**kwargs) -> None` and `transcribe(audio_paths, **kwargs) -> list[TranscriptionResult]`
3. Register the key in `_DRIVER_REGISTRY` dict in `transcriber.py`
4. Add optional deps under `[project.optional-dependencies]` in `pyproject.toml`

## Try It

After cloning the repo:

```bash
python examples/qwen_basic.py audio.wav
python examples/qwen_basic.py audio.wav --timestamps
python examples/qwen_basic.py a.wav b.wav --language English
```

Issues and PRs welcome.

## License

MIT
