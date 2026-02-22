"""Basic Qwen3-ASR transcription example.

Usage:
    python examples/qwen_basic.py audio.wav
    python examples/qwen_basic.py a.wav b.wav --language English
    python examples/qwen_basic.py audio.wav --timestamps
"""

import argparse
import sys

from asr_kit import Transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe WAV files with Qwen3-ASR.")
    parser.add_argument("files", nargs="+", help="WAV file path(s)")
    parser.add_argument(
        "--language", default=None, help="Language name (e.g. 'English', 'Italian'). Default: auto-detect."
    )
    parser.add_argument("--timestamps", action="store_true", help="Include word-level timestamps")
    parser.add_argument("--device", default="auto", help="Torch device (auto / cuda / mps / cpu)")
    args = parser.parse_args()

    load_kwargs = {"device": args.device}
    if args.timestamps:
        load_kwargs["use_forced_aligner"] = True

    transcriber = Transcriber(model="qwen", **load_kwargs)
    results = transcriber.transcribe(args.files, language=args.language, return_timestamps=args.timestamps)

    if not isinstance(results, list):
        results = [results]

    for r in results:
        print(f"[{r.audio_path}]")
        print(r.text)
        if r.timestamps:
            for w in r.timestamps:
                print(f"  {w.start:.2f}s - {w.end:.2f}s  {w.text}")
        print()


if __name__ == "__main__":
    sys.exit(main())