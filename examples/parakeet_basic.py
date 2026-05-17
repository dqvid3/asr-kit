"""Transcribe WAV files with NVIDIA Parakeet."""

import argparse

from asr_kit import Transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with Parakeet.")
    parser.add_argument("files", nargs="+", help="WAV file path(s)")
    parser.add_argument("--device", default="auto", help="Torch device (auto / cuda / mps / cpu)")
    parser.add_argument("--timestamps", action="store_true", help="Include word-level timestamps")
    args = parser.parse_args()

    results = Transcriber(model="parakeet", device=args.device).transcribe(
        args.files,
        return_timestamps=args.timestamps,
    )

    for result in results if isinstance(results, list) else [results]:
        print(f"\n--- {result.audio_path} ---")
        print(result.text)
        if result.language:
            print(f"Language: {result.language}")
        for word in result.timestamps or []:
            print(f"  {word.start:.2f}s - {word.end:.2f}s  {word.text}")


if __name__ == "__main__":
    main()
