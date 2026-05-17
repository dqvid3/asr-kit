"""Transcribe WAV files with Cohere-Transcribe."""

import argparse

from asr_kit import Transcriber


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio with Cohere-Transcribe.")
    parser.add_argument("files", nargs="+", help="WAV file path(s)")
    parser.add_argument("--language", required=True, help="Language code (e.g. 'en', 'it', 'fr')")
    parser.add_argument("--device", default="auto", help="Torch device (auto / cuda / mps / cpu)")
    args = parser.parse_args()

    results = Transcriber(model="cohere", device=args.device).transcribe(
        args.files,
        language=args.language,
    )
    for result in results if isinstance(results, list) else [results]:
        print(f"\n--- {result.audio_path} ---")
        print(result.text)


if __name__ == "__main__":
    main()
