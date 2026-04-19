"""Basic example of using the Cohere-Transcribe driver."""

import argparse
import sys
from asr_kit import Transcriber


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with Cohere-Transcribe.")
    parser.add_argument("audio_paths", nargs="+", help="Path(s) to WAV file(s).")
    parser.add_argument("--language", required=True, help="Language code (e.g. 'en', 'fr').")
    parser.add_argument("--device", default="auto", help="Device to use (cuda, cpu, mps).")

    args = parser.parse_args()

    try:
        # Initialize transcriber with the 'cohere' model
        t = Transcriber(model="cohere", device=args.device)

        # Transcribe
        results = t.transcribe(args.audio_paths, language=args.language)

        # Print results
        if isinstance(results, list):
            for res in results:
                print(f"\n--- {res.audio_path} ---")
                print(res.text)
        else:
            print(f"\n--- {results.audio_path} ---")
            print(results.text)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
