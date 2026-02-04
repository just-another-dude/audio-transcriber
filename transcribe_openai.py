#!/usr/bin/env python3
"""
Simple script to transcribe audio using OpenAI's Whisper API.
Doesn't require installing PyTorch or local Whisper models.
"""

import os
import sys
import argparse
from pathlib import Path
from openai import OpenAI


def transcribe_with_openai_api(
    audio_file: str,
    output_format: str = "text",
    language: str = None,
    api_key: str = None
):
    """
    Transcribe audio using OpenAI's Whisper API.

    Args:
        audio_file: Path to audio file
        output_format: Output format (text, srt, vtt, or verbose_json)
        language: Language code (e.g., 'en', 'es') or None for auto-detection
        api_key: OpenAI API key (or set OPENAI_API_KEY environment variable)

    Returns:
        Transcription result
    """
    # Initialize OpenAI client
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        # Will use OPENAI_API_KEY environment variable
        client = OpenAI()

    audio_path = Path(audio_file)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    print(f"Transcribing: {audio_path.name}")
    print(f"File size: {audio_path.stat().st_size / 1_000_000:.2f} MB")

    # Check file size (OpenAI has 25 MB limit)
    file_size_mb = audio_path.stat().st_size / 1_000_000
    if file_size_mb > 25:
        print(f"\nWARNING: File is {file_size_mb:.2f} MB, but OpenAI's limit is 25 MB")
        print("You may need to compress or split the audio file.")
        sys.exit(1)

    # Prepare request parameters
    params = {
        "model": "whisper-1",
        "response_format": output_format,
    }

    if language:
        params["language"] = language

    # Transcribe
    print(f"\nSending to OpenAI Whisper API...")
    print(f"Format: {output_format}")
    if language:
        print(f"Language: {language}")

    with open(audio_path, "rb") as audio:
        response = client.audio.transcriptions.create(
            file=audio,
            **params
        )

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using OpenAI's Whisper API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transcription (requires OPENAI_API_KEY env var)
  python transcribe_openai.py audio.m4a

  # With API key as argument
  python transcribe_openai.py audio.m4a --api-key sk-...

  # Generate SRT subtitles
  python transcribe_openai.py audio.m4a --format srt -o output.srt

  # Specify language
  python transcribe_openai.py audio.m4a --language es

Pricing:
  $0.006 per minute (e.g., a 30-minute file costs ~$0.18)

API Key:
  Set OPENAI_API_KEY environment variable or use --api-key flag
  Get your key from: https://platform.openai.com/api-keys
        """
    )

    parser.add_argument(
        "audio_file",
        help="Path to audio file (MP3, WAV, M4A, etc.)"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "srt", "vtt", "verbose_json"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "-l", "--language",
        help="Language code (e.g., en, es, fr). Auto-detected if not specified."
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file path. If not specified, prints to stdout."
    )

    parser.add_argument(
        "--api-key",
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )

    args = parser.parse_args()

    # Check for API key
    if not args.api_key and not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key required!")
        print("\nEither:")
        print("  1. Set environment variable: export OPENAI_API_KEY='sk-...'")
        print("  2. Use --api-key flag: python transcribe_openai.py audio.m4a --api-key sk-...")
        print("\nGet your API key from: https://platform.openai.com/api-keys")
        sys.exit(1)

    try:
        # Transcribe
        result = transcribe_with_openai_api(
            audio_file=args.audio_file,
            output_format=args.format,
            language=args.language,
            api_key=args.api_key
        )

        # Handle output
        if args.format == "verbose_json":
            import json
            output = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
        else:
            output = result

        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)

            print(f"\n✅ Transcription saved to: {output_path}")
        else:
            # Print to stdout
            print("\n" + "="*80)
            print("TRANSCRIPTION RESULT")
            print("="*80)
            print(output)
            print("="*80)

        print("\n✅ Transcription completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
