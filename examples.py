#!/usr/bin/env python3
"""
Example usage scripts for Audio Transcriber.

Demonstrates various features and use cases.
"""

from pathlib import Path
from transcribe import Transcriber, TranscriptionResult


def example_1_basic_transcription():
    """Example 1: Basic transcription of a single file."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Transcription")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    # Transcribe a file
    audio_file = "sample_audio.m4a"  # Replace with your file

    print(f"Transcribing: {audio_file}")

    try:
        result = transcriber.transcribe_file(
            audio_file,
            engine="whisper",
            output_format="txt"
        )

        print("\nTranscription:")
        print(result.text)

        print(f"\nLanguage detected: {result.language}")
        print(f"Processing time: {result.duration:.2f}s")

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
        print("Please provide a valid audio file path")


def example_2_batch_processing():
    """Example 2: Batch processing multiple files."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Processing")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    # List of audio files
    audio_files = [
        "audio1.mp3",
        "audio2.m4a",
        "audio3.wav",
    ]

    print(f"Processing {len(audio_files)} files...")

    try:
        results = transcriber.transcribe_batch(
            audio_files,
            engine="whisper",
            output_format="txt"
        )

        # Display results
        for i, (file, result) in enumerate(zip(audio_files, results), 1):
            if result:
                print(f"\n{i}. {file}")
                print(f"   Text: {result.text[:100]}...")
                print(f"   Duration: {result.duration:.2f}s")
            else:
                print(f"\n{i}. {file} - FAILED")

    except Exception as e:
        print(f"Error: {e}")
        print("Please provide valid audio file paths")


def example_3_subtitle_generation():
    """Example 3: Generate subtitles in SRT format."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Subtitle Generation")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    audio_file = "video_audio.m4a"

    print(f"Generating subtitles for: {audio_file}")

    try:
        result = transcriber.transcribe_file(
            audio_file,
            engine="whisper",
            output_format="srt"
        )

        print("\nSRT Subtitles (first 500 chars):")
        print(result.to_srt()[:500])

        # Also generate VTT
        print("\nVTT Subtitles (first 500 chars):")
        print(result.to_vtt()[:500])

        # Save to files
        Path("output.srt").write_text(result.to_srt(), encoding='utf-8')
        Path("output.vtt").write_text(result.to_vtt(), encoding='utf-8')

        print("\n✅ Subtitle files saved: output.srt, output.vtt")

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"Error: {e}")


def example_4_language_detection():
    """Example 4: Language detection and translation."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Language Detection and Translation")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    audio_file = "multilingual_audio.m4a"

    print(f"Detecting language in: {audio_file}")

    try:
        # First, transcribe with auto-detect
        result = transcriber.transcribe_file(
            audio_file,
            engine="whisper",
            language=None  # Auto-detect
        )

        print(f"\nDetected Language: {result.language}")
        print(f"Original Text: {result.text}")

        # Now translate to English
        print("\nTranslating to English...")

        transcriber.config['whisper']['task'] = 'translate'

        result_translated = transcriber.transcribe_file(
            audio_file,
            engine="whisper"
        )

        print(f"Translated Text: {result_translated.text}")

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"Error: {e}")


def example_5_timestamp_extraction():
    """Example 5: Extract word-level timestamps."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Timestamp Extraction")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    audio_file = "sample_audio.m4a"

    print(f"Extracting timestamps from: {audio_file}")

    try:
        result = transcriber.transcribe_file(
            audio_file,
            engine="whisper"
        )

        if result.segments:
            print(f"\nFound {len(result.segments)} segments:\n")

            # Display first 10 segments
            for i, segment in enumerate(result.segments[:10], 1):
                start = segment.get('start', 0)
                end = segment.get('end', 0)
                text = segment.get('text', '').strip()
                confidence = segment.get('confidence', None)

                print(f"{i}. [{start:.2f}s - {end:.2f}s] {text}")
                if confidence:
                    print(f"   Confidence: {confidence:.2f}")

            if len(result.segments) > 10:
                print(f"\n... and {len(result.segments) - 10} more segments")

        else:
            print("No timestamp information available")

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"Error: {e}")


def example_6_engine_comparison():
    """Example 6: Compare results from different engines."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Engine Comparison")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    audio_file = "sample_audio.m4a"

    print(f"Comparing engines on: {audio_file}\n")

    engines = ["whisper", "google", "vosk"]
    results = {}

    for engine in engines:
        try:
            print(f"Transcribing with {engine}...")
            result = transcriber.transcribe_file(
                audio_file,
                engine=engine
            )
            results[engine] = result
            print(f"✅ {engine} completed in {result.duration:.2f}s")

        except Exception as e:
            print(f"❌ {engine} failed: {e}")
            results[engine] = None

    # Display comparison
    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80 + "\n")

    for engine, result in results.items():
        if result:
            print(f"{engine.upper()}:")
            print(f"  Text: {result.text[:100]}...")
            print(f"  Language: {result.language}")
            print(f"  Time: {result.duration:.2f}s")
            print()


def example_7_custom_configuration():
    """Example 7: Using custom configuration."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom Configuration")
    print("="*80 + "\n")

    # Create custom config
    custom_config_path = "custom_config.yaml"

    # Initialize with custom config
    try:
        transcriber = Transcriber(config_path=custom_config_path)
        print("✅ Loaded custom configuration")

        # Check current settings
        print("\nCurrent Settings:")
        print(f"  Default Engine: {transcriber.config.get('default_engine')}")
        print(f"  Whisper Model: {transcriber.config.get('whisper', {}).get('model')}")
        print(f"  Sample Rate: {transcriber.config.get('audio', {}).get('sample_rate')}")

    except FileNotFoundError:
        print(f"Config file not found: {custom_config_path}")
        print("Using default configuration")
        transcriber = Transcriber()


def example_8_json_output():
    """Example 8: Working with JSON output."""
    print("\n" + "="*80)
    print("EXAMPLE 8: JSON Output")
    print("="*80 + "\n")

    import json

    # Initialize transcriber
    transcriber = Transcriber()

    audio_file = "sample_audio.m4a"

    print(f"Transcribing to JSON: {audio_file}")

    try:
        result = transcriber.transcribe_file(
            audio_file,
            engine="whisper",
            output_format="json"
        )

        # Convert to JSON
        json_output = result.to_json()

        print("\nJSON Output:")
        print(json_output)

        # Parse and work with JSON
        data = json.loads(json_output)

        print("\nAccessing data:")
        print(f"  Text: {data['text'][:100]}...")
        print(f"  Engine: {data['engine']}")
        print(f"  Model: {data['model']}")
        print(f"  Segments: {len(data['segments']) if data['segments'] else 0}")

    except FileNotFoundError:
        print(f"File not found: {audio_file}")
        print("Please provide a valid audio file path")
    except Exception as e:
        print(f"Error: {e}")


def example_9_m4a_specific():
    """Example 9: Specific M4A file handling."""
    print("\n" + "="*80)
    print("EXAMPLE 9: M4A File Handling")
    print("="*80 + "\n")

    # Initialize transcriber
    transcriber = Transcriber()

    m4a_files = [
        "recording.m4a",
        "voice_memo.m4a",
        "podcast.m4a",
    ]

    print("Processing M4A files specifically...\n")

    for m4a_file in m4a_files:
        try:
            print(f"Processing: {m4a_file}")

            # Load audio to check format
            audio, sr = transcriber.audio_processor.load_audio(m4a_file)
            duration = transcriber.audio_processor.get_duration(audio, sr)

            print(f"  Duration: {duration:.2f}s")
            print(f"  Sample Rate: {sr} Hz")
            print(f"  Shape: {audio.shape}")

            # Transcribe
            result = transcriber.transcribe_file(
                m4a_file,
                engine="whisper",
                output_format="txt"
            )

            print(f"  Transcription: {result.text[:50]}...")
            print(f"  ✅ Success\n")

        except FileNotFoundError:
            print(f"  File not found: {m4a_file}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}\n")


def example_10_progress_monitoring():
    """Example 10: Monitor progress during batch processing."""
    print("\n" + "="*80)
    print("EXAMPLE 10: Progress Monitoring")
    print("="*80 + "\n")

    from pathlib import Path
    import glob

    # Initialize transcriber
    transcriber = Transcriber()

    # Enable progress bars
    transcriber.config['logging']['progress_bars'] = True

    # Find all audio files in a directory
    audio_dir = "audio_files"  # Replace with your directory

    try:
        audio_files = []
        for ext in transcriber.audio_processor.SUPPORTED_FORMATS:
            audio_files.extend(glob.glob(f"{audio_dir}/*.{ext}"))

        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            print("Creating sample file list for demonstration...")
            audio_files = ["sample1.m4a", "sample2.mp3", "sample3.wav"]

        print(f"Found {len(audio_files)} audio files")
        print("Starting batch transcription with progress bar...\n")

        # Batch process with progress bar
        results = transcriber.transcribe_batch(
            audio_files,
            engine="whisper",
            output_format="txt"
        )

        # Summary
        successful = sum(1 for r in results if r is not None)
        failed = len(results) - successful

        print(f"\n✅ Completed: {successful} successful, {failed} failed")

    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("AUDIO TRANSCRIBER - EXAMPLE SCRIPTS")
    print("="*80)

    examples = [
        ("Basic Transcription", example_1_basic_transcription),
        ("Batch Processing", example_2_batch_processing),
        ("Subtitle Generation", example_3_subtitle_generation),
        ("Language Detection", example_4_language_detection),
        ("Timestamp Extraction", example_5_timestamp_extraction),
        ("Engine Comparison", example_6_engine_comparison),
        ("Custom Configuration", example_7_custom_configuration),
        ("JSON Output", example_8_json_output),
        ("M4A File Handling", example_9_m4a_specific),
        ("Progress Monitoring", example_10_progress_monitoring),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nTo run a specific example, modify the main() function")
    print("or run the example function directly.\n")

    # Uncomment to run specific example
    # example_1_basic_transcription()
    # example_9_m4a_specific()

    print("Examples ready. Uncomment the example you want to run in main()")


if __name__ == '__main__':
    main()
