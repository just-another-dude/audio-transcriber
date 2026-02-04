#!/usr/bin/env python3
"""
Simple, intuitive web UI for audio transcription using OpenAI's Whisper API.
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('/tmp/transcriber.log')  # Persistent log file
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 24  # OpenAI limit is 25MB, use 24 to be safe
SUPPORTED_FORMATS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm', '.mp4', '.mpeg', '.mpga', '.oga', '.opus']


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return Path(file_path).stat().st_size / (1024 * 1024)


def compress_audio(input_path: str, target_size_mb: float = 20) -> str:
    """Compress audio file to target size using FFmpeg. Extracts audio only from video files."""
    input_path = Path(input_path)
    input_size = get_file_size_mb(str(input_path))
    duration = get_audio_duration(str(input_path))

    logger.info(f"Compression requested: {input_path.name}")
    logger.info(f"  Input size: {input_size:.2f}MB, Duration: {duration:.1f}s, Target: {target_size_mb}MB")

    if duration <= 0:
        logger.error("Could not determine audio duration")
        raise ValueError("Could not determine audio duration")

    # Calculate target bitrate (in kbps) to achieve target size
    # target_size (MB) * 8 (bits/byte) * 1024 (KB) / duration (s) = bitrate (kbps)
    raw_bitrate = (target_size_mb * 8 * 1024) / duration * 0.8  # 80% to be safe
    target_bitrate = int(max(32, min(raw_bitrate, 128)))  # Clamp 32-128kbps for speech

    logger.info(f"  Calculated bitrate: {raw_bitrate:.1f}kbps -> using {target_bitrate}kbps")

    # Create temp file for compressed output
    temp_file = tempfile.NamedTemporaryFile(suffix='.m4a', delete=False)
    output_path = temp_file.name
    temp_file.close()

    # Compress with FFmpeg - use -vn to strip video track (critical for MP4 files!)
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-vn',  # No video - extract audio only!
        '-b:a', f'{target_bitrate}k',
        '-ac', '1',  # Mono
        '-ar', '16000',  # 16kHz sample rate
        '-y',  # Overwrite
        output_path
    ]

    logger.debug(f"  FFmpeg command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"FFmpeg failed: {result.stderr}")
        raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")

    output_size = get_file_size_mb(output_path)
    logger.info(f"  Output size: {output_size:.2f}MB (compression ratio: {input_size/output_size:.1f}x)")

    # If still too large, try again with lower bitrate
    if output_size > 24:
        logger.warning(f"  Output still too large ({output_size:.2f}MB), retrying with lower bitrate")
        retry_bitrate = int(target_bitrate * (22 / output_size))  # Scale down to hit 22MB
        retry_bitrate = max(24, retry_bitrate)  # Minimum 24kbps

        logger.info(f"  Retry bitrate: {retry_bitrate}kbps")

        # Build new command with retry bitrate
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-vn',
            '-b:a', f'{retry_bitrate}k',
            '-ac', '1',
            '-ar', '16000',
            '-y',
            output_path
        ]

        logger.debug(f"  Retry FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg retry failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg compression failed: {result.stderr}")

        output_size = get_file_size_mb(output_path)
        logger.info(f"  Retry output size: {output_size:.2f}MB")

    return output_path


def get_audio_duration(file_path: str) -> float:
    """Get audio duration in seconds using FFprobe."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(result.stdout.strip())
        logger.debug(f"Duration for {Path(file_path).name}: {duration:.1f}s")
        return duration
    except ValueError:
        logger.warning(f"Could not get duration for {file_path}: {result.stderr}")
        return 0


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def transcribe(
    audio_file,
    output_format: str = "text",
    language: str = "auto",
    progress=gr.Progress()
) -> tuple:
    """
    Transcribe audio file using OpenAI Whisper API.

    Returns: (status, transcription, output_file, info)
    """
    if audio_file is None:
        return "Please upload an audio file", "", None, ""

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not found. Set OPENAI_API_KEY in .env file", "", None, ""

    try:
        # Handle gr.File input (returns path string or file object)
        if hasattr(audio_file, 'name'):
            audio_file = audio_file.name
        audio_path = Path(audio_file)
        file_size_mb = get_file_size_mb(str(audio_path))
        duration = get_audio_duration(str(audio_path))

        logger.info(f"=== New transcription request ===")
        logger.info(f"File: {audio_path.name}")
        logger.info(f"Size: {file_size_mb:.2f}MB, Duration: {duration:.1f}s")
        logger.info(f"Format: {output_format}, Language: {language}")

        progress(0.1, desc="Checking file...")

        # Check file format
        if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported format: {audio_path.suffix}")
            return f"Unsupported format: {audio_path.suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}", "", None, ""

        # Compress if needed
        transcribe_path = str(audio_path)
        compressed = False

        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.info(f"File exceeds {MAX_FILE_SIZE_MB}MB limit, compressing...")
            progress(0.2, desc=f"Compressing ({file_size_mb:.1f}MB > 24MB limit)...")
            transcribe_path = compress_audio(str(audio_path))
            compressed = True
            new_size = get_file_size_mb(transcribe_path)
            logger.info(f"Compression complete: {new_size:.2f}MB")
            progress(0.4, desc=f"Compressed to {new_size:.1f}MB")

        # CRITICAL: Verify file size before sending to API
        final_size_mb = get_file_size_mb(transcribe_path)
        final_size_bytes = Path(transcribe_path).stat().st_size
        logger.info(f"Final file to send: {final_size_mb:.2f}MB ({final_size_bytes} bytes)")

        if final_size_bytes > 26214400:  # OpenAI's exact limit
            logger.error(f"File still too large: {final_size_bytes} bytes > 26214400 limit")
            if compressed:
                Path(transcribe_path).unlink()
            return f"Error: File too large ({final_size_mb:.2f}MB). OpenAI limit is 25MB.", "", None, ""

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Prepare API parameters
        api_format = "text" if output_format == "txt" else output_format
        params = {
            "model": "whisper-1",
            "response_format": api_format if api_format in ["text", "srt", "vtt", "verbose_json"] else "text",
        }

        if language != "auto":
            params["language"] = language

        logger.info(f"Sending to OpenAI API...")
        progress(0.5, desc="Transcribing with OpenAI Whisper...")

        # Transcribe
        with open(transcribe_path, "rb") as audio:
            response = client.audio.transcriptions.create(file=audio, **params)

        logger.info("Transcription successful!")

        progress(0.9, desc="Processing results...")

        # Handle response
        if api_format == "verbose_json":
            import json
            transcription = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
        else:
            transcription = response if isinstance(response, str) else response.text

        # Clean up compressed file
        if compressed and Path(transcribe_path).exists():
            Path(transcribe_path).unlink()

        # Save output file
        output_suffix = output_format if output_format != "verbose_json" else "json"
        output_file = tempfile.NamedTemporaryFile(
            suffix=f".{output_suffix}",
            prefix=f"{audio_path.stem}_",
            delete=False,
            mode='w',
            encoding='utf-8'
        )
        output_file.write(transcription)
        output_file.close()

        # Calculate cost estimate
        cost = (duration / 60) * 0.006  # $0.006 per minute

        # Build info string
        info_parts = [
            f"Duration: {format_duration(duration)}",
            f"Original size: {file_size_mb:.1f}MB",
        ]
        if compressed:
            info_parts.append("(compressed for API)")
        info_parts.append(f"Est. cost: ${cost:.3f}")

        progress(1.0, desc="Done!")

        return (
            f"Transcription complete!",
            transcription,
            output_file.name,
            " | ".join(info_parts)
        )

    except Exception as e:
        logger.exception(f"Transcription failed: {e}")
        return f"Error: {str(e)}", "", None, ""


def create_app():
    """Create and return the Gradio app."""

    with gr.Blocks() as app:

        gr.Markdown("# Audio Transcriber")
        gr.Markdown("Transcribe audio files to text using OpenAI Whisper. Supports MP3, WAV, M4A, FLAC, OGG, and more. Large files are automatically compressed.")

        with gr.Row():
            with gr.Column(scale=1):
                # Input - use File component to accept all formats including MP4
                audio_input = gr.File(
                    label="Upload Audio/Video File",
                    file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".mpeg", ".mpga", ".oga", ".opus", ".wma", ".aac"],
                )

                with gr.Row():
                    output_format = gr.Dropdown(
                        choices=[
                            ("Plain Text", "txt"),
                            ("SRT Subtitles", "srt"),
                            ("VTT Subtitles", "vtt"),
                            ("JSON (detailed)", "verbose_json"),
                        ],
                        value="txt",
                        label="Output Format",
                        scale=1,
                    )

                    language = gr.Dropdown(
                        choices=[
                            ("Auto-detect", "auto"),
                            ("English", "en"),
                            ("Hebrew", "he"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("German", "de"),
                            ("Italian", "it"),
                            ("Portuguese", "pt"),
                            ("Dutch", "nl"),
                            ("Russian", "ru"),
                            ("Japanese", "ja"),
                            ("Korean", "ko"),
                            ("Chinese", "zh"),
                            ("Arabic", "ar"),
                            ("Hindi", "hi"),
                        ],
                        value="auto",
                        label="Language",
                        scale=1,
                    )

                transcribe_btn = gr.Button(
                    "Transcribe",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # Output
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

                transcription = gr.Textbox(
                    label="Transcription",
                    lines=12,
                    max_lines=20,
                    placeholder="Your transcription will appear here...",
                )

                with gr.Row():
                    download = gr.File(
                        label="Download",
                        scale=2,
                    )
                    info = gr.Textbox(
                        label="Info",
                        interactive=False,
                        scale=1,
                    )

        # Connect
        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, output_format, language],
            outputs=[status, transcription, download, info],
        )

        # Footer
        gr.Markdown("---\n*Powered by OpenAI Whisper API | $0.006/minute*")

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio Transcriber Web App")
    parser.add_argument("-p", "--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
