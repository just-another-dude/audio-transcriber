#!/usr/bin/env python3
"""
Simple, intuitive web UI for audio transcription using OpenAI's Whisper API.
"""

import os
import subprocess
import tempfile
import logging
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple
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


def _transcribe_single(
    audio_path: Path,
    output_format: str,
    language: str,
    client: OpenAI,
) -> Tuple[str, float]:
    """Transcribe a single audio file. Returns (transcription_text, duration)."""
    file_size_mb = get_file_size_mb(str(audio_path))
    duration = get_audio_duration(str(audio_path))

    logger.info(f"File: {audio_path.name}")
    logger.info(f"Size: {file_size_mb:.2f}MB, Duration: {duration:.1f}s")

    if audio_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {audio_path.suffix}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # Compress if needed
    transcribe_path = str(audio_path)
    compressed = False

    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.info(f"File exceeds {MAX_FILE_SIZE_MB}MB limit, compressing...")
        transcribe_path = compress_audio(str(audio_path))
        compressed = True
        logger.info(f"Compression complete: {get_file_size_mb(transcribe_path):.2f}MB")

    # Verify file size before sending to API
    final_size_bytes = Path(transcribe_path).stat().st_size
    if final_size_bytes > 26214400:  # OpenAI's exact limit
        if compressed:
            Path(transcribe_path).unlink()
        raise ValueError(
            f"File too large ({get_file_size_mb(transcribe_path):.2f}MB). OpenAI limit is 25MB."
        )

    # Prepare API parameters
    api_format = "text" if output_format == "txt" else output_format
    params = {
        "model": "whisper-1",
        "response_format": api_format if api_format in ["text", "srt", "vtt", "verbose_json"] else "text",
    }
    if language != "auto":
        params["language"] = language

    logger.info("Sending to OpenAI API...")

    with open(transcribe_path, "rb") as audio:
        response = client.audio.transcriptions.create(file=audio, **params)

    logger.info("Transcription successful!")

    # Clean up compressed file
    if compressed and Path(transcribe_path).exists():
        Path(transcribe_path).unlink()

    # Handle response
    if api_format == "verbose_json":
        import json
        text = json.dumps(response.model_dump(), indent=2, ensure_ascii=False)
    else:
        text = response if isinstance(response, str) else response.text

    return text, duration


def transcribe(
    audio_files,
    output_format: str = "text",
    language: str = "auto",
    progress=gr.Progress()
) -> tuple:
    """
    Transcribe one or more audio files using OpenAI Whisper API.

    Returns: (status, transcription, output_file, info)
    """
    if not audio_files:
        return "Please upload at least one audio file", "", None, ""

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OpenAI API key not found. Set OPENAI_API_KEY in .env file", "", None, ""

    try:
        client = OpenAI(api_key=api_key)

        # Normalise to list of Path objects
        paths = []
        for f in audio_files:
            paths.append(Path(f.name if hasattr(f, 'name') else f))

        logger.info(f"=== New transcription request ({len(paths)} file(s)) ===")
        logger.info(f"Format: {output_format}, Language: {language}")

        # --- Single file (original behaviour) ---
        if len(paths) == 1:
            audio_path = paths[0]
            progress(0.1, desc="Checking file...")

            text, duration = _transcribe_single(
                audio_path, output_format, language, client
            )

            progress(0.9, desc="Processing results...")

            output_suffix = output_format if output_format != "verbose_json" else "json"
            output_file = tempfile.NamedTemporaryFile(
                suffix=f".{output_suffix}",
                prefix=f"{audio_path.stem}_",
                delete=False, mode='w', encoding='utf-8'
            )
            output_file.write(text)
            output_file.close()

            file_size_mb = get_file_size_mb(str(audio_path))
            cost = (duration / 60) * 0.006
            info_parts = [
                f"Duration: {format_duration(duration)}",
                f"Original size: {file_size_mb:.1f}MB",
                f"Est. cost: ${cost:.3f}",
            ]

            progress(1.0, desc="Done!")
            return "Transcription complete!", text, output_file.name, " | ".join(info_parts)

        # --- Multiple files ---
        total = len(paths)
        succeeded = 0
        combined_texts: List[str] = []
        per_file_outputs: List[Tuple[str, str]] = []  # (filename, content)
        total_duration = 0.0
        total_size = 0.0
        errors: List[str] = []

        for idx, audio_path in enumerate(paths):
            step = (idx + 1) / (total + 1)
            progress(step, desc=f"Transcribing {audio_path.name} ({idx+1}/{total})...")
            try:
                text, duration = _transcribe_single(
                    audio_path, output_format, language, client
                )
                succeeded += 1
                total_duration += duration
                total_size += get_file_size_mb(str(audio_path))
                combined_texts.append(f"=== {audio_path.name} ===\n{text}")
                output_suffix = output_format if output_format != "verbose_json" else "json"
                per_file_outputs.append((
                    f"{audio_path.stem}.{output_suffix}", text
                ))
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path.name}: {e}")
                combined_texts.append(f"=== {audio_path.name} ===\n[failed: {e}]")
                errors.append(audio_path.name)

        display_text = "\n\n".join(combined_texts)

        # Build zip with individual output files
        zip_path = tempfile.NamedTemporaryFile(
            suffix=".zip", prefix="transcriptions_", delete=False
        ).name
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name, content in per_file_outputs:
                zf.writestr(name, content)

        cost = (total_duration / 60) * 0.006
        info_parts = [
            f"Files: {succeeded}/{total}",
            f"Total duration: {format_duration(total_duration)}",
            f"Total size: {total_size:.1f}MB",
            f"Est. cost: ${cost:.3f}",
        ]

        status = f"Transcribed {succeeded}/{total} files"
        if errors:
            status += f" (failed: {', '.join(errors)})"

        progress(1.0, desc="Done!")
        return status, display_text, zip_path, " | ".join(info_parts)

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
                    label="Upload Audio/Video File(s)",
                    file_count="multiple",
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
