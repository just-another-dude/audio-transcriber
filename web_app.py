#!/usr/bin/env python3
"""
Web interface for Audio Transcriber using Gradio.

Provides an easy-to-use web UI for transcribing audio files.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from transcribe import Transcriber, TranscriptionResult, WHISPER_AVAILABLE, GOOGLE_SR_AVAILABLE, VOSK_AVAILABLE


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriberWebApp:
    """Web application for audio transcription."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize web app.

        Args:
            config_path: Path to configuration file
        """
        try:
            self.transcriber = Transcriber(config_path)
            logger.info("Transcriber initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transcriber: {e}")
            raise

        # Available engines
        self.available_engines = []
        if WHISPER_AVAILABLE:
            self.available_engines.append("whisper")
        if GOOGLE_SR_AVAILABLE:
            self.available_engines.append("google")
        if VOSK_AVAILABLE:
            self.available_engines.append("vosk")

        if not self.available_engines:
            raise RuntimeError(
                "No transcription engines available. "
                "Install at least one: openai-whisper, SpeechRecognition, or vosk"
            )

        logger.info(f"Available engines: {', '.join(self.available_engines)}")

    def transcribe_audio(
        self,
        audio_file,
        engine: str,
        model: str,
        language: str,
        output_format: str,
        task: str = "transcribe"
    ) -> Tuple[str, str, str, Optional[str]]:
        """Transcribe audio file through Gradio interface.

        Args:
            audio_file: Uploaded audio file from Gradio
            engine: Transcription engine to use
            model: Model size (for Whisper)
            language: Language code or "auto"
            output_format: Output format
            task: Task type (transcribe or translate)

        Returns:
            Tuple of (status_message, transcription_text, download_file_path, metadata)
        """
        if audio_file is None:
            return "❌ Please upload an audio file", "", None, ""

        try:
            # Get file path
            audio_path = Path(audio_file.name if hasattr(audio_file, 'name') else audio_file)

            logger.info(f"Processing: {audio_path.name}")

            # Update config based on user selection
            if engine == "whisper":
                self.transcriber.config['whisper']['model'] = model
                self.transcriber.config['whisper']['task'] = task

                # Clear cached model if model changed
                if 'whisper' in self.transcriber.engines:
                    current_model = self.transcriber.engines['whisper'].model_name
                    if current_model != model:
                        logger.info(f"Model changed from {current_model} to {model}, clearing cache")
                        self.transcriber.engines.pop('whisper')

            # Transcribe
            result = self.transcriber.transcribe_file(
                audio_path,
                engine=engine,
                language=language if language != "auto" else None
            )

            # Format output based on selected format
            if output_format == "txt":
                output_content = result.text
            elif output_format == "json":
                output_content = result.to_json()
            elif output_format == "srt":
                output_content = result.to_srt()
            elif output_format == "vtt":
                output_content = result.to_vtt()
            else:
                output_content = result.text

            # Save to temporary file for download
            output_filename = f"{audio_path.stem}_transcription.{output_format}"
            output_path = Path(f"/tmp/{output_filename}")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output_content)

            # Create metadata
            metadata_lines = [
                f"**Engine:** {result.engine}",
                f"**Model:** {result.model}" if result.model else "",
                f"**Language:** {result.language}" if result.language else "",
                f"**Processing Time:** {result.duration:.2f}s" if result.duration else "",
                f"**Segments:** {len(result.segments)}" if result.segments else "",
            ]
            metadata = "\n".join([line for line in metadata_lines if line])

            status = f"✅ Transcription completed successfully using {engine}"

            return status, output_content, str(output_path), metadata

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            logger.error(f"Transcription failed: {e}", exc_info=True)
            return error_msg, "", None, ""

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="Audio Transcriber",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown(
                """
                # 🎙️ Audio Transcriber

                Transcribe audio files to text using multiple AI engines.
                Supports MP3, WAV, M4A, FLAC, OGG, WMA, AAC, OPUS, WebM, and MP4 formats.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📁 Input")
                    audio_input = gr.Audio(
                        label="Upload Audio File",
                        type="filepath",
                        sources=["upload", "microphone"]
                    )

                    # Engine selection
                    engine_choices = self.available_engines
                    default_engine = self.available_engines[0]

                    engine = gr.Radio(
                        choices=engine_choices,
                        value=default_engine,
                        label="Transcription Engine",
                        info="Choose the AI engine for transcription"
                    )

                    # Model selection (for Whisper)
                    model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        value="base",
                        label="Whisper Model",
                        info="Larger models are more accurate but slower",
                        visible=WHISPER_AVAILABLE
                    )

                    # Language selection
                    language = gr.Dropdown(
                        choices=[
                            "auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
                            "ar", "hi", "nl", "pl", "tr", "vi", "id", "th", "uk", "ro"
                        ],
                        value="auto",
                        label="Language",
                        info="Select language or 'auto' for detection"
                    )

                    # Task selection
                    task = gr.Radio(
                        choices=["transcribe", "translate"],
                        value="transcribe",
                        label="Task",
                        info="Translate converts to English",
                        visible=WHISPER_AVAILABLE
                    )

                    # Output format
                    output_format = gr.Radio(
                        choices=["txt", "json", "srt", "vtt"],
                        value="txt",
                        label="Output Format",
                        info="Choose output format"
                    )

                    # Transcribe button
                    transcribe_btn = gr.Button(
                        "🎯 Transcribe",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=1):
                    # Output section
                    gr.Markdown("### 📝 Results")

                    status_output = gr.Markdown(
                        label="Status",
                        value="Ready to transcribe..."
                    )

                    transcription_output = gr.Textbox(
                        label="Transcription",
                        lines=15,
                        max_lines=20,
                        placeholder="Transcription will appear here...",
                        show_copy_button=True
                    )

                    download_output = gr.File(
                        label="Download Transcription",
                        visible=True
                    )

                    metadata_output = gr.Markdown(
                        label="Metadata"
                    )

            # Examples section
            gr.Markdown("### 💡 Tips")
            gr.Markdown(
                """
                - **Whisper** is the most accurate engine and supports many languages
                - **Google** requires internet connection and works best for clear speech
                - **Vosk** works offline but requires downloading models separately
                - Larger models (medium, large) are slower but more accurate
                - SRT and VTT formats include timestamps for subtitles
                - For best results, use clear audio with minimal background noise
                """
            )

            # Engine visibility controls
            def update_visibility(engine_choice):
                """Update visibility of model and task options based on engine."""
                is_whisper = engine_choice == "whisper"
                return gr.update(visible=is_whisper), gr.update(visible=is_whisper)

            engine.change(
                fn=update_visibility,
                inputs=[engine],
                outputs=[model, task]
            )

            # Connect transcribe button
            transcribe_btn.click(
                fn=self.transcribe_audio,
                inputs=[
                    audio_input,
                    engine,
                    model,
                    language,
                    output_format,
                    task
                ],
                outputs=[
                    status_output,
                    transcription_output,
                    download_output,
                    metadata_output
                ]
            )

            # Footer
            gr.Markdown(
                """
                ---
                **Audio Transcriber v1.0.0** | Built with [Gradio](https://gradio.app)
                | Powered by OpenAI Whisper, Google Speech, and Vosk
                """
            )

        return interface

    def launch(self, **kwargs):
        """Launch the web application.

        Args:
            **kwargs: Additional arguments for gr.Interface.launch()
        """
        interface = self.create_interface()

        # Default launch arguments
        launch_args = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'show_error': True,
            'quiet': False
        }
        launch_args.update(kwargs)

        logger.info(f"Launching web app on {launch_args['server_name']}:{launch_args['server_port']}")

        try:
            interface.launch(**launch_args)
        except Exception as e:
            logger.error(f"Failed to launch web app: {e}")
            raise


def main():
    """Main entry point for web app."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Audio Transcriber Web Interface'
    )

    parser.add_argument(
        '-c', '--config',
        help='Configuration file path (default: config.yaml)'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=7860,
        help='Port to run the server on (default: 7860)'
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public link for sharing'
    )

    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind to (default: 0.0.0.0)'
    )

    args = parser.parse_args()

    try:
        # Initialize and launch app
        app = TranscriberWebApp(config_path=args.config)
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
    except KeyboardInterrupt:
        logger.info("Web app stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start web app: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
