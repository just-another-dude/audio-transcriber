#!/usr/bin/env python3
"""
Audio Transcriber - A comprehensive audio transcription tool

Supports multiple transcription engines (Whisper, Google, Vosk) and various audio formats.
Includes CLI interface and batch processing capabilities.
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available, falling back to pydub only")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    warnings.warn("pydub not available")

# Transcription engines
try:
    import whisper
    import torch
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    warnings.warn("Whisper not available. Install with: pip install openai-whisper")

try:
    import speech_recognition as sr
    GOOGLE_SR_AVAILABLE = True
except ImportError:
    GOOGLE_SR_AVAILABLE = False
    warnings.warn("Google Speech Recognition not available. Install with: pip install SpeechRecognition")

try:
    from vosk import Model, KaldiRecognizer
    import wave
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    warnings.warn("Vosk not available. Install with: pip install vosk")

# Configure logging
try:
    import coloredlogs
    COLOREDLOGS_AVAILABLE = True
except ImportError:
    COLOREDLOGS_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Data class to store transcription results."""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None
    engine: Optional[str] = None
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_srt(self) -> str:
        """Convert to SRT subtitle format."""
        if not self.segments:
            raise ValueError("No segments available for SRT conversion")

        srt_content = []
        for i, segment in enumerate(self.segments, 1):
            start = self._format_timestamp(segment.get('start', 0))
            end = self._format_timestamp(segment.get('end', 0))
            text = segment.get('text', '').strip()

            srt_content.append(f"{i}")
            srt_content.append(f"{start} --> {end}")
            srt_content.append(text)
            srt_content.append("")  # Empty line between subtitles

        return "\n".join(srt_content)

    def to_vtt(self) -> str:
        """Convert to VTT (WebVTT) subtitle format."""
        if not self.segments:
            raise ValueError("No segments available for VTT conversion")

        vtt_content = ["WEBVTT", ""]
        for segment in self.segments:
            start = self._format_timestamp(segment.get('start', 0), vtt=True)
            end = self._format_timestamp(segment.get('end', 0), vtt=True)
            text = segment.get('text', '').strip()

            vtt_content.append(f"{start} --> {end}")
            vtt_content.append(text)
            vtt_content.append("")  # Empty line between subtitles

        return "\n".join(vtt_content)

    @staticmethod
    def _format_timestamp(seconds: float, vtt: bool = False) -> str:
        """Format timestamp for SRT/VTT format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)

        if vtt:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class AudioProcessor:
    """Process and convert audio files for transcription."""

    SUPPORTED_FORMATS = ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac', 'opus', 'webm', 'mp4']

    def __init__(self, config: Dict[str, Any]):
        """Initialize audio processor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.audio_config = config.get('audio', {})
        self.target_sr = self.audio_config.get('sample_rate', 16000)
        self.mono = self.audio_config.get('mono', True)
        self.normalize = self.audio_config.get('normalize', True)
        self.logger = logging.getLogger(__name__)

        # Check for FFmpeg
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Check if FFmpeg is available."""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'],
                                   capture_output=True,
                                   text=True,
                                   timeout=5)
            if result.returncode != 0:
                self.logger.warning("FFmpeg may not be properly installed")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.logger.warning(
                "FFmpeg not found. Install it for full audio format support:\n"
                "  Ubuntu/Debian: sudo apt-get install ffmpeg\n"
                "  macOS: brew install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/"
            )

    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to numpy array.

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        file_ext = file_path.suffix.lower().lstrip('.')
        if file_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {file_ext}. "
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        self.logger.info(f"Loading audio file: {file_path}")

        # Try librosa first (faster and more reliable)
        if LIBROSA_AVAILABLE:
            try:
                audio, sr = librosa.load(
                    str(file_path),
                    sr=self.target_sr,
                    mono=self.mono
                )
                self.logger.debug(f"Loaded with librosa: {audio.shape}, SR: {sr}")
                return audio, sr
            except Exception as e:
                self.logger.warning(f"librosa failed, trying pydub: {e}")

        # Fallback to pydub
        if PYDUB_AVAILABLE:
            try:
                audio = AudioSegment.from_file(str(file_path))

                # Convert to mono if requested
                if self.mono and audio.channels > 1:
                    audio = audio.set_channels(1)

                # Resample if needed
                if audio.frame_rate != self.target_sr:
                    audio = audio.set_frame_rate(self.target_sr)

                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                samples = samples / (2**15)  # Normalize to [-1, 1]

                self.logger.debug(f"Loaded with pydub: {samples.shape}, SR: {audio.frame_rate}")
                return samples, audio.frame_rate
            except Exception as e:
                self.logger.error(f"pydub failed: {e}")
                raise

        raise RuntimeError("No audio loading library available. Install librosa or pydub.")

    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio (normalization, etc.).

        Args:
            audio: Audio data as numpy array
            sr: Sample rate

        Returns:
            Preprocessed audio array
        """
        if self.normalize:
            # Normalize to [-1, 1]
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val

        return audio

    def get_duration(self, audio: np.ndarray, sr: int) -> float:
        """Get audio duration in seconds.

        Args:
            audio: Audio data as numpy array
            sr: Sample rate

        Returns:
            Duration in seconds
        """
        return len(audio) / sr

    def save_audio(self, audio: np.ndarray, sr: int, output_path: Union[str, Path]):
        """Save audio to file.

        Args:
            audio: Audio data as numpy array
            sr: Sample rate
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if LIBROSA_AVAILABLE:
            sf.write(str(output_path), audio, sr)
        elif PYDUB_AVAILABLE:
            # Convert to int16
            audio_int = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_int.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=1
            )
            audio_segment.export(str(output_path), format=output_path.suffix.lstrip('.'))
        else:
            raise RuntimeError("No audio library available for saving")


class WhisperTranscriber:
    """Whisper-based transcription engine."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Whisper transcriber.

        Args:
            config: Configuration dictionary
        """
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not available. Install with: pip install openai-whisper")

        self.config = config.get('whisper', {})
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_name = self.config.get('model', 'base')
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Determine which device to use for inference."""
        device_config = self.config.get('device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                self.logger.info("CUDA not available, using CPU")
        else:
            device = device_config

        return device

    def load_model(self):
        """Load Whisper model."""
        if self.model is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            try:
                self.model = whisper.load_model(self.model_name, device=self.device)
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for transcription

        Returns:
            TranscriptionResult object
        """
        self.load_model()

        # Merge config with kwargs
        options = {
            'language': self.config.get('language', None),
            'task': self.config.get('task', 'transcribe'),
            'fp16': self.config.get('fp16', True) and self.device == 'cuda',
            'verbose': self.config.get('verbose', False),
            'beam_size': self.config.get('beam_size', 5),
            'best_of': self.config.get('best_of', 5),
            'temperature': self.config.get('temperature', 0.0),
        }
        options.update(kwargs)

        # Handle auto language detection
        if options['language'] == 'auto':
            options['language'] = None

        self.logger.info(f"Transcribing with Whisper ({self.model_name})...")
        start_time = time.time()

        try:
            result = self.model.transcribe(str(audio_path), **options)
            duration = time.time() - start_time

            # Extract segments with timestamps
            segments = []
            if 'segments' in result:
                for seg in result['segments']:
                    segments.append({
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text'].strip(),
                        'confidence': seg.get('confidence', None)
                    })

            return TranscriptionResult(
                text=result['text'].strip(),
                language=result.get('language', None),
                duration=duration,
                segments=segments if segments else None,
                engine='whisper',
                model=self.model_name
            )
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise


class GoogleSpeechTranscriber:
    """Google Speech Recognition based transcription engine."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Google Speech transcriber.

        Args:
            config: Configuration dictionary
        """
        if not GOOGLE_SR_AVAILABLE:
            raise RuntimeError(
                "Google Speech Recognition not available. "
                "Install with: pip install SpeechRecognition"
            )

        self.config = config.get('google', {})
        self.logger = logging.getLogger(__name__)
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for transcription

        Returns:
            TranscriptionResult object
        """
        language = kwargs.get('language') or self.config.get('language', 'en-US')
        # Ensure language is always a string, not None
        if language is None or language == 'auto':
            language = 'en-US'
        show_all = kwargs.get('show_all', self.config.get('show_all', False))

        self.logger.info("Transcribing with Google Speech Recognition...")
        start_time = time.time()

        try:
            # Load audio file
            with sr.AudioFile(str(audio_path)) as source:
                audio = self.recognizer.record(source)

            # Perform transcription
            result = self.recognizer.recognize_google(
                audio,
                language=language,
                show_all=show_all
            )
            duration = time.time() - start_time

            if show_all:
                # Extract best alternative
                if result and 'alternative' in result:
                    best = result['alternative'][0]
                    text = best.get('transcript', '')
                    confidence = best.get('confidence', None)
                else:
                    text = ''
                    confidence = None
            else:
                text = result
                confidence = None

            return TranscriptionResult(
                text=text,
                language=language,
                confidence=confidence,
                duration=duration,
                engine='google',
                model='google-speech-api'
            )
        except sr.UnknownValueError:
            self.logger.error("Google Speech Recognition could not understand audio")
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            self.logger.error(f"Google Speech Recognition request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise


class VoskTranscriber:
    """Vosk-based offline transcription engine."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Vosk transcriber.

        Args:
            config: Configuration dictionary
        """
        if not VOSK_AVAILABLE:
            raise RuntimeError("Vosk not available. Install with: pip install vosk")

        self.config = config.get('vosk', {})
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.model_path = self.config.get('model_path', 'models/vosk-model-en-us-0.22')

    def load_model(self):
        """Load Vosk model."""
        if self.model is None:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Vosk model not found at {model_path}. "
                    "Download models from https://alphacephei.com/vosk/models"
                )

            self.logger.info(f"Loading Vosk model from: {model_path}")
            try:
                self.model = Model(str(model_path))
                self.logger.info("Model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
                raise

    def transcribe(self, audio_path: Union[str, Path], **kwargs) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            **kwargs: Additional arguments for transcription

        Returns:
            TranscriptionResult object
        """
        self.load_model()

        sample_rate = self.config.get('sample_rate', 16000)
        show_words = kwargs.get('show_words', self.config.get('show_words', True))

        self.logger.info("Transcribing with Vosk...")
        start_time = time.time()

        try:
            # Open audio file
            wf = wave.open(str(audio_path), "rb")

            # Check format
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
                self.logger.warning(
                    f"Audio file must be WAV format mono PCM @ {sample_rate}Hz. "
                    "Consider preprocessing the audio."
                )

            # Create recognizer
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(show_words)

            # Process audio
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    results.append(result)

            # Get final result
            final_result = json.loads(rec.FinalResult())
            results.append(final_result)

            duration = time.time() - start_time
            wf.close()

            # Combine results
            text = ' '.join([r.get('text', '') for r in results]).strip()

            # Extract segments if available
            segments = []
            for result in results:
                if 'result' in result:
                    for word_info in result['result']:
                        segments.append({
                            'start': word_info.get('start', 0),
                            'end': word_info.get('end', 0),
                            'text': word_info.get('word', ''),
                            'confidence': word_info.get('conf', None)
                        })

            return TranscriptionResult(
                text=text,
                language=self.config.get('language', 'en-US'),
                duration=duration,
                segments=segments if segments else None,
                engine='vosk',
                model=self.model_path
            )
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise


class Transcriber:
    """Main transcription class that orchestrates all components."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize transcriber.

        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.audio_processor = AudioProcessor(self.config)
        self.engines = {}

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        if config_path is None:
            # Look for config.yaml in current directory
            config_path = Path('config.yaml')
        else:
            config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        else:
            # Return default configuration
            return {
                'default_engine': 'whisper',
                'whisper': {'model': 'base'},
                'audio': {'sample_rate': 16000, 'mono': True},
                'output': {'format': 'txt'},
                'logging': {'level': 'INFO'}
            }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            Logger instance
        """
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        # Create logger
        logger = logging.getLogger('audio_transcriber')
        logger.setLevel(level)

        # Remove existing handlers
        logger.handlers = []

        # Console handler
        handler = logging.StreamHandler()
        handler.setLevel(level)

        if COLOREDLOGS_AVAILABLE and log_config.get('colored', True):
            coloredlogs.install(
                level=level,
                logger=logger,
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # File handler if specified
        log_file = log_config.get('log_file')
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def _get_engine(self, engine_name: str):
        """Get or create transcription engine.

        Args:
            engine_name: Name of the engine (whisper, google, vosk)

        Returns:
            Transcription engine instance
        """
        if engine_name not in self.engines:
            if engine_name == 'whisper':
                self.engines[engine_name] = WhisperTranscriber(self.config)
            elif engine_name == 'google':
                self.engines[engine_name] = GoogleSpeechTranscriber(self.config)
            elif engine_name == 'vosk':
                self.engines[engine_name] = VoskTranscriber(self.config)
            else:
                raise ValueError(f"Unknown engine: {engine_name}")

        return self.engines[engine_name]

    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        engine: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe a single audio file.

        Args:
            audio_path: Path to audio file
            engine: Transcription engine to use (whisper, google, vosk)
            output_format: Output format (txt, json, srt, vtt)
            **kwargs: Additional arguments for transcription

        Returns:
            TranscriptionResult object
        """
        audio_path = Path(audio_path)

        # Get engine
        if engine is None:
            engine = self.config.get('default_engine', 'whisper')

        self.logger.info(f"Transcribing: {audio_path.name} with {engine}")

        # Load and preprocess audio
        try:
            audio, sr = self.audio_processor.load_audio(audio_path)
            audio = self.audio_processor.preprocess_audio(audio, sr)

            # For engines that need WAV, save temporary file
            if engine in ['google', 'vosk']:
                temp_path = audio_path.parent / f"temp_{audio_path.stem}.wav"
                self.audio_processor.save_audio(audio, sr, temp_path)
                transcribe_path = temp_path
            else:
                transcribe_path = audio_path

            # Transcribe
            transcription_engine = self._get_engine(engine)
            result = transcription_engine.transcribe(transcribe_path, **kwargs)

            # Clean up temporary file
            if engine in ['google', 'vosk'] and temp_path.exists():
                temp_path.unlink()

            # Save output if format specified
            if output_format:
                self._save_output(result, audio_path, output_format)

            return result

        except Exception as e:
            self.logger.error(f"Failed to transcribe {audio_path}: {e}")
            raise

    def transcribe_batch(
        self,
        file_paths: List[Union[str, Path]],
        engine: Optional[str] = None,
        output_format: Optional[str] = None,
        **kwargs
    ) -> List[TranscriptionResult]:
        """Transcribe multiple audio files.

        Args:
            file_paths: List of paths to audio files
            engine: Transcription engine to use
            output_format: Output format
            **kwargs: Additional arguments for transcription

        Returns:
            List of TranscriptionResult objects
        """
        results = []
        batch_config = self.config.get('batch', {})
        continue_on_error = batch_config.get('continue_on_error', True)
        show_progress = self.config.get('logging', {}).get('progress_bars', True)

        iterator = tqdm(file_paths, desc="Transcribing") if show_progress else file_paths

        for file_path in iterator:
            try:
                result = self.transcribe_file(
                    file_path,
                    engine=engine,
                    output_format=output_format,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to transcribe {file_path}: {e}")
                if not continue_on_error:
                    raise
                results.append(None)

        return results

    def _save_output(
        self,
        result: TranscriptionResult,
        input_path: Path,
        output_format: str
    ):
        """Save transcription result to file.

        Args:
            result: TranscriptionResult object
            input_path: Original input file path
            output_format: Output format (txt, json, srt, vtt, all)
        """
        output_config = self.config.get('output', {})
        output_dir = output_config.get('directory')

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path.parent

        # Determine output filename
        base_name = input_path.stem
        if output_config.get('append_engine_name', False):
            base_name = f"{base_name}_{result.engine}"

        formats = [output_format] if output_format != 'all' else ['txt', 'json', 'srt', 'vtt']

        for fmt in formats:
            output_path = output_dir / f"{base_name}.{fmt}"

            try:
                if fmt == 'txt':
                    content = result.text
                elif fmt == 'json':
                    content = result.to_json()
                elif fmt == 'srt':
                    content = result.to_srt()
                elif fmt == 'vtt':
                    content = result.to_vtt()
                else:
                    self.logger.warning(f"Unknown format: {fmt}")
                    continue

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.logger.info(f"Saved {fmt.upper()} to: {output_path}")
            except Exception as e:
                self.logger.error(f"Failed to save {fmt} output: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Audio Transcriber - Transcribe audio files to text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single file
  python transcribe.py audio.m4a

  # Specify engine and output format
  python transcribe.py audio.mp3 --engine whisper --output-format srt

  # Batch process multiple files
  python transcribe.py audio1.wav audio2.m4a audio3.mp3

  # Use custom configuration
  python transcribe.py audio.m4a --config my_config.yaml

  # Generate subtitles
  python transcribe.py video.mp4 --output-format srt --language en
        """
    )

    # Required arguments
    parser.add_argument(
        'input_files',
        nargs='+',
        help='Input audio file(s) to transcribe'
    )

    # Optional arguments
    parser.add_argument(
        '-e', '--engine',
        choices=['whisper', 'google', 'vosk'],
        help='Transcription engine to use (default: from config)'
    )

    parser.add_argument(
        '-m', '--model',
        help='Model to use (for Whisper: tiny, base, small, medium, large)'
    )

    parser.add_argument(
        '-l', '--language',
        help='Language code (e.g., en, es, fr) or "auto" for detection'
    )

    parser.add_argument(
        '-o', '--output-format',
        choices=['txt', 'json', 'srt', 'vtt', 'all'],
        help='Output format (default: txt)'
    )

    parser.add_argument(
        '-od', '--output-dir',
        help='Output directory (default: same as input)'
    )

    parser.add_argument(
        '-c', '--config',
        help='Configuration file path (default: config.yaml)'
    )

    parser.add_argument(
        '--task',
        choices=['transcribe', 'translate'],
        help='Task type (translate converts to English)'
    )

    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for computation'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Audio Transcriber 1.0.0'
    )

    args = parser.parse_args()

    # Initialize transcriber
    try:
        transcriber = Transcriber(config_path=args.config)

        # Override config with CLI arguments
        if args.model:
            transcriber.config.setdefault('whisper', {})['model'] = args.model

        if args.device:
            transcriber.config.setdefault('whisper', {})['device'] = args.device

        if args.task:
            transcriber.config.setdefault('whisper', {})['task'] = args.task

        if args.verbose:
            transcriber.logger.setLevel(logging.DEBUG)

        if args.output_dir:
            transcriber.config.setdefault('output', {})['directory'] = args.output_dir

        # Process files
        input_files = [Path(f) for f in args.input_files]

        # Check if files exist
        for file_path in input_files:
            if not file_path.exists():
                transcriber.logger.error(f"File not found: {file_path}")
                sys.exit(1)

        # Transcribe
        if len(input_files) == 1:
            result = transcriber.transcribe_file(
                input_files[0],
                engine=args.engine,
                output_format=args.output_format or 'txt',
                language=args.language
            )
            print("\n" + "="*80)
            print("TRANSCRIPTION RESULT")
            print("="*80)
            print(result.text)
            print("="*80)
            if result.language:
                print(f"Language: {result.language}")
            if result.duration:
                print(f"Processing time: {result.duration:.2f}s")
            print("="*80)
        else:
            results = transcriber.transcribe_batch(
                input_files,
                engine=args.engine,
                output_format=args.output_format or 'txt',
                language=args.language
            )
            print(f"\nProcessed {len(results)} files successfully")

    except KeyboardInterrupt:
        print("\nTranscription interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
