# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Security Notice

**This is a PUBLIC repository.** Be extremely careful to never:
- Commit API keys, tokens, or secrets
- Include credentials in code or comments
- Add `.env` files or any sensitive configuration
- Expose personal information

Always verify changes don't contain sensitive data before committing.

## Project Overview

Audio Transcriber is a production-ready audio transcription tool that supports multiple AI engines (OpenAI Whisper, Google Speech Recognition, and Vosk) and various audio formats, with special emphasis on M4A file support.

## Project Structure

```
audio-transcriber/
├── transcribe.py           # Main transcription module with CLI
├── app.py                 # Unified Gradio web interface (OpenAI API + local engines)
├── examples.py            # Usage examples and demonstrations
├── config.yaml            # Configuration file
├── requirements.txt       # Core Python dependencies
├── requirements-gpu.txt   # GPU-specific dependencies
├── setup.py              # Package installation script
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Docker Compose orchestration
├── LICENSE              # MIT License
├── README.md            # Comprehensive documentation
└── .github/
    └── workflows/
        └── ci.yml       # GitHub Actions CI/CD pipeline
```

## Core Architecture

### Main Components

1. **TranscriptionResult** (dataclass in transcribe.py:32-84)
   - Stores transcription output with metadata
   - Methods: `to_dict()`, `to_json()`, `to_srt()`, `to_vtt()`
   - Handles timestamp formatting for subtitles

2. **AudioProcessor** (transcribe.py:87-198)
   - Handles audio loading and preprocessing
   - Uses librosa (primary) with pydub fallback
   - Manages format conversion via FFmpeg
   - Supports all major audio formats including M4A

3. **Transcription Engines** (transcribe.py:201-513)
   - `WhisperTranscriber`: Primary engine, most accurate
   - `GoogleSpeechTranscriber`: Cloud-based, requires internet
   - `VoskTranscriber`: Offline support
   - Each engine implements: `__init__()`, `load_model()`, `transcribe()`

4. **Transcriber** (main orchestrator, transcribe.py:516-752)
   - Coordinates all components
   - Handles configuration loading
   - Manages batch processing
   - Saves output in multiple formats

5. **Web Interface** (app.py)
   - Unified Gradio-based UI with engine selector (OpenAI API + local engines)
   - File upload, engine selection, result download
   - Local engines available only when their dependencies are installed

## Common Development Tasks

### Running the Application

```bash
# CLI transcription
python transcribe.py audio.m4a

# With options
python transcribe.py audio.m4a --engine whisper --model base --output-format srt

# Web interface
python app.py

# With custom port
python app.py --port 8080
```

### Testing

```bash
# Import test
python -c "import transcribe; print('OK')"

# CLI help
python transcribe.py --help
python app.py --help

# Run examples
python examples.py
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up audio-transcriber

# Build manually
docker build -t audio-transcriber .

# Run CLI in Docker
docker run -v $(pwd)/input:/app/input audio-transcriber python transcribe.py /app/input/audio.m4a
```

### Dependencies

Install with:
```bash
pip install -r requirements.txt        # Core dependencies
pip install -r requirements-gpu.txt    # GPU support
pip install .                          # Full installation
```

**Critical dependency:** FFmpeg must be installed on the system for audio format support.

## Key Technical Details

### Audio Format Handling

- **M4A files**: Fully supported via FFmpeg with automatic conversion
- Audio loading uses librosa first, falls back to pydub if needed
- All audio is resampled to 16kHz mono by default (configurable in config.yaml)
- Supported formats defined in `AudioProcessor.SUPPORTED_FORMATS`

### Engine Selection

- **Whisper**: Best accuracy, GPU-accelerated, supports 99+ languages
- **Google**: Fast, requires internet, good for clear speech
- **Vosk**: Offline, requires model download, moderate accuracy

### Configuration System

All settings in `config.yaml`:
- Engine-specific parameters (whisper, google, vosk)
- Audio processing (sample_rate, mono, normalize)
- Output preferences (format, timestamps, directory)
- Subtitle formatting (max_chars_per_line, max_duration)
- Performance tuning (use_gpu, chunk_size, cache_models)

### Timestamp Handling

- SRT format: `HH:MM:SS,mmm` (comma separator)
- VTT format: `HH:MM:SS.mmm` (period separator)
- Implemented in `TranscriptionResult._format_timestamp()` (transcribe.py:74-84)

### Error Handling

- FFmpeg availability check in `AudioProcessor._check_ffmpeg()` (transcribe.py:110-126)
- Graceful fallback between audio libraries
- Batch processing continues on error if `continue_on_error=True`
- Comprehensive logging with colored output (coloredlogs)

## Code Style

- Follows PEP 8 guidelines
- Type hints throughout (Python 3.8+ compatible)
- Comprehensive docstrings for all classes and methods
- Error messages include helpful installation instructions
- Uses pathlib.Path for cross-platform file handling

## Important Implementation Notes

1. **Temporary WAV files**: Google and Vosk engines require WAV format, so temporary files are created and cleaned up in `transcribe_file()` (transcribe.py:630-666)

2. **Model caching**: Whisper models are cached after first load to avoid reloading

3. **GPU detection**: Automatic CUDA detection in `WhisperTranscriber._get_device()` (transcribe.py:237-252)

4. **Subtitle segment extraction**: Whisper provides word-level timestamps, formatted for SRT/VTT output

5. **Progress bars**: Uses tqdm for batch processing progress indication

## Testing Considerations

When testing or modifying:
- Test M4A file support specifically
- Verify FFmpeg integration works correctly
- Test all output formats (txt, json, srt, vtt)
- Check error handling for missing files/formats
- Verify batch processing with multiple files
- Test with and without GPU

## Common Issues and Solutions

1. **FFmpeg not found**: Install FFmpeg using system package manager
2. **CUDA out of memory**: Use smaller Whisper model or force CPU with `--device cpu`
3. **Import errors**: Ensure all dependencies installed with `pip install -r requirements.txt`
4. **Vosk model not found**: Download model from alphacephei.com/vosk/models
5. **Google API errors**: Check internet connection or use Whisper/Vosk instead

## Performance Optimization

- Use `base` or `small` Whisper models for speed
- Enable GPU with `fp16: true` in config for faster inference
- Use WAV format to skip conversion step
- Batch process multiple files together
- Consider using `chunk_size` for very long audio files

## License

This project uses the MIT License, allowing free use, modification, and distribution.
