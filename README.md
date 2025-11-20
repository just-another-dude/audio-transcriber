# 🎙️ Audio Transcriber

A comprehensive, production-ready audio transcription tool supporting multiple AI engines and various audio formats, with special focus on M4A files.

[![CI/CD](https://github.com/yourusername/audio-transcriber/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/yourusername/audio-transcriber/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **Multiple Transcription Engines**
  - 🤖 OpenAI Whisper (primary, most accurate)
  - 🌐 Google Speech Recognition
  - 📱 Vosk (offline support)

- **Extensive Format Support**
  - Audio: MP3, WAV, M4A, FLAC, OGG, WMA, AAC, OPUS
  - Video: WebM, MP4
  - Special focus on M4A files with seamless FFmpeg integration

- **Multiple Output Formats**
  - Plain text (TXT)
  - Structured data (JSON)
  - Subtitles (SRT, VTT)

- **Advanced Features**
  - Batch processing for multiple files
  - Language auto-detection
  - Translation to English
  - Timestamp extraction for subtitles
  - Progress bars and detailed logging
  - GPU acceleration support
  - Docker deployment ready

- **User-Friendly Interfaces**
  - Command-line interface (CLI)
  - Web interface powered by Gradio
  - Python API for integration

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [CLI Usage](#cli-usage)
  - [Web Interface](#web-interface)
  - [Python API](#python-api)
- [Supported Formats](#-supported-formats)
- [Configuration](#-configuration)
- [Docker](#-docker)
- [Performance Tips](#-performance-tips)
- [Examples](#-examples)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Installation

### Prerequisites

**FFmpeg is required** for audio format support. Install it first:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsndfile1
```

#### macOS
```bash
brew install ffmpeg
```

#### Windows
Download from [FFmpeg official website](https://ffmpeg.org/download.html) or use Chocolatey:
```powershell
choco install ffmpeg
```

### Install Audio Transcriber

#### Option 1: Using pip (recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# Install with pip
pip install .

# For GPU support (CUDA required)
pip install .[gpu]

# For development
pip install .[dev]
```

#### Option 2: Install dependencies manually
```bash
# Core dependencies
pip install -r requirements.txt

# For GPU support
pip install -r requirements-gpu.txt
```

### Verify Installation

```bash
# Check CLI
audio-transcriber --version

# Or use directly
python transcribe.py --version
```

## ⚡ Quick Start

### Transcribe a Single File

```bash
# Using CLI
python transcribe.py audio.m4a

# Specify output format
python transcribe.py audio.m4a --output-format srt

# Use different engine
python transcribe.py audio.mp3 --engine google
```

### Launch Web Interface

```bash
# Start web server
python web_app.py

# Access at http://localhost:7860
```

### Python API

```python
from transcribe import Transcriber

# Initialize
transcriber = Transcriber()

# Transcribe
result = transcriber.transcribe_file("audio.m4a")
print(result.text)
```

## 📖 Usage

### CLI Usage

#### Basic Transcription

```bash
# Single file
python transcribe.py audio.m4a

# Multiple files (batch processing)
python transcribe.py audio1.m4a audio2.mp3 audio3.wav

# Specify language
python transcribe.py audio.m4a --language es

# Auto-detect language
python transcribe.py audio.m4a --language auto
```

#### Output Formats

```bash
# Text output (default)
python transcribe.py audio.m4a --output-format txt

# JSON with metadata
python transcribe.py audio.m4a --output-format json

# SRT subtitles
python transcribe.py audio.m4a --output-format srt

# VTT subtitles
python transcribe.py audio.m4a --output-format vtt

# All formats
python transcribe.py audio.m4a --output-format all
```

#### Engine Selection

```bash
# Whisper (default, most accurate)
python transcribe.py audio.m4a --engine whisper --model base

# Google Speech Recognition (requires internet)
python transcribe.py audio.m4a --engine google

# Vosk (offline)
python transcribe.py audio.m4a --engine vosk
```

#### Advanced Options

```bash
# Translate to English
python transcribe.py audio_spanish.m4a --task translate

# Specify output directory
python transcribe.py audio.m4a --output-dir ./transcriptions

# Use custom configuration
python transcribe.py audio.m4a --config my_config.yaml

# Verbose output
python transcribe.py audio.m4a --verbose

# Force CPU (even if GPU available)
python transcribe.py audio.m4a --device cpu
```

### Web Interface

Launch the web interface:

```bash
python web_app.py

# Custom port
python web_app.py --port 8080

# Make publicly accessible
python web_app.py --share

# Custom host
python web_app.py --host 0.0.0.0 --port 7860
```

Features:
- 📁 Drag-and-drop file upload
- 🎤 Direct microphone recording
- ⚙️ Engine and model selection
- 🌍 Language selection
- 📥 Download results in various formats
- 📊 Real-time status updates

### Python API

#### Basic Usage

```python
from transcribe import Transcriber

# Initialize transcriber
transcriber = Transcriber()

# Transcribe a file
result = transcriber.transcribe_file(
    "audio.m4a",
    engine="whisper",
    output_format="txt"
)

# Access results
print(result.text)
print(f"Language: {result.language}")
print(f"Duration: {result.duration}s")
```

#### Batch Processing

```python
from transcribe import Transcriber
from pathlib import Path

transcriber = Transcriber()

# Get all audio files
audio_files = list(Path("audio_dir").glob("*.m4a"))

# Batch transcribe
results = transcriber.transcribe_batch(
    audio_files,
    engine="whisper",
    output_format="srt"
)

# Process results
for file, result in zip(audio_files, results):
    if result:
        print(f"{file.name}: {len(result.text)} characters")
```

#### Working with Segments (Timestamps)

```python
result = transcriber.transcribe_file("audio.m4a")

# Access segments with timestamps
if result.segments:
    for segment in result.segments:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        print(f"[{start:.2f}s - {end:.2f}s] {text}")
```

#### Generate Subtitles

```python
result = transcriber.transcribe_file("video.mp4")

# Save as SRT
with open("output.srt", "w", encoding="utf-8") as f:
    f.write(result.to_srt())

# Save as VTT
with open("output.vtt", "w", encoding="utf-8") as f:
    f.write(result.to_vtt())
```

#### Custom Configuration

```python
from transcribe import Transcriber

# Load custom config
transcriber = Transcriber(config_path="custom_config.yaml")

# Or modify config programmatically
transcriber.config['whisper']['model'] = 'large'
transcriber.config['audio']['sample_rate'] = 16000

result = transcriber.transcribe_file("audio.m4a")
```

## 🎵 Supported Formats

| Format | Extension | Description | M4A Support |
|--------|-----------|-------------|-------------|
| MP3 | .mp3 | MPEG Audio Layer III | ✅ Excellent |
| WAV | .wav | Waveform Audio File | ✅ Excellent |
| M4A | .m4a | MPEG-4 Audio | ✅ **Excellent** |
| FLAC | .flac | Free Lossless Audio Codec | ✅ Excellent |
| OGG | .ogg | Ogg Vorbis | ✅ Good |
| WMA | .wma | Windows Media Audio | ✅ Good |
| AAC | .aac | Advanced Audio Coding | ✅ Good |
| OPUS | .opus | Opus Audio | ✅ Good |
| WebM | .webm | WebM Audio/Video | ✅ Good |
| MP4 | .mp4 | MPEG-4 Video (audio extracted) | ✅ Good |

**Note:** M4A files are fully supported with automatic FFmpeg-based conversion. No manual preprocessing required!

## ⚙️ Configuration

The `config.yaml` file allows extensive customization:

### Whisper Settings

```yaml
whisper:
  model: base  # tiny, base, small, medium, large, large-v2, large-v3
  device: auto  # auto, cpu, cuda
  language: auto  # auto, en, es, fr, de, etc.
  task: transcribe  # transcribe or translate
  fp16: true  # GPU acceleration
  beam_size: 5
  temperature: 0.0
```

### Audio Processing

```yaml
audio:
  sample_rate: 16000  # Target sample rate
  mono: true  # Convert to mono
  normalize: true  # Normalize audio levels
  remove_silence: false  # Remove silence (experimental)
```

### Output Settings

```yaml
output:
  format: txt  # Default format
  timestamps: false  # Include timestamps
  directory: null  # Output directory (null = same as input)
  append_engine_name: false  # Append engine name to filename
```

### Subtitle Settings

```yaml
subtitles:
  max_chars_per_line: 42
  max_lines: 2
  max_duration: 7.0  # Maximum subtitle duration (seconds)
  min_duration: 0.5  # Minimum subtitle duration (seconds)
```

## 🐳 Docker

### Using Docker Compose (Recommended)

#### CPU Version

```bash
# Start web interface
docker-compose up audio-transcriber

# Access at http://localhost:7860
```

#### GPU Version

```bash
# Requires nvidia-docker
docker-compose --profile gpu up audio-transcriber-gpu
```

#### CLI Batch Processing

```bash
# Place audio files in ./input directory
mkdir -p input output

# Run batch transcription
docker-compose --profile cli run audio-transcriber-cli
```

### Using Dockerfile Directly

```bash
# Build image
docker build -t audio-transcriber .

# Run web interface
docker run -p 7860:7860 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  audio-transcriber

# Run CLI
docker run -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  audio-transcriber \
  python transcribe.py /app/input/audio.m4a --output-dir /app/output
```

### GPU Support with Docker

```bash
# Build GPU image
docker build --target gpu -t audio-transcriber:gpu .

# Run with GPU
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  audio-transcriber:gpu
```

## 🚀 Performance Tips

### Model Selection

| Model | Speed | Accuracy | Memory | Use Case |
|-------|-------|----------|--------|----------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | 1 GB | Quick drafts, testing |
| base | ⚡⚡⚡⚡ | ⭐⭐⭐ | 1 GB | General use, fast |
| small | ⚡⚡⚡ | ⭐⭐⭐⭐ | 2 GB | Good balance |
| medium | ⚡⚡ | ⭐⭐⭐⭐⭐ | 5 GB | High accuracy |
| large | ⚡ | ⭐⭐⭐⭐⭐ | 10 GB | Best accuracy |

### GPU Acceleration

```python
# Automatically use GPU if available
transcriber = Transcriber()
transcriber.config['whisper']['device'] = 'auto'

# Force GPU
transcriber.config['whisper']['device'] = 'cuda'

# Enable FP16 for faster GPU inference
transcriber.config['whisper']['fp16'] = True
```

### Batch Processing

```python
# Enable parallel processing (experimental)
transcriber.config['batch']['parallel'] = True
transcriber.config['batch']['workers'] = 4  # Number of parallel workers
```

### Audio Preprocessing

- Use WAV format for fastest processing (no conversion needed)
- Lower sample rates (8kHz-16kHz) are faster but less accurate
- Mono audio is faster than stereo
- Remove silence for faster processing (experimental)

## 📚 Examples

See `examples.py` for comprehensive usage examples:

1. Basic transcription
2. Batch processing
3. Subtitle generation
4. Language detection and translation
5. Timestamp extraction
6. Engine comparison
7. Custom configuration
8. JSON output
9. M4A file handling
10. Progress monitoring

Run examples:

```bash
python examples.py
```

## 📖 API Reference

### Main Classes

#### `Transcriber`

Main class for transcription operations.

```python
transcriber = Transcriber(config_path: Optional[str] = None)
```

**Methods:**

- `transcribe_file(audio_path, engine=None, output_format=None, **kwargs)` - Transcribe single file
- `transcribe_batch(file_paths, engine=None, output_format=None, **kwargs)` - Batch transcribe

#### `TranscriptionResult`

Data class storing transcription results.

**Attributes:**

- `text: str` - Transcribed text
- `language: Optional[str]` - Detected/specified language
- `confidence: Optional[float]` - Confidence score
- `duration: Optional[float]` - Processing time
- `segments: Optional[List[Dict]]` - Timestamp segments
- `engine: Optional[str]` - Engine used
- `model: Optional[str]` - Model used

**Methods:**

- `to_dict()` - Convert to dictionary
- `to_json(indent=2)` - Convert to JSON
- `to_srt()` - Convert to SRT subtitles
- `to_vtt()` - Convert to VTT subtitles

#### `AudioProcessor`

Audio loading and preprocessing.

```python
processor = AudioProcessor(config: Dict)
```

**Methods:**

- `load_audio(file_path)` - Load audio file
- `preprocess_audio(audio, sr)` - Preprocess audio
- `get_duration(audio, sr)` - Get duration
- `save_audio(audio, sr, output_path)` - Save audio

### Engine Classes

- `WhisperTranscriber` - OpenAI Whisper engine
- `GoogleSpeechTranscriber` - Google Speech Recognition
- `VoskTranscriber` - Vosk offline engine

Each engine has:
- `__init__(config)` - Initialize with config
- `transcribe(audio_path, **kwargs)` - Transcribe audio
- `load_model()` - Load transcription model

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/audio-transcriber.git
cd audio-transcriber

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black .

# Lint
flake8 .

# Type check
mypy transcribe.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Robust speech recognition
- [Gradio](https://gradio.app/) - Web interface framework
- [FFmpeg](https://ffmpeg.org/) - Audio/video processing
- [librosa](https://librosa.org/) - Audio analysis
- [Vosk](https://alphacephei.com/vosk/) - Offline speech recognition

## 📧 Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/audio-transcriber/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/audio-transcriber/discussions)
- **Documentation:** [Wiki](https://github.com/yourusername/audio-transcriber/wiki)

## 🗺️ Roadmap

- [ ] Web API (REST/GraphQL)
- [ ] Real-time streaming transcription
- [ ] Speaker diarization (identify multiple speakers)
- [ ] Custom model fine-tuning
- [ ] Cloud storage integration (S3, GCS)
- [ ] More language support
- [ ] Mobile app (React Native)

---

**Made with ❤️ by the Audio Transcriber Team**
