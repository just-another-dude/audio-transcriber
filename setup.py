#!/usr/bin/env python3
"""
Setup script for Audio Transcriber package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#')
        ]
else:
    requirements = [
        'openai-whisper>=20231117',
        'SpeechRecognition>=3.10.0',
        'pydub>=0.25.1',
        'librosa>=0.10.0',
        'soundfile>=0.12.1',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'vosk>=0.3.45',
        'gradio>=4.0.0',
        'PyYAML>=6.0',
        'tqdm>=4.65.0',
        'ffmpeg-python>=0.2.0',
        'typing-extensions>=4.5.0',
        'coloredlogs>=15.0.1',
    ]

setup(
    name="audio-transcriber",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive audio transcription tool supporting multiple engines and formats",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/audio-transcriber",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/audio-transcriber/issues",
        "Documentation": "https://github.com/yourusername/audio-transcriber#readme",
        "Source Code": "https://github.com/yourusername/audio-transcriber",
    },
    py_modules=["transcribe", "web_app", "examples"],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        'gpu': [
            'torch>=2.0.0',
            'torchaudio>=2.0.0',
            'gputil>=1.4.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'audio-transcriber=transcribe:main',
            'audio-transcriber-web=web_app:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="audio transcription speech-to-text whisper asr voice-recognition subtitles",
    include_package_data=True,
    package_data={
        '': ['config.yaml', 'LICENSE', 'README.md'],
    },
    zip_safe=False,
)
