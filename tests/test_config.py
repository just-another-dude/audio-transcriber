"""Tests for configuration loading and validation."""

import yaml
import pytest
from pathlib import Path


class TestConfigYaml:
    """Tests for the config.yaml file."""

    @pytest.fixture
    def config(self):
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_config_loads(self, config):
        assert config is not None
        assert isinstance(config, dict)

    def test_has_default_engine(self, config):
        assert "default_engine" in config
        assert config["default_engine"] in ("whisper", "google", "vosk")

    def test_has_whisper_section(self, config):
        assert "whisper" in config
        assert "model" in config["whisper"]
        assert "device" in config["whisper"]

    def test_has_audio_section(self, config):
        assert "audio" in config
        assert config["audio"]["sample_rate"] == 16000

    def test_has_output_section(self, config):
        assert "output" in config
        assert "format" in config["output"]

    def test_has_diarization_section(self, config):
        assert "diarization" in config
        assert "model" in config["diarization"]
        assert config["diarization"]["model"] == "pyannote/speaker-diarization-3.1"

    def test_diarization_speaker_limits_default_null(self, config):
        assert config["diarization"]["min_speakers"] is None
        assert config["diarization"]["max_speakers"] is None

    def test_has_logging_section(self, config):
        assert "logging" in config
        assert config["logging"]["level"] == "INFO"
