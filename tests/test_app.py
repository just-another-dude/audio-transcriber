"""Tests for app.py utility functions and structure."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestAppHelpers:
    """Tests for app.py helper functions."""

    def test_format_duration_seconds(self):
        from app import format_duration
        assert format_duration(45) == "0:45"

    def test_format_duration_minutes(self):
        from app import format_duration
        assert format_duration(125) == "2:05"

    def test_format_duration_hours(self):
        from app import format_duration
        assert format_duration(3661) == "1:01:01"

    def test_get_file_size_mb(self, tmp_path):
        from app import get_file_size_mb
        f = tmp_path / "test.bin"
        f.write_bytes(b"x" * 1024 * 1024)  # 1MB
        size = get_file_size_mb(str(f))
        assert abs(size - 1.0) < 0.01


class TestAppTranscribeValidation:
    """Tests for input validation in the transcribe function."""

    def test_no_files_returns_error(self):
        from app import transcribe
        status, text, output_file, info = transcribe(
            None, "openai", "txt", "auto", "base", "transcribe", False,
        )
        assert "Please upload" in status

    def test_empty_list_returns_error(self):
        from app import transcribe
        status, text, output_file, info = transcribe(
            [], "openai", "txt", "auto", "base", "transcribe", False,
        )
        assert "Please upload" in status


class TestLocalTranscriberWrapper:
    """Tests for the LocalTranscriberWrapper formatting."""

    @pytest.fixture
    def wrapper_class(self):
        from app import LocalTranscriberWrapper
        return LocalTranscriberWrapper

    def test_format_result_txt(self, wrapper_class):
        from transcribe import TranscriptionResult
        result = TranscriptionResult(text="Hello world")
        assert wrapper_class._format_result(result, "txt") == "Hello world"

    def test_format_result_verbose_json(self, wrapper_class):
        from transcribe import TranscriptionResult
        import json
        result = TranscriptionResult(text="Hello", language="en")
        formatted = wrapper_class._format_result(result, "verbose_json")
        parsed = json.loads(formatted)
        assert parsed["text"] == "Hello"

    def test_format_result_srt(self, wrapper_class):
        from transcribe import TranscriptionResult
        result = TranscriptionResult(
            text="Hello",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )
        srt = wrapper_class._format_result(result, "srt")
        assert "00:00:00,000 --> 00:00:01,000" in srt

    def test_format_result_vtt(self, wrapper_class):
        from transcribe import TranscriptionResult
        result = TranscriptionResult(
            text="Hello",
            segments=[{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )
        vtt = wrapper_class._format_result(result, "vtt")
        assert "WEBVTT" in vtt


class TestPyannoteAvailabilityFlag:
    """Test that the diarization checkbox visibility is tied to PYANNOTE_AVAILABLE."""

    def test_pyannote_flag_is_boolean(self):
        import app
        assert isinstance(app.PYANNOTE_AVAILABLE, bool)
