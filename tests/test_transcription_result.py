"""Tests for the TranscriptionResult dataclass."""

import json
import pytest
from transcribe import TranscriptionResult


class TestTranscriptionResult:
    """Tests for TranscriptionResult creation and serialization."""

    def test_basic_creation(self):
        result = TranscriptionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.language is None
        assert result.confidence is None
        assert result.segments is None
        assert result.speakers is None

    def test_full_creation(self):
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        result = TranscriptionResult(
            text="Hello",
            language="en",
            confidence=0.95,
            duration=1.5,
            segments=segments,
            engine="whisper",
            model="base",
            speakers=["Speaker 1"],
        )
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.engine == "whisper"
        assert result.speakers == ["Speaker 1"]

    def test_to_dict(self):
        result = TranscriptionResult(text="test", language="en", engine="whisper")
        d = result.to_dict()
        assert d["text"] == "test"
        assert d["language"] == "en"
        assert d["engine"] == "whisper"
        assert d["speakers"] is None

    def test_to_dict_with_speakers(self):
        result = TranscriptionResult(
            text="test", speakers=["Speaker 1", "Speaker 2"]
        )
        d = result.to_dict()
        assert d["speakers"] == ["Speaker 1", "Speaker 2"]

    def test_to_json(self):
        result = TranscriptionResult(text="hello", language="en")
        j = result.to_json()
        parsed = json.loads(j)
        assert parsed["text"] == "hello"
        assert parsed["language"] == "en"

    def test_to_json_ensure_ascii_false(self):
        result = TranscriptionResult(text="shalom")
        j = result.to_json()
        # Should not escape non-ASCII
        assert "shalom" in j


class TestTranscriptionResultSRT:
    """Tests for SRT output format."""

    def test_basic_srt(self):
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "How are you"},
        ]
        result = TranscriptionResult(text="Hello world How are you", segments=segments)
        srt = result.to_srt()
        assert "1\n00:00:00,000 --> 00:00:02,500\nHello world" in srt
        assert "2\n00:00:02,500 --> 00:00:05,000\nHow are you" in srt

    def test_srt_with_speaker(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "Speaker 1"},
            {"start": 2.0, "end": 4.0, "text": "Hi there", "speaker": "Speaker 2"},
        ]
        result = TranscriptionResult(text="", segments=segments)
        srt = result.to_srt()
        assert "Speaker 1: Hello" in srt
        assert "Speaker 2: Hi there" in srt

    def test_srt_without_speaker(self):
        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        result = TranscriptionResult(text="", segments=segments)
        srt = result.to_srt()
        assert "Speaker" not in srt
        assert "Hello" in srt

    def test_srt_no_segments_raises(self):
        result = TranscriptionResult(text="Hello")
        with pytest.raises(ValueError, match="No segments"):
            result.to_srt()

    def test_srt_timestamp_format(self):
        segments = [{"start": 3661.123, "end": 3662.456, "text": "test"}]
        result = TranscriptionResult(text="", segments=segments)
        srt = result.to_srt()
        assert "01:01:01,123" in srt
        assert "01:01:02,456" in srt


class TestTranscriptionResultVTT:
    """Tests for VTT output format."""

    def test_basic_vtt(self):
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
        ]
        result = TranscriptionResult(text="Hello world", segments=segments)
        vtt = result.to_vtt()
        assert vtt.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.500" in vtt
        assert "Hello world" in vtt

    def test_vtt_with_speaker(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Hello", "speaker": "Speaker 1"},
        ]
        result = TranscriptionResult(text="", segments=segments)
        vtt = result.to_vtt()
        assert "Speaker 1: Hello" in vtt

    def test_vtt_without_speaker(self):
        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        result = TranscriptionResult(text="", segments=segments)
        vtt = result.to_vtt()
        assert "Speaker" not in vtt

    def test_vtt_no_segments_raises(self):
        result = TranscriptionResult(text="Hello")
        with pytest.raises(ValueError, match="No segments"):
            result.to_vtt()

    def test_vtt_uses_period_separator(self):
        segments = [{"start": 1.5, "end": 3.0, "text": "test"}]
        result = TranscriptionResult(text="", segments=segments)
        vtt = result.to_vtt()
        assert "00:00:01.500" in vtt
        # VTT uses period, not comma
        assert "00:00:01,500" not in vtt


class TestFormatTimestamp:
    """Tests for the _format_timestamp static method."""

    def test_zero(self):
        assert TranscriptionResult._format_timestamp(0) == "00:00:00,000"

    def test_seconds_only(self):
        assert TranscriptionResult._format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert TranscriptionResult._format_timestamp(65.0) == "00:01:05,000"

    def test_hours(self):
        assert TranscriptionResult._format_timestamp(3661.0) == "01:01:01,000"

    def test_vtt_format(self):
        assert TranscriptionResult._format_timestamp(1.5, vtt=True) == "00:00:01.500"

    def test_srt_format(self):
        assert TranscriptionResult._format_timestamp(1.5, vtt=False) == "00:00:01,500"

    def test_millisecond_precision(self):
        assert TranscriptionResult._format_timestamp(0.123) == "00:00:00,123"
