"""Tests for the diarize module.

These tests use mocks so pyannote.audio is not required.
"""

import pytest
from unittest.mock import patch, MagicMock
from transcribe import TranscriptionResult


class TestAssignSpeakers:
    """Tests for SpeakerDiarizer.assign_speakers (static method, no deps needed)."""

    @pytest.fixture(autouse=True)
    def _import_with_mock(self):
        """Import diarize with pyannote mocked so it's always available."""
        with patch.dict("sys.modules", {"pyannote.audio": MagicMock(), "pyannote": MagicMock()}):
            from diarize import SpeakerDiarizer
            self.assign_speakers = SpeakerDiarizer.assign_speakers

    def test_basic_assignment(self):
        transcription_segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello"},
            {"start": 3.0, "end": 6.0, "text": "World"},
        ]
        diarization_segments = [
            {"start": 0.0, "end": 3.0, "speaker": "Speaker 1"},
            {"start": 3.0, "end": 6.0, "speaker": "Speaker 2"},
        ]
        result = self.assign_speakers(transcription_segments, diarization_segments)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"

    def test_overlap_based_assignment(self):
        """Speaker assigned based on highest overlap, not first match."""
        transcription_segments = [
            {"start": 1.0, "end": 4.0, "text": "Hello"},
        ]
        diarization_segments = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},  # 1s overlap
            {"start": 2.0, "end": 5.0, "speaker": "Speaker 2"},  # 2s overlap
        ]
        result = self.assign_speakers(transcription_segments, diarization_segments)
        assert result[0]["speaker"] == "Speaker 2"

    def test_no_overlap(self):
        transcription_segments = [
            {"start": 10.0, "end": 12.0, "text": "Hello"},
        ]
        diarization_segments = [
            {"start": 0.0, "end": 5.0, "speaker": "Speaker 1"},
        ]
        result = self.assign_speakers(transcription_segments, diarization_segments)
        assert "speaker" not in result[0]

    def test_empty_diarization(self):
        transcription_segments = [
            {"start": 0.0, "end": 3.0, "text": "Hello"},
        ]
        result = self.assign_speakers(transcription_segments, [])
        assert result == transcription_segments

    def test_does_not_mutate_original(self):
        original = [{"start": 0.0, "end": 3.0, "text": "Hello"}]
        diarization = [{"start": 0.0, "end": 3.0, "speaker": "Speaker 1"}]
        result = self.assign_speakers(original, diarization)
        assert "speaker" not in original[0]
        assert result[0]["speaker"] == "Speaker 1"

    def test_multiple_speakers(self):
        transcription_segments = [
            {"start": 0.0, "end": 2.0, "text": "A"},
            {"start": 2.0, "end": 4.0, "text": "B"},
            {"start": 4.0, "end": 6.0, "text": "C"},
        ]
        diarization_segments = [
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},
            {"start": 2.0, "end": 4.0, "speaker": "Speaker 2"},
            {"start": 4.0, "end": 6.0, "speaker": "Speaker 1"},
        ]
        result = self.assign_speakers(transcription_segments, diarization_segments)
        assert result[0]["speaker"] == "Speaker 1"
        assert result[1]["speaker"] == "Speaker 2"
        assert result[2]["speaker"] == "Speaker 1"


class TestProcessResult:
    """Tests for SpeakerDiarizer.process_result with mocked pipeline."""

    @pytest.fixture
    def mock_diarizer(self):
        """Create a SpeakerDiarizer with a mocked pipeline."""
        with patch.dict("sys.modules", {"pyannote.audio": MagicMock(), "pyannote": MagicMock()}):
            from diarize import SpeakerDiarizer
            with patch.object(SpeakerDiarizer, "__init__", lambda self, **kw: None):
                diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
                diarizer.hf_token = "fake"
                diarizer.config = {}
                diarizer.model_name = "test"
                diarizer._pipeline = None
                return diarizer

    def test_process_result_with_segments(self, mock_diarizer):
        mock_diarizer.diarize = MagicMock(return_value=[
            {"start": 0.0, "end": 3.0, "speaker": "Speaker 1"},
            {"start": 3.0, "end": 6.0, "speaker": "Speaker 2"},
        ])

        result = TranscriptionResult(
            text="Hello World",
            segments=[
                {"start": 0.0, "end": 3.0, "text": "Hello"},
                {"start": 3.0, "end": 6.0, "text": "World"},
            ],
        )

        result = mock_diarizer.process_result(result, "fake.wav")

        assert result.speakers == ["Speaker 1", "Speaker 2"]
        assert result.segments[0]["speaker"] == "Speaker 1"
        assert result.segments[1]["speaker"] == "Speaker 2"
        assert "Speaker 1: Hello" in result.text
        assert "Speaker 2: World" in result.text

    def test_process_result_speaker_change_only(self, mock_diarizer):
        """Text should only label on speaker change."""
        mock_diarizer.diarize = MagicMock(return_value=[
            {"start": 0.0, "end": 2.0, "speaker": "Speaker 1"},
            {"start": 2.0, "end": 4.0, "speaker": "Speaker 1"},
            {"start": 4.0, "end": 6.0, "speaker": "Speaker 2"},
        ])

        result = TranscriptionResult(
            text="A B C",
            segments=[
                {"start": 0.0, "end": 2.0, "text": "A"},
                {"start": 2.0, "end": 4.0, "text": "B"},
                {"start": 4.0, "end": 6.0, "text": "C"},
            ],
        )

        result = mock_diarizer.process_result(result, "fake.wav")
        lines = result.text.split("\n")
        assert lines[0] == "Speaker 1: A"
        assert lines[1] == "B"  # Same speaker, no prefix
        assert lines[2] == "Speaker 2: C"

    def test_process_result_no_segments(self, mock_diarizer):
        """When there are no transcription segments, prefix entire text."""
        mock_diarizer.diarize = MagicMock(return_value=[
            {"start": 0.0, "end": 5.0, "speaker": "Speaker 1"},
        ])

        result = TranscriptionResult(text="Hello world")
        result = mock_diarizer.process_result(result, "fake.wav")
        assert result.text == "Speaker 1: Hello world"

    def test_process_result_empty_diarization(self, mock_diarizer):
        mock_diarizer.diarize = MagicMock(return_value=[])

        result = TranscriptionResult(text="Hello world")
        original_text = result.text
        result = mock_diarizer.process_result(result, "fake.wav")
        assert result.text == original_text


class TestSpeakerDiarizerInit:
    """Tests for SpeakerDiarizer initialization."""

    def test_missing_pyannote_raises(self):
        """When pyannote is not available, init should raise."""
        with patch.dict("sys.modules", {"pyannote.audio": None, "pyannote": None}):
            # Force re-import to get PYANNOTE_AVAILABLE=False
            import importlib
            import diarize
            original = diarize.PYANNOTE_AVAILABLE
            diarize.PYANNOTE_AVAILABLE = False
            try:
                with pytest.raises(RuntimeError, match="pyannote.audio not available"):
                    diarize.SpeakerDiarizer(hf_token="fake")
            finally:
                diarize.PYANNOTE_AVAILABLE = original

    def test_missing_hf_token_raises(self):
        """When no HF token is provided, init should raise."""
        with patch.dict("sys.modules", {"pyannote.audio": MagicMock(), "pyannote": MagicMock()}):
            import diarize
            original = diarize.PYANNOTE_AVAILABLE
            diarize.PYANNOTE_AVAILABLE = True
            try:
                with patch.dict("os.environ", {}, clear=True):
                    with pytest.raises(RuntimeError, match="HuggingFace token required"):
                        diarize.SpeakerDiarizer()
            finally:
                diarize.PYANNOTE_AVAILABLE = original
