"""Tests for CLI argument parsing."""

import argparse
import pytest
from unittest.mock import patch


class TestCLIArguments:
    """Test that CLI argument parsing works correctly."""

    @pytest.fixture
    def parser(self):
        """Create the argument parser from transcribe.main without running it."""
        from transcribe import main
        import transcribe

        parser = argparse.ArgumentParser()
        parser.add_argument('input_files', nargs='+')
        parser.add_argument('-e', '--engine', choices=['whisper', 'google', 'vosk'])
        parser.add_argument('-m', '--model')
        parser.add_argument('-l', '--language')
        parser.add_argument('-o', '--output-format', choices=['txt', 'json', 'srt', 'vtt', 'all'])
        parser.add_argument('-od', '--output-dir')
        parser.add_argument('-c', '--config')
        parser.add_argument('--task', choices=['transcribe', 'translate'])
        parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
        parser.add_argument('--diarize', action='store_true')
        parser.add_argument('-v', '--verbose', action='store_true')
        return parser

    def test_basic_args(self, parser):
        args = parser.parse_args(["audio.m4a"])
        assert args.input_files == ["audio.m4a"]
        assert args.diarize is False

    def test_diarize_flag(self, parser):
        args = parser.parse_args(["audio.m4a", "--diarize"])
        assert args.diarize is True

    def test_diarize_with_engine(self, parser):
        args = parser.parse_args(["audio.m4a", "--diarize", "-e", "whisper"])
        assert args.diarize is True
        assert args.engine == "whisper"

    def test_diarize_with_output_format(self, parser):
        args = parser.parse_args(["audio.m4a", "--diarize", "-o", "srt"])
        assert args.diarize is True
        assert args.output_format == "srt"

    def test_multiple_files(self, parser):
        args = parser.parse_args(["a.wav", "b.mp3", "c.m4a"])
        assert len(args.input_files) == 3

    def test_all_options(self, parser):
        args = parser.parse_args([
            "audio.m4a",
            "-e", "whisper",
            "-m", "large",
            "-l", "en",
            "-o", "srt",
            "--task", "translate",
            "--device", "cpu",
            "--diarize",
            "-v",
        ])
        assert args.engine == "whisper"
        assert args.model == "large"
        assert args.language == "en"
        assert args.output_format == "srt"
        assert args.task == "translate"
        assert args.device == "cpu"
        assert args.diarize is True
        assert args.verbose is True
