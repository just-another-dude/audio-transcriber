"""
Speaker diarization module using pyannote.audio.

This is an optional post-processing step that identifies and labels different
speakers in transcriptions. It works with all transcription engines by running
diarization separately and merging results by timestamp overlap.

Requires: pip install -r requirements-diarization.txt
Also requires a HuggingFace token with access to pyannote models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio."""

    def __init__(self, hf_token: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize diarizer.

        Args:
            hf_token: HuggingFace token. Falls back to HF_TOKEN env var.
            config: Optional diarization config dict from config.yaml.
        """
        if not PYANNOTE_AVAILABLE:
            raise RuntimeError(
                "pyannote.audio not available. Install with:\n"
                "  pip install -r requirements-diarization.txt\n"
                "You also need a HuggingFace token with access to pyannote models."
            )

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError(
                "HuggingFace token required for pyannote models. "
                "Set HF_TOKEN in your .env file or pass hf_token parameter.\n"
                "You must also accept the model terms at:\n"
                "  https://huggingface.co/pyannote/speaker-diarization-3.1"
            )

        self.config = config or {}
        self.model_name = self.config.get("model", "pyannote/speaker-diarization-3.1")
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the diarization pipeline."""
        if self._pipeline is None:
            logger.info(f"Loading diarization pipeline: {self.model_name}")
            self._pipeline = Pipeline.from_pretrained(
                self.model_name, use_auth_token=self.hf_token
            )
            logger.info("Diarization pipeline loaded")

    def diarize(self, audio_path: Union[str, Path]) -> List[Dict]:
        """Run diarization on an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of dicts with keys: start, end, speaker.
            Speaker labels are normalized to "Speaker 1", "Speaker 2", etc.
        """
        self._load_pipeline()

        params = {}
        min_speakers = self.config.get("min_speakers")
        max_speakers = self.config.get("max_speakers")
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        logger.info(f"Running diarization on {Path(audio_path).name}...")
        diarization = self._pipeline(str(audio_path), **params)

        # Normalize speaker labels to "Speaker 1", "Speaker 2", etc.
        speaker_map = {}
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"Speaker {len(speaker_map) + 1}"
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker_map[speaker],
            })

        logger.info(
            f"Diarization complete: {len(segments)} segments, "
            f"{len(speaker_map)} speakers"
        )
        return segments

    @staticmethod
    def assign_speakers(
        transcription_segments: List[Dict],
        diarization_segments: List[Dict],
    ) -> List[Dict]:
        """Assign speaker labels to transcription segments by timestamp overlap.

        For each transcription segment, finds the diarization segment with the
        highest temporal overlap and assigns that speaker.

        Args:
            transcription_segments: Segments from transcription (with start/end/text).
            diarization_segments: Segments from diarization (with start/end/speaker).

        Returns:
            Transcription segments with 'speaker' key added.
        """
        if not diarization_segments:
            return transcription_segments

        result = []
        for seg in transcription_segments:
            seg = dict(seg)  # copy
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)

            best_speaker = None
            best_overlap = 0.0

            for d_seg in diarization_segments:
                overlap_start = max(seg_start, d_seg["start"])
                overlap_end = min(seg_end, d_seg["end"])
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = d_seg["speaker"]

            if best_speaker:
                seg["speaker"] = best_speaker

            result.append(seg)

        return result

    def process_result(self, result, audio_path: Union[str, Path]):
        """Run diarization and merge with a TranscriptionResult.

        Modifies result in place: adds speaker labels to segments and
        rewrites result.text with speaker prefixes where speaker changes.

        Args:
            result: A TranscriptionResult object.
            audio_path: Path to the audio file (for diarization).

        Returns:
            The modified TranscriptionResult.
        """
        diarization_segments = self.diarize(audio_path)

        if not diarization_segments:
            logger.warning("No speakers detected by diarization")
            return result

        # Collect unique speakers
        speakers = sorted(set(d["speaker"] for d in diarization_segments))
        result.speakers = speakers

        if result.segments:
            # Assign speakers to existing segments
            result.segments = self.assign_speakers(result.segments, diarization_segments)

            # Rebuild text with speaker labels (only on speaker change)
            lines = []
            current_speaker = None
            for seg in result.segments:
                speaker = seg.get("speaker")
                text = seg.get("text", "").strip()
                if not text:
                    continue
                if speaker and speaker != current_speaker:
                    current_speaker = speaker
                    lines.append(f"{speaker}: {text}")
                else:
                    lines.append(text)
            result.text = "\n".join(lines)
        else:
            # No segments — assign based on full audio duration
            # This happens with Google Speech which returns no segments
            logger.warning(
                "No transcription segments available. Speaker labels will be "
                "based on diarization timeline only."
            )
            # Build text from diarization timeline with the full transcription
            # Since we can't split the text, just prefix with the first speaker
            if diarization_segments:
                result.text = f"{diarization_segments[0]['speaker']}: {result.text}"

        return result
