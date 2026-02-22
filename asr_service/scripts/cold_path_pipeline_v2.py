#!/usr/bin/env python3
"""
Cold Path Pipeline for High-Accuracy Transcription (MLX Version)

This pipeline implements:
1. Speaker Diarization (Pyannote.audio 3.1+)
2. ASR Transcription (MLX-Whisper large-v3-turbo with Metal acceleration)
3. Word-level Alignment (WhisperX)

Note: MLX version does NOT support context chaining via initial_prompt (causes hallucinations)

Usage:
    from cold_path_pipeline_v2 import ColdPathPipeline_MLX

    pipeline = ColdPathPipeline_MLX()
    result = pipeline.process("audio.mp3")

    for segment in result['segments']:
        print(f"[{segment['speaker']}] {segment['text']}")
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import warnings
import concurrent.futures
import os
import contextlib

# Suppress known harmless warnings BEFORE importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

import torch  # noqa: E402
import whisperx  # noqa: E402
import mlx_whisper  # noqa: E402
import soundfile as sf  # noqa: E402
from pyannote.audio import Pipeline as DiarizationPipeline  # noqa: E402


class Timer:
    """Context manager for timing operations."""
    def __init__(self, name: str, verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        if self.verbose:
            print(f"⏱️  {self.name}...", end="", flush=True)
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            print(f" ✓ ({self.elapsed:.2f}s)")


class ColdPathPipeline_MLX:
    """
    High-accuracy transcription pipeline for post-meeting processing using MLX.

    This pipeline uses MLX-Whisper for native M4 Metal acceleration:
    - Pyannote for robust speaker diarization
    - MLX-Whisper large-v3-turbo for fast, accurate transcription
    - WhisperX for word-level alignment with diarization

    Note: Does NOT support context chaining (causes hallucinations in MLX)
    """

    def __init__(
        self,
        whisper_model: str = "mlx-community/whisper-large-v3-turbo",
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        use_diarization: bool = True,
        hf_token: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Cold Path pipeline with MLX.

        Args:
            whisper_model: MLX Whisper model to use (mlx-community/whisper-large-v3-turbo recommended)
            diarization_model: Pyannote diarization model to use
            device: Device to use for diarization (auto-detected if None)
            use_diarization: Whether to perform speaker diarization
            hf_token: HuggingFace token for Pyannote models (required for diarization)
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.use_diarization = use_diarization
        self.whisper_model_name = whisper_model

        # Auto-detect device for diarization (MLX uses Metal automatically)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.verbose:
            print("🔧 Initializing Cold Path Pipeline (MLX)")
            print(f"   Device (diarization): {self.device}")
            print("   Device (MLX-Whisper): Metal")
            print(f"   Whisper model: {whisper_model}")
            print(f"   Diarization: {'enabled' if use_diarization else 'disabled'}")

        # MLX-Whisper loads models on first use - no pre-loading needed
        with Timer("Loading MLX-Whisper model", verbose=self.verbose):
            # First transcription will trigger model download/cache
            pass

        # Load diarization pipeline if enabled
        self.diarization_pipeline = None
        if use_diarization:
            if hf_token is None:
                warnings.warn(
                    "Diarization requested but no HuggingFace token provided. "
                    "Set hf_token parameter or HF_TOKEN environment variable. "
                    "Disabling diarization."
                )
                self.use_diarization = False
            else:
                with Timer("Loading Diarization model", verbose=self.verbose):
                    self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                        diarization_model,
                        token=hf_token
                    )
                    # Move to device
                    if self.device != "cpu":
                        self.diarization_pipeline.to(torch.device(self.device))

    def transcribe(self, audio_path: str | Path, language: str = "en") -> Dict[str, Any]:
        """
        Transcribe audio using MLX-Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")

        Returns:
            Dictionary with transcription results
        """
        with Timer("Transcribing audio with MLX", verbose=self.verbose):
            # Suppress MLX-Whisper's progress bars
            with contextlib.redirect_stdout(open(os.devnull, 'w')), \
                 contextlib.redirect_stderr(open(os.devnull, 'w')):

                result = mlx_whisper.transcribe(
                    str(audio_path),
                    path_or_hf_repo=self.whisper_model_name,
                    language=language,

                    # CRITICAL: NO initial_prompt - causes severe hallucinations in MLX
                    # initial_prompt=None,

                    # Hallucination prevention
                    condition_on_previous_text=False,

                    # Anti-hallucination parameters
                    temperature=0.0,  # Greedy decoding only
                    compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.6,

                    # Output format
                    word_timestamps=True,
                    verbose=False
                )

        # Convert to faster-whisper compatible format
        segments = result.get('segments', [])

        # Get actual audio duration (don't trust MLX segments - they can hallucinate beyond audio end)
        try:
            audio_info = sf.info(str(audio_path))
            duration = audio_info.duration
        except Exception:
            # Fallback to segment-based estimation
            duration = segments[-1]['end'] if segments else 0.0

        # Filter out segments that are beyond the actual audio duration
        # (MLX can hallucinate "Thank you" etc. on silence after audio ends)
        segments = [seg for seg in segments if seg['start'] < duration]

        return {
            "segments": segments,
            "language": language,
            "duration": duration
        }

    def diarize(self, audio_path: str | Path) -> Optional[Any]:
        """
        Perform speaker diarization on audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Diarization annotation or None if disabled
        """
        if not self.use_diarization or self.diarization_pipeline is None:
            return None

        with Timer("Performing speaker diarization", verbose=self.verbose):
            # Load audio as in-memory dictionary to avoid AudioDecoder issues
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))

            # Pyannote expects mono audio, convert if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Create audio dictionary for Pyannote
            audio_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            diarization = self.diarization_pipeline(audio_dict)

        return diarization

    def align_with_whisperx(
        self,
        audio_path: str | Path,
        transcription_result: Dict[str, Any],
        diarization: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Use WhisperX to align transcription with diarization.

        Args:
            audio_path: Path to audio file
            transcription_result: Result from transcribe()
            diarization: Diarization annotation from diarize()

        Returns:
            Aligned segments with speaker labels
        """
        with Timer("Aligning with WhisperX", verbose=self.verbose):
            try:
                # Load audio using WhisperX
                audio = whisperx.load_audio(str(audio_path))
            except Exception as e:
                if self.verbose:
                    print(f"Warning: WhisperX audio loading failed ({e}), using librosa fallback")
                # Fallback to librosa if WhisperX fails
                import librosa
                audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)

            # MLX-Whisper segments are already in dict format
            segments_dict = transcription_result['segments']

            result = {"segments": segments_dict, "language": transcription_result['language']}

            # Align whisper output
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=transcription_result['language'],
                    device=self.device
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    self.device,
                    return_char_alignments=False
                )
            except Exception as e:
                if self.verbose:
                    print(f"Warning: WhisperX alignment failed ({e}), skipping alignment")
                # Return segments without word-level alignment
                pass

            # Assign speaker labels if diarization available
            if diarization is not None:
                try:
                    # Access the actual annotation from DiarizeOutput
                    annotation = diarization.speaker_diarization

                    # Manual speaker assignment - iterate through segments and match with diarization
                    for segment in result.get("segments", []):
                        segment_start = segment["start"]
                        segment_end = segment["end"]

                        # Find speaker by cropping annotation to segment window
                        from pyannote.core import Segment as PyannoteSegment
                        window = PyannoteSegment(segment_start, segment_end)
                        speakers_in_window = annotation.crop(window)

                        # Get the speaker who speaks most in this window
                        if speakers_in_window:
                            speaker = speakers_in_window.argmax()
                            if speaker:
                                segment["speaker"] = speaker

                    if self.verbose:
                        print(f" (assigned {len([s for s in result['segments'] if 'speaker' in s])} of {len(result['segments'])} segments)")

                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Speaker assignment failed ({e}), continuing without speakers")

        return result

    def process(
        self,
        audio_path: str | Path,
        language: str = "en",
        use_alignment: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio file through the complete Cold Path pipeline.

        Args:
            audio_path: Path to audio file
            language: Language code
            use_alignment: Whether to use WhisperX alignment
            parallel: Whether to run diarization and transcription in parallel (default: True)

        Returns:
            Complete transcription with speaker labels and timestamps
        """
        audio_path = Path(audio_path)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {audio_path.name}")
            print(f"{'='*60}\n")

        # Run diarization and transcription in parallel for speed
        if parallel and self.use_diarization:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both tasks
                future_transcription = executor.submit(self.transcribe, audio_path, language)
                future_diarization = executor.submit(self.diarize, audio_path)

                # Wait for both to complete
                transcription = future_transcription.result()
                diarization = future_diarization.result()
        else:
            # Sequential processing
            transcription = self.transcribe(audio_path, language)
            diarization = None
            if self.use_diarization:
                diarization = self.diarize(audio_path)

        # Step 3: Align (if requested)
        if use_alignment:
            result = self.align_with_whisperx(audio_path, transcription, diarization)
        else:
            # Just return raw transcription
            result = {
                "segments": [
                    {
                        "start": seg['start'],
                        "end": seg['end'],
                        "text": seg['text'],
                        "speaker": None
                    }
                    for seg in transcription['segments']
                ]
            }

        # Step 4: Merge short segments (MLX doesn't have VAD filtering, creates micro-segments)
        if result.get('segments'):
            result['segments'] = self.merge_short_segments(
                result['segments'],
                min_gap=0.5,      # Merge if gap < 500ms
                min_duration=1.0  # Merge if segment < 1 second
            )

        # Add metadata
        result['duration'] = transcription['duration']
        result['language'] = transcription['language']

        return result

    def merge_short_segments(
        self,
        segments: list[Dict[str, Any]],
        min_gap: float = 0.5,
        min_duration: float = 1.0
    ) -> list[Dict[str, Any]]:
        """
        Merge segments that are too short or too close together.

        MLX-Whisper doesn't support VAD filtering, so it creates micro-segments
        at every tiny pause. This function merges them back together.

        Args:
            segments: List of segments with start, end, text, speaker
            min_gap: Minimum gap (seconds) between segments before merging
            min_duration: Merge segments shorter than this (seconds)

        Returns:
            List of merged segments
        """
        if not segments:
            return segments

        merged = []
        current = segments[0].copy()

        for seg in segments[1:]:
            gap = seg['start'] - current['end']
            same_speaker = seg.get('speaker') == current.get('speaker')
            is_short = (current['end'] - current['start']) < min_duration

            # Merge if: same speaker AND (gap is small OR current segment is very short)
            if same_speaker and (gap < min_gap or is_short):
                # Extend current segment
                current['end'] = seg['end']
                current['text'] = current['text'].strip() + ' ' + seg['text'].strip()
            else:
                # Finalize current segment
                merged.append(current)
                current = seg.copy()

        # Don't forget the last segment
        merged.append(current)

        return merged

    def format_transcript(self, result: Dict[str, Any]) -> str:
        """
        Format transcription result as human-readable text.

        Args:
            result: Result from process()

        Returns:
            Formatted transcript
        """
        lines = []
        lines.append(f"Duration: {result['duration']:.1f}s")
        lines.append(f"Language: {result['language']}")
        lines.append(f"\n{'='*60}\n")

        for segment in result['segments']:
            timestamp = f"[{segment['start']:.1f}s -> {segment['end']:.1f}s]"
            speaker = segment.get('speaker', 'UNKNOWN')
            text = segment['text'].strip()

            if speaker and speaker != 'UNKNOWN':
                lines.append(f"{timestamp} [{speaker}] {text}")
            else:
                lines.append(f"{timestamp} {text}")

        return "\n".join(lines)


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cold_path_pipeline_v2.py <audio_file> [--no-diarization]")
        sys.exit(1)

    audio_path = sys.argv[1]
    use_diarization = "--no-diarization" not in sys.argv

    # Get HF token from environment
    import os
    hf_token = os.getenv("HF_TOKEN")

    # Create pipeline
    pipeline = ColdPathPipeline_MLX(
        use_diarization=use_diarization,
        hf_token=hf_token,
        verbose=True
    )

    # Process audio
    result = pipeline.process(audio_path)

    # Print result
    print("\n" + pipeline.format_transcript(result))


if __name__ == "__main__":
    main()
