#!/usr/bin/env python3
"""
Cold Path Pipeline for High-Accuracy Transcription

This pipeline implements:
1. Speaker Diarization (Pyannote.audio 3.1+)
2. Voice Activity Detection (Silero VAD via faster-whisper)
3. ASR Transcription (Whisper large-v3-turbo)
4. Word-level Alignment (WhisperX)

Usage:
    from cold_path_pipeline import ColdPathPipeline

    pipeline = ColdPathPipeline()
    result = pipeline.process("audio.mp3")

    for segment in result['segments']:
        print(f"[{segment['speaker']}] {segment['text']}")
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import warnings

# Suppress known harmless warnings BEFORE importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.core.io")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

import torch
import whisperx
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline as DiarizationPipeline


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


class ColdPathPipeline:
    """
    High-accuracy transcription pipeline for post-meeting processing.

    This pipeline sacrifices speed for accuracy, using:
    - Pyannote for robust speaker diarization
    - Silero VAD to strip silence
    - Whisper large-v3-turbo for fast, accurate transcription
    - WhisperX for word-level alignment with diarization
    """

    def __init__(
        self,
        whisper_model: str = "large-v3-turbo",
        diarization_model: str = "pyannote/speaker-diarization-3.1",
        device: Optional[str] = None,
        compute_type: str = "float32",
        use_diarization: bool = True,
        hf_token: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the Cold Path pipeline.

        Args:
            whisper_model: Whisper model to use (large-v3-turbo recommended)
            diarization_model: Pyannote diarization model to use
            device: Device to use (auto-detected if None)
            compute_type: Compute type for faster-whisper
            use_diarization: Whether to perform speaker diarization
            hf_token: HuggingFace token for Pyannote models (required for diarization)
            verbose: Whether to print progress
        """
        self.verbose = verbose
        self.use_diarization = use_diarization

        # Auto-detect device
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
            print(f"🔧 Initializing Cold Path Pipeline")
            print(f"   Device: {self.device}")
            print(f"   Whisper model: {whisper_model}")
            print(f"   Diarization: {'enabled' if use_diarization else 'disabled'}")

        # Load Whisper model
        with Timer("Loading Whisper model", verbose=self.verbose):
            self.whisper_model = WhisperModel(
                whisper_model,
                device=self.device if self.device != "mps" else "cpu",  # faster-whisper doesn't support MPS
                compute_type=compute_type
            )

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
        Transcribe audio with VAD filtering and word-level timestamps.

        Args:
            audio_path: Path to audio file
            language: Language code (default: "en")

        Returns:
            Dictionary with transcription results
        """
        with Timer("Transcribing audio", verbose=self.verbose):
            segments, info = self.whisper_model.transcribe(
                str(audio_path),
                language=language,

                # Hallucination prevention
                condition_on_previous_text=False,

                # VAD filtering (uses Silero VAD internally)
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Minimum silence duration to consider
                    threshold=0.5,                # VAD threshold
                    min_speech_duration_ms=250,   # Minimum speech duration
                ),

                # Word-level timestamps
                word_timestamps=True,

                # Prevent repetition
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

            # Convert to list
            segments = list(segments)

        return {
            "segments": segments,
            "info": info,
            "language": info.language,
            "duration": info.duration
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

            # Convert faster-whisper segments to WhisperX format
            segments_dict = []
            for seg in transcription_result['segments']:
                segments_dict.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text
                })

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
                        segment_mid = (segment_start + segment_end) / 2

                        # Find speaker at segment midpoint by cropping annotation
                        # Crop returns speakers active in this time window
                        from pyannote.core import Segment as PyannoteSegment
                        window = PyannoteSegment(segment_start, segment_end)
                        speakers_in_window = annotation.crop(window)

                        # Get the speaker who speaks most in this window
                        if speakers_in_window:
                            # argmax returns the most active speaker
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
        use_alignment: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio file through the complete Cold Path pipeline.

        Args:
            audio_path: Path to audio file
            language: Language code
            use_alignment: Whether to use WhisperX alignment

        Returns:
            Complete transcription with speaker labels and timestamps
        """
        audio_path = Path(audio_path)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {audio_path.name}")
            print(f"{'='*60}\n")

        # Step 1: Transcribe with VAD
        transcription = self.transcribe(audio_path, language)

        # Step 2: Diarize (if enabled)
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
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "speaker": None
                    }
                    for seg in transcription['segments']
                ]
            }

        # Add metadata
        result['duration'] = transcription['duration']
        result['language'] = transcription['language']

        return result

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
        print("Usage: python cold_path_pipeline.py <audio_file> [--no-diarization]")
        sys.exit(1)

    audio_path = sys.argv[1]
    use_diarization = "--no-diarization" not in sys.argv

    # Get HF token from environment
    import os
    hf_token = os.getenv("HF_TOKEN")

    # Create pipeline
    pipeline = ColdPathPipeline(
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
