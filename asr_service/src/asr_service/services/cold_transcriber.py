"""
Cold path post-processor service.

High-quality post-processing using cold path pipeline with smart chunking.
Handles long audio by splitting on silence boundaries.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import torch

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import TranscriptionError


class ColdPathPostProcessor:
    """
    High-quality post-processing using cold path pipeline.

    Features:
    - Smart chunking on silence boundaries for long audio
    - Speaker diarization with Pyannote
    - Word-level alignment with WhisperX
    - MLX-Whisper for fast, accurate transcription
    """

    def __init__(self, pipeline=None):
        """
        Initialize cold path post-processor.

        Args:
            pipeline: ColdPathPipeline_MLX instance (if None, will load on demand)
        """
        self.pipeline = pipeline
        self._vad_model = None

    def _ensure_pipeline(self):
        """Lazy-load pipeline if not provided."""
        if self.pipeline is not None:
            return

        logger.info("Lazy-loading cold path pipeline...")

        # Add scripts directory to path
        scripts_dir = Path(__file__).parent.parent.parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))

        from cold_path_pipeline_v2 import ColdPathPipeline_MLX

        self.pipeline = ColdPathPipeline_MLX(
            whisper_model=settings.MLX_WHISPER_MODEL,
            diarization_model=settings.DIARIZATION_MODEL,
            hf_token=settings.HF_TOKEN,
            use_diarization=True,
            verbose=False,
        )

        logger.info("Cold path pipeline loaded")

    def _ensure_vad_model(self):
        """Lazy-load VAD model for chunking."""
        if self._vad_model is not None:
            return

        logger.info("Loading VAD model for chunking...")
        self._vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            verbose=False,
        )
        logger.info("VAD model loaded")

    def process_long_audio(
        self,
        audio_path: Path | str,
        chunk_duration: int | None = None,
        overlap: int | None = None,
    ) -> Dict[str, Any]:
        """
        Process long audio by chunking on silence boundaries.

        Strategy:
        1. If audio ≤ chunk_duration, process directly
        2. Otherwise, split at silence near chunk boundaries
        3. Process each chunk with cold_path_pipeline_v2
        4. Adjust timestamps and merge results

        Args:
            audio_path: Path to audio file
            chunk_duration: Target chunk duration in seconds (default from settings)
            overlap: Overlap between chunks in seconds (default from settings)

        Returns:
            Dictionary with 'segments', 'duration', 'language'
        """
        self._ensure_pipeline()

        audio_path = Path(audio_path)
        chunk_duration = chunk_duration or settings.COLD_PATH_CHUNK_DURATION
        overlap = overlap or settings.COLD_PATH_OVERLAP

        logger.info(f"Processing audio: {audio_path.name}")

        # Load full audio
        audio, sr = sf.read(audio_path)

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f}s")

        # If short enough, process directly
        if duration <= chunk_duration:
            logger.info(f"Audio ≤ {chunk_duration}s, processing directly")
            return self.pipeline.process(str(audio_path))

        # Find silence boundaries for chunking
        logger.info(f"Finding silence boundaries for {chunk_duration}s chunks...")
        chunks = self._find_silence_chunks(audio, sr, chunk_duration, overlap)

        logger.info(f"Split into {len(chunks)} chunks")

        # Process each chunk
        all_segments = []
        for chunk_idx, (start_sample, end_sample) in enumerate(chunks):
            logger.info(
                f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                f"({start_sample / sr:.1f}s - {end_sample / sr:.1f}s)"
            )

            # Extract chunk
            chunk_audio = audio[start_sample:end_sample]
            chunk_offset = start_sample / sr

            # Save temporary chunk file
            chunk_path = audio_path.parent / f"{audio_path.stem}_chunk_{chunk_idx}.wav"
            sf.write(chunk_path, chunk_audio, sr)

            try:
                # Process chunk
                result = self.pipeline.process(str(chunk_path))

                # Adjust timestamps
                for seg in result['segments']:
                    seg['start'] += chunk_offset
                    seg['end'] += chunk_offset

                all_segments.extend(result['segments'])

                logger.info(f"Chunk {chunk_idx + 1} done ({len(result['segments'])} segments)")

            except Exception as e:
                logger.error(f"Chunk {chunk_idx + 1} failed: {e}", exc_info=True)
                raise TranscriptionError(f"Chunk {chunk_idx + 1} failed: {e}")

            finally:
                # Cleanup temp file
                chunk_path.unlink(missing_ok=True)

        logger.info(f"Cold path processing complete ({len(all_segments)} total segments)")

        return {'segments': all_segments, 'duration': duration, 'language': 'en'}

    def _find_silence_chunks(
        self,
        audio: np.ndarray,
        sr: int,
        target_chunk_duration: int,
        overlap: int,
    ) -> List[Tuple[int, int]]:
        """
        Find chunk boundaries at silence points.

        Strategy:
        1. Run Silero VAD on full audio
        2. Find silence regions (vad_prob < 0.3)
        3. Split at silence midpoints near target chunk boundaries
        4. Fallback to hard split if no silence found

        Args:
            audio: Audio array
            sr: Sample rate
            target_chunk_duration: Target duration in seconds
            overlap: Overlap in seconds

        Returns:
            List of (start_sample, end_sample) tuples
        """
        self._ensure_vad_model()

        logger.debug("Running VAD on full audio...")

        # Run VAD on entire audio
        chunk_size = 512  # 32ms at 16kHz
        vad_scores = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                break
            vad_prob = self._vad_model(torch.from_numpy(chunk), sr).item()
            vad_scores.append((i, vad_prob))

        # Find silence regions (vad_prob < 0.3)
        silence_regions = []
        current_silence_start = None
        for sample_idx, vad_prob in vad_scores:
            if vad_prob < 0.3:
                if current_silence_start is None:
                    current_silence_start = sample_idx
            else:
                if current_silence_start is not None:
                    silence_regions.append((current_silence_start, sample_idx))
                    current_silence_start = None

        logger.debug(f"Found {len(silence_regions)} silence regions")

        # Split at silence near target boundaries
        chunks = []
        target_samples = target_chunk_duration * sr
        current_start = 0

        while current_start < len(audio):
            target_end = current_start + target_samples

            # Find nearest silence to target_end
            best_split = target_end
            if silence_regions:
                min_distance = float('inf')
                for silence_start, silence_end in silence_regions:
                    silence_mid = (silence_start + silence_end) // 2
                    distance = abs(silence_mid - target_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_split = silence_mid

            # Ensure we don't exceed audio length
            best_split = min(best_split, len(audio))

            chunks.append((current_start, best_split))

            # Next chunk starts at (best_split - overlap)
            current_start = max(best_split - int(overlap * sr), best_split)

            # Break if we've reached the end
            if best_split >= len(audio):
                break

        logger.debug(f"Created {len(chunks)} chunks")
        return chunks
