"""
Cold path post-processor service.

High-quality post-processing using cold path pipeline with smart chunking.
Handles long audio by splitting on silence boundaries.
"""

import sys
import time
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import TranscriptionError
from ..utils.file_ops import get_project_root


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
        try:
            repo_root = get_project_root()
            scripts_dir = repo_root / "scripts"
        except FileNotFoundError:
            logger.warning("Could not find project root, using relative path")
            scripts_dir = Path("scripts")

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
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )

        # CRITICAL: Convert VAD model to float32 to match input audio dtype
        # Some Silero VAD installations default to float64, causing dtype mismatches
        self._vad_model = self._vad_model.float()

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

        # Ensure audio is float32 (Silero VAD requirement)
        # soundfile.read() returns float32 by default, but explicitly enforce it
        audio = audio.astype(np.float32)

        # Convert stereo to mono if needed
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f}s")

        # If short enough, process directly
        if duration <= chunk_duration:
            logger.info(f"Audio ≤ {chunk_duration}s, processing directly")
            # CRITICAL: Serialize pipeline access (pyannote/Numba not thread-safe)
            from .model_manager import ModelManager
            with ModelManager._cold_pipeline_lock:
                return self.pipeline.process(str(audio_path))

        # Find silence boundaries for chunking
        logger.info(f"Finding silence boundaries for {chunk_duration}s chunks...")
        chunks = self._find_silence_chunks(audio, sr, chunk_duration, overlap)

        logger.info(f"Split into {len(chunks)} chunks")

        # Process each chunk using secure temporary directory
        all_segments = []
        import shutil
        temp_dir = tempfile.mkdtemp(prefix=f"asr_chunks_{uuid.uuid4().hex[:8]}_")
        try:
            for chunk_idx, (start_sample, end_sample) in tqdm(
                enumerate(chunks),
                total=len(chunks),
                desc="Transcribing",
                unit="chunk",
                miniters=max(1, len(chunks)//20),
                leave=False
            ):
                logger.debug(
                    f"Processing chunk {chunk_idx + 1}/{len(chunks)} "
                    f"({start_sample / sr:.1f}s - {end_sample / sr:.1f}s)"
                )

                # Extract chunk
                chunk_audio = audio[start_sample:end_sample]
                chunk_offset = start_sample / sr

                # Save temporary chunk file with UUID-based name for security
                chunk_path = Path(temp_dir) / f"chunk_{uuid.uuid4().hex}.wav"
                sf.write(str(chunk_path), chunk_audio, sr)

                try:
                    # Process chunk (with lock for thread safety)
                    from .model_manager import ModelManager
                    import threading
                    chunk_start_time = time.time()
                    logger.debug(f"Running Whisper + Diarization on chunk {chunk_idx + 1}...")

                    # Heartbeat thread to show progress during long transcription
                    heartbeat_stop = threading.Event()
                    def heartbeat():
                        while not heartbeat_stop.is_set():
                            heartbeat_stop.wait(60)  # Every 60 seconds
                            if not heartbeat_stop.is_set():
                                elapsed = time.time() - chunk_start_time
                                logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} still processing ({elapsed:.0f}s elapsed)")

                    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                    heartbeat_thread.start()

                    try:
                        with ModelManager._cold_pipeline_lock:
                            result = self.pipeline.process(str(chunk_path))
                    finally:
                        heartbeat_stop.set()
                        heartbeat_thread.join(timeout=1.0)

                    chunk_elapsed = time.time() - chunk_start_time

                    # Adjust timestamps
                    for seg in result["segments"]:
                        seg["start"] += chunk_offset
                        seg["end"] += chunk_offset

                    all_segments.extend(result["segments"])

                    logger.info(
                        f"Chunk {chunk_idx + 1}/{len(chunks)} completed in {chunk_elapsed:.1f}s ({len(result['segments'])} segments)"
                    )

                except Exception as e:
                    logger.error(f"Chunk {chunk_idx + 1} failed: {e}", exc_info=True)
                    raise TranscriptionError(f"Chunk {chunk_idx + 1} failed: {e}")

                finally:
                    # Cleanup temp file
                    chunk_path.unlink(missing_ok=True)

        finally:
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary chunk directory: {temp_dir}")

        logger.info(
            f"Cold path processing complete ({len(all_segments)} total segments)"
        )

        return {"segments": all_segments, "duration": duration, "language": "en"}

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

        # Run VAD on entire audio with tqdm progress bar
        chunk_size = 512  # 32ms at 16kHz
        vad_scores = []
        total_chunks = len(audio) // chunk_size

        start_time = time.time()

        # Use tqdm with miniters to update every 5%
        with tqdm(total=total_chunks, desc="VAD Analysis", unit="chunk", miniters=total_chunks//20, leave=False) as pbar:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    break
                vad_prob = self._vad_model(torch.from_numpy(chunk), sr).item()
                vad_scores.append((i, vad_prob))
                pbar.update(1)

        # Find silence regions (vad_prob < 0.3)
        silence_build_start = time.time()
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

        silence_build_elapsed = time.time() - silence_build_start
        vad_elapsed = time.time() - start_time
        logger.info(f"VAD analysis complete: {len(silence_regions)} silence regions found in {vad_elapsed:.1f}s")
        logger.debug(f"  VAD inference: {vad_elapsed - silence_build_elapsed:.1f}s, silence building: {silence_build_elapsed:.1f}s")

        # Split at silence near target boundaries
        logger.debug("Creating chunks at silence boundaries...")
        chunking_start = time.time()
        chunks = []
        target_samples = target_chunk_duration * sr
        current_start = 0
        chunk_iteration = 0

        # Estimate max chunks for progress bar (audio_duration / chunk_duration)
        estimated_chunks = int(len(audio) / (target_chunk_duration * sr)) + 2

        with tqdm(total=estimated_chunks, desc="Chunking", unit="chunk", miniters=max(1, estimated_chunks//20), leave=False) as pbar:
            while current_start < len(audio):
                chunk_iteration += 1

                target_end = current_start + target_samples

                # Find nearest silence to target_end, but enforce minimum chunk size
                # Don't split if it would create a chunk < 200s (2/3 of target)
                min_chunk_samples = int(0.66 * target_samples)  # 200s for 300s target

                best_split = target_end
                if silence_regions:
                    min_distance = float("inf")
                    for silence_start, silence_end in silence_regions:
                        silence_mid = (silence_start + silence_end) // 2

                        # Skip this silence if it would create too-short chunks
                        chunk_length = silence_mid - current_start
                        if chunk_length < min_chunk_samples:
                            continue

                        distance = abs(silence_mid - target_end)
                        if distance < min_distance:
                            min_distance = distance
                            best_split = silence_mid

                # Ensure we don't exceed audio length
                best_split = min(best_split, len(audio))

                chunks.append((current_start, best_split))
                pbar.update(1)

                # Break if we've reached the end
                if best_split >= len(audio):
                    logger.debug(f"Reached end of audio at chunk {chunk_iteration}")
                    break

                # Next chunk starts at (best_split - overlap)
                # CRITICAL: Ensure we always move forward, never backwards!
                next_start = best_split - int(overlap * sr)
                if next_start <= current_start:
                    # Overlap would cause backwards movement - advance by minimum amount
                    next_start = current_start + int(sr)  # Move forward by 1 second minimum
                    logger.debug(f"  Overlap adjustment: forcing forward progress at chunk {chunk_iteration}")

                current_start = next_start

                # Safety check: prevent infinite loops
                if chunk_iteration > 1000:
                    logger.warning(f"  WARNING: Chunking loop exceeded 1000 iterations!")
                    logger.warning(f"    current_start={current_start:,d}, audio_len={len(audio):,d}")
                    logger.warning(f"    best_split={best_split:,d}")
                    break

        # Final summary
        chunking_total = time.time() - chunking_start
        if chunks:
            total_samples = sum(end - start for start, end in chunks)
            avg_chunk_duration = total_samples / len(chunks) / sr

            logger.info(
                f"Created {len(chunks)} chunks (avg {avg_chunk_duration:.1f}s each) in {chunking_total:.1f}s"
            )
        return chunks
