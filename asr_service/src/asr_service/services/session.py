"""
Active session service.

Manages a single multi-source recording session with full lifecycle.
Orchestrates N SourcePipelines, ChronologicalMerger, and WebSocket broadcasting.
"""

import time
import asyncio
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import concurrent.futures

from ..core.config import settings
from ..core.logging import logger
from ..core.exceptions import ModelLoadingError, TranscriptionError
from ..schemas.transcription import (
    SessionState,
    Utterance,
    SourceConfig,
    TranscriptDocument,
    ColdTranscriptResult,
)
from ..schemas.websocket import (
    WSStateChangeMessage,
    WSUtteranceMessage,
    WSFinalTranscriptMessage,
)
from .source_pipeline import SourcePipeline
from .transcript_merger import ChronologicalMerger
from .audio_mixer import AudioMixer
from .cold_transcriber import ColdPathPostProcessor
from .vad_producer import VADAudioProducer
from .screencapture_producer import ScreenCaptureAudioProducer


class ActiveSession:
    """
    Manages a single multi-source recording session.

    State machine: INITIALIZING → RECORDING → STOPPING → PROCESSING → COMPLETED/FAILED

    Owns:
    - N SourcePipelines (one per audio source)
    - ChronologicalMerger (fan-in)
    - AudioMixer (multi-source → mono)
    - WebSocket broadcaster
    """

    def __init__(
        self,
        session_id: str,
        sources: List[SourceConfig],
        model_manager,
        output_dir: Path | str,
    ):
        """
        Initialize active session.

        Args:
            session_id: Unique session identifier
            sources: List of audio sources to capture
            model_manager: ModelManager instance
            output_dir: Directory to save output files
        """
        self.session_id = session_id
        self.sources = sources
        self.model_manager = model_manager

        # Create timestamped subdirectory (YYYY-MM-DD-HH-MM)
        base_output_dir = Path(output_dir).expanduser().resolve()
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.output_dir = base_output_dir / timestamp
        self.base_output_dir = base_output_dir

        # State machine
        self.state = SessionState.INITIALIZING
        self._state_lock = threading.Lock()

        # Timing
        self.started_at: Optional[datetime] = None
        self.ended_at: Optional[datetime] = None
        self._session_start_time: float = 0.0

        # Components
        self.pipelines: List[SourcePipeline] = []
        self.merger = ChronologicalMerger()
        self.mixer = AudioMixer()

        # WebSocket connections
        self._websocket_clients: List[WebSocket] = []
        self._ws_lock = asyncio.Lock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = (
            None  # Store event loop for thread-safe async calls
        )

        # Results
        self.live_transcript: Optional[List[Utterance]] = None
        self.final_transcript: Optional[ColdTranscriptResult] = None
        self.audio_path: Optional[Path] = None
        self.transcript_path: Optional[Path] = None

        logger.info(
            f"Session {session_id} initialized with {len(sources)} sources "
            f"(output: {self.output_dir})"
        )

    async def initialize(self):
        """
        INITIALIZING → RECORDING transition.

        Steps:
        1. Load models (via ModelManager)
        2. Create N SourcePipelines
        3. Register merger callback for WebSocket broadcasting
        4. Transition to RECORDING
        """
        try:
            logger.info(f"Initializing session {self.session_id}...")

            # Store event loop for thread-safe async calls from pipeline threads
            self._event_loop = asyncio.get_running_loop()
            logger.debug(f"Captured event loop: {self._event_loop}")

            # Load models
            await self.model_manager.load_models()

            # Create pipelines for each source
            for idx, source_config in enumerate(self.sources):
                # Create appropriate producer based on source_type
                producer = self._create_producer(idx, source_config)

                # Create pipeline with producer
                pipeline = SourcePipeline(
                    source_id=idx,
                    producer=producer,
                    whisper_model_name=self.model_manager.whisper_model_name,
                    utterance_callback=self._on_utterance,
                    language="en",
                )
                self.pipelines.append(pipeline)

            # Register merger → WebSocket broadcast
            self.merger.add_listener(self._broadcast_utterance_sync)

            # Transition state
            self._set_state(SessionState.RECORDING)

            logger.info(f"Session {self.session_id} initialized successfully")

        except Exception as e:
            logger.error(
                f"Session {self.session_id} initialization failed: {e}", exc_info=True
            )
            self._set_state(SessionState.FAILED)
            raise

    def start_recording(self):
        """
        Start all source pipelines.

        Begins audio capture and live transcription.
        """
        logger.info(f"Starting recording for session {self.session_id}...")

        self.started_at = datetime.now()
        self._session_start_time = time.time()

        # Start all pipelines
        for pipeline in self.pipelines:
            pipeline.start(self._session_start_time)

        logger.info(f"Session {self.session_id} recording started")

    async def stop_recording(self):
        """
        RECORDING → STOPPING → PROCESSING transition (cold path runs in background).

        Steps:
        1. Stop all pipelines gracefully
        2. Collect audio from each source
        3. Mix to mono
        4. Save mixed audio
        5. Transition to PROCESSING
        6. Schedule cold path post-processing in background (returns immediately)

        Cold path completes asynchronously and transitions to COMPLETED when done.
        """
        logger.info(f"Stopping recording for session {self.session_id}...")
        self._set_state(SessionState.STOPPING)

        try:
            # Stop all pipelines (in parallel)
            logger.info(f"Stopping {len(self.pipelines)} pipelines...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.map(lambda p: p.stop(), self.pipelines)

            # Collect audio from each source
            logger.info("Collecting audio from sources...")
            audio_sources = [p.get_audio() for p in self.pipelines]

            # Ensure output directory exists only when saving outputs
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Mix and save
            logger.info("Mixing audio to mono...")
            mixed_path = self.output_dir / f"{self.session_id}_mixed.wav"
            self.audio_path = self.mixer.mix_to_mono(audio_sources, mixed_path)

            # Also save multi-channel for debugging
            multi_path = self.output_dir / f"{self.session_id}_multichannel.wav"
            self.mixer.save_multi_channel(audio_sources, multi_path)

            # Get live transcript
            self.live_transcript = self.merger.get_all_utterances()
            logger.info(f"Live transcript: {len(self.live_transcript)} utterances")

            # Record end time
            self.ended_at = datetime.now()

            # Transition to processing
            self._set_state(SessionState.PROCESSING)

            # Schedule cold path to run in background (fire-and-forget)
            asyncio.create_task(self._run_cold_path_background())

            logger.info(
                f"Session {self.session_id} stop completed, cold path processing in background"
            )

        except Exception as e:
            logger.error(f"Session {self.session_id} stop failed: {e}", exc_info=True)
            self._set_state(SessionState.FAILED)
            self._cleanup_empty_output_dir()
            raise

    def _cleanup_empty_output_dir(self):
        """Remove output directory if it exists and is empty."""
        try:
            if self.output_dir.exists() and not any(self.output_dir.iterdir()):
                self.output_dir.rmdir()
                logger.info(
                    f"Session {self.session_id}: Removed empty output directory {self.output_dir}"
                )
        except OSError as e:
            logger.warning(
                f"Session {self.session_id}: Failed to remove empty output directory: {e}"
            )

    async def _run_cold_path_background(self):
        """
        Run cold path post-processing in background and transition to COMPLETED.

        This is called as a background task after stop_recording() completes,
        allowing the frontend to start a new recording immediately while the
        cold pipeline processes in the background.
        """
        try:
            await self._run_cold_path()
            # Transition to completed after cold path finishes
            self._set_state(SessionState.COMPLETED)
            logger.info(
                f"Session {self.session_id} completed successfully (background)"
            )
        except (ModelLoadingError, TranscriptionError, FileNotFoundError, RuntimeError) as e:
            logger.error(
                f"Session {self.session_id} cold path failed: {e}", exc_info=True
            )
            self._set_state(SessionState.FAILED)

    async def _run_cold_path(self):
        """Run cold path post-processing in executor."""
        logger.info(f"Running cold path for session {self.session_id}...")

        loop = asyncio.get_event_loop()

        def _process():
            processor = ColdPathPostProcessor(self.model_manager.get_cold_pipeline())
            return processor.process_long_audio(
                self.audio_path, chunk_duration=settings.COLD_PATH_CHUNK_DURATION
            )

        # Run in executor to not block event loop
        result = await loop.run_in_executor(None, _process)

        # Convert to ColdTranscriptResult
        self.final_transcript = ColdTranscriptResult(
            segments=result["segments"],
            duration=result["duration"],
            language=result["language"],
        )

        logger.info(
            f"Cold path complete: {len(self.final_transcript.segments)} segments, "
            f"{self.final_transcript.duration:.2f}s"
        )

        # Save transcripts (JSON + Markdown)
        self._save_transcript()
        self._save_transcript_markdown()

        # Broadcast final transcript
        await self._broadcast_message(
            WSFinalTranscriptMessage(
                type="final_transcript", transcript=self.final_transcript
            )
        )

    def _create_producer(self, source_id: int, source_config: SourceConfig):
        """
        Create an appropriate audio producer based on source_type.

        Args:
            source_id: Unique source identifier
            source_config: Source configuration with device_index, device_name, source_type

        Returns:
            AudioProducerBase instance (VADAudioProducer or ScreenCaptureAudioProducer)
        """
        source_type = getattr(source_config, "source_type", "sounddevice")

        if source_type == "screencapture":
            logger.info(f"Creating ScreenCaptureAudioProducer for source {source_id}")
            return ScreenCaptureAudioProducer(
                source_id=source_id,
                device_name=source_config.device_name,
                output_queue=queue.Queue(),  # Will be overwritten by SourcePipeline
            )
        else:
            # Default: sounddevice
            logger.info(
                f"Creating VADAudioProducer for source {source_id} "
                f"(device_index={source_config.device_index})"
            )
            return VADAudioProducer(
                source_id=source_id,
                device_index=source_config.device_index,
                device_name=source_config.device_name,
                vad_model=self.model_manager.vad_model,
                output_queue=queue.Queue(),  # Will be overwritten by SourcePipeline
                device_channels=source_config.device_channels,
            )

    def _on_utterance(self, utterance: Utterance):
        """
        Callback from SourcePipeline → ChronologicalMerger.

        Runs in pipeline thread context.
        """
        self.merger.add_utterance(utterance)

    def _broadcast_utterance_sync(self, utterance: Utterance):
        """
        Callback from ChronologicalMerger → WebSocket clients.

        Runs in pipeline thread context, so we need to schedule async broadcast.
        """
        logger.debug(
            f"Broadcasting utterance from source {utterance.source_id}: "
            f"'{utterance.text[:50]}...' (clients: {len(self._websocket_clients)})"
        )

        # Use stored event loop (set during initialize())
        if self._event_loop is None:
            logger.warning(
                f"Event loop not set - utterance from source {utterance.source_id} NOT sent"
            )
            return

        # Schedule async broadcast using thread-safe method (fire-and-forget)
        message = WSUtteranceMessage(type="utterance", data=utterance)
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message(message), self._event_loop
            )
            logger.debug(
                f"Utterance broadcast scheduled to event loop {self._event_loop}"
            )
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to schedule utterance broadcast: {e}", exc_info=True)

    async def _broadcast_message(self, message):
        """
        Send message to all connected WebSocket clients.

        Args:
            message: WSMessage (Pydantic model)
        """
        async with self._ws_lock:
            disconnected = []
            for ws in self._websocket_clients:
                try:
                    await ws.send_json(message.model_dump())
                except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
                    logger.warning(f"WebSocket send failed: {e}")
                    disconnected.append(ws)

            # Remove disconnected clients
            for ws in disconnected:
                self._websocket_clients.remove(ws)

    async def add_websocket_client(self, websocket: WebSocket):
        """
        Add WebSocket client and send backlog.

        Args:
            websocket: FastAPI WebSocket connection
        """
        async with self._ws_lock:
            self._websocket_clients.append(websocket)

        logger.info(
            f"WebSocket client added to session {self.session_id} "
            f"(total: {len(self._websocket_clients)})"
        )

        # Send current state
        await websocket.send_json(
            WSStateChangeMessage(type="state_change", state=self.state).model_dump()
        )

        # Send backlog (last 20 utterances)
        for utterance in self.merger.get_recent_utterances(count=20):
            await websocket.send_json(
                WSUtteranceMessage(type="utterance", data=utterance).model_dump()
            )

    async def remove_websocket_client(self, websocket: WebSocket):
        """
        Remove WebSocket client.

        Args:
            websocket: FastAPI WebSocket connection
        """
        async with self._ws_lock:
            if websocket in self._websocket_clients:
                self._websocket_clients.remove(websocket)

        logger.info(
            f"WebSocket client removed from session {self.session_id} "
            f"(remaining: {len(self._websocket_clients)})"
        )

    def _set_state(self, new_state: SessionState):
        """
        Update state and broadcast to clients.

        Args:
            new_state: New session state
        """
        with self._state_lock:
            old_state = self.state
            self.state = new_state

        logger.info(f"Session {self.session_id}: {old_state.value} → {new_state.value}")

        # Broadcast state change (fire and forget)
        if self._event_loop is None:
            # No event loop yet - this is expected during initialization before initialize() is called
            logger.debug(
                f"Event loop not yet set, skipping state broadcast for {new_state.value}"
            )
            return

        # Schedule async broadcast using thread-safe method
        message = WSStateChangeMessage(type="state_change", state=new_state)
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message(message), self._event_loop
            )
        except (RuntimeError, ValueError) as e:
            logger.error(f"Failed to schedule state broadcast: {e}", exc_info=True)

    def get_document(self) -> TranscriptDocument:
        """
        Build transcript document for this session.

        Returns:
            TranscriptDocument with all session data
        """
        duration = (
            (self.ended_at - self.started_at).total_seconds()
            if self.ended_at and self.started_at
            else 0.0
        )

        return TranscriptDocument(
            session_id=self.session_id,
            started_at=self.started_at or datetime.now(),
            ended_at=self.ended_at,
            duration_seconds=duration,
            state=self.state,
            utterances=self.live_transcript or [],
            audio_file_path=str(self.audio_path) if self.audio_path else None,
        )

    def _save_transcript(self):
        """
        Save final cold path transcript to JSON file.

        Saves the final transcript result to a JSON file in the session directory.
        """
        if not self.final_transcript:
            logger.warning(f"Session {self.session_id}: No final transcript to save")
            return

        try:
            self.transcript_path = self.output_dir / "transcript.json"
            transcript_data = self.final_transcript.model_dump(mode="json")

            self.transcript_path.write_text(
                __import__("json").dumps(transcript_data, indent=2, default=str)
            )
            logger.info(
                f"Session {self.session_id}: Transcript saved to {self.transcript_path}"
            )
        except (OSError, ValueError) as e:
            logger.error(
                f"Session {self.session_id}: Failed to save transcript: {e}",
                exc_info=True,
            )

    def _save_transcript_markdown(self):
        """
        Save final cold path transcript to Markdown file.

        Creates a human-readable markdown transcript with speaker labels and timestamps.
        """
        if not self.final_transcript:
            logger.warning(
                f"Session {self.session_id}: No final transcript to save as markdown"
            )
            return

        try:
            markdown_path = self.output_dir / "transcript.md"

            # Format: convert seconds to HH:MM:SS
            def format_timestamp(seconds: float) -> str:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            # Build markdown content
            lines = []

            # Header with date
            date_str = (
                self.started_at.strftime("%Y-%m-%d")
                if self.started_at
                else "Unknown Date"
            )
            lines.append(f"# Meeting Transcript: {date_str}")

            # Metadata
            duration = self.final_transcript.duration
            language = self.final_transcript.language
            lines.append(
                f"**Duration:** {format_timestamp(duration)} | **Language:** {language}"
            )
            lines.append("---")
            lines.append("## Transcript")

            # Segments
            for segment in self.final_transcript.segments:
                timestamp = format_timestamp(segment.start)
                speaker = segment.speaker or "UNKNOWN"
                text = segment.text.strip()
                if text:
                    lines.append(f"**[{timestamp}] {speaker}:** {text}")

            # Write to file
            markdown_path.write_text("\n".join(lines))
            logger.info(
                f"Session {self.session_id}: Markdown transcript saved to {markdown_path}"
            )
        except (OSError, ValueError) as e:
            logger.error(
                f"Session {self.session_id}: Failed to save markdown transcript: {e}",
                exc_info=True,
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dictionary with session stats
        """
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "source_count": len(self.pipelines),
            "websocket_clients": len(self._websocket_clients),
            "live_utterances": len(self.live_transcript) if self.live_transcript else 0,
            "final_segments": (
                len(self.final_transcript.segments) if self.final_transcript else 0
            ),
            "merger_stats": self.merger.get_stats(),
            "pipelines": [p.get_stats() for p in self.pipelines],
        }
