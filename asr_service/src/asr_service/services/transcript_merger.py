"""
Transcript merger service.

Fan-in component that merges utterances from N sources chronologically.
Detects and marks overlapping speech segments.
"""

import threading
from typing import List, Callable
from sortedcontainers import SortedList

from ..core.logging import logger
from ..schemas.transcription import Utterance


class ChronologicalMerger:
    """
    Fan-in component that merges utterances from N sources chronologically.

    Uses SortedList for efficient chronological insertion.
    Detects and marks overlapping segments bidirectionally.
    Thread-safe with lock for concurrent source updates.
    """

    def __init__(self):
        """Initialize chronological merger."""
        # SortedList maintains chronological order automatically
        # Key function: sort by start_time
        self._utterances: SortedList[Utterance] = SortedList(key=lambda u: u.start_time)

        # Thread safety for concurrent source updates
        self._lock = threading.Lock()

        # Callbacks for real-time updates
        self._listeners: List[Callable[[Utterance], None]] = []

        logger.info("ChronologicalMerger initialized")

    def add_listener(self, callback: Callable[[Utterance], None]):
        """
        Register callback for real-time utterance updates.

        Args:
            callback: Function to call when new utterance is added
        """
        self._listeners.append(callback)
        logger.debug(f"Added listener to ChronologicalMerger (total: {len(self._listeners)})")

    def add_utterance(self, utterance: Utterance):
        """
        Add utterance from any source, maintaining chronological order.

        Detects overlaps with existing utterances and marks them bidirectionally.

        Args:
            utterance: Utterance to add

        Overlap detection:
            Two utterances overlap if:
            (new.start < existing.end) AND (new.end > existing.start)
        """
        with self._lock:
            # Mark overlaps
            utterance.overlaps_with = self._find_overlaps(utterance)

            # Insert in chronological order
            self._utterances.add(utterance)

            logger.debug(
                f"Added utterance from source {utterance.source_id}: "
                f"[{utterance.start_time:.2f}-{utterance.end_time:.2f}] "
                f"'{utterance.text[:30]}...' (overlaps: {utterance.overlaps_with})"
            )

        # Notify listeners (outside lock to prevent deadlock)
        for listener in self._listeners:
            try:
                listener(utterance)
            except Exception as e:
                logger.error(f"Listener callback error: {e}", exc_info=True)

    def _find_overlaps(self, new_utterance: Utterance) -> List[int]:
        """
        Find all utterances that overlap with the new one.

        Overlap condition:
            (new.start < existing.end) AND (new.end > existing.start)

        Args:
            new_utterance: Utterance to check for overlaps

        Returns:
            List of source_ids that overlap with new_utterance
        """
        overlaps = []

        for existing in self._utterances:
            # Early exit if existing is way before new (no overlap possible)
            if existing.end_time < new_utterance.start_time:
                continue

            # Early exit if existing is way after new (sorted list, no more overlaps)
            if existing.start_time > new_utterance.end_time:
                break

            # Check overlap condition
            if (
                new_utterance.start_time < existing.end_time
                and new_utterance.end_time > existing.start_time
            ):
                # Overlap detected
                overlaps.append(existing.source_id)

                # Mark the existing utterance too (bidirectional)
                if new_utterance.source_id not in existing.overlaps_with:
                    existing.overlaps_with.append(new_utterance.source_id)
                    logger.debug(
                        f"Bidirectional overlap: source {new_utterance.source_id} ↔ {existing.source_id}"
                    )

        return overlaps

    def get_all_utterances(self) -> List[Utterance]:
        """
        Get all utterances in chronological order.

        Returns:
            List of all utterances sorted by start_time
        """
        with self._lock:
            return list(self._utterances)

    def get_utterances_since(self, start_time: float) -> List[Utterance]:
        """
        Get utterances after a given timestamp.

        Args:
            start_time: Unix timestamp

        Returns:
            List of utterances with start_time >= start_time
        """
        with self._lock:
            # Use binary search to find first utterance after start_time
            # SortedList provides efficient slicing
            result = []
            for utterance in self._utterances:
                if utterance.start_time >= start_time:
                    result.append(utterance)
            return result

    def get_recent_utterances(self, count: int = 20) -> List[Utterance]:
        """
        Get the N most recent utterances.

        Args:
            count: Number of recent utterances to return

        Returns:
            List of most recent utterances (chronologically sorted)
        """
        with self._lock:
            if len(self._utterances) <= count:
                return list(self._utterances)
            return list(self._utterances[-count:])

    def clear(self):
        """Reset merger state (remove all utterances)."""
        with self._lock:
            self._utterances.clear()
            logger.info("ChronologicalMerger cleared")

    def get_stats(self) -> dict:
        """
        Get merger statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_utterances = len(self._utterances)
            overlapping_utterances = sum(
                1 for u in self._utterances if len(u.overlaps_with) > 0
            )

            return {
                'total_utterances': total_utterances,
                'overlapping_utterances': overlapping_utterances,
                'overlap_percentage': (
                    (overlapping_utterances / total_utterances * 100)
                    if total_utterances > 0
                    else 0.0
                ),
                'listener_count': len(self._listeners),
            }
