"""
Memory effectiveness scoring system.

Tracks and manages memory effectiveness scores based on user feedback,
usage patterns, and time decay with configurable adjustments.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any


class FeedbackType(Enum):
    """Types of feedback for memory scoring."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class ScoringConfig:
    """Configuration for memory scoring system."""

    default_score: float = 5.0
    positive_adjustment: float = 0.3
    negative_adjustment: float = -0.3
    min_score: float = 0.0
    max_score: float = 10.0
    decay_rate: float = 0.01  # Daily decay for unused memories
    usage_boost: float = 0.1  # Boost per usage

    def __post_init__(self):
        """Validate configuration values."""
        if self.min_score >= self.max_score:
            raise ValueError("min_score must be less than max_score")
        if not self.min_score <= self.default_score <= self.max_score:
            raise ValueError(f"default_score must be between {self.min_score} and {self.max_score}")
        if self.positive_adjustment <= 0:
            raise ValueError("positive_adjustment must be positive")
        if self.negative_adjustment >= 0:
            raise ValueError("negative_adjustment must be negative")


@dataclass
class MemoryScore:
    """Score record for a memory item."""

    memory_id: str
    current_score: float
    initial_score: float
    usage_count: int = 0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    score_history: list[tuple[datetime, float]] = field(default_factory=list)

    @property
    def net_feedback(self) -> int:
        """Calculate net feedback (positive - negative)."""
        return self.positive_feedback_count - self.negative_feedback_count


class EffectivenessScorer:
    """
    Manages effectiveness scoring for memories.
    
    Provides thread-safe operations for scoring, feedback application,
    usage tracking, and score decay with configurable parameters.
    """

    def __init__(self, config: ScoringConfig | None = None):
        """
        Initialize the effectiveness scorer.
        
        Args:
            config: Scoring configuration (uses defaults if None)
        """
        self.config = config or ScoringConfig()
        self.scores: dict[str, MemoryScore] = {}
        self._lock = Lock()  # Thread safety for concurrent operations

    def initialize_score(self, memory_id: str) -> MemoryScore:
        """
        Initialize score for a new memory.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Initialized memory score record
        """
        with self._lock:
            if memory_id not in self.scores:
                score = MemoryScore(
                    memory_id=memory_id,
                    current_score=self.config.default_score,
                    initial_score=self.config.default_score,
                    score_history=[(datetime.now(), self.config.default_score)]
                )
                self.scores[memory_id] = score
            return self.scores[memory_id]

    def apply_feedback(
        self,
        memory_or_id: Any = None,  # Can be MemoryItem object or memory_id string
        is_positive_or_type: Any = True,  # Can be bool or FeedbackType
        **kwargs  # For additional parameters
    ) -> float:
        """
        Apply feedback to adjust memory score.
        
        Args:
            memory_or_id: MemoryItem object or memory_id string
            is_positive_or_type: Boolean (is_positive) or FeedbackType enum
            **kwargs: Additional parameters (is_positive, memory_id, feedback_type)
            
        Returns:
            New score after adjustment
        """
        # Parse arguments to support both old and new interfaces
        memory = None
        mem_id = None
        update_memory_obj = False

        # Determine memory ID
        if memory_or_id is not None:
            if isinstance(memory_or_id, str):
                # Old interface: first arg is memory_id
                mem_id = memory_or_id
            elif hasattr(memory_or_id, 'id'):
                # New interface: first arg is MemoryItem
                memory = memory_or_id
                mem_id = memory.id
                update_memory_obj = True
            else:
                # Could be memory object without id attribute
                mem_id = str(memory_or_id)

        # Check kwargs for alternative specifications
        if 'memory_id' in kwargs and kwargs['memory_id']:
            mem_id = kwargs['memory_id']
        elif 'memory' in kwargs and kwargs['memory'] and hasattr(kwargs['memory'], 'id'):
            memory = kwargs['memory']
            mem_id = memory.id
            update_memory_obj = True

        if not mem_id:
            raise ValueError("Either memory object or memory_id must be provided")

        # Determine feedback type
        if isinstance(is_positive_or_type, FeedbackType):
            # Old interface: second arg is FeedbackType
            fb_type = is_positive_or_type
        elif isinstance(is_positive_or_type, bool):
            # New interface: second arg is boolean
            fb_type = FeedbackType.POSITIVE if is_positive_or_type else FeedbackType.NEGATIVE
        elif 'feedback_type' in kwargs and kwargs['feedback_type']:
            # Specified in kwargs
            fb_type = kwargs['feedback_type']
        elif 'is_positive' in kwargs:
            # Specified as kwarg
            fb_type = FeedbackType.POSITIVE if kwargs['is_positive'] else FeedbackType.NEGATIVE
        else:
            # Default to positive
            fb_type = FeedbackType.POSITIVE

        with self._lock:
            if mem_id not in self.scores:
                self.initialize_score(mem_id)

            score_record = self.scores[mem_id]
            old_score = score_record.current_score

            # Apply adjustment based on feedback type
            if fb_type == FeedbackType.POSITIVE:
                adjustment = self.config.positive_adjustment
                score_record.positive_feedback_count += 1
            elif fb_type == FeedbackType.NEGATIVE:
                adjustment = self.config.negative_adjustment
                score_record.negative_feedback_count += 1
            else:
                adjustment = 0  # Neutral feedback

            # Calculate new score with boundaries
            new_score = old_score + adjustment
            new_score = max(self.config.min_score, min(self.config.max_score, new_score))

            # Update record
            score_record.current_score = new_score
            score_record.last_updated = datetime.now()
            score_record.score_history.append((datetime.now(), new_score))

            # Update memory object if provided
            if update_memory_obj and memory and hasattr(memory, 'effectiveness_score'):
                memory.effectiveness_score = new_score

            return new_score

    def track_usage(self, memory: Any) -> int:
        """
        Track memory usage (increment usage count and apply boost).
        
        Args:
            memory: MemoryItem object
            
        Returns:
            Updated usage count
        """
        if not memory or not hasattr(memory, 'id'):
            raise ValueError("Valid memory object required")

        mem_id = memory.id

        with self._lock:
            if mem_id not in self.scores:
                self.initialize_score(mem_id)

            score_record = self.scores[mem_id]

            # Increment usage count
            score_record.usage_count += 1
            score_record.last_accessed = datetime.now()

            # Apply usage boost to score
            old_score = score_record.current_score
            new_score = min(old_score + self.config.usage_boost, self.config.max_score)
            score_record.current_score = new_score
            score_record.last_updated = datetime.now()
            score_record.score_history.append((datetime.now(), new_score))

            # Update memory object
            if hasattr(memory, 'usage_count'):
                memory.usage_count = score_record.usage_count
            if hasattr(memory, 'effectiveness_score'):
                memory.effectiveness_score = new_score
            if hasattr(memory, 'last_accessed'):
                memory.last_accessed = score_record.last_accessed

            return score_record.usage_count

    def record_usage(
        self,
        memory_id: str,
        apply_boost: bool = False
    ) -> int:
        """
        Record memory usage and optionally apply usage boost.
        
        Args:
            memory_id: Memory identifier
            apply_boost: Whether to apply usage boost to score
            
        Returns:
            New score if boost applied, else usage count
        """
        with self._lock:
            if memory_id not in self.scores:
                self.initialize_score(memory_id)

            score_record = self.scores[memory_id]
            score_record.usage_count += 1
            score_record.last_accessed = datetime.now()

            if apply_boost:
                old_score = score_record.current_score
                new_score = min(
                    old_score + self.config.usage_boost,
                    self.config.max_score
                )
                score_record.current_score = new_score
                score_record.last_updated = datetime.now()
                score_record.score_history.append((datetime.now(), new_score))
                return new_score

            return score_record.usage_count

    def apply_decay(self, memory_id: str) -> float:
        """
        Apply time-based decay to memory score.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            New score after decay
        """
        with self._lock:
            if memory_id not in self.scores:
                return self.config.default_score

            score_record = self.scores[memory_id]

            # Calculate days since last access
            days_inactive = (datetime.now() - score_record.last_accessed).days

            if days_inactive > 0:
                # Apply decay
                decay_amount = self.config.decay_rate * days_inactive
                old_score = score_record.current_score
                new_score = max(
                    self.config.min_score,
                    old_score - decay_amount
                )

                score_record.current_score = new_score
                score_record.last_updated = datetime.now()
                score_record.score_history.append((datetime.now(), new_score))

                return new_score

            return score_record.current_score

    def apply_batch_feedback(
        self,
        feedback_map: dict[str, FeedbackType]
    ) -> dict[str, float]:
        """
        Apply feedback to multiple memories in batch.
        
        Args:
            feedback_map: Map of memory_id to feedback type
            
        Returns:
            Map of memory_id to new score
        """
        results = {}
        for memory_id, feedback_type in feedback_map.items():
            results[memory_id] = self.apply_feedback(memory_id, feedback_type)
        return results

    def normalize_scores(self) -> dict[str, float]:
        """
        Normalize all scores to maintain relative ordering.
        
        Returns:
            Map of memory_id to normalized score
        """
        with self._lock:
            if not self.scores:
                return {}

            # Get all current scores
            scores_list = [(mid, s.current_score) for mid, s in self.scores.items()]

            # Sort by score to maintain ordering
            scores_list.sort(key=lambda x: x[1])

            # Create normalized mapping (maintains relative ordering)
            normalized = {}
            for mid, _ in scores_list:
                normalized[mid] = self.scores[mid].current_score

            return normalized

    def get_top_memories(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get memories with highest scores.
        
        Args:
            n: Number of top memories to return
            
        Returns:
            List of (memory_id, score) tuples sorted by score
        """
        with self._lock:
            if not self.scores:
                return []

            # Sort by current score
            sorted_scores = sorted(
                [(mid, s.current_score) for mid, s in self.scores.items()],
                key=lambda x: x[1],
                reverse=True
            )

            return sorted_scores[:n]

    def filter_by_score(
        self,
        min_score: float | None = None,
        max_score: float | None = None
    ) -> list[str]:
        """
        Filter memories by score range.
        
        Args:
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            
        Returns:
            List of memory IDs within score range
        """
        with self._lock:
            filtered = []
            for memory_id, score_record in self.scores.items():
                score = score_record.current_score
                if min_score is not None and score < min_score:
                    continue
                if max_score is not None and score > max_score:
                    continue
                filtered.append(memory_id)
            return filtered

    def reset_score(self, memory_id: str) -> None:
        """
        Reset a memory's score to default.
        
        Args:
            memory_id: Memory identifier
        """
        with self._lock:
            if memory_id in self.scores:
                self.scores[memory_id] = MemoryScore(
                    memory_id=memory_id,
                    current_score=self.config.default_score,
                    initial_score=self.config.default_score,
                    score_history=[(datetime.now(), self.config.default_score)]
                )

    def get_statistics(self) -> dict[str, Any]:
        """
        Get scoring statistics across all memories.
        
        Returns:
            Dictionary of statistical metrics
        """
        with self._lock:
            if not self.scores:
                return {
                    'total_memories': 0,
                    'average_score': 0.0,
                    'median_score': 0.0,
                    'total_usage': 0,
                    'top_score': 0.0,
                    'bottom_score': 0.0
                }

            scores_list = [s.current_score for s in self.scores.values()]
            usage_list = [s.usage_count for s in self.scores.values()]

            return {
                'total_memories': len(self.scores),
                'average_score': statistics.mean(scores_list),
                'median_score': statistics.median(scores_list),
                'total_usage': sum(usage_list),
                'top_score': max(scores_list),
                'bottom_score': min(scores_list)
            }

    def export_scores(self) -> dict[str, Any]:
        """
        Export all scores for persistence.
        
        Returns:
            Serializable dictionary of all scores
        """
        with self._lock:
            export_data = {}
            for memory_id, score_record in self.scores.items():
                # Convert datetime objects to ISO format strings
                export_data[memory_id] = {
                    'current_score': score_record.current_score,
                    'initial_score': score_record.initial_score,
                    'usage_count': score_record.usage_count,
                    'positive_feedback_count': score_record.positive_feedback_count,
                    'negative_feedback_count': score_record.negative_feedback_count,
                    'last_accessed': score_record.last_accessed.isoformat(),
                    'last_updated': score_record.last_updated.isoformat(),
                    'score_history': [
                        (dt.isoformat(), score)
                        for dt, score in score_record.score_history
                    ]
                }
            return export_data

    def import_scores(self, data: dict[str, Any]) -> None:
        """
        Import scores from exported data.
        
        Args:
            data: Previously exported score data
        """
        with self._lock:
            self.scores.clear()
            for memory_id, score_data in data.items():
                # Convert ISO format strings back to datetime objects
                score_record = MemoryScore(
                    memory_id=memory_id,
                    current_score=score_data['current_score'],
                    initial_score=score_data['initial_score'],
                    usage_count=score_data['usage_count'],
                    positive_feedback_count=score_data['positive_feedback_count'],
                    negative_feedback_count=score_data['negative_feedback_count'],
                    last_accessed=datetime.fromisoformat(score_data['last_accessed']),
                    last_updated=datetime.fromisoformat(score_data['last_updated']),
                    score_history=[
                        (datetime.fromisoformat(dt), score)
                        for dt, score in score_data['score_history']
                    ]
                )
                self.scores[memory_id] = score_record
