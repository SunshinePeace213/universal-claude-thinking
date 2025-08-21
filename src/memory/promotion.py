"""
Memory promotion pipeline for automated layer transitions.

Implements the automated promotion system that moves memories between layers
(STM→WM→LTM) based on effectiveness scores and usage counts, with scheduled
evaluations using APScheduler.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from src.memory.layers.base import MemoryItem
from src.memory.storage.base import StorageBackend

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PromotionCriteria:
    """
    Configuration for memory promotion thresholds and intervals.
    
    Defines the criteria for promoting memories between layers and
    the scheduling intervals for automatic evaluations.
    """

    # Promotion thresholds
    stm_to_wm_threshold: float = 5.0  # Effectiveness score for STM→WM
    wm_to_ltm_score: float = 8.0      # Minimum score for WM→LTM
    wm_to_ltm_uses: int = 5           # Minimum usage count for WM→LTM

    # Evaluation intervals (in seconds)
    stm_check_interval: int = 3600    # 1 hour
    wm_check_interval: int = 86400    # 1 day

    # Optional batch size limits
    batch_size: int = 100              # Max memories to process per evaluation

    def __post_init__(self):
        """Validate criteria values."""
        if self.stm_to_wm_threshold < 0 or self.stm_to_wm_threshold > 10:
            raise ValueError("STM to WM threshold must be between 0 and 10")
        if self.wm_to_ltm_score < 0 or self.wm_to_ltm_score > 10:
            raise ValueError("WM to LTM score must be between 0 and 10")
        if self.wm_to_ltm_uses < 0:
            raise ValueError("WM to LTM usage count must be non-negative")
        if self.stm_check_interval <= 0 or self.wm_check_interval <= 0:
            raise ValueError("Check intervals must be positive")


class PromotionPipeline:
    """
    Orchestrates the automated promotion of memories between layers.
    
    Manages scheduled evaluations, applies promotion criteria, and tracks
    statistics for the memory promotion system.
    """

    def __init__(
        self,
        stm: Any | None = None,  # ShortTermMemory instance
        wm: Any | None = None,   # WorkingMemory instance
        ltm: Any | None = None,  # LongTermMemory instance
        storage: StorageBackend | None = None,  # For backward compatibility
        criteria: PromotionCriteria | None = None,
        scorer: Any | None = None,
        config: Any | None = None
    ):
        """
        Initialize the promotion pipeline.
        
        Args:
            stm: ShortTermMemory instance
            wm: WorkingMemory instance
            ltm: LongTermMemory instance
            storage: Storage backend for memory operations (deprecated, use memory layers instead)
            criteria: Promotion criteria configuration (uses defaults if None)
            scorer: Optional effectiveness scorer for memory evaluation
            config: Optional memory configuration object
        """
        # Support both new (layer-based) and old (storage-based) initialization
        self.stm = stm
        self.wm = wm
        self.ltm = ltm
        self.storage = storage  # Keep for backward compatibility

        self.criteria = criteria or PromotionCriteria()
        self.scorer = scorer  # Store scorer for compatibility
        self.config = config  # Store config for compatibility
        self.scheduler = BackgroundScheduler()

        # Statistics tracking
        self._stats = {
            'total_promotions': 0,
            'stm_to_wm': 0,
            'wm_to_ltm': 0,
            'last_stm_evaluation': None,
            'last_wm_evaluation': None,
            'failed_promotions': 0
        }

        # Lock for concurrent operations
        self._promotion_lock = asyncio.Lock()

    def start(self):
        """
        Start the promotion scheduler.
        
        Registers scheduled jobs for STM and WM evaluations.
        """
        # Add STM evaluation job
        self.scheduler.add_job(
            func=self.evaluate_stm_memories,
            trigger=IntervalTrigger(seconds=self.criteria.stm_check_interval),
            id='stm_evaluation',
            replace_existing=True
        )

        # Add WM evaluation job
        self.scheduler.add_job(
            func=self.evaluate_wm_memories,
            trigger=IntervalTrigger(seconds=self.criteria.wm_check_interval),
            id='wm_evaluation',
            replace_existing=True
        )

        # Start the scheduler
        self.scheduler.start()
        logger.info("Promotion scheduler started")

    def stop(self):
        """
        Stop the promotion scheduler.
        
        Gracefully shuts down scheduled jobs.
        """
        self.scheduler.shutdown(wait=False)
        logger.info("Promotion scheduler stopped")

    async def evaluate_stm_memories(self) -> list[MemoryItem]:
        """
        Evaluate STM memories for promotion to WM.
        
        Checks all STM memories against the effectiveness threshold
        and promotes eligible ones to Working Memory.
        
        Returns:
            List of promoted memory items
        """
        async with self._promotion_lock:
            promoted = []

            try:
                # Get all STM memories
                memories = await self.storage.get_memories_by_type("stm")

                for memory in memories:
                    # Check if memory is at least 1 hour old
                    age = datetime.now() - memory.created_at
                    if age < timedelta(hours=1):
                        continue

                    # Check effectiveness threshold
                    if memory.effectiveness_score >= self.criteria.stm_to_wm_threshold:
                        # Promote to WM
                        success = await self._promote_memory_internal(
                            memory, "stm", "wm",
                            f"Effectiveness score {memory.effectiveness_score} >= {self.criteria.stm_to_wm_threshold}"
                        )

                        if success:
                            promoted.append(memory)
                            self._stats['stm_to_wm'] += 1
                            self._stats['total_promotions'] += 1

                self._stats['last_stm_evaluation'] = datetime.now()
                logger.info(f"STM evaluation completed: {len(promoted)} memories promoted")

            except Exception as e:
                logger.error(f"Error during STM evaluation: {e}")
                self._stats['failed_promotions'] += 1

            return promoted

    async def evaluate_wm_memories(self) -> list[MemoryItem]:
        """
        Evaluate WM memories for promotion to LTM.
        
        Checks all WM memories against both score and usage count
        criteria and promotes eligible ones to Long-Term Memory.
        
        Returns:
            List of promoted memory items
        """
        async with self._promotion_lock:
            promoted = []

            try:
                # Get all WM memories
                memories = await self.storage.get_memories_by_type("wm")

                for memory in memories:
                    # Check both score and usage criteria
                    if (memory.effectiveness_score >= self.criteria.wm_to_ltm_score and
                        memory.usage_count >= self.criteria.wm_to_ltm_uses):

                        # Promote to LTM
                        success = await self._promote_memory_internal(
                            memory, "wm", "ltm",
                            f"Score {memory.effectiveness_score} >= {self.criteria.wm_to_ltm_score} "
                            f"and usage {memory.usage_count} >= {self.criteria.wm_to_ltm_uses}"
                        )

                        if success:
                            promoted.append(memory)
                            self._stats['wm_to_ltm'] += 1
                            self._stats['total_promotions'] += 1

                self._stats['last_wm_evaluation'] = datetime.now()
                logger.info(f"WM evaluation completed: {len(promoted)} memories promoted")

            except Exception as e:
                logger.error(f"Error during WM evaluation: {e}")
                self._stats['failed_promotions'] += 1

            return promoted

    async def promote_memory(self, memory_id: str, target_layer: str) -> bool:
        """
        Manually promote a specific memory to a target layer.
        
        Args:
            memory_id: ID of the memory to promote
            target_layer: Target layer ("wm", "ltm", or "swarm")
            
        Returns:
            True if promotion successful, False otherwise
        """
        try:
            # Get the memory
            memory = await self.storage.get_memory(memory_id)
            if not memory:
                logger.error(f"Memory {memory_id} not found")
                return False

            # Check privacy constraints for SWARM promotion
            if target_layer == "swarm" and memory.metadata and memory.metadata.get("private"):
                logger.warning(f"Cannot promote private memory {memory_id} to SWARM")
                return False

            # Perform promotion
            success = await self._promote_memory_internal(
                memory, memory.memory_type, target_layer,
                "Manual promotion"
            )

            if success:
                self._stats['total_promotions'] += 1

                # Track specific promotion type
                if memory.memory_type == "stm" and target_layer == "wm":
                    self._stats['stm_to_wm'] += 1
                elif memory.memory_type == "wm" and target_layer == "ltm":
                    self._stats['wm_to_ltm'] += 1

            return success

        except Exception as e:
            logger.error(f"Error promoting memory {memory_id}: {e}")
            self._stats['failed_promotions'] += 1
            return False

    async def promote_batch(self, memory_ids: list[str], target_layer: str) -> list[bool]:
        """
        Promote multiple memories in batch.
        
        Args:
            memory_ids: List of memory IDs to promote
            target_layer: Target layer for all memories
            
        Returns:
            List of success flags for each memory
        """
        results = []

        try:
            # Get all memories in batch
            memories = await self.storage.get_memories_batch(memory_ids)

            for memory in memories:
                # Check privacy for SWARM
                if target_layer == "swarm" and memory.metadata and memory.metadata.get("private"):
                    results.append(False)
                    continue

                # Promote each memory
                success = await self._promote_memory_internal(
                    memory, memory.memory_type, target_layer,
                    "Batch promotion"
                )

                results.append(success)

                if success:
                    self._stats['total_promotions'] += 1

                    # Track specific promotion type
                    if memory.memory_type == "stm" and target_layer == "wm":
                        self._stats['stm_to_wm'] += 1
                    elif memory.memory_type == "wm" and target_layer == "ltm":
                        self._stats['wm_to_ltm'] += 1

        except Exception as e:
            logger.error(f"Error in batch promotion: {e}")
            # Fill remaining results with False
            while len(results) < len(memory_ids):
                results.append(False)
                self._stats['failed_promotions'] += 1

        return results

    async def _promote_memory_internal(
        self,
        memory: MemoryItem,
        from_layer: str,
        to_layer: str,
        reason: str
    ) -> bool:
        """
        Internal method to perform memory promotion.
        
        Args:
            memory: Memory item to promote
            from_layer: Source layer
            to_layer: Target layer
            reason: Reason for promotion
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update memory type
            await self.storage.update_memory_type(memory.id, from_layer, to_layer)

            # Update TTL based on target layer
            new_expiry = None
            if to_layer == "wm":
                # Set 7-day TTL for Working Memory
                new_expiry = datetime.now() + timedelta(days=7)
            elif to_layer == "ltm":
                # No expiry for Long-Term Memory
                new_expiry = None
            elif to_layer == "stm":
                # 2-hour TTL for Short-Term Memory
                new_expiry = datetime.now() + timedelta(hours=2)

            # Update expiry time
            await self.storage.update_expiry(memory.id, new_expiry)

            # Record promotion
            await self.storage.record_promotion(
                memory_id=memory.id,
                from_type=from_layer,
                to_type=to_layer,
                score=memory.effectiveness_score,
                reason=reason
            )

            logger.debug(f"Promoted memory {memory.id} from {from_layer} to {to_layer}")
            return True

        except Exception as e:
            logger.error(f"Failed to promote memory {memory.id}: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get promotion pipeline statistics.
        
        Returns:
            Dictionary containing promotion statistics
        """
        return dict(self._stats)
