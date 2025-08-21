"""
Short-Term Memory (STM) implementation.

Provides a 2-hour TTL memory layer with in-memory caching for
immediate context and recent interactions.
"""

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np

from .base import MemoryItem, MemoryLayer, MemoryType

logger = logging.getLogger(__name__)


class ShortTermMemory(MemoryLayer):
    """
    Short-Term Memory layer with 2-hour TTL.
    
    Uses an in-memory LRU cache for fast access to recent memories
    with automatic expiration after 2 hours.
    """

    def __init__(
        self,
        cache_size: int = 1000,
        ttl_hours: float = 2.0,
    ):
        """
        Initialize Short-Term Memory layer.
        
        Args:
            cache_size: Maximum number of items in cache
            ttl_hours: Time-to-live in hours (default: 2)
        """
        super().__init__(
            layer_type=MemoryType.STM,
            ttl=timedelta(hours=ttl_hours),
            max_size=cache_size
        )

        self.cache_size = cache_size
        self.ttl_hours = ttl_hours
        self.memory_type = "stm"  # String version for compatibility

        # Use OrderedDict for LRU cache implementation
        self._cache: OrderedDict[str, MemoryItem] = OrderedDict()
        self._embeddings: dict[str, np.ndarray] = {}

    async def initialize(self) -> None:
        """Initialize the STM layer."""
        if self._initialized:
            return

        logger.info(f"Initializing STM with {self.cache_size} cache size, {self.ttl_hours}h TTL")

        # Start background task for TTL cleanup
        asyncio.create_task(self._cleanup_task())

        self._initialized = True

    async def store(self, memory: MemoryItem) -> str:
        """
        Store a memory in STM.
        
        Args:
            memory: Memory item to store
            
        Returns:
            ID of stored memory
        """
        # Set TTL if not already set
        if memory.expires_at is None:
            memory.set_ttl(ttl_hours=self.ttl_hours)

        # Set memory type
        memory.memory_type = "stm"

        # Handle cache size limit (LRU eviction)
        if len(self._cache) >= self.cache_size:
            # Remove oldest item (first in OrderedDict)
            oldest_id = next(iter(self._cache))
            del self._cache[oldest_id]
            if oldest_id in self._embeddings:
                del self._embeddings[oldest_id]
            logger.debug(f"Evicted oldest memory {oldest_id} from STM cache")

        # Store in cache (moves to end if exists)
        self._cache[memory.id] = memory
        if memory.embedding is not None:
            self._embeddings[memory.id] = memory.embedding

        # Move to end to mark as most recently used
        self._cache.move_to_end(memory.id)

        logger.debug(f"Stored memory {memory.id} in STM")
        return memory.id

    async def retrieve(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 10
    ) -> MemoryItem | list[MemoryItem] | None:
        """
        Retrieve memory or memories from STM.
        
        Args:
            memory_id: Specific memory ID to retrieve
            user_id: User ID to filter memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            Single memory, list of memories, or None
        """
        # Clean up expired items first
        await self.cleanup_expired()

        if memory_id:
            # Retrieve specific memory
            if memory_id in self._cache:
                memory = self._cache[memory_id]
                memory.update_access()
                # Move to end (most recently accessed)
                self._cache.move_to_end(memory_id)
                return memory
            return None

        # Retrieve memories by user_id
        results = []
        for memory in reversed(self._cache.values()):  # Most recent first
            if user_id and memory.user_id != user_id:
                continue
            if not memory.is_expired():
                results.append(memory)
                if len(results) >= limit:
                    break

        return results if results else []

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from STM.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        if memory_id in self._cache:
            del self._cache[memory_id]
            if memory_id in self._embeddings:
                del self._embeddings[memory_id]
            logger.debug(f"Deleted memory {memory_id} from STM")
            return True
        return False

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.85
    ) -> list[MemoryItem]:
        """
        Search for similar memories using embedding similarity.
        
        Args:
            query_embedding: Query vector for similarity search
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of similar memories
        """
        # Clean up expired items first
        await self.cleanup_expired()

        if not self._embeddings:
            return []

        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        # Calculate similarities
        similarities = []
        for memory_id, embedding in self._embeddings.items():
            if memory_id not in self._cache:
                continue

            memory = self._cache[memory_id]
            if memory.is_expired():
                continue

            # Normalize stored embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # Cosine similarity
            similarity = float(np.dot(query_embedding, embedding))

            if similarity >= threshold:
                similarities.append((memory, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in similarities[:k]]

    async def cleanup_expired(self) -> int:
        """
        Remove expired memories from STM.
        
        Returns:
            Number of memories removed
        """
        from datetime import datetime
        current_time = datetime.now()
        expired_ids = []

        for memory_id, memory in list(self._cache.items()):
            # Check expiration more explicitly
            if memory.expires_at and memory.expires_at <= current_time:
                expired_ids.append(memory_id)
                logger.debug(f"Memory {memory_id} expired: {memory.expires_at} <= {current_time}")

        for memory_id in expired_ids:
            await self.delete(memory_id)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired memories from STM")

        return len(expired_ids)

    async def _cleanup_task(self) -> None:
        """Background task to periodically clean up expired memories."""
        while self._initialized:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in STM cleanup task: {e}")

    async def get_promotion_candidates(
        self,
        min_score: float = 5.0
    ) -> list[MemoryItem]:
        """
        Get memories that are candidates for promotion to WM.
        
        Args:
            min_score: Minimum effectiveness score for promotion
            
        Returns:
            List of memories eligible for promotion
        """
        candidates = []

        for memory in self._cache.values():
            # Check if memory has been in STM for at least 1 hour
            age = datetime.now() - memory.created_at
            if age >= timedelta(hours=1):
                # Check effectiveness score
                if memory.effectiveness_score > min_score:
                    candidates.append(memory)

        return candidates

    async def clear(self) -> None:
        """Clear all memories from STM."""
        self._cache.clear()
        self._embeddings.clear()
        logger.info("Cleared all memories from STM")

    async def close(self) -> None:
        """Clean up STM resources."""
        self._initialized = False
        await self.clear()
        logger.info("STM closed")
