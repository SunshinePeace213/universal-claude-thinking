"""
Base storage backend interface for memory system.

Defines the abstract interface that all storage backends must implement
for storing, retrieving, and managing memory items.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.memory.layers.base import MemoryItem


class StorageBackend(ABC):
    """
    Abstract base class for memory storage backends.
    
    Defines the common interface for all storage implementations
    including SQLite, PostgreSQL, or in-memory storage.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.
        
        Set up database connections, create tables, and prepare indexes.
        """
        pass

    @abstractmethod
    async def store(self, memory: MemoryItem) -> None:
        """
        Store a memory item in the backend.
        
        Args:
            memory: Memory item to store
        """
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str) -> MemoryItem | None:
        """
        Retrieve a memory item by ID.
        
        Args:
            memory_id: Unique identifier of the memory
            
        Returns:
            Memory item if found, None otherwise
        """
        pass

    @abstractmethod
    async def update(self, memory: MemoryItem) -> None:
        """
        Update an existing memory item.
        
        Args:
            memory: Memory item with updated values
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> None:
        """
        Delete a memory item.
        
        Args:
            memory_id: Unique identifier of the memory to delete
        """
        pass

    @abstractmethod
    async def list_by_user(
        self,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 100
    ) -> list[MemoryItem]:
        """
        List memories for a specific user.
        
        Args:
            user_id: User identifier
            memory_type: Optional filter by memory type
            limit: Maximum number of results
            
        Returns:
            List of memory items
        """
        pass

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.5,
        user_id: str | None = None,
        memory_type: str | None = None
    ) -> list[tuple[MemoryItem, float]]:
        """
        Search for similar memories using embedding similarity.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            user_id: Optional filter by user
            memory_type: Optional filter by memory type
            
        Returns:
            List of tuples (memory, similarity_score)
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close storage backend and clean up resources.
        """
        pass

    async def batch_store(self, memories: list[MemoryItem]) -> None:
        """
        Store multiple memory items in batch.
        
        Default implementation calls store() for each item.
        Subclasses can override for optimized batch operations.
        
        Args:
            memories: List of memory items to store
        """
        for memory in memories:
            await self.store(memory)

    async def batch_update(self, memories: list[MemoryItem]) -> None:
        """
        Update multiple memory items in batch.
        
        Default implementation calls update() for each item.
        Subclasses can override for optimized batch operations.
        
        Args:
            memories: List of memory items to update
        """
        for memory in memories:
            await self.update(memory)

    async def cleanup_expired(self) -> int:
        """
        Remove expired memories from storage.
        
        Returns:
            Number of memories removed
        """
        # Default implementation - subclasses should override
        return 0

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'backend_type': self.__class__.__name__,
            'initialized': True
        }
