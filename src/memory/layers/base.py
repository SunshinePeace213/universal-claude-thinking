"""
Base classes for memory layer implementations.

Provides abstract base classes and common data structures for the
5-layer memory system (STM, WM, LTM, SWARM, Privacy).
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class MemoryType(Enum):
    """Types of memory layers in the system."""
    STM = "stm"  # Short-term memory
    WM = "wm"    # Working memory
    LTM = "ltm"  # Long-term memory
    SWARM = "swarm"  # Community memory


@dataclass
class MemoryItem:
    """
    Individual memory item stored in the system.
    
    Represents a single memory with content, embedding, metadata,
    and tracking information for promotion and effectiveness scoring.
    """

    id: str
    user_id: str
    memory_type: str  # Using string instead of enum for flexibility
    content: dict[str, Any]
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
    effectiveness_score: float = 5.0
    usage_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    promoted_from: str | None = None
    promoted_at: datetime | None = None
    promotion_reason: str | None = None

    def __post_init__(self):
        """Validate and initialize memory item."""
        # Generate ID if not provided
        if not self.id:
            self.id = str(uuid.uuid4())

        # Ensure embedding is numpy array if provided
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)

        # Initialize metadata if not provided
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert memory item to dictionary for storage.
        
        Returns:
            Dictionary representation of the memory item
        """
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'memory_type': self.memory_type,
            'content': self.content,
            'metadata': self.metadata,
            'effectiveness_score': self.effectiveness_score,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
        }

        # Add optional fields if present
        if self.embedding is not None:
            data['embedding_shape'] = self.embedding.shape
            # Don't serialize full embedding in dict, too large
            data['has_embedding'] = True

        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()

        if self.promoted_from:
            data['promoted_from'] = self.promoted_from

        if self.promoted_at:
            data['promoted_at'] = self.promoted_at.isoformat()

        if self.promotion_reason:
            data['promotion_reason'] = self.promotion_reason

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], embedding: np.ndarray | None = None) -> 'MemoryItem':
        """
        Create memory item from dictionary.
        
        Args:
            data: Dictionary with memory data
            embedding: Optional embedding array (loaded separately)
            
        Returns:
            MemoryItem instance
        """
        # Parse datetime fields
        created_at = datetime.fromisoformat(data['created_at']) if isinstance(data['created_at'], str) else data['created_at']
        last_accessed = datetime.fromisoformat(data['last_accessed']) if isinstance(data['last_accessed'], str) else data['last_accessed']

        expires_at = None
        if 'expires_at' in data and data['expires_at']:
            expires_at = datetime.fromisoformat(data['expires_at']) if isinstance(data['expires_at'], str) else data['expires_at']

        promoted_at = None
        if 'promoted_at' in data and data['promoted_at']:
            promoted_at = datetime.fromisoformat(data['promoted_at']) if isinstance(data['promoted_at'], str) else data['promoted_at']

        return cls(
            id=data['id'],
            user_id=data['user_id'],
            memory_type=data['memory_type'],
            content=data['content'],
            embedding=embedding,
            metadata=data.get('metadata', {}),
            effectiveness_score=data.get('effectiveness_score', 5.0),
            usage_count=data.get('usage_count', 0),
            created_at=created_at,
            last_accessed=last_accessed,
            expires_at=expires_at,
            promoted_from=data.get('promoted_from'),
            promoted_at=promoted_at,
            promotion_reason=data.get('promotion_reason')
        )

    def is_expired(self) -> bool:
        """
        Check if memory has expired based on TTL.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False  # No expiration (e.g., LTM)
        return datetime.now() > self.expires_at

    def update_access(self) -> None:
        """Update last accessed timestamp and increment usage count."""
        self.last_accessed = datetime.now()
        self.usage_count += 1

    def set_ttl(self, ttl_hours: float | None = None, ttl_days: float | None = None) -> None:
        """
        Set time-to-live for the memory.
        
        Args:
            ttl_hours: TTL in hours
            ttl_days: TTL in days
        """
        if ttl_hours is not None:
            self.expires_at = datetime.now() + timedelta(hours=ttl_hours)
        elif ttl_days is not None:
            self.expires_at = datetime.now() + timedelta(days=ttl_days)
        else:
            self.expires_at = None  # No expiration


class MemoryLayer(ABC):
    """
    Abstract base class for memory layers.
    
    Defines the common interface that all memory layers (STM, WM, LTM, SWARM)
    must implement for storing, retrieving, and managing memories.
    """

    def __init__(
        self,
        layer_type: MemoryType,
        ttl: timedelta | None = None,
        max_size: int | None = None
    ):
        """
        Initialize memory layer.
        
        Args:
            layer_type: Type of memory layer
            ttl: Time-to-live for memories in this layer
            max_size: Maximum number of memories to store
        """
        self.layer_type = layer_type
        self.ttl = ttl
        self.max_size = max_size
        self.memories: dict[str, MemoryItem] = {}
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the memory layer and its storage backend."""
        pass

    @abstractmethod
    async def store(self, memory: MemoryItem) -> str:
        """
        Store a memory in this layer.
        
        Args:
            memory: Memory item to store
            
        Returns:
            ID of stored memory
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        memory_id: str | None = None,
        user_id: str | None = None,
        limit: int = 10
    ) -> MemoryItem | list[MemoryItem] | None:
        """
        Retrieve memory or memories from this layer.
        
        Args:
            memory_id: Specific memory ID to retrieve
            user_id: User ID to filter memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            Single memory, list of memories, or None
        """
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory from this layer.
        
        Args:
            memory_id: ID of memory to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """
        Remove expired memories from this layer.
        
        Returns:
            Number of memories removed
        """
        pass

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about this memory layer.
        
        Returns:
            Dictionary of statistics
        """
        total_memories = len(self.memories)
        expired_count = sum(1 for m in self.memories.values() if m.is_expired())

        avg_score = 0.0
        avg_usage = 0.0
        if total_memories > 0:
            avg_score = sum(m.effectiveness_score for m in self.memories.values()) / total_memories
            avg_usage = sum(m.usage_count for m in self.memories.values()) / total_memories

        return {
            'layer_type': self.layer_type.value,
            'total_memories': total_memories,
            'expired_count': expired_count,
            'average_effectiveness': avg_score,
            'average_usage': avg_usage,
            'max_size': self.max_size,
            'ttl': str(self.ttl) if self.ttl else 'None'
        }

    async def close(self) -> None:
        """Clean up resources used by this layer."""
        self.memories.clear()
        self._initialized = False
