"""
In-memory cache for memory system.

Provides fast in-memory caching with LRU eviction for frequently
accessed memories, particularly for STM layer.
"""

import threading
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any

from src.memory.layers.base import MemoryItem


class InMemoryCache:
    """
    In-memory cache implementation with LRU eviction.
    
    Provides fast access to frequently used memories with automatic
    eviction of least recently used items when capacity is reached.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int | None = None
    ):
        """
        Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of items in cache
            ttl_seconds: Optional TTL for cached items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[MemoryItem, datetime | None]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        """Return the current number of items in cache."""
        return len(self._cache)

    def set(self, key: str, value: MemoryItem) -> None:
        """
        Set an item in the cache.
        
        Args:
            key: Cache key
            value: Memory item to cache
        """
        with self._lock:
            # Remove if already exists (to update position)
            if key in self._cache:
                del self._cache[key]

            # Calculate expiry time if TTL is set
            expiry = None
            if self.ttl_seconds:
                expiry = datetime.now() + timedelta(seconds=self.ttl_seconds)

            # Add to end (most recently used)
            self._cache[key] = (value, expiry)

            # Evict LRU if over capacity
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)  # Remove oldest

    def get(self, key: str) -> MemoryItem | None:
        """
        Get an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached memory item or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None

            value, expiry = self._cache[key]

            # Check cache TTL expiry
            if expiry and datetime.now() > expiry:
                del self._cache[key]
                self.misses += 1
                return None

            # Check memory item's own expiry
            if value.expires_at and datetime.now() > value.expires_at:
                del self._cache[key]
                self.misses += 1
                return None

            # Move to end (most recently used)
            del self._cache[key]
            self._cache[key] = (value, expiry)

            self.hits += 1
            return value

    def delete(self, key: str) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def cleanup_expired(self) -> int:
        """
        Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        if not self.ttl_seconds:
            return 0

        with self._lock:
            current_time = datetime.now()
            expired_keys = []

            for key, (value, expiry) in self._cache.items():
                if expiry and current_time > expiry:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def get_statistics(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self.ttl_seconds
            }
