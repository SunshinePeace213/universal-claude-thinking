"""
Unit tests for memory storage backends.

Tests SQLite storage with connection pooling and in-memory cache with LRU eviction.
"""

import asyncio
import json
import pytest
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import numpy as np

# These imports will fail initially (TDD)
from src.memory.storage.base import StorageBackend
from src.memory.storage.sqlite_storage import SQLiteStorage
from src.memory.storage.memory_cache import InMemoryCache
from src.memory.layers.base import MemoryItem


class TestStorageBackend:
    """Test abstract storage backend interface."""
    
    def test_storage_backend_interface(self):
        """Test that StorageBackend defines required methods."""
        # StorageBackend should be abstract
        with pytest.raises(TypeError):
            StorageBackend()
            
        # Check required methods are defined
        assert hasattr(StorageBackend, 'initialize')
        assert hasattr(StorageBackend, 'store')
        assert hasattr(StorageBackend, 'retrieve')
        assert hasattr(StorageBackend, 'update')
        assert hasattr(StorageBackend, 'delete')
        assert hasattr(StorageBackend, 'list_by_user')
        assert hasattr(StorageBackend, 'search_by_embedding')
        assert hasattr(StorageBackend, 'close')


@pytest.mark.asyncio
class TestSQLiteStorage:
    """Test SQLite storage backend."""
    
    @pytest.fixture
    async def storage(self):
        """Create SQLite storage with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
            
        storage = SQLiteStorage(db_path=db_path, pool_size=3)
        await storage.initialize()
        
        yield storage
        
        await storage.close()
        Path(db_path).unlink(missing_ok=True)
        
    async def test_sqlite_initialization(self, storage):
        """Test SQLite storage initializes with schema."""
        # Check tables exist
        async with storage._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            table_names = [t[0] for t in tables]
            
            assert 'memories' in table_names
            assert 'memory_promotions' in table_names
            assert 'memory_vectors' in table_names  # Fallback table if vec0 not available
            
    async def test_sqlite_store_and_retrieve(self, storage):
        """Test storing and retrieving memory items."""
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            user_id="test_user",
            memory_type="wm",
            content={"text": "Test memory", "metadata": {"tag": "test"}},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=7.5,
            usage_count=3,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)
        )
        
        # Store memory
        await storage.store(memory)
        
        # Retrieve by ID
        retrieved = await storage.retrieve(memory.id)
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.user_id == memory.user_id
        assert retrieved.content == memory.content
        assert retrieved.effectiveness_score == memory.effectiveness_score
        assert retrieved.usage_count == memory.usage_count
        assert np.allclose(retrieved.embedding, memory.embedding, rtol=1e-5)
        
    async def test_sqlite_update_memory(self, storage):
        """Test updating an existing memory."""
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            user_id="test_user",
            memory_type="stm",
            content={"text": "Original"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        await storage.store(memory)
        
        # Update memory
        memory.effectiveness_score = 6.5
        memory.usage_count = 2
        memory.content = {"text": "Updated"}
        
        await storage.update(memory)
        
        # Verify update
        retrieved = await storage.retrieve(memory.id)
        assert retrieved.effectiveness_score == 6.5
        assert retrieved.usage_count == 2
        assert retrieved.content == {"text": "Updated"}
        
    async def test_sqlite_delete_memory(self, storage):
        """Test deleting a memory."""
        memory = MemoryItem(
            id=str(uuid.uuid4()),
            user_id="test_user",
            memory_type="stm",
            content={"text": "Delete me"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        await storage.store(memory)
        assert await storage.retrieve(memory.id) is not None
        
        # Delete memory
        await storage.delete(memory.id)
        assert await storage.retrieve(memory.id) is None
        
    async def test_sqlite_list_by_user(self, storage):
        """Test listing memories by user ID."""
        user_id = "test_user"
        other_user = "other_user"
        
        # Store memories for different users
        for i in range(5):
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id=user_id if i < 3 else other_user,
                memory_type="wm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0 + i,
                usage_count=i
            )
            await storage.store(memory)
            
        # List memories for test_user
        user_memories = await storage.list_by_user(user_id, memory_type="wm")
        assert len(user_memories) == 3
        assert all(m.user_id == user_id for m in user_memories)
        
    async def test_sqlite_search_by_embedding(self, storage):
        """Test searching memories by embedding similarity."""
        # Store memories with different embeddings
        base_embedding = np.random.rand(4096).astype(np.float32)
        
        memories = []
        for i in range(10):
            # Create embeddings with varying similarity to base
            noise = np.random.rand(4096).astype(np.float32) * (i * 0.1)
            embedding = base_embedding + noise
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id="test_user",
                memory_type="wm",
                content={"text": f"Memory {i}"},
                embedding=embedding,
                effectiveness_score=5.0,
                usage_count=0
            )
            memories.append(memory)
            await storage.store(memory)
            
        # Search for similar memories
        results = await storage.search_by_embedding(
            embedding=base_embedding,
            k=5,
            min_similarity=0.5
        )
        
        assert len(results) <= 5
        # Results should be ordered by similarity (descending)
        if len(results) > 1:
            similarities = [r.similarity for r in results]
            assert similarities == sorted(similarities, reverse=True)
            
    async def test_sqlite_connection_pooling(self, storage):
        """Test connection pooling for concurrent access."""
        # Run multiple concurrent operations
        async def store_memory(idx):
            memory = MemoryItem(
                id=f"concurrent_{idx}",
                user_id="test_user",
                memory_type="stm",
                content={"text": f"Concurrent {idx}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0,
                usage_count=0
            )
            await storage.store(memory)
            return await storage.retrieve(memory.id)
            
        # Execute concurrent operations
        tasks = [store_memory(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(r is not None for r in results)
        
    async def test_sqlite_batch_operations(self, storage):
        """Test batch insert and update operations."""
        memories = []
        for i in range(100):
            memory = MemoryItem(
                id=f"batch_{i}",
                user_id="test_user",
                memory_type="wm",
                content={"text": f"Batch {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0 + (i * 0.1),
                usage_count=i % 10
            )
            memories.append(memory)
            
        # Batch insert
        await storage.batch_store(memories)
        
        # Verify all stored
        for memory in memories[:10]:  # Check first 10
            retrieved = await storage.retrieve(memory.id)
            assert retrieved is not None
            assert retrieved.id == memory.id
            
    async def test_sqlite_expired_memory_cleanup(self, storage):
        """Test automatic cleanup of expired memories."""
        # Store expired and non-expired memories
        expired = MemoryItem(
            id="expired",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Expired"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0,
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        valid = MemoryItem(
            id="valid",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Valid"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        await storage.store(expired)
        await storage.store(valid)
        
        # Run cleanup
        await storage.cleanup_expired()
        
        # Expired should be gone
        assert await storage.retrieve("expired") is None
        # Valid should remain
        assert await storage.retrieve("valid") is not None
        
    async def test_sqlite_promotion_tracking(self, storage):
        """Test tracking memory promotions."""
        memory_id = str(uuid.uuid4())
        
        # Record promotion
        await storage.record_promotion(
            memory_id=memory_id,
            from_type="stm",
            to_type="wm",
            promotion_score=6.5,
            reason="Effectiveness threshold met"
        )
        
        # Get promotion history
        history = await storage.get_promotion_history(memory_id)
        assert len(history) == 1
        assert history[0]['from_type'] == 'stm'
        assert history[0]['to_type'] == 'wm'
        assert history[0]['promotion_score'] == 6.5


class TestInMemoryCache:
    """Test in-memory cache with LRU eviction."""
    
    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return InMemoryCache(max_size=5)
        
    def test_cache_initialization(self, cache):
        """Test cache initializes empty with correct size."""
        assert cache.max_size == 5
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
        
    def test_cache_set_and_get(self, cache):
        """Test setting and getting items from cache."""
        memory = MemoryItem(
            id="test",
            user_id="user",
            memory_type="stm",
            content={"text": "Cached"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        # Set item
        cache.set("test", memory)
        assert len(cache) == 1
        
        # Get item (hit)
        retrieved = cache.get("test")
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert cache.hits == 1
        assert cache.misses == 0
        
        # Get non-existent (miss)
        missing = cache.get("missing")
        assert missing is None
        assert cache.hits == 1
        assert cache.misses == 1
        
    def test_cache_lru_eviction(self, cache):
        """Test LRU eviction when cache is full."""
        # Fill cache beyond capacity
        for i in range(7):
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id="user",
                memory_type="stm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0,
                usage_count=0
            )
            cache.set(f"mem_{i}", memory)
            
        # Cache should only have last 5 items
        assert len(cache) == 5
        assert cache.get("mem_0") is None  # Evicted
        assert cache.get("mem_1") is None  # Evicted
        assert cache.get("mem_2") is not None  # Still in cache
        
        # Access mem_2 to make it recently used
        cache.get("mem_2")
        
        # Add new item, mem_3 should be evicted (least recently used)
        new_memory = MemoryItem(
            id="new",
            user_id="user",
            memory_type="stm",
            content={"text": "New"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        cache.set("new", new_memory)
        
        assert cache.get("mem_3") is None  # Evicted
        assert cache.get("mem_2") is not None  # Still there (recently accessed)
        assert cache.get("new") is not None  # New item is there
        
    def test_cache_delete(self, cache):
        """Test deleting items from cache."""
        memory = MemoryItem(
            id="delete_me",
            user_id="user",
            memory_type="stm",
            content={"text": "Delete"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        cache.set("delete_me", memory)
        assert len(cache) == 1
        
        # Delete item
        cache.delete("delete_me")
        assert len(cache) == 0
        assert cache.get("delete_me") is None
        
    def test_cache_clear(self, cache):
        """Test clearing all items from cache."""
        # Add multiple items
        for i in range(3):
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id="user",
                memory_type="stm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0,
                usage_count=0
            )
            cache.set(f"mem_{i}", memory)
            
        assert len(cache) == 3
        
        # Clear cache
        cache.clear()
        assert len(cache) == 0
        assert cache.get("mem_0") is None
        
    def test_cache_expiration(self, cache):
        """Test cache respects item expiration times."""
        # Create expired memory
        expired = MemoryItem(
            id="expired",
            user_id="user",
            memory_type="stm",
            content={"text": "Expired"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0,
            expires_at=datetime.now() - timedelta(minutes=1)
        )
        
        # Create valid memory
        valid = MemoryItem(
            id="valid",
            user_id="user",
            memory_type="stm",
            content={"text": "Valid"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        cache.set("expired", expired)
        cache.set("valid", valid)
        
        # Should not return expired item
        assert cache.get("expired") is None
        # Should return valid item
        assert cache.get("valid") is not None
        
    def test_cache_statistics(self, cache):
        """Test cache hit/miss statistics."""
        # Add items
        for i in range(3):
            memory = MemoryItem(
                id=f"mem_{i}",
                user_id="user",
                memory_type="stm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0,
                usage_count=0
            )
            cache.set(f"mem_{i}", memory)
            
        # Generate hits and misses
        cache.get("mem_0")  # Hit
        cache.get("mem_1")  # Hit
        cache.get("mem_0")  # Hit
        cache.get("missing")  # Miss
        cache.get("not_there")  # Miss
        
        stats = cache.get_statistics()
        assert stats['hits'] == 3
        assert stats['misses'] == 2
        assert stats['hit_rate'] == 0.6
        assert stats['size'] == 3
        assert stats['max_size'] == 5