"""
End-to-end tests for complete memory lifecycle.

Tests the full journey of memories from creation through STM to WM to LTM,
including TTL expiration, cross-session persistence, and retrieval.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os

from src.memory import (
    ShortTermMemory,
    WorkingMemory,
    LongTermMemory,
    MemoryItem,
    MemoryType,
    MemoryEmbedder,
    EffectivenessScorer,
)
from src.memory.config import MemoryConfig


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except:
        pass


@pytest.fixture
async def memory_system(temp_db):
    """Create a complete memory system for testing."""
    config = MemoryConfig()
    
    # Create memory layers
    stm = ShortTermMemory(cache_size=100, ttl_hours=2.0)
    wm = WorkingMemory(db_path=temp_db, ttl_days=7)
    ltm = LongTermMemory(db_path=temp_db)
    
    # Initialize layers
    await stm.initialize()
    await wm.initialize()
    await ltm.initialize()
    
    # Create supporting systems
    embedder = MemoryEmbedder(cache_size=100)
    scorer = EffectivenessScorer()
    
    yield {
        "stm": stm,
        "wm": wm,
        "ltm": ltm,
        "embedder": embedder,
        "scorer": scorer,
        "config": config,
        "db_path": temp_db
    }
    
    # Cleanup
    await wm.close()
    await ltm.close()


@pytest.mark.asyncio
class TestMemoryLifecycle:
    """Test complete memory lifecycle from creation to retrieval."""
    
    async def test_memory_creation_and_storage(self, memory_system):
        """Test creating and storing memories in STM."""
        stm = memory_system["stm"]
        
        # Create a new memory
        memory = MemoryItem(
            id="test_memory_1",
            user_id="user_123",
            memory_type="stm",
            content={
                "text": "User prefers dark mode for coding",
                "context": "IDE settings discussion",
                "timestamp": datetime.now().isoformat()
            },
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=1
        )
        
        # Store in STM
        memory_id = await stm.store(memory)
        assert memory_id == "test_memory_1"
        
        # Retrieve and verify
        retrieved = await stm.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved.user_id == "user_123"
        assert retrieved.content["text"] == "User prefers dark mode for coding"
        assert retrieved.expires_at is not None
        
        # Verify TTL is set correctly (2 hours)
        ttl_delta = retrieved.expires_at - datetime.now()
        assert 115 < ttl_delta.total_seconds() / 60 < 125  # ~2 hours
    
    async def test_stm_to_wm_promotion(self, memory_system):
        """Test promoting memory from STM to WM based on effectiveness."""
        stm = memory_system["stm"]
        wm = memory_system["wm"]
        scorer = memory_system["scorer"]
        
        # Create high-value memory
        memory = MemoryItem(
            id="promote_to_wm",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Critical user preference"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=3.0,  # Start low
            usage_count=1
        )
        
        # Store in STM
        await stm.store(memory)
        
        # Simulate positive feedback to increase score
        for _ in range(3):
            scorer.apply_feedback(memory, is_positive=True)
        
        assert memory.effectiveness_score > 5.0  # Above WM threshold
        
        # Get promotion candidates
        candidates = await stm.get_promotion_candidates(min_score=5.0)
        assert len(candidates) > 0
        
        # Promote to WM
        promoted = candidates[0]
        promoted.memory_type = "wm"
        promoted.promoted_from = "stm"
        promoted.promoted_at = datetime.now()
        promoted.expires_at = datetime.now() + timedelta(days=7)
        
        await wm.store(promoted)
        
        # Verify in WM
        wm_memory = await wm.retrieve(promoted.id, user_id="user_123")
        assert wm_memory is not None
        assert wm_memory.memory_type == "wm"
        assert wm_memory.promoted_from == "stm"
        
        # Remove from STM
        await stm.delete(promoted.id)
        assert await stm.retrieve(promoted.id) is None
    
    async def test_wm_to_ltm_promotion(self, memory_system):
        """Test promoting memory from WM to LTM based on score and usage."""
        wm = memory_system["wm"]
        ltm = memory_system["ltm"]
        
        # Create highly valuable memory
        memory = MemoryItem(
            id="promote_to_ltm",
            user_id="user_123",
            memory_type="wm",
            content={"text": "Core user workflow pattern"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=8.5,  # Above LTM threshold
            usage_count=10,  # Above usage threshold
            promoted_from="stm",
            promoted_at=datetime.now() - timedelta(days=3)
        )
        
        # Store in WM
        await wm.store(memory)
        
        # Get promotion candidates
        candidates = await wm.get_candidates_for_promotion(
            min_effectiveness=8.0,
            min_usage=5
        )
        assert len(candidates) > 0
        
        # Promote to LTM
        promoted = candidates[0]
        promoted.memory_type = "ltm"
        promoted.promoted_from = "wm"
        promoted.promoted_at = datetime.now()
        promoted.expires_at = None  # LTM is permanent
        
        await ltm.store(promoted)
        
        # Verify in LTM
        ltm_memory = await ltm.retrieve(promoted.id, user_id="user_123")
        assert ltm_memory is not None
        assert ltm_memory.memory_type == "ltm"
        assert ltm_memory.expires_at is None
        assert ltm_memory.effectiveness_score >= 8.0
    
    async def test_ttl_expiration(self, memory_system):
        """Test automatic TTL expiration for STM and WM."""
        stm = memory_system["stm"]
        wm = memory_system["wm"]
        
        # Create STM memory with very short TTL
        stm_memory = MemoryItem(
            id="expire_stm",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Temporary context"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=3.0,
            usage_count=1
        )
        
        # Manually set short TTL for testing
        stm_memory.expires_at = datetime.now() + timedelta(seconds=1)
        await stm.store(stm_memory)
        
        # Memory should exist initially
        assert await stm.retrieve("expire_stm") is not None
        
        # Wait for expiration
        await asyncio.sleep(2)
        
        # Trigger cleanup
        expired_count = await stm.cleanup_expired()
        assert expired_count > 0
        
        # Memory should be gone
        assert await stm.retrieve("expire_stm") is None
        
        # Test WM expiration
        wm_memory = MemoryItem(
            id="expire_wm",
            user_id="user_123",
            memory_type="wm",
            content={"text": "Weekly context"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=6.0,
            usage_count=3,
            expires_at=datetime.now() + timedelta(seconds=1)
        )
        
        await wm.store(wm_memory)
        assert await wm.retrieve("expire_wm", user_id="user_123") is not None
        
        await asyncio.sleep(2)
        expired_count = await wm.cleanup_expired()
        assert expired_count > 0
        
        assert await wm.retrieve("expire_wm", user_id="user_123") is None
    
    async def test_cross_session_persistence(self, memory_system):
        """Test that memories persist across sessions."""
        db_path = memory_system["db_path"]
        
        # Create and store LTM memory
        ltm1 = LongTermMemory(db_path=db_path)
        await ltm1.initialize()
        
        memory = MemoryItem(
            id="persistent_memory",
            user_id="user_123",
            memory_type="ltm",
            content={"text": "Permanent user preference"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=9.0,
            usage_count=20
        )
        
        await ltm1.store(memory)
        await ltm1.close()
        
        # Create new LTM instance (simulating new session)
        ltm2 = LongTermMemory(db_path=db_path)
        await ltm2.initialize()
        
        # Memory should persist
        retrieved = await ltm2.retrieve("persistent_memory", user_id="user_123")
        assert retrieved is not None
        assert retrieved.content["text"] == "Permanent user preference"
        assert retrieved.effectiveness_score == 9.0
        
        await ltm2.close()
    
    async def test_memory_retrieval_with_embeddings(self, memory_system):
        """Test retrieving memories using embedding similarity."""
        wm = memory_system["wm"]
        embedder = memory_system["embedder"]
        
        # Mock the embedder
        with patch.object(embedder, 'encode') as mock_encode:
            # Create base embedding
            base_embedding = np.random.rand(4096).astype(np.float32)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Create similar embeddings (high cosine similarity)
            similar1 = base_embedding + np.random.randn(4096) * 0.1
            similar1 = similar1 / np.linalg.norm(similar1)
            
            similar2 = base_embedding + np.random.randn(4096) * 0.1
            similar2 = similar2 / np.linalg.norm(similar2)
            
            # Create dissimilar embedding
            dissimilar = np.random.rand(4096).astype(np.float32)
            dissimilar = dissimilar / np.linalg.norm(dissimilar)
            
            # Store memories
            memories = [
                MemoryItem(
                    id="similar_1",
                    user_id="user_123",
                    memory_type="wm",
                    content={"text": "Python coding preference"},
                    embedding=similar1,
                    effectiveness_score=7.0,
                    usage_count=5
                ),
                MemoryItem(
                    id="similar_2",
                    user_id="user_123",
                    memory_type="wm",
                    content={"text": "Python debugging tips"},
                    embedding=similar2,
                    effectiveness_score=6.5,
                    usage_count=3
                ),
                MemoryItem(
                    id="dissimilar",
                    user_id="user_123",
                    memory_type="wm",
                    content={"text": "Coffee preferences"},
                    embedding=dissimilar,
                    effectiveness_score=5.0,
                    usage_count=2
                )
            ]
            
            for memory in memories:
                await wm.store(memory)
            
            # Mock encode to return base embedding for query
            mock_encode.return_value = base_embedding
            
            # Search for similar memories
            results = await wm.search(
                query_embedding=base_embedding,
                user_id="user_123",
                limit=2,
                min_similarity=0.5
            )
            
            # Should get the two similar memories
            assert len(results) == 2
            result_ids = [r.id for r in results]
            assert "similar_1" in result_ids
            assert "similar_2" in result_ids
            assert "dissimilar" not in result_ids
    
    async def test_memory_usage_tracking(self, memory_system):
        """Test that memory usage is tracked correctly."""
        stm = memory_system["stm"]
        scorer = memory_system["scorer"]
        
        memory = MemoryItem(
            id="track_usage",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Frequently accessed info"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0
        )
        
        await stm.store(memory)
        
        # Simulate multiple accesses
        for i in range(5):
            retrieved = await stm.retrieve("track_usage")
            scorer.track_usage(retrieved)
        
        # Check usage count increased
        final_memory = await stm.retrieve("track_usage")
        assert final_memory.usage_count == 5
        
        # Score should have increased with usage
        assert final_memory.effectiveness_score > 5.0
    
    async def test_batch_memory_operations(self, memory_system):
        """Test batch operations for efficiency."""
        wm = memory_system["wm"]
        
        # Create batch of memories
        memories = []
        for i in range(10):
            memory = MemoryItem(
                id=f"batch_{i}",
                user_id="user_123",
                memory_type="wm",
                content={"text": f"Batch memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0 + i * 0.5,
                usage_count=i
            )
            memories.append(memory)
        
        # Batch store
        start_time = datetime.now()
        for memory in memories:
            await wm.store(memory)
        batch_time = (datetime.now() - start_time).total_seconds()
        
        # Should complete reasonably fast
        assert batch_time < 5.0  # 5 seconds for 10 memories
        
        # Verify all stored
        for i in range(10):
            retrieved = await wm.retrieve(f"batch_{i}", user_id="user_123")
            assert retrieved is not None
            assert retrieved.content["text"] == f"Batch memory {i}"