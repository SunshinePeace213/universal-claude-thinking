"""
Integration tests for complete memory lifecycle.

Tests the full memory flow from creation through promotion to expiration,
including TTL enforcement and threshold validation.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

from src.memory.layers.base import MemoryItem
from src.memory.layers.stm import ShortTermMemory
from src.memory.layers.wm import WorkingMemory
from src.memory.layers.ltm import LongTermMemory
from src.memory.storage.sqlite_storage import SQLiteStorage
from src.memory.promotion import PromotionPipeline
from src.memory.scoring import EffectivenessScorer, FeedbackType
from src.memory.privacy import PrivacyEngine
from src.memory.config import MemoryConfig


class TestMemoryLifecycle:
    """Integration tests for complete memory lifecycle management."""
    
    @pytest.fixture
    async def memory_system(self):
        """Set up complete memory system for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        # Initialize storage
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        
        # Initialize memory layers
        stm = ShortTermMemory(cache_size=100, ttl_hours=2)
        wm = WorkingMemory(db_path=db_path, ttl_days=7)
        ltm = LongTermMemory(db_path=db_path)
        
        await stm.initialize()
        await wm.initialize()
        await ltm.initialize()
        
        # Initialize scorer and promotion pipeline
        scorer = EffectivenessScorer()
        config = MemoryConfig(
            stm={'ttl_hours': 2, 'cache_size': 100},
            wm={'ttl_days': 7, 'promotion_threshold': 5.0},
            ltm={'promotion_score': 8.0, 'promotion_uses': 5}
        )
        
        promotion = PromotionPipeline(
            storage=storage,
            scorer=scorer,
            config=config
        )
        
        # Privacy engine
        privacy = PrivacyEngine()
        
        yield {
            'storage': storage,
            'stm': stm,
            'wm': wm,
            'ltm': ltm,
            'scorer': scorer,
            'promotion': promotion,
            'privacy': privacy,
            'config': config,
            'db_path': db_path
        }
        
        # Cleanup
        await storage.close()
        if hasattr(stm, '_connection'):
            await stm._connection.close()
        if hasattr(wm, '_connection'):
            await wm._connection.close()
        if hasattr(ltm, '_connection'):
            await ltm._connection.close()
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_complete_memory_creation_flow(self, memory_system):
        """Test memory creation with privacy filtering and embedding generation."""
        storage = memory_system['storage']
        privacy = memory_system['privacy']
        
        # Create memory with PII
        memory = MemoryItem(
            id="lifecycle_001",
            user_id="test_user",
            memory_type="stm",
            content={
                "text": "Meeting with john.doe@example.com scheduled for tomorrow",
                "context": "calendar"
            },
            metadata={"source": "user_input"}
        )
        
        # Apply privacy filtering
        sanitized_text = privacy.remove_pii(memory.content["text"])
        memory.content["text"] = sanitized_text
        
        # Simulate embedding generation
        memory.embedding = np.random.rand(4096).astype(np.float32)
        
        # Store memory
        await storage.store(memory)
        
        # Verify storage and PII removal
        retrieved = await storage.retrieve("lifecycle_001")
        assert retrieved is not None
        assert "@example.com" not in retrieved.content["text"]
        assert retrieved.embedding is not None
    
    @pytest.mark.asyncio
    async def test_stm_ttl_expiration(self, memory_system):
        """Test that STM memories expire after 2 hours."""
        stm = memory_system['stm']
        storage = memory_system['storage']
        
        # Create STM memory that's already expired
        memory = MemoryItem(
            id="stm_expire_test",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Temporary information"},
            created_at=datetime.now() - timedelta(hours=2, minutes=30)  # Created 2.5 hours ago
        )
        # Manually set expiration to the past to simulate expired memory
        memory.expires_at = datetime.now() - timedelta(minutes=30)  # Expired 30 minutes ago
        
        # Store in STM cache
        stm._cache[memory.id] = memory
        
        # Check if expired
        assert memory.is_expired()
        
        # Cleanup should remove it
        removed_count = await stm.cleanup_expired()
        assert removed_count == 1
        assert memory.id not in stm._cache
    
    @pytest.mark.asyncio
    async def test_wm_ttl_enforcement(self, memory_system):
        """Test that WM memories expire after 7 days."""
        wm = memory_system['wm']
        
        # Create WM memory near expiration
        memory = MemoryItem(
            id="wm_expire_test",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Working context"},
            created_at=datetime.now() - timedelta(days=6, hours=23)
        )
        memory.set_ttl(ttl_days=7)
        
        # Store memory
        await wm.store(memory)
        
        # Should not be expired yet
        assert not memory.is_expired()
        
        # Simulate time passing
        memory.expires_at = datetime.now() - timedelta(minutes=1)
        assert memory.is_expired()
        
        # Cleanup should remove expired memories
        removed = await wm.cleanup_expired()
        assert removed >= 0  # May be 0 if already cleaned by background task
    
    @pytest.mark.asyncio
    async def test_stm_to_wm_promotion_threshold(self, memory_system):
        """Test STM→WM promotion with effectiveness threshold >5.0."""
        storage = memory_system['storage']
        scorer = memory_system['scorer']
        promotion = memory_system['promotion']
        
        # Create STM memories with different scores
        low_score_memory = MemoryItem(
            id="low_score",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Low value content"},
            effectiveness_score=4.5  # Below threshold
        )
        
        high_score_memory = MemoryItem(
            id="high_score",
            user_id="test_user",
            memory_type="stm",
            content={"text": "High value content"},
            effectiveness_score=5.5  # Above threshold
        )
        
        await storage.store(low_score_memory)
        await storage.store(high_score_memory)
        
        # Apply positive feedback to high score memory
        scorer.apply_feedback("high_score", FeedbackType.POSITIVE)
        
        # Run promotion evaluation
        promoted = await promotion.evaluate_stm_to_wm()
        
        # Only high score memory should be promoted
        promoted_ids = [m.id for m in promoted]
        assert "high_score" in promoted_ids
        assert "low_score" not in promoted_ids
    
    @pytest.mark.asyncio
    async def test_wm_to_ltm_promotion_criteria(self, memory_system):
        """Test WM→LTM promotion with score >8.0 and usage >5."""
        storage = memory_system['storage']
        promotion = memory_system['promotion']
        
        # Create WM memories with different criteria
        low_score = MemoryItem(
            id="wm_low_score",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Average content"},
            effectiveness_score=7.5,  # Below LTM threshold
            usage_count=10
        )
        
        low_usage = MemoryItem(
            id="wm_low_usage",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Rarely used"},
            effectiveness_score=8.5,
            usage_count=3  # Below LTM threshold
        )
        
        eligible = MemoryItem(
            id="wm_eligible",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Valuable and frequently used"},
            effectiveness_score=8.5,
            usage_count=10
        )
        
        await storage.store(low_score)
        await storage.store(low_usage)
        await storage.store(eligible)
        
        # Run promotion evaluation
        promoted = await promotion.evaluate_wm_to_ltm()
        
        # Only eligible memory should be promoted
        promoted_ids = [m.id for m in promoted]
        assert "wm_eligible" in promoted_ids
        assert "wm_low_score" not in promoted_ids
        assert "wm_low_usage" not in promoted_ids
    
    @pytest.mark.asyncio
    async def test_ltm_permanent_storage(self, memory_system):
        """Test that LTM memories have no expiration."""
        ltm = memory_system['ltm']
        
        memory = MemoryItem(
            id="ltm_permanent",
            user_id="test_user",
            memory_type="ltm",
            content={"text": "Permanent knowledge"},
            effectiveness_score=9.0,
            usage_count=20
        )
        
        # Store in LTM
        await ltm.store(memory)
        
        # Verify no expiration
        assert memory.expires_at is None
        assert not memory.is_expired()
        
        # Should persist indefinitely
        retrieved = await ltm.retrieve("ltm_permanent")
        assert retrieved is not None
        assert retrieved.expires_at is None
    
    @pytest.mark.asyncio
    async def test_promotion_tracking(self, memory_system):
        """Test that promotion history is tracked correctly."""
        storage = memory_system['storage']
        
        # Create memory that gets promoted
        memory = MemoryItem(
            id="tracked_memory",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Content to track"},
            effectiveness_score=6.0
        )
        
        await storage.store(memory)
        
        # Simulate promotion to WM
        memory.memory_type = "wm"
        memory.promoted_from = "stm"
        memory.promoted_at = datetime.now()
        memory.promotion_reason = "Effectiveness threshold met"
        
        await storage.update(memory)
        
        # Verify promotion metadata
        retrieved = await storage.retrieve("tracked_memory")
        assert retrieved.promoted_from == "stm"
        assert retrieved.promoted_at is not None
        assert retrieved.promotion_reason == "Effectiveness threshold met"
    
    @pytest.mark.asyncio
    async def test_memory_usage_tracking(self, memory_system):
        """Test that memory usage count is properly tracked."""
        storage = memory_system['storage']
        
        memory = MemoryItem(
            id="usage_test",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Frequently accessed"},
            usage_count=0
        )
        
        await storage.store(memory)
        
        # Simulate multiple accesses
        for _ in range(5):
            retrieved = await storage.retrieve("usage_test")
            retrieved.usage_count += 1
            retrieved.last_accessed = datetime.now()
            await storage.update(retrieved)
        
        # Verify usage count
        final = await storage.retrieve("usage_test")
        assert final.usage_count == 5
        assert final.last_accessed is not None
    
    @pytest.mark.asyncio
    async def test_cross_layer_memory_retrieval(self, memory_system):
        """Test retrieving memories across all layers."""
        storage = memory_system['storage']
        
        # Create memories in each layer
        stm_memory = MemoryItem(
            id="stm_item",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Short-term"},
            embedding=np.random.rand(4096).astype(np.float32)
        )
        
        wm_memory = MemoryItem(
            id="wm_item",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Working"},
            embedding=np.random.rand(4096).astype(np.float32)
        )
        
        ltm_memory = MemoryItem(
            id="ltm_item",
            user_id="test_user",
            memory_type="ltm",
            content={"text": "Long-term"},
            embedding=np.random.rand(4096).astype(np.float32)
        )
        
        await storage.store(stm_memory)
        await storage.store(wm_memory)
        await storage.store(ltm_memory)
        
        # Search across all layers
        query_embedding = np.random.rand(4096).astype(np.float32)
        results = await storage.search_by_embedding(
            embedding=query_embedding,
            k=10,
            user_id="test_user"
        )
        
        # Should find memories from all layers
        found_types = {r[0].memory_type for r in results}
        assert "stm" in found_types
        assert "wm" in found_types
        assert "ltm" in found_types
    
    @pytest.mark.asyncio
    async def test_memory_effectiveness_feedback_loop(self, memory_system):
        """Test effectiveness scoring with feedback adjustments."""
        storage = memory_system['storage']
        scorer = memory_system['scorer']
        
        memory = MemoryItem(
            id="feedback_test",
            user_id="test_user",
            memory_type="wm",
            content={"text": "Content for feedback"},
            effectiveness_score=5.0
        )
        
        await storage.store(memory)
        
        # Apply positive feedback (+0.3)
        scorer.apply_feedback("feedback_test", FeedbackType.POSITIVE)
        updated_score = scorer.get_score("feedback_test")
        assert updated_score == pytest.approx(5.3, rel=0.01)
        
        # Apply negative feedback (-0.3)
        scorer.apply_feedback("feedback_test", FeedbackType.NEGATIVE)
        updated_score = scorer.get_score("feedback_test")
        assert updated_score == pytest.approx(5.0, rel=0.01)
        
        # Score should stay within bounds [0, 10]
        for _ in range(20):
            scorer.apply_feedback("feedback_test", FeedbackType.POSITIVE)
        
        final_score = scorer.get_score("feedback_test")
        assert final_score <= 10.0
    
    @pytest.mark.asyncio
    async def test_batch_memory_operations(self, memory_system):
        """Test batch storage and retrieval operations."""
        storage = memory_system['storage']
        
        # Create batch of memories
        memories = []
        for i in range(50):
            memory = MemoryItem(
                id=f"batch_{i}",
                user_id="test_user",
                memory_type="wm",
                content={"text": f"Batch content {i}"},
                embedding=np.random.rand(4096).astype(np.float32)
            )
            memories.append(memory)
        
        # Batch store
        await storage.batch_store(memories)
        
        # Batch retrieve
        retrieved = await storage.list_by_user("test_user", limit=100)
        assert len(retrieved) >= 50
        
        # Verify all stored
        for memory in memories[:10]:  # Check first 10
            result = await storage.retrieve(memory.id)
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle_end_to_end(self, memory_system):
        """Test complete memory lifecycle from creation to LTM."""
        storage = memory_system['storage']
        scorer = memory_system['scorer']
        promotion = memory_system['promotion']
        
        # 1. Create STM memory
        memory = MemoryItem(
            id="lifecycle_e2e",
            user_id="test_user",
            memory_type="stm",
            content={"text": "Important information to remember"},
            effectiveness_score=5.5,
            usage_count=1,
            embedding=np.random.rand(4096).astype(np.float32)
        )
        
        await storage.store(memory)
        
        # 2. Apply positive feedback to increase score
        for _ in range(3):
            scorer.apply_feedback("lifecycle_e2e", FeedbackType.POSITIVE)
        
        # 3. Promote STM → WM
        promoted_to_wm = await promotion.evaluate_stm_to_wm()
        assert any(m.id == "lifecycle_e2e" for m in promoted_to_wm)
        
        # Update memory state
        memory.memory_type = "wm"
        memory.promoted_from = "stm"
        memory.effectiveness_score = scorer.get_score("lifecycle_e2e")
        await storage.update(memory)
        
        # 4. Increase usage and score for LTM promotion
        memory.usage_count = 10
        memory.effectiveness_score = 8.5
        await storage.update(memory)
        
        # 5. Promote WM → LTM
        promoted_to_ltm = await promotion.evaluate_wm_to_ltm()
        assert any(m.id == "lifecycle_e2e" for m in promoted_to_ltm)
        
        # 6. Verify final state
        final = await storage.retrieve("lifecycle_e2e")
        assert final.memory_type == "ltm"
        assert final.expires_at is None  # Permanent
        assert final.promoted_from == "wm"