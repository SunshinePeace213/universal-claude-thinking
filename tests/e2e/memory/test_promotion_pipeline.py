"""
End-to-end tests for memory promotion pipeline.

Tests automatic and manual promotion of memories through the hierarchy,
including threshold evaluation, feedback adjustments, and scheduling.
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
    PromotionPipeline,
    EffectivenessScorer,
    MemoryConfig,
)


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
async def promotion_system(temp_db):
    """Create a complete promotion pipeline system."""
    config = MemoryConfig()
    
    # Create memory layers
    stm = ShortTermMemory(cache_size=100, ttl_hours=2.0)
    wm = WorkingMemory(db_path=temp_db, ttl_days=7)
    ltm = LongTermMemory(db_path=temp_db)
    
    # Initialize layers
    await stm.initialize()
    await wm.initialize()
    await ltm.initialize()
    
    # Create promotion pipeline
    pipeline = PromotionPipeline(
        stm=stm,
        wm=wm,
        ltm=ltm,
        config=config
    )
    
    # Create scorer
    scorer = EffectivenessScorer()
    
    yield {
        "pipeline": pipeline,
        "stm": stm,
        "wm": wm,
        "ltm": ltm,
        "scorer": scorer,
        "config": config
    }
    
    # Cleanup
    await pipeline.stop()
    await wm.close()
    await ltm.close()


@pytest.mark.asyncio
class TestPromotionPipeline:
    """Test the automated memory promotion pipeline."""
    
    async def test_automatic_stm_to_wm_promotion(self, promotion_system):
        """Test automatic promotion from STM to WM based on effectiveness."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        scorer = promotion_system["scorer"]
        
        # Create memories with varying effectiveness
        memories = [
            MemoryItem(
                id=f"auto_stm_{i}",
                user_id="user_123",
                memory_type="stm",
                content={"text": f"Memory {i}", "importance": i},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=3.0 + i,  # Scores: 3, 4, 5, 6, 7
                usage_count=i + 1,
                created_at=datetime.now() - timedelta(hours=1)  # Old enough for promotion
            )
            for i in range(5)
        ]
        
        # Store all memories in STM
        for memory in memories:
            await stm.store(memory)
        
        # Run promotion evaluation
        promoted_count = await pipeline.evaluate_stm_promotions()
        
        # Should promote memories with score > 5.0 (indices 3 and 4)
        assert promoted_count == 2
        
        # Verify promoted memories are in WM
        for i in [3, 4]:
            wm_memory = await wm.retrieve(f"auto_stm_{i}", user_id="user_123")
            assert wm_memory is not None
            assert wm_memory.memory_type == "wm"
            assert wm_memory.promoted_from == "stm"
            
            # Should be removed from STM
            stm_memory = await stm.retrieve(f"auto_stm_{i}")
            assert stm_memory is None
        
        # Low-score memories should remain in STM
        for i in [0, 1, 2]:
            stm_memory = await stm.retrieve(f"auto_stm_{i}")
            assert stm_memory is not None
            assert stm_memory.memory_type == "stm"
    
    async def test_automatic_wm_to_ltm_promotion(self, promotion_system):
        """Test automatic promotion from WM to LTM based on score and usage."""
        pipeline = promotion_system["pipeline"]
        wm = promotion_system["wm"]
        ltm = promotion_system["ltm"]
        
        # Create WM memories with varying scores and usage
        memories = [
            MemoryItem(
                id="low_score_low_usage",
                user_id="user_123",
                memory_type="wm",
                content={"text": "Rarely used"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0,  # Below LTM threshold
                usage_count=2,  # Below usage threshold
                promoted_from="stm"
            ),
            MemoryItem(
                id="high_score_low_usage",
                user_id="user_123",
                memory_type="wm",
                content={"text": "Good but rarely used"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=8.5,  # Above score threshold
                usage_count=2,  # Below usage threshold
                promoted_from="stm"
            ),
            MemoryItem(
                id="low_score_high_usage",
                user_id="user_123",
                memory_type="wm",
                content={"text": "Frequently used but low score"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0,  # Below score threshold
                usage_count=10,  # Above usage threshold
                promoted_from="stm"
            ),
            MemoryItem(
                id="high_score_high_usage",
                user_id="user_123",
                memory_type="wm",
                content={"text": "Valuable and frequently used"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=9.0,  # Above score threshold
                usage_count=15,  # Above usage threshold
                promoted_from="stm"
            )
        ]
        
        # Store all in WM
        for memory in memories:
            await wm.store(memory)
        
        # Run WM promotion evaluation
        promoted_count = await pipeline.evaluate_wm_promotions()
        
        # Only the memory meeting both criteria should be promoted
        assert promoted_count == 1
        
        # Verify correct memory was promoted
        ltm_memory = await ltm.retrieve("high_score_high_usage", user_id="user_123")
        assert ltm_memory is not None
        assert ltm_memory.memory_type == "ltm"
        assert ltm_memory.promoted_from == "wm"
        assert ltm_memory.expires_at is None  # LTM is permanent
        
        # Others should remain in WM
        for memory_id in ["low_score_low_usage", "high_score_low_usage", "low_score_high_usage"]:
            wm_memory = await wm.retrieve(memory_id, user_id="user_123")
            assert wm_memory is not None
            assert wm_memory.memory_type == "wm"
    
    async def test_feedback_driven_promotion(self, promotion_system):
        """Test that positive feedback drives promotion."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        scorer = promotion_system["scorer"]
        
        # Create memory below promotion threshold
        memory = MemoryItem(
            id="feedback_driven",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Initially low value"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=3.0,  # Below WM threshold
            usage_count=1,
            created_at=datetime.now() - timedelta(hours=1)
        )
        
        await stm.store(memory)
        
        # Initially should not be promoted
        candidates = await stm.get_promotion_candidates(min_score=5.0)
        assert len(candidates) == 0
        
        # Apply positive feedback multiple times
        for _ in range(10):
            retrieved = await stm.retrieve("feedback_driven")
            scorer.apply_feedback(retrieved, is_positive=True)
            await stm.store(retrieved)  # Update with new score
        
        # Now should be eligible for promotion
        final_memory = await stm.retrieve("feedback_driven")
        assert final_memory.effectiveness_score > 5.0
        
        # Run promotion
        promoted_count = await pipeline.evaluate_stm_promotions()
        assert promoted_count == 1
        
        # Verify promoted to WM
        wm_memory = await wm.retrieve("feedback_driven", user_id="user_123")
        assert wm_memory is not None
        assert wm_memory.memory_type == "wm"
    
    async def test_negative_feedback_prevents_promotion(self, promotion_system):
        """Test that negative feedback prevents promotion."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        scorer = promotion_system["scorer"]
        
        # Create memory near promotion threshold
        memory = MemoryItem(
            id="negative_feedback",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Potentially valuable"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.5,  # Just above threshold
            usage_count=3,
            created_at=datetime.now() - timedelta(hours=1)
        )
        
        await stm.store(memory)
        
        # Apply negative feedback
        for _ in range(3):
            retrieved = await stm.retrieve("negative_feedback")
            scorer.apply_feedback(retrieved, is_positive=False)
            await stm.store(retrieved)
        
        # Score should have decreased
        final_memory = await stm.retrieve("negative_feedback")
        assert final_memory.effectiveness_score < 5.0
        
        # Should not be promoted
        promoted_count = await pipeline.evaluate_stm_promotions()
        assert promoted_count == 0
    
    async def test_manual_promotion_override(self, promotion_system):
        """Test manual promotion overrides automatic thresholds."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        ltm = promotion_system["ltm"]
        
        # Create low-score memory
        memory = MemoryItem(
            id="manual_promote",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Manually important"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=2.0,  # Well below threshold
            usage_count=1
        )
        
        await stm.store(memory)
        
        # Manually promote STM -> WM
        success = await pipeline.promote_manually(
            memory_id="manual_promote",
            from_layer="stm",
            to_layer="wm"
        )
        assert success is True
        
        # Verify in WM
        wm_memory = await wm.retrieve("manual_promote", user_id="user_123")
        assert wm_memory is not None
        assert wm_memory.memory_type == "wm"
        assert wm_memory.promoted_from == "stm"
        
        # Manually promote WM -> LTM
        success = await pipeline.promote_manually(
            memory_id="manual_promote",
            from_layer="wm",
            to_layer="ltm"
        )
        assert success is True
        
        # Verify in LTM
        ltm_memory = await ltm.retrieve("manual_promote", user_id="user_123")
        assert ltm_memory is not None
        assert ltm_memory.memory_type == "ltm"
        assert ltm_memory.expires_at is None
    
    async def test_batch_promotion_efficiency(self, promotion_system):
        """Test efficient batch promotion of multiple memories."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        
        # Create 20 eligible memories
        memories = []
        for i in range(20):
            memory = MemoryItem(
                id=f"batch_promote_{i}",
                user_id="user_123",
                memory_type="stm",
                content={"text": f"Batch memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0 + (i * 0.1),  # All above threshold
                usage_count=i + 2,
                created_at=datetime.now() - timedelta(hours=2)
            )
            memories.append(memory)
            await stm.store(memory)
        
        # Time batch promotion
        start_time = datetime.now()
        promoted_count = await pipeline.evaluate_stm_promotions()
        promotion_time = (datetime.now() - start_time).total_seconds()
        
        # All should be promoted
        assert promoted_count == 20
        
        # Should complete in reasonable time
        assert promotion_time < 10.0  # 10 seconds for 20 memories
        
        # Verify all in WM
        for i in range(20):
            wm_memory = await wm.retrieve(f"batch_promote_{i}", user_id="user_123")
            assert wm_memory is not None
            assert wm_memory.memory_type == "wm"
    
    async def test_promotion_with_ttl_update(self, promotion_system):
        """Test that promotion updates TTL appropriately."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        ltm = promotion_system["ltm"]
        
        # Create STM memory
        memory = MemoryItem(
            id="ttl_test",
            user_id="user_123",
            memory_type="stm",
            content={"text": "TTL test memory"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=6.0,
            usage_count=3,
            created_at=datetime.now() - timedelta(hours=1)
        )
        
        await stm.store(memory)
        stm_memory = await stm.retrieve("ttl_test")
        
        # STM should have ~2 hour TTL
        stm_ttl = stm_memory.expires_at - datetime.now()
        assert 0 < stm_ttl.total_seconds() < 7200  # Less than 2 hours
        
        # Promote to WM
        await pipeline.evaluate_stm_promotions()
        wm_memory = await wm.retrieve("ttl_test", user_id="user_123")
        
        # WM should have ~7 day TTL
        wm_ttl = wm_memory.expires_at - datetime.now()
        assert 600000 < wm_ttl.total_seconds() < 605000  # ~7 days
        
        # Update for LTM promotion
        wm_memory.effectiveness_score = 9.0
        wm_memory.usage_count = 10
        await wm.store(wm_memory)
        
        # Promote to LTM
        await pipeline.evaluate_wm_promotions()
        ltm_memory = await ltm.retrieve("ttl_test", user_id="user_123")
        
        # LTM should have no expiration
        assert ltm_memory.expires_at is None
    
    async def test_promotion_failure_handling(self, promotion_system):
        """Test graceful handling of promotion failures."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        wm = promotion_system["wm"]
        
        # Create memory for promotion
        memory = MemoryItem(
            id="fail_promotion",
            user_id="user_123",
            memory_type="stm",
            content={"text": "Will fail to promote"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=7.0,
            usage_count=5,
            created_at=datetime.now() - timedelta(hours=1)
        )
        
        await stm.store(memory)
        
        # Mock WM store to fail
        with patch.object(wm, 'store', side_effect=Exception("Storage error")):
            promoted_count = await pipeline.evaluate_stm_promotions()
            
            # Should handle failure gracefully
            assert promoted_count == 0
            
            # Memory should remain in STM
            stm_memory = await stm.retrieve("fail_promotion")
            assert stm_memory is not None
            assert stm_memory.memory_type == "stm"
    
    async def test_promotion_statistics_tracking(self, promotion_system):
        """Test that promotion statistics are tracked correctly."""
        pipeline = promotion_system["pipeline"]
        stm = promotion_system["stm"]
        
        # Create memories for promotion
        for i in range(5):
            memory = MemoryItem(
                id=f"stats_{i}",
                user_id="user_123",
                memory_type="stm",
                content={"text": f"Stats memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0 + i,
                usage_count=i + 2,
                created_at=datetime.now() - timedelta(hours=1)
            )
            await stm.store(memory)
        
        # Get initial stats
        initial_stats = pipeline.get_statistics()
        initial_stm_to_wm = initial_stats.get("stm_to_wm_promotions", 0)
        
        # Run promotion
        promoted_count = await pipeline.evaluate_stm_promotions()
        
        # Get updated stats
        final_stats = pipeline.get_statistics()
        final_stm_to_wm = final_stats.get("stm_to_wm_promotions", 0)
        
        # Stats should reflect promotions
        assert final_stm_to_wm == initial_stm_to_wm + promoted_count
        assert final_stats.get("last_stm_evaluation") is not None