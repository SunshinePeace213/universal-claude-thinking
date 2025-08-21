"""
Unit tests for memory promotion pipeline.

Tests the automated promotion system that moves memories between layers
based on effectiveness scores and usage counts, using APScheduler for
scheduled evaluations.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call
import uuid

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

# These imports will fail initially (TDD) - we'll implement them
from src.memory.promotion import PromotionPipeline, PromotionCriteria
from src.memory.layers.base import MemoryItem
from src.memory.storage import StorageBackend


class TestPromotionCriteria:
    """Test promotion criteria configuration."""
    
    def test_default_criteria(self):
        """Test default promotion criteria values."""
        criteria = PromotionCriteria()
        
        assert criteria.stm_to_wm_threshold == 5.0
        assert criteria.wm_to_ltm_score == 8.0
        assert criteria.wm_to_ltm_uses == 5
        assert criteria.stm_check_interval == 3600  # 1 hour
        assert criteria.wm_check_interval == 86400  # 1 day
    
    def test_custom_criteria(self):
        """Test custom promotion criteria configuration."""
        criteria = PromotionCriteria(
            stm_to_wm_threshold=6.0,
            wm_to_ltm_score=9.0,
            wm_to_ltm_uses=10,
            stm_check_interval=1800,
            wm_check_interval=43200
        )
        
        assert criteria.stm_to_wm_threshold == 6.0
        assert criteria.wm_to_ltm_score == 9.0
        assert criteria.wm_to_ltm_uses == 10
        assert criteria.stm_check_interval == 1800
        assert criteria.wm_check_interval == 43200


class TestPromotionPipeline:
    """Test the promotion pipeline implementation."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock(spec=StorageBackend)
        storage.get_memories_by_type = AsyncMock(return_value=[])
        storage.update_memory_type = AsyncMock()
        storage.record_promotion = AsyncMock()
        storage.get_memory = AsyncMock(return_value=None)
        storage.update_expiry = AsyncMock()
        storage.get_memories_batch = AsyncMock(return_value=[])
        return storage
    
    @pytest.fixture
    def pipeline(self, mock_storage):
        """Create a promotion pipeline instance."""
        criteria = PromotionCriteria()
        return PromotionPipeline(storage=mock_storage, criteria=criteria)
    
    def test_pipeline_initialization(self, pipeline, mock_storage):
        """Test pipeline initialization with storage and criteria."""
        assert pipeline.storage == mock_storage
        assert isinstance(pipeline.criteria, PromotionCriteria)
        assert isinstance(pipeline.scheduler, BackgroundScheduler)
        assert not pipeline.scheduler.running
    
    def test_start_scheduler(self, pipeline):
        """Test starting the promotion scheduler."""
        with patch.object(pipeline.scheduler, 'add_job') as mock_add_job:
            with patch.object(pipeline.scheduler, 'start') as mock_start:
                pipeline.start()
                
                # Should add two jobs - STM and WM evaluation
                assert mock_add_job.call_count == 2
                
                # Check STM evaluation job
                stm_call = mock_add_job.call_args_list[0]
                assert stm_call[1]['func'] == pipeline.evaluate_stm_memories
                assert stm_call[1]['id'] == 'stm_evaluation'
                assert isinstance(stm_call[1]['trigger'], IntervalTrigger)
                
                # Check WM evaluation job
                wm_call = mock_add_job.call_args_list[1]
                assert wm_call[1]['func'] == pipeline.evaluate_wm_memories
                assert wm_call[1]['id'] == 'wm_evaluation'
                assert isinstance(wm_call[1]['trigger'], IntervalTrigger)
                
                mock_start.assert_called_once()
    
    def test_stop_scheduler(self, pipeline):
        """Test stopping the promotion scheduler."""
        with patch.object(pipeline.scheduler, 'shutdown') as mock_shutdown:
            pipeline.stop()
            mock_shutdown.assert_called_once_with(wait=False)
    
    @pytest.mark.asyncio
    async def test_evaluate_stm_memories_for_promotion(self, pipeline, mock_storage):
        """Test evaluating STM memories for promotion to WM."""
        # Create test memories with different effectiveness scores
        memories = [
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="stm",
                content={"text": "Low score memory"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=3.0,
                usage_count=2,
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1)
            ),
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="stm",
                content={"text": "High score memory"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=7.0,
                usage_count=5,
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1)
            ),
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="stm",
                content={"text": "Borderline memory"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0,
                usage_count=3,
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1)
            )
        ]
        
        mock_storage.get_memories_by_type.return_value = memories
        
        # Run evaluation
        promoted = await pipeline.evaluate_stm_memories()
        
        # Check that correct memories were promoted
        assert len(promoted) == 2  # High score and borderline
        assert memories[1] in promoted  # High score
        assert memories[2] in promoted  # Borderline (5.0 threshold)
        assert memories[0] not in promoted  # Low score
        
        # Verify storage calls
        mock_storage.get_memories_by_type.assert_called_once_with("stm")
        assert mock_storage.update_memory_type.call_count == 2
        assert mock_storage.record_promotion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_evaluate_wm_memories_for_promotion(self, pipeline, mock_storage):
        """Test evaluating WM memories for promotion to LTM."""
        # Create test memories with different scores and usage counts
        memories = [
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="wm",
                content={"text": "Low score, low usage"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0,
                usage_count=2,
                created_at=datetime.now() - timedelta(days=3),
                expires_at=datetime.now() + timedelta(days=4)
            ),
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="wm",
                content={"text": "High score, high usage"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=9.0,
                usage_count=10,
                created_at=datetime.now() - timedelta(days=3),
                expires_at=datetime.now() + timedelta(days=4)
            ),
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="wm",
                content={"text": "High score, low usage"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=8.5,
                usage_count=3,
                created_at=datetime.now() - timedelta(days=3),
                expires_at=datetime.now() + timedelta(days=4)
            ),
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id="user1",
                memory_type="wm",
                content={"text": "Borderline case"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=8.0,
                usage_count=5,
                created_at=datetime.now() - timedelta(days=3),
                expires_at=datetime.now() + timedelta(days=4)
            )
        ]
        
        mock_storage.get_memories_by_type.return_value = memories
        
        # Run evaluation
        promoted = await pipeline.evaluate_wm_memories()
        
        # Check that correct memories were promoted (score >= 8.0 AND uses >= 5)
        assert len(promoted) == 2
        assert memories[1] in promoted  # High score, high usage
        assert memories[3] in promoted  # Borderline case (exactly meets criteria)
        assert memories[0] not in promoted  # Low score
        assert memories[2] not in promoted  # High score but low usage
        
        # Verify storage calls
        mock_storage.get_memories_by_type.assert_called_once_with("wm")
        assert mock_storage.update_memory_type.call_count == 2
        assert mock_storage.record_promotion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_manual_promotion(self, pipeline, mock_storage):
        """Test manually promoting a specific memory."""
        memory_id = str(uuid.uuid4())
        memory = MemoryItem(
            id=memory_id,
            user_id="user1",
            memory_type="stm",
            content={"text": "Manual promotion"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=4.0,  # Below threshold
            usage_count=1,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=2)
        )
        
        mock_storage.get_memory.return_value = memory
        
        # Manually promote to WM
        result = await pipeline.promote_memory(memory_id, target_layer="wm")
        
        assert result is True
        mock_storage.update_memory_type.assert_called_once_with(
            memory_id, "stm", "wm"
        )
        mock_storage.record_promotion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_promotion(self, pipeline, mock_storage):
        """Test promoting multiple memories in batch."""
        memory_ids = [str(uuid.uuid4()) for _ in range(3)]
        memories = [
            MemoryItem(
                id=mid,
                user_id="user1",
                memory_type="stm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0,
                usage_count=3,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=2)
            )
            for i, mid in enumerate(memory_ids)
        ]
        
        mock_storage.get_memories_batch.return_value = memories
        
        # Batch promote to WM
        results = await pipeline.promote_batch(memory_ids, target_layer="wm")
        
        assert all(results)
        assert mock_storage.update_memory_type.call_count == 3
        assert mock_storage.record_promotion.call_count == 3
    
    @pytest.mark.asyncio
    async def test_promotion_with_ttl_update(self, pipeline, mock_storage):
        """Test that promotion updates TTL appropriately."""
        memory_id = str(uuid.uuid4())
        memory = MemoryItem(
            id=memory_id,
            user_id="user1",
            memory_type="stm",
            content={"text": "TTL update test"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=7.0,
            usage_count=5,
            created_at=datetime.now() - timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=1)  # STM expiry
        )
        
        mock_storage.get_memory.return_value = memory
        
        # Promote to WM
        await pipeline.promote_memory(memory_id, target_layer="wm")
        
        # Check that TTL was updated for WM (7 days)
        update_call = mock_storage.update_memory_type.call_args
        assert update_call is not None
        
        # Should also update expiry time
        mock_storage.update_expiry.assert_called_once()
        expiry_call = mock_storage.update_expiry.call_args
        new_expiry = expiry_call[0][1]
        
        # New expiry should be ~7 days from now
        expected_expiry = datetime.now() + timedelta(days=7)
        time_diff = abs((new_expiry - expected_expiry).total_seconds())
        assert time_diff < 60  # Within 1 minute tolerance
    
    @pytest.mark.asyncio
    async def test_promotion_to_ltm_removes_expiry(self, pipeline, mock_storage):
        """Test that promoting to LTM removes expiry time."""
        memory_id = str(uuid.uuid4())
        memory = MemoryItem(
            id=memory_id,
            user_id="user1",
            memory_type="wm",
            content={"text": "LTM promotion"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=9.0,
            usage_count=10,
            created_at=datetime.now() - timedelta(days=3),
            expires_at=datetime.now() + timedelta(days=4)
        )
        
        mock_storage.get_memory.return_value = memory
        
        # Promote to LTM
        await pipeline.promote_memory(memory_id, target_layer="ltm")
        
        # Check that expiry was set to None for LTM
        mock_storage.update_expiry.assert_called_once_with(memory_id, None)
    
    @pytest.mark.asyncio
    async def test_promotion_failure_handling(self, pipeline, mock_storage):
        """Test handling of promotion failures."""
        memory_id = str(uuid.uuid4())
        
        # Simulate storage error
        mock_storage.get_memory.side_effect = Exception("Storage error")
        
        # Attempt promotion
        result = await pipeline.promote_memory(memory_id, target_layer="wm")
        
        assert result is False
        mock_storage.update_memory_type.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_promotions(self, pipeline, mock_storage):
        """Test handling concurrent promotion evaluations."""
        # Create many memories
        memories = [
            MemoryItem(
                id=str(uuid.uuid4()),
                user_id=f"user{i % 3}",
                memory_type="stm",
                content={"text": f"Memory {i}"},
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=6.0 + (i % 3),
                usage_count=i,
                created_at=datetime.now() - timedelta(hours=1),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            for i in range(100)
        ]
        
        mock_storage.get_memories_by_type.return_value = memories
        
        # Run multiple concurrent evaluations
        tasks = [
            pipeline.evaluate_stm_memories()
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All evaluations should complete without errors
        assert all(isinstance(r, list) for r in results)
    
    def test_promotion_statistics(self, pipeline):
        """Test tracking promotion statistics."""
        stats = pipeline.get_statistics()
        
        assert 'total_promotions' in stats
        assert 'stm_to_wm' in stats
        assert 'wm_to_ltm' in stats
        assert 'last_stm_evaluation' in stats
        assert 'last_wm_evaluation' in stats
        assert stats['total_promotions'] == 0
    
    @pytest.mark.asyncio
    async def test_promotion_with_privacy_check(self, pipeline, mock_storage):
        """Test that promotions respect privacy settings."""
        memory_id = str(uuid.uuid4())
        memory = MemoryItem(
            id=memory_id,
            user_id="user1",
            memory_type="wm",
            content={"text": "Private memory", "private": True},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=9.0,
            usage_count=10,
            created_at=datetime.now() - timedelta(days=3),
            expires_at=datetime.now() + timedelta(days=4),
            metadata={"private": True}
        )
        
        mock_storage.get_memory.return_value = memory
        
        # Attempt to promote to SWARM (should fail for private memories)
        result = await pipeline.promote_memory(memory_id, target_layer="swarm")
        
        assert result is False
        mock_storage.update_memory_type.assert_not_called()