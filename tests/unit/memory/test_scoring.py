"""
Unit tests for memory effectiveness scoring system.

Tests the scoring mechanism that tracks memory effectiveness based on
user feedback and usage patterns, with +0.3/-0.3 adjustments.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import numpy as np

# These imports will fail initially (TDD) - we'll implement them
from src.memory.scoring import (
    EffectivenessScorer,
    FeedbackType,
    ScoringConfig,
    MemoryScore
)
from src.memory.layers.base import MemoryItem


class TestScoringConfig:
    """Test scoring configuration."""
    
    def test_default_config(self):
        """Test default scoring configuration values."""
        config = ScoringConfig()
        
        assert config.default_score == 5.0
        assert config.positive_adjustment == 0.3
        assert config.negative_adjustment == -0.3
        assert config.min_score == 0.0
        assert config.max_score == 10.0
        assert config.decay_rate == 0.01  # Daily decay for unused memories
        assert config.usage_boost == 0.1  # Boost per usage
    
    def test_custom_config(self):
        """Test custom scoring configuration."""
        config = ScoringConfig(
            default_score=6.0,
            positive_adjustment=0.5,
            negative_adjustment=-0.5,
            min_score=1.0,
            max_score=9.0,
            decay_rate=0.02,
            usage_boost=0.2
        )
        
        assert config.default_score == 6.0
        assert config.positive_adjustment == 0.5
        assert config.negative_adjustment == -0.5
        assert config.min_score == 1.0
        assert config.max_score == 9.0
        assert config.decay_rate == 0.02
        assert config.usage_boost == 0.2
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid score range
        with pytest.raises(ValueError, match="min_score must be less than max_score"):
            ScoringConfig(min_score=10.0, max_score=5.0)
        
        # Invalid default score
        with pytest.raises(ValueError, match="default_score must be between"):
            ScoringConfig(default_score=11.0)
        
        # Invalid adjustment values
        with pytest.raises(ValueError, match="positive_adjustment must be positive"):
            ScoringConfig(positive_adjustment=-0.5)
        
        with pytest.raises(ValueError, match="negative_adjustment must be negative"):
            ScoringConfig(negative_adjustment=0.5)


class TestMemoryScore:
    """Test memory score data structure."""
    
    def test_memory_score_creation(self):
        """Test creating a memory score record."""
        score = MemoryScore(
            memory_id="test_memory_1",
            current_score=5.0,
            initial_score=5.0,
            usage_count=0,
            positive_feedback_count=0,
            negative_feedback_count=0,
            last_accessed=datetime.now(),
            last_updated=datetime.now()
        )
        
        assert score.memory_id == "test_memory_1"
        assert score.current_score == 5.0
        assert score.initial_score == 5.0
        assert score.usage_count == 0
        assert score.positive_feedback_count == 0
        assert score.negative_feedback_count == 0
    
    def test_memory_score_history(self):
        """Test score history tracking."""
        score = MemoryScore(
            memory_id="test_memory_1",
            current_score=7.0,
            initial_score=5.0,
            usage_count=10,
            positive_feedback_count=5,
            negative_feedback_count=2,
            last_accessed=datetime.now(),
            last_updated=datetime.now(),
            score_history=[
                (datetime.now() - timedelta(hours=3), 5.0),
                (datetime.now() - timedelta(hours=2), 5.3),
                (datetime.now() - timedelta(hours=1), 6.6),
                (datetime.now(), 7.0)
            ]
        )
        
        assert len(score.score_history) == 4
        assert score.score_history[-1][1] == 7.0
        assert score.net_feedback == 3  # 5 positive - 2 negative


class TestEffectivenessScorer:
    """Test the effectiveness scoring implementation."""
    
    @pytest.fixture
    def scorer(self):
        """Create an effectiveness scorer instance."""
        config = ScoringConfig()
        return EffectivenessScorer(config=config)
    
    @pytest.fixture
    def sample_memory(self):
        """Create a sample memory item."""
        return MemoryItem(
            id=str(uuid.uuid4()),
            user_id="user1",
            memory_type="stm",
            content={"text": "Test memory"},
            embedding=np.random.rand(4096).astype(np.float32),
            effectiveness_score=5.0,
            usage_count=0,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=2)
        )
    
    def test_scorer_initialization(self, scorer):
        """Test scorer initialization with default config."""
        assert isinstance(scorer.config, ScoringConfig)
        assert scorer.config.default_score == 5.0
        assert len(scorer.scores) == 0
    
    def test_initial_score_assignment(self, scorer, sample_memory):
        """Test assigning initial score to new memory."""
        score = scorer.initialize_score(sample_memory.id)
        
        assert score.memory_id == sample_memory.id
        assert score.current_score == 5.0
        assert score.initial_score == 5.0
        assert score.usage_count == 0
        assert sample_memory.id in scorer.scores
    
    def test_positive_feedback(self, scorer, sample_memory):
        """Test applying positive feedback to memory score."""
        # Initialize score
        scorer.initialize_score(sample_memory.id)
        
        # Apply positive feedback
        new_score = scorer.apply_feedback(
            sample_memory.id, 
            FeedbackType.POSITIVE
        )
        
        assert new_score == 5.3  # 5.0 + 0.3
        assert scorer.scores[sample_memory.id].current_score == 5.3
        assert scorer.scores[sample_memory.id].positive_feedback_count == 1
    
    def test_negative_feedback(self, scorer, sample_memory):
        """Test applying negative feedback to memory score."""
        # Initialize score
        scorer.initialize_score(sample_memory.id)
        
        # Apply negative feedback
        new_score = scorer.apply_feedback(
            sample_memory.id,
            FeedbackType.NEGATIVE
        )
        
        assert new_score == 4.7  # 5.0 - 0.3
        assert scorer.scores[sample_memory.id].current_score == 4.7
        assert scorer.scores[sample_memory.id].negative_feedback_count == 1
    
    def test_score_boundaries(self, scorer, sample_memory):
        """Test that scores respect min/max boundaries."""
        # Initialize score
        scorer.initialize_score(sample_memory.id)
        
        # Apply many positive feedbacks
        for _ in range(20):
            scorer.apply_feedback(sample_memory.id, FeedbackType.POSITIVE)
        
        # Score should be capped at max_score (10.0)
        assert scorer.scores[sample_memory.id].current_score == 10.0
        
        # Apply many negative feedbacks
        for _ in range(50):
            scorer.apply_feedback(sample_memory.id, FeedbackType.NEGATIVE)
        
        # Score should be capped at min_score (0.0)
        assert scorer.scores[sample_memory.id].current_score == 0.0
    
    def test_usage_tracking(self, scorer, sample_memory):
        """Test tracking memory usage count."""
        # Initialize score
        scorer.initialize_score(sample_memory.id)
        
        # Record usage
        scorer.record_usage(sample_memory.id)
        assert scorer.scores[sample_memory.id].usage_count == 1
        
        # Multiple usages
        for _ in range(5):
            scorer.record_usage(sample_memory.id)
        
        assert scorer.scores[sample_memory.id].usage_count == 6
    
    def test_usage_boost(self, scorer, sample_memory):
        """Test score boost from usage."""
        # Initialize score
        scorer.initialize_score(sample_memory.id)
        
        # Record usage with boost
        new_score = scorer.record_usage(sample_memory.id, apply_boost=True)
        
        assert new_score == 5.1  # 5.0 + 0.1 usage boost
        assert scorer.scores[sample_memory.id].usage_count == 1
        assert scorer.scores[sample_memory.id].current_score == 5.1
    
    def test_score_decay(self, scorer, sample_memory):
        """Test score decay for unused memories."""
        # Initialize score with higher value
        scorer.initialize_score(sample_memory.id)
        scorer.scores[sample_memory.id].current_score = 7.0
        scorer.scores[sample_memory.id].last_accessed = datetime.now() - timedelta(days=5)
        
        # Apply decay
        new_score = scorer.apply_decay(sample_memory.id)
        
        # Should decay by 0.01 per day * 5 days = 0.05
        expected_score = 7.0 - (0.01 * 5)
        assert abs(new_score - expected_score) < 0.001
    
    def test_batch_scoring(self, scorer):
        """Test scoring multiple memories in batch."""
        memory_ids = [str(uuid.uuid4()) for _ in range(10)]
        
        # Initialize all scores
        for mid in memory_ids:
            scorer.initialize_score(mid)
        
        # Apply batch feedback
        feedback_map = {
            memory_ids[0]: FeedbackType.POSITIVE,
            memory_ids[1]: FeedbackType.POSITIVE,
            memory_ids[2]: FeedbackType.NEGATIVE,
            memory_ids[3]: FeedbackType.POSITIVE,
            memory_ids[4]: FeedbackType.NEGATIVE,
        }
        
        results = scorer.apply_batch_feedback(feedback_map)
        
        assert results[memory_ids[0]] == 5.3
        assert results[memory_ids[1]] == 5.3
        assert results[memory_ids[2]] == 4.7
        assert results[memory_ids[3]] == 5.3
        assert results[memory_ids[4]] == 4.7
    
    def test_score_normalization(self, scorer):
        """Test score normalization across all memories."""
        # Create memories with various scores
        memory_ids = [str(uuid.uuid4()) for _ in range(5)]
        scores = [3.0, 5.0, 7.0, 9.0, 2.0]
        
        for mid, score in zip(memory_ids, scores):
            scorer.initialize_score(mid)
            scorer.scores[mid].current_score = score
        
        # Normalize scores
        normalized = scorer.normalize_scores()
        
        # Check that normalization maintains relative ordering
        sorted_original = sorted(zip(memory_ids, scores), key=lambda x: x[1])
        sorted_normalized = sorted(normalized.items(), key=lambda x: x[1])
        
        for i in range(len(memory_ids)):
            assert sorted_original[i][0] == sorted_normalized[i][0]
    
    def test_get_top_memories(self, scorer):
        """Test retrieving memories with highest scores."""
        # Create memories with various scores
        memory_ids = [str(uuid.uuid4()) for _ in range(10)]
        scores = [3.0, 8.0, 5.0, 9.0, 2.0, 7.0, 6.0, 4.0, 8.5, 5.5]
        
        for mid, score in zip(memory_ids, scores):
            scorer.initialize_score(mid)
            scorer.scores[mid].current_score = score
        
        # Get top 3 memories
        top_3 = scorer.get_top_memories(n=3)
        
        assert len(top_3) == 3
        assert top_3[0][0] == memory_ids[3]  # Score 9.0
        assert top_3[0][1] == 9.0
        assert top_3[1][0] == memory_ids[8]  # Score 8.5
        assert top_3[1][1] == 8.5
        assert top_3[2][0] == memory_ids[1]  # Score 8.0
        assert top_3[2][1] == 8.0
    
    def test_score_persistence(self, scorer, sample_memory):
        """Test saving and loading scores."""
        # Initialize and modify scores
        scorer.initialize_score(sample_memory.id)
        scorer.apply_feedback(sample_memory.id, FeedbackType.POSITIVE)
        scorer.record_usage(sample_memory.id)
        
        # Save scores
        saved_data = scorer.export_scores()
        
        # Create new scorer and import
        new_scorer = EffectivenessScorer(config=ScoringConfig())
        new_scorer.import_scores(saved_data)
        
        # Verify scores were restored
        assert sample_memory.id in new_scorer.scores
        assert new_scorer.scores[sample_memory.id].current_score == 5.3
        assert new_scorer.scores[sample_memory.id].usage_count == 1
        assert new_scorer.scores[sample_memory.id].positive_feedback_count == 1
    
    def test_concurrent_scoring(self, scorer):
        """Test thread-safe concurrent scoring operations."""
        memory_id = str(uuid.uuid4())
        scorer.initialize_score(memory_id)
        
        # Simulate concurrent feedback
        async def apply_feedback_async():
            await asyncio.sleep(0.001)  # Small delay
            scorer.apply_feedback(memory_id, FeedbackType.POSITIVE)
        
        # Run multiple concurrent operations
        async def run_concurrent_test():
            tasks = [apply_feedback_async() for _ in range(10)]
            await asyncio.gather(*tasks)
        
        # Use asyncio.run for proper event loop handling
        asyncio.run(run_concurrent_test())
        
        # All feedbacks should be applied correctly
        assert scorer.scores[memory_id].positive_feedback_count == 10
        expected_score = 5.0 + (0.3 * 10)
        assert scorer.scores[memory_id].current_score == pytest.approx(min(expected_score, 10.0), 0.001)
    
    def test_score_statistics(self, scorer):
        """Test computing scoring statistics."""
        # Create memories with various scores
        memory_ids = [str(uuid.uuid4()) for _ in range(10)]
        scores = [3.0, 8.0, 5.0, 9.0, 2.0, 7.0, 6.0, 4.0, 8.5, 5.5]
        
        for mid, score in zip(memory_ids, scores):
            scorer.initialize_score(mid)
            scorer.scores[mid].current_score = score
            scorer.scores[mid].usage_count = int(score * 2)
        
        # Get statistics
        stats = scorer.get_statistics()
        
        assert stats['total_memories'] == 10
        assert stats['average_score'] == pytest.approx(5.8, 0.1)
        assert stats['median_score'] == 5.75  # Median of 10 values: (5.5 + 6.0) / 2
        assert stats['total_usage'] == sum(int(s * 2) for s in scores)
        assert stats['top_score'] == 9.0
        assert stats['bottom_score'] == 2.0
    
    def test_reset_score(self, scorer, sample_memory):
        """Test resetting a memory's score to default."""
        # Initialize and modify score
        scorer.initialize_score(sample_memory.id)
        scorer.apply_feedback(sample_memory.id, FeedbackType.POSITIVE)
        scorer.apply_feedback(sample_memory.id, FeedbackType.POSITIVE)
        scorer.record_usage(sample_memory.id)
        
        assert scorer.scores[sample_memory.id].current_score == 5.6
        assert scorer.scores[sample_memory.id].usage_count == 1
        
        # Reset score
        scorer.reset_score(sample_memory.id)
        
        assert scorer.scores[sample_memory.id].current_score == 5.0
        assert scorer.scores[sample_memory.id].usage_count == 0
        assert scorer.scores[sample_memory.id].positive_feedback_count == 0
        assert scorer.scores[sample_memory.id].negative_feedback_count == 0
    
    def test_score_filtering(self, scorer):
        """Test filtering memories by score threshold."""
        # Create memories with various scores
        memory_ids = [str(uuid.uuid4()) for _ in range(10)]
        scores = [3.0, 8.0, 5.0, 9.0, 2.0, 7.0, 6.0, 4.0, 8.5, 5.5]
        
        for mid, score in zip(memory_ids, scores):
            scorer.initialize_score(mid)
            scorer.scores[mid].current_score = score
        
        # Filter memories with score >= 7.0
        high_score_memories = scorer.filter_by_score(min_score=7.0)
        
        assert len(high_score_memories) == 4
        assert all(scorer.scores[mid].current_score >= 7.0 for mid in high_score_memories)
        
        # Filter memories with score between 4.0 and 6.0
        mid_score_memories = scorer.filter_by_score(min_score=4.0, max_score=6.0)
        
        assert len(mid_score_memories) == 4
        assert all(4.0 <= scorer.scores[mid].current_score <= 6.0 
                  for mid in mid_score_memories)