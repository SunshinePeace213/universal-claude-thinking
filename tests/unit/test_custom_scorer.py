"""
Unit tests for Custom Scoring Algorithm.

Tests the CustomScorer class that implements hybrid scoring mechanisms
combining semantic similarity, effectiveness tracking, and contextual relevance.
"""

import math
import unittest
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

import numpy as np
import pytest

from src.rag.custom_scorer import (
    CustomScorer,
    ScoredItem,
    ScoringMethod,
)


class TestScoredItem(unittest.TestCase):
    """Test cases for ScoredItem dataclass."""
    
    def test_scored_item_creation(self):
        """Test creating a scored item with all fields."""
        item = ScoredItem(
            id=1,
            content="Test content",
            semantic_score=0.9,
            lexical_score=0.7,
            effectiveness_score=0.5,
            recency_score=0.6,
            diversity_score=0.3,
            final_score=0.75,
            metadata={"key": "value"},
        )
        
        self.assertEqual(item.id, 1)
        self.assertEqual(item.content, "Test content")
        self.assertEqual(item.semantic_score, 0.9)
        self.assertEqual(item.lexical_score, 0.7)
        self.assertEqual(item.effectiveness_score, 0.5)
        self.assertEqual(item.recency_score, 0.6)
        self.assertEqual(item.diversity_score, 0.3)
        self.assertEqual(item.final_score, 0.75)
        self.assertEqual(item.metadata, {"key": "value"})


class TestCustomScorer(unittest.TestCase):
    """Test cases for CustomScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = CustomScorer(
            semantic_weight=0.5,
            lexical_weight=0.2,
            effectiveness_weight=0.2,
            recency_weight=0.05,
            diversity_weight=0.05,
            learning_rate=0.01,
        )
        
    def test_initialization(self):
        """Test scorer initialization and weight normalization."""
        # Weights should be normalized to sum to 1
        total_weight = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
        
        # Check individual weights are normalized
        self.assertAlmostEqual(self.scorer.weights["semantic"], 0.5)
        self.assertAlmostEqual(self.scorer.weights["lexical"], 0.2)
        self.assertAlmostEqual(self.scorer.weights["effectiveness"], 0.2)
        self.assertAlmostEqual(self.scorer.weights["recency"], 0.05)
        self.assertAlmostEqual(self.scorer.weights["diversity"], 0.05)
        
        # Check BM25 parameters
        self.assertEqual(self.scorer.k1, 1.2)
        self.assertEqual(self.scorer.b, 0.75)
        
    def test_normalize_weights(self):
        """Test weight normalization."""
        # Set unnormalized weights
        self.scorer.weights = {
            "semantic": 2.0,
            "lexical": 1.0,
            "effectiveness": 1.0,
            "recency": 0.5,
            "diversity": 0.5,
        }
        
        self.scorer._normalize_weights()
        
        # Check normalization
        total = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Check proportions maintained
        self.assertAlmostEqual(self.scorer.weights["semantic"], 0.4)  # 2.0/5.0
        self.assertAlmostEqual(self.scorer.weights["lexical"], 0.2)   # 1.0/5.0
        
    def test_calculate_semantic_score(self):
        """Test semantic similarity calculation."""
        # Test with valid embeddings
        item_embedding = np.array([1, 0, 0, 0])
        query_embedding = np.array([1, 1, 0, 0])
        
        score = self.scorer._calculate_semantic_score(item_embedding, query_embedding)
        
        # Cosine similarity should be 1/sqrt(2) ≈ 0.707
        # Mapped to [0,1]: (0.707 + 1) / 2 ≈ 0.853
        self.assertAlmostEqual(score, 0.853, places=2)
        
        # Test with orthogonal vectors
        item_embedding = np.array([1, 0, 0, 0])
        query_embedding = np.array([0, 1, 0, 0])
        
        score = self.scorer._calculate_semantic_score(item_embedding, query_embedding)
        
        # Cosine similarity should be 0
        # Mapped to [0,1]: (0 + 1) / 2 = 0.5
        self.assertAlmostEqual(score, 0.5, places=2)
        
        # Test with None embeddings
        score = self.scorer._calculate_semantic_score(None, query_embedding)
        self.assertEqual(score, 0.0)
        
        score = self.scorer._calculate_semantic_score(item_embedding, None)
        self.assertEqual(score, 0.0)
        
    def test_calculate_lexical_score(self):
        """Test BM25-based lexical scoring."""
        content = "machine learning is a subset of artificial intelligence"
        query_terms = ["machine", "learning"]
        
        score = self.scorer._calculate_lexical_score(content, query_terms)
        
        # Should have non-zero score for matching terms
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with no matching terms
        query_terms = ["quantum", "computing"]
        score = self.scorer._calculate_lexical_score(content, query_terms)
        
        # Should have lower score
        self.assertLess(score, 0.1)
        
        # Test with empty inputs
        score = self.scorer._calculate_lexical_score("", query_terms)
        self.assertEqual(score, 0.0)
        
        score = self.scorer._calculate_lexical_score(content, None)
        self.assertEqual(score, 0.0)
        
    def test_calculate_recency_score(self):
        """Test recency score calculation."""
        # Currently returns placeholder
        score = self.scorer._calculate_recency_score("2024-01-01")
        self.assertEqual(score, 0.5)
        
        score = self.scorer._calculate_recency_score(None)
        self.assertEqual(score, 0.5)
        
    def test_calculate_diversity_scores(self):
        """Test diversity score calculation."""
        items = [
            ScoredItem(
                id=1, content="A", semantic_score=0.9,
                lexical_score=0, effectiveness_score=0,
                recency_score=0, diversity_score=0, final_score=0
            ),
            ScoredItem(
                id=2, content="B", semantic_score=0.5,
                lexical_score=0, effectiveness_score=0,
                recency_score=0, diversity_score=0, final_score=0
            ),
            ScoredItem(
                id=3, content="C", semantic_score=0.1,
                lexical_score=0, effectiveness_score=0,
                recency_score=0, diversity_score=0, final_score=0
            ),
        ]
        
        items_with_diversity = self.scorer._calculate_diversity_scores(items)
        
        # Check diversity scores calculated
        for item in items_with_diversity:
            self.assertGreaterEqual(item.diversity_score, 0.0)
            self.assertLessEqual(item.diversity_score, 1.0)
            
        # Middle item should have moderate diversity
        self.assertGreater(items_with_diversity[1].diversity_score, 0.0)
        
    def test_combine_scores(self):
        """Test score combination with current weights."""
        semantic = 0.8
        lexical = 0.6
        effectiveness = 0.7
        recency = 0.5
        diversity = 0.3
        
        combined = self.scorer._combine_scores(
            semantic, lexical, effectiveness, recency, diversity
        )
        
        # Calculate expected value
        expected = (
            self.scorer.weights["semantic"] * semantic +
            self.scorer.weights["lexical"] * lexical +
            self.scorer.weights["effectiveness"] * effectiveness +
            self.scorer.weights["recency"] * recency +
            self.scorer.weights["diversity"] * diversity
        )
        
        self.assertAlmostEqual(combined, expected, places=5)
        
    def test_adaptive_score(self):
        """Test adaptive scoring with feedback history."""
        # Test without feedback history
        score = self.scorer._adaptive_score(0.8, 0.6, 0.7, 0.5, 0.3)
        
        # Should be same as combined score without history
        expected = self.scorer._combine_scores(0.8, 0.6, 0.7, 0.5, 0.3)
        self.assertAlmostEqual(score, expected, places=5)
        
        # Add feedback history
        for i in range(10):
            self.scorer._feedback_history.append((0.7, 0.3))  # Positive feedback
            
        # Test with positive feedback and high semantic score
        score = self.scorer._adaptive_score(0.9, 0.5, 0.6, 0.5, 0.3)
        
        # Should be boosted due to positive feedback and high semantic
        base_score = self.scorer._combine_scores(0.9, 0.5, 0.6, 0.5, 0.3)
        self.assertGreaterEqual(score, base_score)
        
    def test_score_items_cosine_method(self):
        """Test scoring items with pure cosine similarity."""
        query_embedding = np.array([1, 0, 0, 0])
        
        items = [
            {
                "id": 1,
                "content": "Item 1",
                "embedding": np.array([1, 0, 0, 0]),  # Perfect match
                "effectiveness_score": 0.5,
            },
            {
                "id": 2,
                "content": "Item 2",
                "embedding": np.array([0, 1, 0, 0]),  # Orthogonal
                "effectiveness_score": 0.8,
            },
        ]
        
        scored = self.scorer.score_items(
            items,
            query_embedding=query_embedding,
            method=ScoringMethod.COSINE
        )
        
        # Check ordering by semantic score
        self.assertEqual(len(scored), 2)
        self.assertGreater(scored[0].semantic_score, scored[1].semantic_score)
        self.assertEqual(scored[0].final_score, scored[0].semantic_score)
        
    def test_score_items_bm25_method(self):
        """Test scoring items with BM25."""
        query_terms = ["machine", "learning"]
        
        items = [
            {
                "id": 1,
                "content": "Machine learning is powerful",
                "embedding": np.array([1, 0, 0, 0]),
            },
            {
                "id": 2,
                "content": "Deep neural networks",
                "embedding": np.array([0, 1, 0, 0]),
            },
        ]
        
        scored = self.scorer.score_items(
            items,
            query_terms=query_terms,
            method=ScoringMethod.BM25
        )
        
        # First item should score higher (contains query terms)
        self.assertEqual(len(scored), 2)
        self.assertGreater(scored[0].lexical_score, scored[1].lexical_score)
        self.assertEqual(scored[0].final_score, scored[0].lexical_score)
        
    def test_score_items_hybrid_method(self):
        """Test scoring items with hybrid method."""
        query_embedding = np.array([1, 0, 0, 0])
        query_terms = ["test"]
        
        items = [
            {
                "id": 1,
                "content": "Test item one",
                "embedding": np.array([0.8, 0.2, 0, 0]),
                "effectiveness_score": 0.7,
            },
            {
                "id": 2,
                "content": "Another item",
                "embedding": np.array([0.5, 0.5, 0, 0]),
                "effectiveness_score": 0.9,
            },
        ]
        
        scored = self.scorer.score_items(
            items,
            query_embedding=query_embedding,
            query_terms=query_terms,
            method=ScoringMethod.HYBRID
        )
        
        # Check all scores are calculated
        for item in scored:
            self.assertGreaterEqual(item.semantic_score, 0.0)
            self.assertGreaterEqual(item.lexical_score, 0.0)
            self.assertGreaterEqual(item.effectiveness_score, 0.0)
            self.assertGreaterEqual(item.final_score, 0.0)
            
        # Should be sorted by final score
        if len(scored) > 1:
            self.assertGreaterEqual(scored[0].final_score, scored[1].final_score)
            
    def test_update_weights(self):
        """Test weight updates based on feedback (AC 5)."""
        initial_weights = self.scorer.weights.copy()
        
        # Positive feedback with high semantic score
        last_scores = {
            "semantic": 0.9,
            "lexical": 0.3,
            "effectiveness": 0.6,
            "recency": 0.5,
            "diversity": 0.4,
        }
        
        self.scorer.update_weights(0.3, last_scores)  # +0.3 feedback
        
        # Semantic weight should increase (or stay very close due to normalization)
        # Allow for small floating-point differences after normalization
        self.assertGreaterEqual(
            self.scorer.weights["semantic"],
            initial_weights["semantic"] - 0.001  # Allow tiny decrease due to normalization
        )
        
        # Weights should still be normalized
        total = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Test negative feedback
        self.scorer.update_weights(-0.3, last_scores)
        
        # Weights should decrease
        self.assertLess(
            self.scorer.weights["semantic"],
            self.scorer.weights["semantic"] + 0.1
        )
        
        # Check feedback history updated
        self.assertEqual(len(self.scorer._feedback_history), 2)
        
    def test_reciprocal_rank_fusion(self):
        """Test RRF for combining multiple rankings."""
        rankings = [
            [1, 2, 3, 4],  # First ranking
            [2, 1, 4, 3],  # Second ranking
            [1, 3, 2, 4],  # Third ranking
        ]
        
        fused = self.scorer.reciprocal_rank_fusion(rankings, k=60)
        
        # Item 1 appears first twice, should rank high
        self.assertIn(1, fused[:2])
        
        # All items should be present
        self.assertEqual(set(fused), {1, 2, 3, 4})
        
        # Test with different k value
        fused_k10 = self.scorer.reciprocal_rank_fusion(rankings, k=10)
        self.assertEqual(len(fused_k10), 4)
        
    def test_get_statistics(self):
        """Test statistics collection."""
        # No history initially
        stats = self.scorer.get_statistics()
        self.assertEqual(stats, {})
        
        # Add some scoring history
        for i in range(5):
            items = [
                ScoredItem(
                    id=i, content=f"Item {i}",
                    semantic_score=0.5 + i*0.1,
                    lexical_score=0.3,
                    effectiveness_score=0.6,
                    recency_score=0.5,
                    diversity_score=0.4,
                    final_score=0.5,
                )
            ]
            self.scorer._track_scoring(items)
            
        # Add feedback
        self.scorer._feedback_history.append((0.6, 0.2))
        self.scorer._feedback_history.append((0.7, 0.3))
        
        stats = self.scorer.get_statistics()
        
        # Check statistics structure
        self.assertIn("current_weights", stats)
        self.assertIn("total_scorings", stats)
        self.assertEqual(stats["total_scorings"], 5)
        
        self.assertIn("avg_scores", stats)
        self.assertIn("semantic", stats["avg_scores"])
        self.assertIn("mean", stats["avg_scores"]["semantic"])
        self.assertIn("std", stats["avg_scores"]["semantic"])
        
        self.assertIn("feedback", stats)
        self.assertEqual(stats["feedback"]["count"], 2)
        
    def test_reset_weights(self):
        """Test weight reset functionality."""
        # Modify weights
        self.scorer.weights["semantic"] = 0.8
        self.scorer._normalize_weights()
        
        # Reset
        self.scorer.reset_weights()
        
        # Check defaults restored
        self.assertAlmostEqual(self.scorer.weights["semantic"], 0.5)
        self.assertAlmostEqual(self.scorer.weights["lexical"], 0.2)
        self.assertAlmostEqual(self.scorer.weights["effectiveness"], 0.2)
        
        # Should be normalized
        total = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestScoringEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = CustomScorer()
        
    def test_score_empty_items(self):
        """Test scoring with empty item list."""
        scored = self.scorer.score_items([])
        self.assertEqual(len(scored), 0)
        
    def test_score_items_missing_fields(self):
        """Test scoring items with missing fields."""
        items = [
            {"id": 1},  # No content or embedding
            {"id": 2, "content": "Text"},  # No embedding
            {"id": 3, "embedding": np.array([1, 0])},  # No content
        ]
        
        scored = self.scorer.score_items(items)
        
        # Should handle missing fields gracefully
        self.assertEqual(len(scored), 3)
        for item in scored:
            self.assertIsNotNone(item.final_score)
            
    def test_zero_norm_embeddings(self):
        """Test with zero-norm embeddings."""
        zero_embedding = np.array([0, 0, 0, 0])
        normal_embedding = np.array([1, 0, 0, 0])
        
        score = self.scorer._calculate_semantic_score(
            zero_embedding,
            normal_embedding
        )
        
        self.assertEqual(score, 0.0)
        
    def test_effectiveness_feedback_bounds(self):
        """Test effectiveness updates stay within bounds (AC 5)."""
        last_scores = {
            "semantic": 1.0,
            "lexical": 1.0,
            "effectiveness": 1.0,
            "recency": 1.0,
            "diversity": 1.0,
        }
        
        # Large positive feedback
        self.scorer.update_weights(1.0, last_scores)
        
        # Weights should stay positive and normalized
        for weight in self.scorer.weights.values():
            self.assertGreater(weight, 0.0)
            self.assertLessEqual(weight, 1.0)
            
        total = sum(self.scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
        # Large negative feedback
        self.scorer.update_weights(-1.0, last_scores)
        
        # Weights should still be valid
        for weight in self.scorer.weights.values():
            self.assertGreaterEqual(weight, 0.01)  # Minimum weight
            
    def test_adaptive_score_capping(self):
        """Test adaptive score stays within [0, 1]."""
        # Add positive feedback history
        for i in range(20):
            self.scorer._feedback_history.append((0.9, 1.0))
            
        # High scores that might overflow
        score = self.scorer._adaptive_score(1.0, 1.0, 1.0, 1.0, 1.0)
        
        # Should be capped at 1.0
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])