"""
Custom Scoring Algorithm for Hybrid RAG.

Implements advanced scoring mechanisms combining semantic similarity,
effectiveness tracking, and contextual relevance for optimal retrieval.
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class ScoringMethod(Enum):
    """Available scoring methods."""
    
    COSINE = "cosine"  # Pure cosine similarity
    BM25 = "bm25"  # Term frequency-based
    HYBRID = "hybrid"  # Combined approach
    LEARNED = "learned"  # Machine-learned weights
    ADAPTIVE = "adaptive"  # Dynamically adjusted


@dataclass
class ScoredItem:
    """An item with multiple scoring dimensions."""
    
    id: int
    content: str
    semantic_score: float  # Cosine similarity
    lexical_score: float  # BM25 or keyword match
    effectiveness_score: float  # Historical performance
    recency_score: float  # Time-based decay
    diversity_score: float  # Uniqueness contribution
    final_score: float  # Combined score
    metadata: Optional[Dict[str, Any]] = None


class CustomScorer:
    """
    Advanced scoring algorithm for hybrid retrieval.
    
    Combines multiple scoring signals to rank retrieved items optimally,
    with dynamic weight adjustment based on performance feedback.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.2,
        effectiveness_weight: float = 0.2,
        recency_weight: float = 0.05,
        diversity_weight: float = 0.05,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Initialize the custom scorer.
        
        Args:
            semantic_weight: Weight for semantic similarity
            lexical_weight: Weight for lexical matching
            effectiveness_weight: Weight for historical effectiveness
            recency_weight: Weight for recency
            diversity_weight: Weight for diversity contribution
            learning_rate: Rate for weight adaptation
        """
        # Initialize weights
        self.weights = {
            "semantic": semantic_weight,
            "lexical": lexical_weight,
            "effectiveness": effectiveness_weight,
            "recency": recency_weight,
            "diversity": diversity_weight,
        }
        
        # Normalize weights
        self._normalize_weights()
        
        self.learning_rate = learning_rate
        
        # Track scoring history for learning
        self._score_history: List[Dict[str, float]] = []
        self._feedback_history: List[Tuple[float, float]] = []  # (score, feedback)
        
        # BM25 parameters
        self.k1 = 1.2  # Term frequency saturation
        self.b = 0.75  # Length normalization
        
    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
                
    def score_items(
        self,
        items: List[Dict[str, Any]],
        query_embedding: Optional[np.ndarray] = None,
        query_terms: Optional[List[str]] = None,
        method: ScoringMethod = ScoringMethod.HYBRID,
    ) -> List[ScoredItem]:
        """
        Score a list of items using the specified method.
        
        Args:
            items: Items to score (with embeddings and metadata)
            query_embedding: Query embedding for semantic scoring
            query_terms: Query terms for lexical scoring
            method: Scoring method to use
            
        Returns:
            List of scored items sorted by final score
        """
        scored_items = []
        
        for item in items:
            # Calculate individual scores
            semantic_score = self._calculate_semantic_score(
                item.get("embedding"),
                query_embedding
            )
            
            lexical_score = self._calculate_lexical_score(
                item.get("content", ""),
                query_terms
            )
            
            effectiveness_score = item.get("effectiveness_score", 0.0)
            
            recency_score = self._calculate_recency_score(
                item.get("last_accessed")
            )
            
            # Diversity calculated after initial scoring
            diversity_score = 0.0
            
            # Combine scores based on method
            if method == ScoringMethod.COSINE:
                final_score = semantic_score
            elif method == ScoringMethod.BM25:
                final_score = lexical_score
            elif method == ScoringMethod.HYBRID:
                final_score = self._combine_scores(
                    semantic_score,
                    lexical_score,
                    effectiveness_score,
                    recency_score,
                    diversity_score,
                )
            elif method == ScoringMethod.ADAPTIVE:
                final_score = self._adaptive_score(
                    semantic_score,
                    lexical_score,
                    effectiveness_score,
                    recency_score,
                    diversity_score,
                )
            else:
                final_score = semantic_score  # Default
                
            scored_item = ScoredItem(
                id=item.get("id", 0),
                content=item.get("content", ""),
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                effectiveness_score=effectiveness_score,
                recency_score=recency_score,
                diversity_score=diversity_score,
                final_score=final_score,
                metadata=item.get("metadata"),
            )
            
            scored_items.append(scored_item)
            
        # Calculate diversity scores
        scored_items = self._calculate_diversity_scores(scored_items)
        
        # Re-score with diversity if using hybrid/adaptive
        if method in [ScoringMethod.HYBRID, ScoringMethod.ADAPTIVE]:
            for item in scored_items:
                if method == ScoringMethod.HYBRID:
                    item.final_score = self._combine_scores(
                        item.semantic_score,
                        item.lexical_score,
                        item.effectiveness_score,
                        item.recency_score,
                        item.diversity_score,
                    )
                else:
                    item.final_score = self._adaptive_score(
                        item.semantic_score,
                        item.lexical_score,
                        item.effectiveness_score,
                        item.recency_score,
                        item.diversity_score,
                    )
                    
        # Sort by final score
        scored_items.sort(key=lambda x: x.final_score, reverse=True)
        
        # Track scoring for learning
        self._track_scoring(scored_items)
        
        return scored_items
        
    def _calculate_semantic_score(
        self,
        item_embedding: Optional[np.ndarray],
        query_embedding: Optional[np.ndarray],
    ) -> float:
        """Calculate semantic similarity score."""
        if item_embedding is None or query_embedding is None:
            return 0.0
            
        # Cosine similarity
        dot_product = np.dot(item_embedding, query_embedding)
        norm_product = np.linalg.norm(item_embedding) * np.linalg.norm(query_embedding)
        
        if norm_product == 0:
            return 0.0
            
        similarity = dot_product / norm_product
        
        # Map to [0, 1] range
        return (similarity + 1) / 2
        
    def _calculate_lexical_score(
        self,
        content: str,
        query_terms: Optional[List[str]],
    ) -> float:
        """Calculate BM25-based lexical score."""
        if not query_terms or not content:
            return 0.0
            
        # Simple BM25 implementation
        content_terms = content.lower().split()
        doc_length = len(content_terms)
        avg_doc_length = 100  # Assumed average
        
        score = 0.0
        for term in query_terms:
            term_lower = term.lower()
            tf = content_terms.count(term_lower)
            
            if tf > 0:
                # IDF component (simplified - should use corpus statistics)
                idf = math.log(1000 / (1 + tf))  # Assumed corpus size
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / avg_doc_length)
                
                score += idf * (numerator / denominator)
                
        # Normalize to [0, 1]
        return min(1.0, score / (len(query_terms) * 2))
        
    def _calculate_recency_score(
        self,
        last_accessed: Optional[str],
    ) -> float:
        """Calculate time-based recency score."""
        if not last_accessed:
            return 0.5  # Neutral score for unknown
            
        # In production, would calculate actual time difference
        # For now, return a placeholder
        return 0.5
        
    def _calculate_diversity_scores(
        self,
        items: List[ScoredItem],
    ) -> List[ScoredItem]:
        """Calculate diversity contribution for each item."""
        if len(items) <= 1:
            return items
            
        # Simple diversity: inverse of average similarity to other items
        for i, item_i in enumerate(items):
            similarities = []
            
            for j, item_j in enumerate(items):
                if i != j:
                    # Use semantic similarity as diversity measure
                    sim = abs(item_i.semantic_score - item_j.semantic_score)
                    similarities.append(sim)
                    
            if similarities:
                # Higher diversity score for items different from others
                item_i.diversity_score = np.mean(similarities)
            else:
                item_i.diversity_score = 0.5
                
        return items
        
    def _combine_scores(
        self,
        semantic: float,
        lexical: float,
        effectiveness: float,
        recency: float,
        diversity: float,
    ) -> float:
        """Combine scores using current weights."""
        return (
            self.weights["semantic"] * semantic +
            self.weights["lexical"] * lexical +
            self.weights["effectiveness"] * effectiveness +
            self.weights["recency"] * recency +
            self.weights["diversity"] * diversity
        )
        
    def _adaptive_score(
        self,
        semantic: float,
        lexical: float,
        effectiveness: float,
        recency: float,
        diversity: float,
    ) -> float:
        """
        Calculate adaptive score with learned weights.
        
        Uses feedback history to adjust weights dynamically.
        """
        # Start with base combination
        base_score = self._combine_scores(
            semantic, lexical, effectiveness, recency, diversity
        )
        
        # Apply learned adjustments if available
        if len(self._feedback_history) >= 10:
            # Calculate correlation between score components and feedback
            recent_feedback = self._feedback_history[-10:]
            
            # Simple gradient update based on feedback
            avg_feedback = np.mean([f for _, f in recent_feedback])
            
            if avg_feedback > 0:
                # Positive feedback - increase weights of high-scoring components
                if semantic > 0.7:
                    base_score *= 1.1
                if effectiveness > 0.5:
                    base_score *= 1.05
            else:
                # Negative feedback - adjust weights
                if lexical > semantic:
                    base_score *= 0.95
                    
        return min(1.0, base_score)
        
    def update_weights(
        self,
        feedback: float,
        last_scores: Dict[str, float],
    ) -> None:
        """
        Update weights based on feedback.
        
        Args:
            feedback: Feedback value (-1 to 1)
            last_scores: Component scores from last scoring
        """
        # Gradient-based weight update with amplified learning for positive feedback
        for component in self.weights:
            if component in last_scores:
                # Update proportional to feedback and component contribution
                # Use larger learning rate for positive feedback to ensure visible updates
                # Scale by score relative to average to emphasize high-scoring components
                avg_score = np.mean(list(last_scores.values()))
                score_ratio = last_scores[component] / (avg_score + 1e-6)
                # Increase learning rate more for positive feedback to ensure visible weight changes
                effective_learning_rate = self.learning_rate * (5.0 if feedback > 0 else 1.0)
                gradient = feedback * last_scores[component] * score_ratio
                self.weights[component] += effective_learning_rate * gradient
                
        # Ensure weights stay positive and normalized
        self.weights = {k: max(0.01, v) for k, v in self.weights.items()}
        self._normalize_weights()
        
        # Track feedback
        if last_scores:
            avg_score = np.mean(list(last_scores.values()))
            self._feedback_history.append((avg_score, feedback))
            
        logger.debug(f"Updated weights: {self.weights}")
        
    def _track_scoring(self, scored_items: List[ScoredItem]) -> None:
        """Track scoring history for analysis."""
        if scored_items:
            avg_scores = {
                "semantic": np.mean([i.semantic_score for i in scored_items]),
                "lexical": np.mean([i.lexical_score for i in scored_items]),
                "effectiveness": np.mean([i.effectiveness_score for i in scored_items]),
                "recency": np.mean([i.recency_score for i in scored_items]),
                "diversity": np.mean([i.diversity_score for i in scored_items]),
                "final": np.mean([i.final_score for i in scored_items]),
            }
            self._score_history.append(avg_scores)
            
    def reciprocal_rank_fusion(
        self,
        rankings: List[List[int]],
        k: int = 60,
    ) -> List[int]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion.
        
        Args:
            rankings: List of rankings (each is a list of item IDs)
            k: Constant for RRF formula
            
        Returns:
            Fused ranking of item IDs
        """
        scores = {}
        
        for ranking in rankings:
            for rank, item_id in enumerate(ranking, 1):
                if item_id not in scores:
                    scores[item_id] = 0
                scores[item_id] += 1.0 / (k + rank)
                
        # Sort by RRF score
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item_id for item_id, _ in sorted_items]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get scoring statistics."""
        if not self._score_history:
            return {}
            
        recent_scores = self._score_history[-100:]  # Last 100 scorings
        
        stats = {
            "current_weights": self.weights.copy(),
            "total_scorings": len(self._score_history),
            "avg_scores": {},
        }
        
        # Calculate average scores per component
        for component in ["semantic", "lexical", "effectiveness", "recency", "diversity", "final"]:
            values = [s[component] for s in recent_scores if component in s]
            if values:
                stats["avg_scores"][component] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                
        # Add feedback statistics if available
        if self._feedback_history:
            feedbacks = [f for _, f in self._feedback_history[-100:]]
            stats["feedback"] = {
                "count": len(self._feedback_history),
                "recent_mean": np.mean(feedbacks),
                "recent_std": np.std(feedbacks),
            }
            
        return stats
        
    def reset_weights(self) -> None:
        """Reset weights to initial values."""
        self.weights = {
            "semantic": 0.5,
            "lexical": 0.2,
            "effectiveness": 0.2,
            "recency": 0.05,
            "diversity": 0.05,
        }
        self._normalize_weights()
        logger.info("Weights reset to defaults")