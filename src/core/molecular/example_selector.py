"""
Dynamic Example Selection Module.

Implements intelligent example selection based on semantic similarity
and effectiveness tracking for few-shot learning optimization.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .context_builder import MoleculeContextBuilder
from .vector_store import VectorStore, VectorSearchResult

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Example selection strategies."""
    
    SIMILARITY = "similarity"  # Pure semantic similarity
    EFFECTIVENESS = "effectiveness"  # Weighted by effectiveness scores
    HYBRID = "hybrid"  # Combination of similarity and effectiveness
    DIVERSE = "diverse"  # Maximize diversity while maintaining relevance


@dataclass
class Example:
    """An example for few-shot learning."""
    
    id: int
    input: str
    output: str
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    effectiveness_score: float = 0.0
    usage_count: int = 0
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for context building."""
        return {
            "input": self.input,
            "output": self.output,
        }


@dataclass
class SelectionResult:
    """Result from example selection process."""
    
    examples: List[Example]
    total_candidates: int
    selection_time_ms: float
    average_similarity: float
    average_effectiveness: float
    strategy_used: SelectionStrategy


class ExampleSelector:
    """
    Selects the most relevant examples for few-shot learning.
    
    Uses semantic similarity and effectiveness tracking to dynamically
    select examples that maximize context quality.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        context_builder: MoleculeContextBuilder,
        default_strategy: SelectionStrategy = SelectionStrategy.HYBRID,
        similarity_weight: float = 0.7,
        effectiveness_weight: float = 0.3,
    ) -> None:
        """
        Initialize the example selector.
        
        Args:
            vector_store: Vector storage backend
            context_builder: Context builder for similarity calculation
            default_strategy: Default selection strategy
            similarity_weight: Weight for similarity in hybrid strategy
            effectiveness_weight: Weight for effectiveness in hybrid strategy
        """
        self.vector_store = vector_store
        self.context_builder = context_builder
        self.default_strategy = default_strategy
        self.similarity_weight = similarity_weight
        self.effectiveness_weight = effectiveness_weight
        
        # Ensure weights sum to 1
        total_weight = similarity_weight + effectiveness_weight
        self.similarity_weight /= total_weight
        self.effectiveness_weight /= total_weight
        
        # Cache for frequently used examples
        self._example_cache: Dict[int, Example] = {}
        
    async def select_examples(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        strategy: Optional[SelectionStrategy] = None,
        min_similarity: float = 0.85,
    ) -> SelectionResult:
        """
        Select k most relevant examples for the query.
        
        Args:
            query_embedding: Query embedding vector (4096D)
            k: Number of examples to select
            strategy: Selection strategy (uses default if None)
            min_similarity: Minimum similarity threshold
            
        Returns:
            Selection result with examples and metrics
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        # Search for similar vectors (get more than k for filtering)
        search_results = await self.vector_store.search_similar(
            query_embedding,
            k=k * 3,  # Get extra for filtering and diversity
            include_embeddings=True,
        )
        
        if not search_results:
            logger.warning("No examples found in vector store")
            return SelectionResult(
                examples=[],
                total_candidates=0,
                selection_time_ms=0,
                average_similarity=0,
                average_effectiveness=0,
                strategy_used=strategy,
            )
            
        # Convert to Example objects
        examples = await self._convert_to_examples(search_results)
        
        # Apply selection strategy
        if strategy == SelectionStrategy.SIMILARITY:
            selected = self._select_by_similarity(examples, k, min_similarity)
        elif strategy == SelectionStrategy.EFFECTIVENESS:
            selected = self._select_by_effectiveness(examples, k, min_similarity)
        elif strategy == SelectionStrategy.HYBRID:
            selected = self._select_hybrid(examples, k, min_similarity)
        elif strategy == SelectionStrategy.DIVERSE:
            selected = self._select_diverse(examples, k, min_similarity)
        else:
            selected = examples[:k]
            
        # Calculate metrics
        selection_time_ms = (time.time() - start_time) * 1000
        avg_similarity = np.mean([ex.similarity_score for ex in selected]) if selected else 0
        avg_effectiveness = np.mean([ex.effectiveness_score for ex in selected]) if selected else 0
        
        logger.info(
            f"Selected {len(selected)} examples using {strategy.value} strategy "
            f"(avg similarity: {avg_similarity:.3f}, avg effectiveness: {avg_effectiveness:.3f})"
        )
        
        return SelectionResult(
            examples=selected,
            total_candidates=len(search_results),
            selection_time_ms=selection_time_ms,
            average_similarity=avg_similarity,
            average_effectiveness=avg_effectiveness,
            strategy_used=strategy,
        )
        
    async def _convert_to_examples(
        self,
        search_results: List[VectorSearchResult],
    ) -> List[Example]:
        """Convert search results to Example objects."""
        examples = []
        
        for result in search_results:
            # Check cache first
            if result.id in self._example_cache:
                example = self._example_cache[result.id]
                example.similarity_score = result.similarity
            else:
                # Get full vector data
                vector_data = await self.vector_store.get_vector(
                    result.id,
                    include_embedding=True,
                )
                
                if not vector_data:
                    continue
                    
                # Parse content (expected format: "Input: ... Output: ...")
                content = vector_data.get("content", "")
                input_text, output_text = self._parse_example_content(content)
                
                example = Example(
                    id=result.id,
                    input=input_text,
                    output=output_text,
                    embedding=vector_data.get("embedding"),
                    similarity_score=result.similarity,
                    effectiveness_score=vector_data.get("effectiveness_score", 0.0),
                    usage_count=vector_data.get("usage_count", 0),
                    metadata=result.metadata,
                )
                
                # Cache the example
                self._example_cache[result.id] = example
                
            examples.append(example)
            
        return examples
        
    def _parse_example_content(self, content: str) -> Tuple[str, str]:
        """Parse example content into input and output."""
        # Simple parsing - can be enhanced based on actual format
        if "Output:" in content:
            parts = content.split("Output:", 1)
            input_part = parts[0].replace("Input:", "").strip()
            output_part = parts[1].strip() if len(parts) > 1 else ""
            return input_part, output_part
        else:
            # Treat entire content as input if no output marker
            return content, ""
            
    def _select_by_similarity(
        self,
        examples: List[Example],
        k: int,
        min_similarity: float,
    ) -> List[Example]:
        """Select examples purely by similarity score."""
        # Filter by minimum similarity
        filtered = [ex for ex in examples if ex.similarity_score >= min_similarity]
        
        # Sort by similarity (already sorted from vector store, but ensure)
        filtered.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return filtered[:k]
        
    def _select_by_effectiveness(
        self,
        examples: List[Example],
        k: int,
        min_similarity: float,
    ) -> List[Example]:
        """Select examples by effectiveness score."""
        # Filter by minimum similarity
        filtered = [ex for ex in examples if ex.similarity_score >= min_similarity]
        
        # Sort by effectiveness
        filtered.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return filtered[:k]
        
    def _select_hybrid(
        self,
        examples: List[Example],
        k: int,
        min_similarity: float,
    ) -> List[Example]:
        """Select examples using hybrid scoring."""
        # Filter by minimum similarity
        filtered = [ex for ex in examples if ex.similarity_score >= min_similarity]
        
        # Calculate hybrid scores
        for ex in filtered:
            ex.hybrid_score = (
                self.similarity_weight * ex.similarity_score +
                self.effectiveness_weight * ex.effectiveness_score
            )
            
        # Sort by hybrid score
        filtered.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return filtered[:k]
        
    def _select_diverse(
        self,
        examples: List[Example],
        k: int,
        min_similarity: float,
    ) -> List[Example]:
        """
        Select diverse examples while maintaining relevance.
        
        Uses Maximum Marginal Relevance (MMR) algorithm to balance
        relevance and diversity.
        """
        # Filter by minimum similarity
        candidates = [ex for ex in examples if ex.similarity_score >= min_similarity]
        
        if not candidates:
            return []
            
        selected = []
        remaining = candidates.copy()
        
        # Select first example (highest similarity)
        first = remaining.pop(0)
        selected.append(first)
        
        # Select remaining examples using MMR
        lambda_param = 0.7  # Balance between relevance and diversity
        
        while len(selected) < k and remaining:
            best_score = -1
            best_idx = -1
            
            for i, candidate in enumerate(remaining):
                # Relevance score (similarity to query)
                relevance = candidate.similarity_score
                
                # Diversity score (min similarity to selected examples)
                if selected and candidate.embedding is not None:
                    similarities_to_selected = [
                        self.context_builder.calculate_similarity(
                            candidate.embedding,
                            sel.embedding
                        )
                        for sel in selected
                        if sel.embedding is not None
                    ]
                    diversity = 1.0 - max(similarities_to_selected) if similarities_to_selected else 1.0
                else:
                    diversity = 1.0
                    
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
                    
            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))
                
        return selected
        
    async def update_effectiveness(
        self,
        example_id: int,
        feedback: float,
    ) -> None:
        """
        Update effectiveness score based on feedback.
        
        Args:
            example_id: ID of the example
            feedback: Feedback value (+0.3 for positive, -0.3 for negative)
        """
        await self.vector_store.update_effectiveness(example_id, feedback)
        
        # Update cache if present
        if example_id in self._example_cache:
            self._example_cache[example_id].effectiveness_score += feedback
            self._example_cache[example_id].usage_count += 1
            
        logger.debug(f"Updated effectiveness for example {example_id}: {feedback:+.2f}")
        
    async def add_example(
        self,
        input_text: str,
        output_text: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a new example to the pool.
        
        Args:
            input_text: Input text for the example
            output_text: Output text for the example
            embedding: Embedding vector (4096D)
            metadata: Optional metadata
            
        Returns:
            ID of the added example
        """
        # Format content
        content = f"Input: {input_text}\nOutput: {output_text}"
        
        # Store in vector store
        example_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content=content,
            metadata=metadata,
        )
        
        logger.info(f"Added new example {example_id}")
        return example_id
        
    async def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about the example pool."""
        stats = await self.vector_store.get_statistics()
        
        # Add cache statistics
        stats["cache_size"] = len(self._example_cache)
        stats["cache_hit_rate"] = self._calculate_cache_hit_rate()
        
        return stats
        
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (placeholder for actual implementation)."""
        # This would track actual hits/misses in production
        return 0.0
        
    def clear_cache(self) -> None:
        """Clear the example cache."""
        self._example_cache.clear()
        logger.info("Example cache cleared")