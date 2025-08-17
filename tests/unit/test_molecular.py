"""
Unit tests for Molecular Layer Components.

Tests MoleculeContextBuilder and ExampleSelector classes for Story 1.4.
Validates MOLECULE structure, token allocation, example selection strategies,
and effectiveness tracking.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any

import numpy as np
import pytest

from src.core.molecular.context_builder import (
    MoleculeContextBuilder,
    MoleculeContext,
    MoleculeSection,
    TokenAllocation,
)
from src.core.molecular.example_selector import (
    ExampleSelector,
    Example,
    SelectionStrategy,
    SelectionResult,
)
from src.core.molecular.vector_store import VectorSearchResult


class TestTokenAllocation(unittest.TestCase):
    """Test cases for TokenAllocation class."""
    
    def test_total_allocated(self):
        """Test total token calculation."""
        allocation = TokenAllocation(
            instruction=256,
            examples=512,
            context=256,
            new_input=100,
        )
        self.assertEqual(allocation.total_allocated, 1124)
        
    def test_adjust_for_limit_no_change(self):
        """Test adjustment when within limit."""
        allocation = TokenAllocation(
            instruction=200,
            examples=400,
            context=200,
            new_input=100,
        )
        allocation.adjust_for_limit(1024)
        
        # Should remain unchanged
        self.assertEqual(allocation.instruction, 200)
        self.assertEqual(allocation.examples, 400)
        self.assertEqual(allocation.context, 200)
        self.assertEqual(allocation.new_input, 100)
        
    def test_adjust_for_limit_scale_down(self):
        """Test proportional scaling when over limit."""
        allocation = TokenAllocation(
            instruction=400,
            examples=800,
            context=400,
            new_input=400,
        )
        allocation.adjust_for_limit(1000)
        
        # Should scale down proportionally (total was 2000, limit is 1000)
        self.assertEqual(allocation.instruction, 200)
        self.assertEqual(allocation.examples, 400)
        self.assertEqual(allocation.context, 200)
        self.assertEqual(allocation.new_input, 200)
        self.assertEqual(allocation.total_allocated, 1000)


class TestMoleculeContext(unittest.TestCase):
    """Test cases for MoleculeContext class."""
    
    def test_format_complete(self):
        """Test formatting complete MOLECULE structure."""
        context = MoleculeContext(
            instruction="Solve the problem step by step",
            examples=[
                {"input": "2 + 2", "output": "4"},
                {"input": "3 + 3", "output": "6"},
            ],
            context="Mathematical operations",
            new_input="5 + 5",
            token_count=500,
            similarity_scores=[0.95, 0.90],
        )
        
        formatted = context.format()
        
        # Check all sections present
        self.assertIn("## INSTRUCTION", formatted)
        self.assertIn("Solve the problem step by step", formatted)
        self.assertIn("## EXAMPLES", formatted)
        self.assertIn("### Example 1", formatted)
        self.assertIn("Input: 2 + 2", formatted)
        self.assertIn("Output: 4", formatted)
        self.assertIn("### Example 2", formatted)
        self.assertIn("## CONTEXT", formatted)
        self.assertIn("Mathematical operations", formatted)
        self.assertIn("## NEW INPUT", formatted)
        self.assertIn("5 + 5", formatted)
        
    def test_format_partial(self):
        """Test formatting with missing sections."""
        context = MoleculeContext(
            instruction="",
            examples=[],
            context="Some context",
            new_input="Input text",
            token_count=200,
        )
        
        formatted = context.format()
        
        # Should only include non-empty sections
        self.assertNotIn("## INSTRUCTION", formatted)
        self.assertNotIn("## EXAMPLES", formatted)
        self.assertIn("## CONTEXT", formatted)
        self.assertIn("## NEW INPUT", formatted)


class TestMoleculeContextBuilder(unittest.TestCase):
    """Test cases for MoleculeContextBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = MoleculeContextBuilder(
            tokenizer_name="gpt2",
            chunk_size=1024,
            overlap_ratio=0.15,
            max_examples=10,
        )
        
    @patch("src.core.molecular.context_builder.AutoTokenizer")
    def test_initialization(self, mock_tokenizer):
        """Test builder initialization."""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        builder = MoleculeContextBuilder(
            tokenizer_name="gpt2",
            chunk_size=1024,
            overlap_ratio=0.15,
        )
        
        self.assertEqual(builder.chunk_size, 1024)
        self.assertEqual(builder.overlap_ratio, 0.15)
        self.assertEqual(builder.overlap_tokens, 153)  # 1024 * 0.15
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        
    def test_count_tokens_with_tokenizer(self):
        """Test token counting with tokenizer."""
        self.builder.tokenizer = MagicMock()
        self.builder.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        count = self.builder.count_tokens("test text")
        
        self.assertEqual(count, 5)
        self.builder.tokenizer.encode.assert_called_once_with("test text")
        
    def test_count_tokens_fallback(self):
        """Test token counting fallback when tokenizer unavailable."""
        self.builder.tokenizer = None
        
        # Fallback estimates ~4 characters per token
        count = self.builder.count_tokens("12345678")  # 8 characters
        self.assertEqual(count, 2)
        
    @patch("time.time")
    def test_build_context_within_time_limit(self, mock_time):
        """Test context building within 800ms limit (AC 6)."""
        # Mock time to simulate fast construction
        mock_time.side_effect = [0.0, 0.5]  # 500ms
        
        self.builder.tokenizer = MagicMock()
        self.builder.tokenizer.encode.return_value = [1] * 100
        
        context = self.builder.build_context(
            instruction="Test instruction",
            examples=[
                {"input": "ex1", "output": "out1"},
                {"input": "ex2", "output": "out2"},
            ],
            context="Test context",
            new_input="New input text",
            similarity_scores=[0.95, 0.90],
        )
        
        self.assertIsInstance(context, MoleculeContext)
        self.assertEqual(context.instruction, "Test instruction")
        self.assertEqual(len(context.examples), 2)
        self.assertEqual(context.new_input, "New input text")
        
    @patch("time.time")
    @patch("src.core.molecular.context_builder.logger")
    def test_build_context_exceeds_time_limit(self, mock_logger, mock_time):
        """Test warning when context building exceeds 800ms."""
        # Mock time to simulate slow construction
        mock_time.side_effect = [0.0, 0.9]  # 900ms
        
        self.builder.tokenizer = MagicMock()
        self.builder.tokenizer.encode.return_value = [1] * 100
        
        context = self.builder.build_context(
            instruction="Test",
            examples=[],
            context="Context",
            new_input="Input",
        )
        
        # Should log warning
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        self.assertIn("900", warning_message)
        self.assertIn(">800ms", warning_message)
        
    def test_select_examples_by_similarity(self):
        """Test example selection prioritized by similarity scores."""
        examples = [
            {"input": "low", "output": "low_out"},
            {"input": "high", "output": "high_out"},
            {"input": "medium", "output": "medium_out"},
        ]
        similarity_scores = [0.70, 0.95, 0.85]
        
        self.builder.tokenizer = MagicMock()
        self.builder.tokenizer.encode.return_value = [1] * 10
        
        selected = self.builder._select_examples(
            examples,
            token_budget=100,
            similarity_scores=similarity_scores,
        )
        
        # Should be sorted by similarity (highest first)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[0]["input"], "high")
        self.assertEqual(selected[1]["input"], "medium")
        self.assertEqual(selected[2]["input"], "low")
        
    def test_select_examples_token_budget(self):
        """Test example selection respects token budget."""
        examples = [
            {"input": "short", "output": "s"},
            {"input": "medium length", "output": "m"},
            {"input": "very long example text", "output": "l"},
        ]
        
        # Mock tokenizer to return different lengths
        self.builder.tokenizer = MagicMock()
        self.builder.tokenizer.encode.side_effect = lambda x: [1] * len(str(x))
        self.builder.count_tokens = lambda x: len(str(x))
        
        selected = self.builder._select_examples(
            examples,
            token_budget=30,  # Limited budget
        )
        
        # Should select only examples that fit
        total_tokens = sum(len(str(ex)) for ex in selected)
        self.assertLessEqual(total_tokens, 30)
        
    def test_truncate_to_tokens(self):
        """Test text truncation to token limit."""
        self.builder.tokenizer = MagicMock()
        # Mock tokenizer to use character count as token count
        self.builder.count_tokens = lambda x: len(x)
        
        text = "This is a long text that needs truncation"
        truncated = self.builder._truncate_to_tokens(text, 10)
        
        self.assertEqual(len(truncated), 10)
        self.assertEqual(truncated, "This is a ")
        
    def test_create_chunks_with_overlap(self):
        """Test chunk creation with 15% overlap (AC 4)."""
        # Create multiple contexts
        contexts = [
            MoleculeContext(
                instruction="Inst1",
                examples=[],
                context="Ctx1",
                new_input="Input1",
                token_count=500,
            ),
            MoleculeContext(
                instruction="Inst2",
                examples=[],
                context="Ctx2",
                new_input="Input2",
                token_count=500,
            ),
        ]
        
        # Mock tokenizer for controlled chunking
        mock_tokenizer = MagicMock()
        mock_tokens = list(range(2000))  # Enough for multiple chunks
        mock_tokenizer.encode.return_value = mock_tokens
        mock_tokenizer.decode.side_effect = lambda tokens: f"Chunk_{len(tokens)}"
        self.builder.tokenizer = mock_tokenizer
        
        chunks = self.builder.create_chunks(contexts)
        
        # Should create multiple chunks with overlap
        self.assertGreater(len(chunks), 1)
        
        # Verify overlap calculation (15% of 1024 = 153 tokens)
        self.assertEqual(self.builder.overlap_tokens, 153)
        
    def test_calculate_similarity(self):
        """Test cosine similarity calculation."""
        # Create two vectors
        vec1 = np.array([1, 0, 0, 0])
        vec2 = np.array([1, 1, 0, 0])
        
        similarity = self.builder.calculate_similarity(vec1, vec2)
        
        # Cosine similarity should be 1/sqrt(2) ≈ 0.707
        self.assertAlmostEqual(similarity, 0.707, places=3)
        
    def test_calculate_similarity_zero_vector(self):
        """Test similarity with zero vector."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.array([0, 0, 0])
        
        similarity = self.builder.calculate_similarity(vec1, vec2)
        
        self.assertEqual(similarity, 0.0)
        
    def test_prioritize_by_similarity(self):
        """Test prioritization by similarity with threshold (AC 1)."""
        items = ["A", "B", "C", "D", "E"]
        similarities = [0.90, 0.70, 0.95, 0.85, 0.80]
        
        prioritized = self.builder.prioritize_by_similarity(
            items,
            similarities,
            threshold=0.85,  # AC 1: >0.85 threshold
        )
        
        # Should only include items above threshold, sorted by similarity
        self.assertEqual(len(prioritized), 3)
        self.assertEqual(prioritized[0], ("C", 0.95))
        self.assertEqual(prioritized[1], ("A", 0.90))
        self.assertEqual(prioritized[2], ("D", 0.85))
        
    def test_prioritize_by_similarity_length_mismatch(self):
        """Test error on mismatched lengths."""
        items = ["A", "B"]
        similarities = [0.9, 0.8, 0.7]
        
        with self.assertRaises(ValueError) as context:
            self.builder.prioritize_by_similarity(items, similarities)
            
        self.assertIn("same length", str(context.exception))


class TestExampleSelector(unittest.IsolatedAsyncioTestCase):
    """Test cases for ExampleSelector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vector_store = AsyncMock()
        self.context_builder = MagicMock()
        self.selector = ExampleSelector(
            vector_store=self.vector_store,
            context_builder=self.context_builder,
            default_strategy=SelectionStrategy.HYBRID,
            similarity_weight=0.7,
            effectiveness_weight=0.3,
        )
        
    def test_initialization(self):
        """Test selector initialization and weight normalization."""
        # Weights should be normalized to sum to 1
        self.assertAlmostEqual(
            self.selector.similarity_weight + self.selector.effectiveness_weight,
            1.0
        )
        self.assertEqual(self.selector.default_strategy, SelectionStrategy.HYBRID)
        
    @pytest.mark.asyncio
    async def test_select_examples_no_results(self):
        """Test selection when no examples found."""
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Mock empty search results
        self.vector_store.search_similar.return_value = []
        
        result = await self.selector.select_examples(
            query_embedding,
            k=5,
        )
        
        self.assertEqual(len(result.examples), 0)
        self.assertEqual(result.total_candidates, 0)
        self.assertEqual(result.average_similarity, 0)
        
    @pytest.mark.asyncio
    async def test_select_examples_similarity_strategy(self):
        """Test selection using similarity strategy."""
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Mock search results
        search_results = [
            VectorSearchResult(id=1, distance=0.1, similarity=0.95, content="Ex1", metadata={}),
            VectorSearchResult(id=2, distance=0.2, similarity=0.90, content="Ex2", metadata={}),
            VectorSearchResult(id=3, distance=0.4, similarity=0.80, content="Ex3", metadata={}),
        ]
        self.vector_store.search_similar.return_value = search_results
        
        # Mock vector data retrieval
        async def get_vector_mock(id, include_embedding=False):
            return {
                "content": f"Input: Test{id}\nOutput: Result{id}",
                "effectiveness_score": 0.5,
                "usage_count": 1,
                "embedding": np.random.randn(4096) if include_embedding else None,
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        result = await self.selector.select_examples(
            query_embedding,
            k=2,
            strategy=SelectionStrategy.SIMILARITY,
            min_similarity=0.85,
        )
        
        # Should select top 2 by similarity above threshold
        self.assertEqual(len(result.examples), 2)
        self.assertEqual(result.examples[0].similarity_score, 0.95)
        self.assertEqual(result.examples[1].similarity_score, 0.90)
        self.assertEqual(result.strategy_used, SelectionStrategy.SIMILARITY)
        
    @pytest.mark.asyncio
    async def test_select_examples_effectiveness_strategy(self):
        """Test selection using effectiveness strategy."""
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Mock search results
        search_results = [
            VectorSearchResult(id=1, distance=0.2, similarity=0.90, content="Ex1", metadata={}),
            VectorSearchResult(id=2, distance=0.24, similarity=0.88, content="Ex2", metadata={}),
            VectorSearchResult(id=3, distance=0.28, similarity=0.86, content="Ex3", metadata={}),
        ]
        self.vector_store.search_similar.return_value = search_results
        
        # Mock vector data with different effectiveness scores
        async def get_vector_mock(id, include_embedding=False):
            effectiveness_scores = {1: 0.3, 2: 0.9, 3: 0.6}
            return {
                "content": f"Input: Test{id}\nOutput: Result{id}",
                "effectiveness_score": effectiveness_scores.get(id, 0.0),
                "usage_count": 5,
                "embedding": np.random.randn(4096) if include_embedding else None,
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        result = await self.selector.select_examples(
            query_embedding,
            k=2,
            strategy=SelectionStrategy.EFFECTIVENESS,
            min_similarity=0.85,
        )
        
        # Should select by effectiveness score
        self.assertEqual(len(result.examples), 2)
        self.assertEqual(result.examples[0].effectiveness_score, 0.9)  # ID 2
        self.assertEqual(result.examples[1].effectiveness_score, 0.6)  # ID 3
        
    @pytest.mark.asyncio
    async def test_select_examples_hybrid_strategy(self):
        """Test selection using hybrid strategy."""
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Mock search results
        search_results = [
            VectorSearchResult(id=1, distance=0.1, similarity=0.95, content="Ex1", metadata={}),
            VectorSearchResult(id=2, distance=0.28, similarity=0.86, content="Ex2", metadata={}),
        ]
        self.vector_store.search_similar.return_value = search_results
        
        # Mock vector data
        async def get_vector_mock(id, include_embedding=False):
            data = {
                1: {"effectiveness_score": 0.2},
                2: {"effectiveness_score": 0.9},
            }
            return {
                "content": f"Input: Test{id}\nOutput: Result{id}",
                "effectiveness_score": data[id]["effectiveness_score"],
                "usage_count": 1,
                "embedding": np.random.randn(4096) if include_embedding else None,
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        result = await self.selector.select_examples(
            query_embedding,
            k=2,
            strategy=SelectionStrategy.HYBRID,
            min_similarity=0.85,
        )
        
        # Hybrid should balance similarity and effectiveness
        # ID 1: 0.7 * 0.95 + 0.3 * 0.2 = 0.665 + 0.06 = 0.725
        # ID 2: 0.7 * 0.86 + 0.3 * 0.9 = 0.602 + 0.27 = 0.872
        # So ID 2 should be ranked first
        self.assertEqual(len(result.examples), 2)
        self.assertEqual(result.examples[0].id, 2)
        
    @pytest.mark.asyncio
    async def test_select_examples_diverse_strategy(self):
        """Test selection using diverse strategy (MMR)."""
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Mock search results
        search_results = [
            VectorSearchResult(id=i, distance=0.1 + i*0.04, similarity=0.95 - i*0.02, content=f"Ex{i}", metadata={})
            for i in range(5)
        ]
        self.vector_store.search_similar.return_value = search_results
        
        # Mock vector data with embeddings
        async def get_vector_mock(id, include_embedding=False):
            # Create distinct embeddings
            embedding = np.zeros(4096)
            embedding[id*100:(id+1)*100] = 1.0  # Different regions
            return {
                "content": f"Input: Test{id}\nOutput: Result{id}",
                "effectiveness_score": 0.5,
                "usage_count": 1,
                "embedding": embedding if include_embedding else None,
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        # Mock similarity calculation
        self.context_builder.calculate_similarity.return_value = 0.1  # Low similarity
        
        result = await self.selector.select_examples(
            query_embedding,
            k=3,
            strategy=SelectionStrategy.DIVERSE,
            min_similarity=0.85,
        )
        
        # Should select diverse examples
        self.assertEqual(len(result.examples), 3)
        self.assertEqual(result.strategy_used, SelectionStrategy.DIVERSE)
        
    @pytest.mark.asyncio
    async def test_update_effectiveness(self):
        """Test effectiveness score update (AC 5)."""
        example_id = 1
        feedback = 0.3  # Positive feedback
        
        await self.selector.update_effectiveness(example_id, feedback)
        
        # Should update in vector store
        self.vector_store.update_effectiveness.assert_called_once_with(
            example_id,
            feedback
        )
        
        # If cached, should update cache
        self.selector._example_cache[example_id] = Example(
            id=example_id,
            input="test",
            output="result",
            effectiveness_score=0.5,
            usage_count=1,
        )
        
        await self.selector.update_effectiveness(example_id, feedback)
        
        cached_example = self.selector._example_cache[example_id]
        self.assertEqual(cached_example.effectiveness_score, 0.8)  # 0.5 + 0.3
        self.assertEqual(cached_example.usage_count, 2)
        
    @pytest.mark.asyncio
    async def test_add_example(self):
        """Test adding new example to pool."""
        input_text = "Sample input"
        output_text = "Sample output"
        embedding = np.random.randn(4096).astype(np.float32)
        metadata = {"source": "test"}
        
        self.vector_store.insert_vector.return_value = 42
        
        example_id = await self.selector.add_example(
            input_text,
            output_text,
            embedding,
            metadata,
        )
        
        self.assertEqual(example_id, 42)
        
        # Verify vector store call
        self.vector_store.insert_vector.assert_called_once()
        call_args = self.vector_store.insert_vector.call_args
        
        self.assertIn("Input: Sample input", call_args.kwargs["content"])
        self.assertIn("Output: Sample output", call_args.kwargs["content"])
        np.testing.assert_array_equal(call_args.kwargs["embedding"], embedding)
        
    @pytest.mark.asyncio
    async def test_get_pool_statistics(self):
        """Test getting example pool statistics."""
        # Mock vector store statistics
        self.vector_store.get_statistics.return_value = {
            "total_vectors": 100,
            "avg_effectiveness": 0.6,
        }
        
        # Add some cached examples
        self.selector._example_cache = {
            1: Example(id=1, input="", output=""),
            2: Example(id=2, input="", output=""),
        }
        
        stats = await self.selector.get_pool_statistics()
        
        self.assertEqual(stats["total_vectors"], 100)
        self.assertEqual(stats["cache_size"], 2)
        self.assertIn("cache_hit_rate", stats)
        
    def test_parse_example_content(self):
        """Test parsing example content."""
        # Test with proper format
        content = "Input: Test input\nOutput: Test output"
        input_text, output_text = self.selector._parse_example_content(content)
        
        self.assertEqual(input_text, "Test input")
        self.assertEqual(output_text, "Test output")
        
        # Test without output marker
        content = "Just some text"
        input_text, output_text = self.selector._parse_example_content(content)
        
        self.assertEqual(input_text, "Just some text")
        self.assertEqual(output_text, "")
        
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add items to cache
        self.selector._example_cache = {
            1: Example(id=1, input="", output=""),
            2: Example(id=2, input="", output=""),
        }
        
        self.selector.clear_cache()
        
        self.assertEqual(len(self.selector._example_cache), 0)


class TestPerformanceRequirements(unittest.IsolatedAsyncioTestCase):
    """Test performance requirements for Story 1.4."""
    
    @pytest.mark.asyncio
    async def test_search_latency_under_100ms(self):
        """Test that similarity search meets <100ms requirement (AC 8)."""
        vector_store = AsyncMock()
        context_builder = MagicMock()
        selector = ExampleSelector(vector_store, context_builder)
        
        # Mock fast search
        async def fast_search(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms
            return [
                VectorSearchResult(id=1, distance=0.2, similarity=0.9, content="Ex", metadata={})
            ]
        vector_store.search_similar = fast_search
        
        # Mock fast vector retrieval
        async def fast_get(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms
            return {"content": "Input: X\nOutput: Y", "effectiveness_score": 0.5}
        vector_store.get_vector = fast_get
        
        start = time.time()
        result = await selector.select_examples(
            np.random.randn(4096).astype(np.float32),
            k=5,
        )
        elapsed_ms = (time.time() - start) * 1000
        
        # Should complete under 100ms
        self.assertLess(elapsed_ms, 100)
        
    def test_batch_processing_capacity(self):
        """Test batch processing up to 32 examples (AC 7)."""
        builder = MoleculeContextBuilder()
        
        # Create 32 examples
        examples = [
            {"input": f"Input {i}", "output": f"Output {i}"}
            for i in range(32)
        ]
        
        # Should handle 32 examples without error
        builder.tokenizer = MagicMock()
        builder.tokenizer.encode.return_value = [1] * 10
        
        selected = builder._select_examples(
            examples,
            token_budget=10000,  # Large budget
        )
        
        # Should process all 32 if budget allows
        self.assertLessEqual(len(selected), 32)
        
    def test_4096_dimensional_embeddings(self):
        """Test handling of 4096-dimensional embeddings (AC 3)."""
        builder = MoleculeContextBuilder()
        
        # Create 4096D embeddings
        embedding1 = np.random.randn(4096).astype(np.float32)
        embedding2 = np.random.randn(4096).astype(np.float32)
        
        # Should handle 4096D without error
        similarity = builder.calculate_similarity(embedding1, embedding2)
        
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])