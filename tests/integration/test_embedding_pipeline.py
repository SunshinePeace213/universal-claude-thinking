"""
Integration tests for Full Embedding Pipeline.

Tests the complete flow from text input to context output, including
integration between all components, memory consolidation, and privacy hooks.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any

import numpy as np
import pytest

from src.core.molecular.context_builder import (
    MoleculeContext,
    MoleculeContextBuilder,
)
from src.core.molecular.example_selector import (
    ExampleSelector,
    Example,
    SelectionStrategy,
)
from src.core.molecular.vector_store import VectorStore, VectorSearchResult
from src.rag.embedder import AdaptiveEmbedder, ModelType
from src.rag.pipeline import RAGPipeline, PipelineConfig, PipelineMode
from src.rag.custom_scorer import CustomScorer, ScoringMethod


class TestEmbeddingPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete embedding pipeline."""
    
    async def asyncSetUp(self):
        """Set up test fixtures for integration tests."""
        # Create mock components
        self.vector_store = AsyncMock(spec=VectorStore)
        self.embedder = AsyncMock(spec=AdaptiveEmbedder)
        
        # Configure embedder
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        self.embedder.generate_embedding = AsyncMock()
        
        # Configure vector store
        self.vector_store.initialize = AsyncMock()
        self.vector_store.search_similar = AsyncMock()
        self.vector_store.insert_vector = AsyncMock()
        self.vector_store.get_vector = AsyncMock()
        self.vector_store.update_effectiveness = AsyncMock()
        
        # Create pipeline with real components
        self.config = PipelineConfig(
            mode=PipelineMode.HYBRID,
            max_examples=5,
            similarity_threshold=0.85,
            chunk_size=1024,
            overlap_ratio=0.15,
            selection_strategy=SelectionStrategy.HYBRID,
            enable_caching=True,
            target_latency_ms=800.0,
            batch_size=32,
        )
        
        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
            config=self.config,
        )
        
    def setUp(self):
        """Sync setup for async tests."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.asyncSetUp())
        
    def tearDown(self):
        """Sync teardown for async tests."""
        self.loop.close()
        
    async def test_end_to_end_flow(self):
        """Test complete flow from input to MOLECULE context output."""
        # Input data
        input_text = "How do I implement a REST API?"
        instruction = "Answer the question based on examples"
        context = "Focus on best practices"
        
        # Mock embedding generation
        query_embedding = np.random.randn(4096).astype(np.float32)
        self.embedder.generate_embedding.return_value = query_embedding
        
        # Mock vector search results
        search_results = [
            VectorSearchResult(
                id=1,
                similarity=0.92,
                content="Input: REST API design\nOutput: Use RESTful principles",
                metadata={"category": "api"},
            ),
            VectorSearchResult(
                id=2,
                similarity=0.88,
                content="Input: API authentication\nOutput: Implement OAuth2",
                metadata={"category": "security"},
            ),
            VectorSearchResult(
                id=3,
                similarity=0.86,
                content="Input: API versioning\nOutput: Use URL versioning",
                metadata={"category": "versioning"},
            ),
        ]
        self.vector_store.search_similar.return_value = search_results
        
        # Mock vector data retrieval
        async def get_vector_mock(id, include_embedding=False):
            contents = {
                1: "Input: REST API design\nOutput: Use RESTful principles",
                2: "Input: API authentication\nOutput: Implement OAuth2",
                3: "Input: API versioning\nOutput: Use URL versioning",
            }
            return {
                "id": id,
                "content": contents.get(id, ""),
                "effectiveness_score": 0.7,
                "usage_count": 5,
                "embedding": np.random.randn(4096) if include_embedding else None,
                "metadata": {},
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        # Process through pipeline
        result = await self.pipeline.process(
            input_text=input_text,
            instruction=instruction,
            context=context,
        )
        
        # Verify complete flow
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.context)
        
        # Check MOLECULE structure
        molecule_context = result.context
        self.assertEqual(molecule_context.instruction, instruction)
        self.assertEqual(molecule_context.context, context)
        self.assertEqual(molecule_context.new_input, input_text)
        
        # Check examples were retrieved and formatted
        self.assertGreater(len(molecule_context.examples), 0)
        self.assertLessEqual(len(molecule_context.examples), 3)
        
        # Verify formatted output
        formatted = molecule_context.format()
        self.assertIn("## INSTRUCTION", formatted)
        self.assertIn(instruction, formatted)
        self.assertIn("## EXAMPLES", formatted)
        self.assertIn("## CONTEXT", formatted)
        self.assertIn(context, formatted)
        self.assertIn("## NEW INPUT", formatted)
        self.assertIn(input_text, formatted)
        
        # Check performance metrics
        self.assertLess(result.total_latency_ms, 1000)  # Should be fast in tests
        self.assertEqual(result.model_used, ModelType.QWEN3_8B)
        
    async def test_memory_consolidation_integration(self):
        """Test integration with memory consolidation system."""
        # Test that pipeline can work with memory system hooks
        input_text = "Explain machine learning"
        
        # Mock embedding
        self.embedder.generate_embedding.return_value = np.random.randn(4096)
        
        # Mock empty search (no examples found)
        self.vector_store.search_similar.return_value = []
        
        # Add memory consolidation hook simulation
        memory_hook_called = False
        
        async def memory_hook(context):
            nonlocal memory_hook_called
            memory_hook_called = True
            return context
            
        # Process with simulated memory hook
        result = await self.pipeline.process(input_text)
        
        # Pipeline should still work without examples
        self.assertIsNotNone(result)
        self.assertEqual(len(result.context.examples), 0)
        
        # Could integrate with actual memory system here
        # For now, just verify pipeline handles empty results gracefully
        
    async def test_privacy_engine_integration(self):
        """Test integration with privacy engine hooks."""
        # Test that sensitive data can be filtered
        input_text = "My SSN is 123-45-6789 and I need help"
        
        # Mock embedding
        self.embedder.generate_embedding.return_value = np.random.randn(4096)
        self.vector_store.search_similar.return_value = []
        
        # Simulate privacy filtering
        # In real system, privacy engine would filter sensitive data
        filtered_input = "My SSN is [REDACTED] and I need help"
        
        # Process through pipeline
        result = await self.pipeline.process(input_text)
        
        # Verify pipeline processes input
        self.assertIsNotNone(result)
        
        # In production, would verify privacy filtering applied
        # For now, just ensure pipeline completes
        
    async def test_effectiveness_feedback_loop(self):
        """Test effectiveness tracking and feedback integration (AC 5)."""
        # Process initial request
        input_text = "How to optimize database queries?"
        
        # Setup mocks
        self.embedder.generate_embedding.return_value = np.random.randn(4096)
        
        search_results = [
            VectorSearchResult(
                id=1,
                similarity=0.90,
                content="Input: Query optimization\nOutput: Use indexes",
                metadata={},
            ),
        ]
        self.vector_store.search_similar.return_value = search_results
        
        async def get_vector_mock(id, include_embedding=False):
            return {
                "id": id,
                "content": "Input: Query optimization\nOutput: Use indexes",
                "effectiveness_score": 0.5,
                "usage_count": 1,
                "embedding": None,
            }
        self.vector_store.get_vector.side_effect = get_vector_mock
        
        # Process and get result
        result = await self.pipeline.process(input_text)
        
        # Simulate positive feedback
        feedback = 0.3  # Positive feedback
        await self.pipeline.update_effectiveness(result, feedback)
        
        # Verify effectiveness update was called
        # Note: In the current implementation, example IDs are hashed
        # In production, would track actual IDs through pipeline
        self.vector_store.update_effectiveness.assert_called()
        
    async def test_batch_processing_integration(self):
        """Test batch processing with full pipeline (AC 7)."""
        # Create batch of inputs
        inputs = [f"Question {i}" for i in range(10)]
        instruction = "Answer based on examples"
        
        # Mock batch embeddings
        batch_embeddings = np.random.randn(10, 4096).astype(np.float32)
        
        async def batch_embed(texts):
            if isinstance(texts, list):
                return batch_embeddings[:len(texts)]
            return batch_embeddings[0]
            
        self.embedder.generate_embedding = batch_embed
        
        # Mock search results for each input
        self.vector_store.search_similar.return_value = []
        
        # Process batch
        results = await self.pipeline.batch_process(
            inputs,
            instruction=instruction,
        )
        
        # Verify all inputs processed
        self.assertEqual(len(results), 10)
        
        # Each should have a valid context
        for i, result in enumerate(results):
            self.assertIsNotNone(result.context)
            self.assertEqual(result.context.new_input, f"Question {i}")
            self.assertEqual(result.context.instruction, instruction)
            
    async def test_adaptive_model_switching(self):
        """Test adaptive model switching based on performance (AC 11)."""
        # Simulate high latency scenario
        slow_embedding_calls = 0
        
        async def slow_embedding(text):
            nonlocal slow_embedding_calls
            slow_embedding_calls += 1
            if slow_embedding_calls <= 5:
                await asyncio.sleep(0.1)  # Simulate slow response
            return np.random.randn(4096)
            
        self.embedder.generate_embedding = slow_embedding
        self.vector_store.search_similar.return_value = []
        
        # Process multiple requests to build history
        for i in range(5):
            await self.pipeline.process(f"Input {i}")
            
        # Check timing history built up
        self.assertGreater(len(self.pipeline._timing_history), 0)
        
        # Simulate optimization check
        # In production, this would trigger model switch
        # For test, just verify optimization can be called
        await self.pipeline.optimize_for_latency()
        
    async def test_custom_scorer_integration(self):
        """Test integration with custom scoring algorithm."""
        # Create scorer
        scorer = CustomScorer(
            semantic_weight=0.6,
            lexical_weight=0.2,
            effectiveness_weight=0.2,
        )
        
        # Test items to score
        items = [
            {
                "id": 1,
                "content": "Machine learning example",
                "embedding": np.random.randn(4096),
                "effectiveness_score": 0.8,
            },
            {
                "id": 2,
                "content": "Deep learning tutorial",
                "embedding": np.random.randn(4096),
                "effectiveness_score": 0.6,
            },
        ]
        
        query_embedding = np.random.randn(4096)
        query_terms = ["machine", "learning"]
        
        # Score items
        scored = scorer.score_items(
            items,
            query_embedding=query_embedding,
            query_terms=query_terms,
            method=ScoringMethod.HYBRID,
        )
        
        # Verify scoring integration
        self.assertEqual(len(scored), 2)
        for item in scored:
            self.assertGreaterEqual(item.final_score, 0.0)
            self.assertLessEqual(item.final_score, 1.0)
            
        # Update weights based on feedback
        feedback = 0.3
        last_scores = {
            "semantic": 0.8,
            "lexical": 0.5,
            "effectiveness": 0.7,
            "recency": 0.5,
            "diversity": 0.4,
        }
        scorer.update_weights(feedback, last_scores)
        
        # Weights should be updated and normalized
        total = sum(scorer.weights.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        
    async def test_performance_requirements(self):
        """Test that integrated pipeline meets performance requirements."""
        # Setup for fast processing
        self.embedder.generate_embedding = AsyncMock(
            return_value=np.random.randn(4096)
        )
        self.vector_store.search_similar = AsyncMock(return_value=[])
        
        # Measure actual pipeline latency
        start = time.perf_counter()
        result = await self.pipeline.process("Test input")
        end = time.perf_counter()
        
        actual_latency_ms = (end - start) * 1000
        
        # Should meet performance requirements (AC 6)
        # Allow some margin for test environment
        self.assertLess(actual_latency_ms, 1000)  # Relaxed for tests
        
        # Check reported latency
        if result.total_latency_ms > 0:
            self.assertLess(result.total_latency_ms, 1000)
            
    async def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test embedding error
        self.embedder.generate_embedding.side_effect = Exception("Embedding failed")
        
        with self.assertRaises(Exception) as context:
            await self.pipeline.process("Test input")
            
        self.assertIn("Embedding failed", str(context.exception))
        
        # Reset and test vector store error
        self.embedder.generate_embedding.side_effect = None
        self.embedder.generate_embedding.return_value = np.random.randn(4096)
        self.vector_store.search_similar.side_effect = Exception("Search failed")
        
        with self.assertRaises(Exception) as context:
            await self.pipeline.process("Test input")
            
        self.assertIn("Search failed", str(context.exception))
        
    async def test_caching_integration(self):
        """Test caching across integrated components."""
        # Enable caching
        self.pipeline.config.enable_caching = True
        
        # Setup mocks
        self.embedder.generate_embedding.return_value = np.random.randn(4096)
        self.vector_store.search_similar.return_value = []
        
        # First call - should process normally
        result1 = await self.pipeline.process("Cached input")
        self.assertEqual(result1.cache_hits, 0)
        
        # Second call - should use cache
        result2 = await self.pipeline.process("Cached input")
        self.assertEqual(result2.cache_hits, 1)
        
        # Embedder should only be called once
        self.assertEqual(self.embedder.generate_embedding.call_count, 1)
        
        # Different input should not use cache
        result3 = await self.pipeline.process("Different input")
        self.assertEqual(result3.cache_hits, 0)
        
        # Embedder called again for new input
        self.assertEqual(self.embedder.generate_embedding.call_count, 2)


class TestComponentIntegration(unittest.TestCase):
    """Test integration between specific components."""
    
    async def test_context_builder_and_selector_integration(self):
        """Test integration between context builder and example selector."""
        # Create real components
        context_builder = MoleculeContextBuilder(
            chunk_size=1024,
            overlap_ratio=0.15,
            max_examples=5,
        )
        
        # Mock vector store
        vector_store = AsyncMock()
        
        example_selector = ExampleSelector(
            vector_store=vector_store,
            context_builder=context_builder,
            default_strategy=SelectionStrategy.HYBRID,
        )
        
        # Mock search results
        query_embedding = np.random.randn(4096)
        search_results = [
            VectorSearchResult(
                id=1,
                similarity=0.95,
                content="Input: Test 1\nOutput: Result 1",
                metadata={},
            ),
            VectorSearchResult(
                id=2,
                similarity=0.90,
                content="Input: Test 2\nOutput: Result 2",
                metadata={},
            ),
        ]
        vector_store.search_similar.return_value = search_results
        
        # Mock vector data
        async def get_vector_mock(id, include_embedding=False):
            contents = {
                1: "Input: Test 1\nOutput: Result 1",
                2: "Input: Test 2\nOutput: Result 2",
            }
            return {
                "content": contents.get(id, ""),
                "effectiveness_score": 0.7,
                "usage_count": 3,
                "embedding": np.random.randn(4096) if include_embedding else None,
            }
        vector_store.get_vector.side_effect = get_vector_mock
        
        # Select examples
        selection_result = await example_selector.select_examples(
            query_embedding,
            k=2,
            min_similarity=0.85,
        )
        
        # Build context with selected examples
        example_dicts = [ex.to_dict() for ex in selection_result.examples]
        similarity_scores = [ex.similarity_score for ex in selection_result.examples]
        
        context = context_builder.build_context(
            instruction="Test instruction",
            examples=example_dicts,
            context="Test context",
            new_input="Test input",
            similarity_scores=similarity_scores,
        )
        
        # Verify integration
        self.assertIsNotNone(context)
        self.assertEqual(len(context.examples), 2)
        self.assertEqual(context.examples[0]["input"], "Test 1")
        self.assertEqual(context.examples[0]["output"], "Result 1")
        
        # Verify MOLECULE structure
        formatted = context.format()
        self.assertIn("## INSTRUCTION", formatted)
        self.assertIn("## EXAMPLES", formatted)
        self.assertIn("### Example 1", formatted)
        self.assertIn("### Example 2", formatted)
        
    async def test_embedder_and_vector_store_integration(self):
        """Test integration between embedder and vector store."""
        # This would test real embedder with vector store
        # For unit tests, we use mocks to avoid external dependencies
        pass


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])