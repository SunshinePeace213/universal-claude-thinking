"""
Unit tests for RAG Pipeline Orchestration.

Tests the main RAGPipeline class that coordinates embedding generation,
vector search, example selection, and context assembly.
Validates <800ms total latency and batch processing capabilities.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any

import numpy as np
import pytest

from src.rag.pipeline import (
    RAGPipeline,
    PipelineConfig,
    PipelineMode,
    PipelineResult,
)
from src.core.molecular.context_builder import MoleculeContext
from src.core.molecular.example_selector import (
    Example,
    SelectionResult,
    SelectionStrategy,
)
from src.rag.embedder import ModelType


class TestPipelineConfig(unittest.TestCase):
    """Test cases for PipelineConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        self.assertEqual(config.mode, PipelineMode.HYBRID)
        self.assertEqual(config.max_examples, 10)
        self.assertEqual(config.similarity_threshold, 0.85)
        self.assertEqual(config.chunk_size, 1024)
        self.assertEqual(config.overlap_ratio, 0.15)
        self.assertEqual(config.selection_strategy, SelectionStrategy.HYBRID)
        self.assertTrue(config.enable_caching)
        self.assertEqual(config.target_latency_ms, 800.0)
        self.assertEqual(config.batch_size, 32)
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            mode=PipelineMode.RETRIEVAL,
            max_examples=5,
            similarity_threshold=0.90,
            batch_size=16,
        )
        
        self.assertEqual(config.mode, PipelineMode.RETRIEVAL)
        self.assertEqual(config.max_examples, 5)
        self.assertEqual(config.similarity_threshold, 0.90)
        self.assertEqual(config.batch_size, 16)


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAGPipeline class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vector_store = AsyncMock()
        self.embedder = AsyncMock()
        self.config = PipelineConfig(
            enable_caching=True,
            target_latency_ms=800.0,
        )
        
        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
            config=self.config,
        )
        
    async def test_initialization(self):
        """Test pipeline initialization."""
        # Not yet initialized
        self.assertFalse(self.pipeline._initialized)
        
        await self.pipeline.initialize()
        
        # Should initialize components
        self.vector_store.initialize.assert_called_once()
        self.embedder.initialize.assert_called_once()
        self.assertTrue(self.pipeline._initialized)
        
        # Idempotent initialization
        await self.pipeline.initialize()
        self.vector_store.initialize.assert_called_once()  # Still once
        
    async def test_process_simple_input(self):
        """Test processing simple input through pipeline."""
        # Setup mocks
        input_text = "Test input"
        instruction = "Process this"
        context = "Additional context"
        
        # Mock embedding generation
        embedding = np.random.randn(4096).astype(np.float32)
        self.embedder.generate_embedding.return_value = embedding
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Mock example selection
        examples = [
            Example(
                id=1,
                input="Ex1",
                output="Out1",
                similarity_score=0.90,
                effectiveness_score=0.5,
            ),
            Example(
                id=2,
                input="Ex2",
                output="Out2",
                similarity_score=0.88,
                effectiveness_score=0.6,
            ),
        ]
        
        selection_result = SelectionResult(
            examples=examples,
            total_candidates=10,
            selection_time_ms=50.0,
            average_similarity=0.89,
            average_effectiveness=0.55,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        # Create proper mock for example_selector
        with patch.object(self.pipeline.example_selector, 'select_examples', return_value=selection_result) as mock_select:
            # Mock context builder
            with patch.object(self.pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction=instruction,
                    examples=[{"input": "Ex1", "output": "Out1"}],
                    context=context,
                    new_input=input_text,
                    token_count=500,
                    similarity_scores=[0.90],
                )
                mock_build.return_value = mock_context
                
                # Process input
                result = await self.pipeline.process(
                    input_text=input_text,
                    instruction=instruction,
                    context=context,
                )
                
                # Verify result
                self.assertIsInstance(result, PipelineResult)
                self.assertEqual(result.context, mock_context)
                self.assertEqual(result.examples_retrieved, 2)
                self.assertEqual(result.embeddings_generated, 1)
                self.assertEqual(result.model_used, ModelType.QWEN3_8B)
                
                # Verify embedder was called
                self.embedder.generate_embedding.assert_called_once()
                call_text = self.embedder.generate_embedding.call_args[0][0]
                self.assertIn(input_text, call_text)
                
                # Verify example selector was called
                mock_select.assert_called_once()
                np.testing.assert_array_equal(
                    mock_select.call_args.kwargs['query_embedding'],
                    embedding
                )
                
    async def test_process_with_caching(self):
        """Test caching functionality."""
        input_text = "Cached input"
        
        # Setup mocks
        embedding = np.random.randn(4096).astype(np.float32)
        self.embedder.generate_embedding.return_value = embedding
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Mock selection result
        selection_result = SelectionResult(
            examples=[],
            total_candidates=0,
            selection_time_ms=10.0,
            average_similarity=0,
            average_effectiveness=0,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        with patch.object(self.pipeline.example_selector, 'select_examples', return_value=selection_result):
            with patch.object(self.pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction="",
                    examples=[],
                    context="",
                    new_input=input_text,
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                # First call - should process and cache
                result1 = await self.pipeline.process(input_text)
                self.assertEqual(result1.cache_hits, 0)
                
                # Second call - should use cache
                result2 = await self.pipeline.process(input_text)
                self.assertEqual(result2.cache_hits, 1)
                
                # Embedder should only be called once
                self.embedder.generate_embedding.assert_called_once()
                
    async def test_process_exceeds_latency_target(self):
        """Test warning when latency exceeds target (AC 6)."""
        # Setup to simulate slow processing
        async def slow_embedding(*args, **kwargs):
            await asyncio.sleep(0.5)  # 500ms
            return np.random.randn(4096).astype(np.float32)
            
        self.embedder.generate_embedding = slow_embedding
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Fast selection to isolate embedding latency
        selection_result = SelectionResult(
            examples=[],
            total_candidates=0,
            selection_time_ms=10.0,
            average_similarity=0,
            average_effectiveness=0,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        with patch.object(self.pipeline.example_selector, 'select_examples', return_value=selection_result):
            with patch.object(self.pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction="",
                    examples=[],
                    context="",
                    new_input="test",
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                # Set low latency target
                self.pipeline.config.target_latency_ms = 100.0
                
                result = await self.pipeline.process("test input")
                
                # Should have warning about latency
                self.assertIsNotNone(result.warnings)
                self.assertTrue(any("latency" in w.lower() for w in result.warnings))
                
    async def test_process_dimension_mismatch_warning(self):
        """Test warning on embedding dimension mismatch."""
        # Wrong dimension embedding
        wrong_dim_embedding = np.random.randn(2048).astype(np.float32)
        self.embedder.generate_embedding.return_value = wrong_dim_embedding
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        selection_result = SelectionResult(
            examples=[],
            total_candidates=0,
            selection_time_ms=10.0,
            average_similarity=0,
            average_effectiveness=0,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        with patch.object(self.pipeline.example_selector, 'select_examples', return_value=selection_result):
            with patch.object(self.pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction="",
                    examples=[],
                    context="",
                    new_input="test",
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                result = await self.pipeline.process("test")
                
                # Should have warning about dimension
                self.assertIsNotNone(result.warnings)
                self.assertTrue(any("dimension" in w.lower() for w in result.warnings))
                
    async def test_batch_process(self):
        """Test batch processing capability (AC 7)."""
        # Test with 35 inputs (more than batch size of 32)
        inputs = [f"Input {i}" for i in range(35)]
        instruction = "Process batch"
        
        # Mock batch embedding generation
        batch_embeddings = np.random.randn(35, 4096).astype(np.float32)
        
        # Mock to return appropriate batch sizes
        call_count = 0
        
        async def batch_embed(texts):
            nonlocal call_count
            if call_count == 0:
                # First batch (32 items)
                result = batch_embeddings[:32]
            else:
                # Second batch (3 items)
                result = batch_embeddings[32:35]
            call_count += 1
            return result
            
        self.embedder.generate_embedding = batch_embed
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Mock selection
        selection_result = SelectionResult(
            examples=[],
            total_candidates=0,
            selection_time_ms=10.0,
            average_similarity=0,
            average_effectiveness=0,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        with patch.object(self.pipeline.example_selector, 'select_examples', return_value=selection_result):
            with patch.object(self.pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction=instruction,
                    examples=[],
                    context="",
                    new_input="",
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                results = await self.pipeline.batch_process(
                    inputs,
                    instruction=instruction,
                )
                
                # Should process all inputs
                self.assertEqual(len(results), 35)
                
                # Should call embedder twice (32 + 3)
                self.assertEqual(call_count, 2)
                
                # All results should have the model type
                for result in results:
                    self.assertEqual(result.model_used, ModelType.QWEN3_8B)
                    
    async def test_update_effectiveness(self):
        """Test effectiveness score updates (AC 5)."""
        # Create a mock result
        result = PipelineResult(
            context=MoleculeContext(
                instruction="Test",
                examples=[
                    {"input": "Ex1", "output": "Out1"},
                    {"input": "Ex2", "output": "Out2"},
                ],
                context="",
                new_input="Test",
                token_count=100,
            ),
            examples_retrieved=2,
            embeddings_generated=1,
            total_latency_ms=100,
            breakdown={},
            model_used=ModelType.QWEN3_8B,
        )
        
        # Update effectiveness
        feedback = 0.3  # Positive feedback
        
        with patch.object(self.pipeline.example_selector, 'update_effectiveness') as mock_update:
            await self.pipeline.update_effectiveness(result, feedback)
            
            # Should call update for each example
            self.assertEqual(mock_update.call_count, 2)
            
            # Check feedback value
            for call in mock_update.call_args_list:
                self.assertEqual(call.kwargs['feedback'], feedback)
                
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some timing history
        self.pipeline._timing_history = [
            {
                "embedding_generation": 100.0,
                "example_retrieval": 50.0,
                "context_building": 30.0,
                "total": 180.0,
            },
            {
                "embedding_generation": 120.0,
                "example_retrieval": 40.0,
                "context_building": 35.0,
                "total": 195.0,
            },
        ]
        
        # Add cache entries
        self.pipeline._cache = {
            "key1": MagicMock(cache_hits=2),
            "key2": MagicMock(cache_hits=1),
        }
        
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        metrics = self.pipeline.get_performance_metrics()
        
        # Check average calculations
        self.assertAlmostEqual(metrics["avg_embedding_generation_ms"], 110.0)
        self.assertAlmostEqual(metrics["avg_example_retrieval_ms"], 45.0)
        self.assertAlmostEqual(metrics["avg_context_building_ms"], 32.5)
        self.assertAlmostEqual(metrics["avg_total_ms"], 187.5)
        
        # Check cache statistics
        self.assertTrue(metrics["cache_enabled"])
        self.assertEqual(metrics["cache_size"], 2)
        
        # Check model info
        self.assertEqual(metrics["active_model"], ModelType.QWEN3_8B.value)
        self.assertEqual(metrics["batch_size"], 32)
        self.assertEqual(metrics["total_requests"], 2)
        
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add items to cache
        self.pipeline._cache = {
            "key1": MagicMock(),
            "key2": MagicMock(),
        }
        
        self.pipeline.clear_cache()
        
        self.assertEqual(len(self.pipeline._cache), 0)
        
    async def test_optimize_for_latency(self):
        """Test latency optimization."""
        # Add history with high latency
        self.pipeline._timing_history = [
            {"total": 1000.0} for _ in range(15)  # All exceed 800ms target
        ]
        
        # Setup embedder for model switching
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B
        self.embedder.fallback_embedder = None
        
        with patch('src.rag.pipeline.Qwen8B4BitEmbedder') as mock_4bit:
            mock_4bit_instance = AsyncMock()
            mock_4bit.return_value = mock_4bit_instance
            
            await self.pipeline.optimize_for_latency()
            
            # Should create and switch to 4bit model
            mock_4bit.assert_called_once()
            mock_4bit_instance.initialize.assert_called_once()
            self.assertEqual(self.pipeline.embedder.active_embedder, mock_4bit_instance)
            
    async def test_optimize_reduce_batch_size(self):
        """Test optimization by reducing batch size."""
        # Add history with high latency
        self.pipeline._timing_history = [
            {"total": 900.0} for _ in range(15)
        ]
        
        # Setup embedder already using 4bit
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B_4BIT
        self.embedder.fallback_embedder = MagicMock()
        
        # Start with large batch size
        self.pipeline.config.batch_size = 32
        
        await self.pipeline.optimize_for_latency()
        
        # Should reduce batch size
        self.assertEqual(self.pipeline.config.batch_size, 16)
        
    async def test_optimize_reduce_examples(self):
        """Test optimization by reducing max examples."""
        # Add history with high latency
        self.pipeline._timing_history = [
            {"total": 900.0} for _ in range(15)
        ]
        
        # Setup embedder already using 4bit
        self.embedder.active_embedder = MagicMock()
        self.embedder.active_embedder.model_type = ModelType.QWEN3_8B_4BIT
        
        # Already reduced batch size
        self.pipeline.config.batch_size = 8
        self.pipeline.config.max_examples = 10
        
        await self.pipeline.optimize_for_latency()
        
        # Should reduce max examples
        self.assertEqual(self.pipeline.config.max_examples, 8)
        
    async def test_close(self):
        """Test resource cleanup."""
        # Add cache items
        self.pipeline._cache = {"key": MagicMock()}
        
        await self.pipeline.close()
        
        # Should close components
        self.embedder.close.assert_called_once()
        self.vector_store.close.assert_called_once()
        
        # Should clear cache
        self.assertEqual(len(self.pipeline._cache), 0)
        
    def test_get_cache_key(self):
        """Test cache key generation."""
        key1 = self.pipeline._get_cache_key("input", "instruction", "context")
        self.assertEqual(key1, "input|instruction|context")
        
        key2 = self.pipeline._get_cache_key("input", None, None)
        self.assertEqual(key2, "input")
        
        key3 = self.pipeline._get_cache_key("input", "instruction", None)
        self.assertEqual(key3, "input|instruction")


class TestPipelinePerformance(unittest.TestCase):
    """Test performance requirements for RAG pipeline."""
    
    async def test_total_latency_under_800ms(self):
        """Test that total pipeline latency is under 800ms (AC 6)."""
        vector_store = AsyncMock()
        embedder = AsyncMock()
        config = PipelineConfig(
            enable_caching=False,  # Disable cache for this test
            target_latency_ms=800.0,
        )
        
        pipeline = RAGPipeline(vector_store, embedder, config)
        
        # Mock fast operations
        async def fast_embed(*args, **kwargs):
            await asyncio.sleep(0.2)  # 200ms
            return np.random.randn(4096).astype(np.float32)
            
        embedder.generate_embedding = fast_embed
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B_4BIT
        
        # Mock fast selection
        async def fast_select(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms
            return SelectionResult(
                examples=[],
                total_candidates=0,
                selection_time_ms=50.0,
                average_similarity=0,
                average_effectiveness=0,
                strategy_used=SelectionStrategy.HYBRID,
            )
            
        with patch.object(pipeline.example_selector, 'select_examples', side_effect=fast_select):
            with patch.object(pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction="",
                    examples=[],
                    context="",
                    new_input="test",
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                start = time.time()
                result = await pipeline.process("test input")
                total_time_ms = (time.time() - start) * 1000
                
                # Should complete under 800ms
                self.assertLess(total_time_ms, 800)
                self.assertLess(result.total_latency_ms, 800)
                
                # Should not have latency warning
                if result.warnings:
                    self.assertFalse(any("latency" in w.lower() for w in result.warnings))
                    
    async def test_batch_size_32_support(self):
        """Test support for batch size of 32 (AC 7)."""
        vector_store = AsyncMock()
        embedder = AsyncMock()
        config = PipelineConfig(batch_size=32)
        
        pipeline = RAGPipeline(vector_store, embedder, config)
        
        # Create exactly 32 inputs
        inputs = [f"Input {i}" for i in range(32)]
        
        # Mock batch embedding that handles 32 items
        batch_embeddings = np.random.randn(32, 4096).astype(np.float32)
        embedder.generate_embedding.return_value = batch_embeddings
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Mock selection
        selection_result = SelectionResult(
            examples=[],
            total_candidates=0,
            selection_time_ms=10.0,
            average_similarity=0,
            average_effectiveness=0,
            strategy_used=SelectionStrategy.HYBRID,
        )
        
        with patch.object(pipeline.example_selector, 'select_examples', return_value=selection_result):
            with patch.object(pipeline.context_builder, 'build_context') as mock_build:
                mock_context = MoleculeContext(
                    instruction="",
                    examples=[],
                    context="",
                    new_input="",
                    token_count=100,
                )
                mock_build.return_value = mock_context
                
                results = await pipeline.batch_process(inputs)
                
                # Should process all 32 in one batch
                self.assertEqual(len(results), 32)
                embedder.generate_embedding.assert_called_once()


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])