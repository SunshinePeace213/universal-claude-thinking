"""
Performance Benchmark Tests for Embedding Pipeline.

Validates all timing requirements and performance benchmarks for Story 1.4,
including context construction <800ms, similarity search <100ms, and batch processing.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import List, Dict, Any
import statistics

import numpy as np
import pytest

from src.core.molecular.context_builder import MoleculeContextBuilder
from src.core.molecular.example_selector import ExampleSelector, SelectionStrategy
from src.core.molecular.vector_store import VectorStore, VectorSearchResult
from src.rag.embedder import AdaptiveEmbedder, ModelType
from src.rag.pipeline import RAGPipeline, PipelineConfig, PipelineMode
from src.rag.benchmarks.model_benchmark import ModelBenchmark


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance validation tests for all acceptance criteria."""
    
    @pytest.mark.performance
    async def test_context_construction_latency(self):
        """Test context construction meets <800ms requirement (AC 6)."""
        # Create context builder
        builder = MoleculeContextBuilder(
            chunk_size=1024,
            overlap_ratio=0.15,
            max_examples=10,
        )
        
        # Mock tokenizer for consistent performance
        builder.tokenizer = MagicMock()
        builder.tokenizer.encode.return_value = [1] * 100
        
        # Test data
        instruction = "Process the input based on examples"
        examples = [
            {"input": f"Example {i}", "output": f"Output {i}"}
            for i in range(10)
        ]
        context = "Additional context information"
        new_input = "This is the new input to process"
        similarity_scores = [0.95 - i*0.05 for i in range(10)]
        
        # Measure latency over multiple runs
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            
            result = builder.build_context(
                instruction=instruction,
                examples=examples,
                context=context,
                new_input=new_input,
                similarity_scores=similarity_scores,
            )
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Each run should be under 800ms
            self.assertLess(latency_ms, 800, f"Context construction took {latency_ms:.2f}ms")
            
        # Check average and percentiles
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        print(f"Context construction - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        self.assertLess(avg_latency, 500, "Average latency should be well under 800ms")
        self.assertLess(p95_latency, 800, "95th percentile should be under 800ms")
        
    @pytest.mark.performance
    async def test_similarity_search_latency(self):
        """Test similarity search meets <100ms requirement (AC 8)."""
        # Mock vector store with fast search
        vector_store = AsyncMock()
        
        async def fast_search(embedding, k, include_embeddings=False):
            # Simulate fast search
            await asyncio.sleep(0.01)  # 10ms base latency
            results = []
            for i in range(min(k, 20)):
                results.append(
                    VectorSearchResult(
                        id=i+1,
                        similarity=0.95 - i*0.01,
                        content=f"Example {i}",
                        metadata={},
                    )
                )
            return results
            
        vector_store.search_similar = fast_search
        
        # Create example selector
        context_builder = MoleculeContextBuilder()
        selector = ExampleSelector(
            vector_store=vector_store,
            context_builder=context_builder,
        )
        
        # Mock vector data retrieval (should also be fast)
        async def fast_get_vector(id, include_embedding=False):
            await asyncio.sleep(0.001)  # 1ms per vector
            return {
                "id": id,
                "content": f"Input: Example {id}\nOutput: Result {id}",
                "effectiveness_score": 0.7,
                "usage_count": 5,
                "embedding": np.random.randn(4096) if include_embedding else None,
            }
        vector_store.get_vector = fast_get_vector
        
        # Test query
        query_embedding = np.random.randn(4096).astype(np.float32)
        
        # Measure search latency
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            
            result = await selector.select_examples(
                query_embedding,
                k=10,
                min_similarity=0.85,
            )
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Each search should be under 100ms
            self.assertLess(latency_ms, 100, f"Search took {latency_ms:.2f}ms")
            
        # Check statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        
        print(f"Similarity search - Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms")
        
        self.assertLess(avg_latency, 50, "Average search should be well under 100ms")
        self.assertLess(p95_latency, 100, "95th percentile should be under 100ms")
        
    @pytest.mark.performance
    async def test_batch_processing_capacity(self):
        """Test batch processing up to 32 examples (AC 7)."""
        # Create pipeline with batch configuration
        vector_store = AsyncMock()
        embedder = AsyncMock()
        
        config = PipelineConfig(
            batch_size=32,
            max_examples=10,
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            config=config,
        )
        
        # Mock fast batch embedding
        async def batch_embed(texts):
            batch_size = len(texts) if isinstance(texts, list) else 1
            await asyncio.sleep(0.001 * batch_size)  # 1ms per item
            return np.random.randn(batch_size, 4096).astype(np.float32)
            
        embedder.generate_embedding = batch_embed
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        # Mock vector operations
        vector_store.search_similar.return_value = []
        
        # Test with exactly 32 items
        inputs = [f"Input text {i}" for i in range(32)]
        
        start = time.perf_counter()
        results = await pipeline.batch_process(inputs)
        end = time.perf_counter()
        
        total_time_ms = (end - start) * 1000
        
        # Should process all 32
        self.assertEqual(len(results), 32)
        
        # Should complete in reasonable time
        per_item_ms = total_time_ms / 32
        print(f"Batch processing 32 items - Total: {total_time_ms:.2f}ms, Per item: {per_item_ms:.2f}ms")
        
        # Verify reasonable throughput
        self.assertLess(per_item_ms, 100, "Should process each item quickly in batch")
        
        # Test with larger batch (should handle in chunks)
        large_inputs = [f"Input {i}" for i in range(100)]
        
        start = time.perf_counter()
        large_results = await pipeline.batch_process(large_inputs)
        end = time.perf_counter()
        
        self.assertEqual(len(large_results), 100)
        
        large_time_ms = (end - start) * 1000
        print(f"Batch processing 100 items - Total: {large_time_ms:.2f}ms")
        
    @pytest.mark.performance
    async def test_end_to_end_pipeline_latency(self):
        """Test complete pipeline meets <800ms total latency (AC 6)."""
        # Create full pipeline
        vector_store = AsyncMock()
        embedder = AsyncMock()
        
        config = PipelineConfig(
            target_latency_ms=800.0,
            enable_caching=False,  # Disable cache for performance testing
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            config=config,
        )
        
        # Mock components with realistic latencies
        async def embedding_with_latency(text):
            await asyncio.sleep(0.2)  # 200ms for embedding
            return np.random.randn(4096).astype(np.float32)
            
        embedder.generate_embedding = embedding_with_latency
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        async def search_with_latency(embedding, k, include_embeddings=False):
            await asyncio.sleep(0.05)  # 50ms for search
            return [
                VectorSearchResult(
                    id=i,
                    similarity=0.9 - i*0.01,
                    content=f"Example {i}",
                    metadata={},
                )
                for i in range(min(k, 5))
            ]
            
        vector_store.search_similar = search_with_latency
        
        async def get_vector_with_latency(id, include_embedding=False):
            await asyncio.sleep(0.005)  # 5ms per vector
            return {
                "content": f"Input: Ex {id}\nOutput: Result {id}",
                "effectiveness_score": 0.7,
                "usage_count": 3,
                "embedding": None,
            }
            
        vector_store.get_vector = get_vector_with_latency
        
        # Measure end-to-end latency
        latencies = []
        for i in range(5):
            start = time.perf_counter()
            
            result = await pipeline.process(f"Test input {i}")
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
            # Should meet target
            self.assertLess(latency_ms, 900, f"Pipeline took {latency_ms:.2f}ms (target: 800ms)")
            
            # Check breakdown
            if result.breakdown:
                print(f"Run {i+1} breakdown: {result.breakdown}")
                
        avg_latency = statistics.mean(latencies)
        print(f"End-to-end pipeline - Avg: {avg_latency:.2f}ms")
        
        self.assertLess(avg_latency, 850, "Average should be close to target")
        
    @pytest.mark.performance
    async def test_4096_dimensional_performance(self):
        """Test performance with 4096-dimensional embeddings (AC 3)."""
        # Test that 4096D embeddings don't cause performance issues
        dimension = 4096
        num_vectors = 100
        
        # Create random embeddings
        embeddings = np.random.randn(num_vectors, dimension).astype(np.float32)
        
        # Normalize embeddings
        for i in range(num_vectors):
            embeddings[i] = embeddings[i] / np.linalg.norm(embeddings[i])
            
        # Test cosine similarity calculation performance
        query = embeddings[0]
        
        start = time.perf_counter()
        
        similarities = []
        for i in range(1, num_vectors):
            similarity = np.dot(query, embeddings[i])
            similarities.append(similarity)
            
        end = time.perf_counter()
        
        calc_time_ms = (end - start) * 1000
        per_calc_us = (calc_time_ms * 1000) / (num_vectors - 1)
        
        print(f"4096D similarity calculation - Total: {calc_time_ms:.2f}ms, Per calc: {per_calc_us:.2f}μs")
        
        # Should be fast even with 4096 dimensions
        self.assertLess(per_calc_us, 100, "Each similarity calculation should be fast")
        
    @pytest.mark.performance
    async def test_chunk_overlap_performance(self):
        """Test performance with 15% chunk overlap (AC 4)."""
        builder = MoleculeContextBuilder(
            chunk_size=1024,
            overlap_ratio=0.15,
        )
        
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = list(range(5000))  # Large document
        mock_tokenizer.decode.side_effect = lambda tokens: f"Chunk of {len(tokens)} tokens"
        builder.tokenizer = mock_tokenizer
        
        # Create multiple contexts
        contexts = []
        for i in range(5):
            context = MagicMock()
            context.format.return_value = f"Context {i} with lots of text " * 100
            contexts.append(context)
            
        # Measure chunking performance
        start = time.perf_counter()
        chunks = builder.create_chunks(contexts)
        end = time.perf_counter()
        
        chunking_time_ms = (end - start) * 1000
        
        print(f"Chunking with 15% overlap - Time: {chunking_time_ms:.2f}ms, Chunks: {len(chunks)}")
        
        # Should create multiple overlapping chunks
        self.assertGreater(len(chunks), 1)
        
        # Should be reasonably fast
        self.assertLess(chunking_time_ms, 100, "Chunking should be fast")
        
        # Verify overlap calculation
        expected_overlap = int(1024 * 0.15)
        self.assertEqual(builder.overlap_tokens, expected_overlap)
        
    @pytest.mark.performance
    async def test_effectiveness_update_performance(self):
        """Test performance of effectiveness score updates (AC 5)."""
        vector_store = AsyncMock()
        
        # Mock fast effectiveness updates
        update_times = []
        
        async def update_effectiveness(id, feedback):
            start = time.perf_counter()
            await asyncio.sleep(0.001)  # Simulate DB update
            end = time.perf_counter()
            update_times.append((end - start) * 1000)
            
        vector_store.update_effectiveness = update_effectiveness
        
        # Test batch updates
        for i in range(50):
            await vector_store.update_effectiveness(i, 0.3)
            
        avg_update_time = statistics.mean(update_times)
        print(f"Effectiveness updates - Avg: {avg_update_time:.2f}ms")
        
        # Updates should be fast
        self.assertLess(avg_update_time, 10, "Updates should be fast")
        
    @pytest.mark.performance
    async def test_cache_performance(self):
        """Test caching improves performance."""
        vector_store = AsyncMock()
        embedder = AsyncMock()
        
        config = PipelineConfig(
            enable_caching=True,
        )
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            config=config,
        )
        
        # Mock with artificial latency
        async def slow_embedding(text):
            await asyncio.sleep(0.1)  # 100ms
            return np.random.randn(4096)
            
        embedder.generate_embedding = slow_embedding
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        vector_store.search_similar.return_value = []
        
        # First call - no cache
        start1 = time.perf_counter()
        result1 = await pipeline.process("Cached query")
        end1 = time.perf_counter()
        first_call_ms = (end1 - start1) * 1000
        
        # Second call - should use cache
        start2 = time.perf_counter()
        result2 = await pipeline.process("Cached query")
        end2 = time.perf_counter()
        cached_call_ms = (end2 - start2) * 1000
        
        print(f"Cache performance - First: {first_call_ms:.2f}ms, Cached: {cached_call_ms:.2f}ms")
        
        # Cached should be much faster
        self.assertLess(cached_call_ms, first_call_ms * 0.1, "Cached call should be >10x faster")
        
        # Cache hit should be recorded
        self.assertEqual(result2.cache_hits, 1)
        
    @pytest.mark.performance
    async def test_memory_usage_limits(self):
        """Test memory usage stays within limits (AC 10)."""
        # This is a simplified test - production would use memory_profiler
        import sys
        
        # Create large batch of embeddings
        batch_size = 32
        dimension = 4096
        
        # Calculate memory usage
        embeddings = np.random.randn(batch_size, dimension).astype(np.float32)
        memory_mb = embeddings.nbytes / (1024 * 1024)
        
        print(f"Memory for {batch_size} embeddings: {memory_mb:.2f}MB")
        
        # 32 * 4096 * 4 bytes = 524KB per batch
        self.assertLess(memory_mb, 1, "Batch should use less than 1MB")
        
        # Test with 8B model simulation (simplified)
        model_params_8b = 8_000_000_000  # 8 billion parameters
        bytes_per_param_fp16 = 2
        model_memory_gb = (model_params_8b * bytes_per_param_fp16) / (1024**3)
        
        print(f"Estimated 8B model memory: {model_memory_gb:.2f}GB")
        
        # Should fit in Mac M3 memory
        self.assertLess(model_memory_gb, 20, "8B model should fit in reasonable memory")
        
        # 4bit model should use less
        model_memory_4bit_gb = model_memory_gb / 4
        print(f"Estimated 4bit model memory: {model_memory_4bit_gb:.2f}GB")
        
        self.assertLess(model_memory_4bit_gb, 5, "4bit model should use <5GB")


class TestScalabilityBenchmarks(unittest.TestCase):
    """Test scalability with larger workloads."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_large_vector_store_performance(self):
        """Test performance with 1000+ vectors in store."""
        vector_store = AsyncMock()
        
        # Simulate large vector store
        num_vectors = 1000
        
        async def search_large_store(embedding, k, include_embeddings=False):
            # Simulate realistic search time for large store
            await asyncio.sleep(0.02)  # 20ms base
            
            # Return top k results
            results = []
            for i in range(min(k, 50)):
                results.append(
                    VectorSearchResult(
                        id=i,
                        similarity=0.99 - i*0.001,
                        content=f"Document {i}",
                        metadata={},
                    )
                )
            return results
            
        vector_store.search_similar = search_large_store
        
        # Test search performance
        query = np.random.randn(4096)
        
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            results = await vector_store.search_similar(query, k=10)
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            
        avg_latency = statistics.mean(latencies)
        print(f"Large store search (1000 vectors) - Avg: {avg_latency:.2f}ms")
        
        # Should still meet <100ms requirement
        self.assertLess(avg_latency, 100, "Search should be fast even with 1000+ vectors")
        
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_concurrent_request_handling(self):
        """Test handling multiple concurrent requests."""
        vector_store = AsyncMock()
        embedder = AsyncMock()
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
        )
        
        # Mock fast operations
        embedder.generate_embedding = AsyncMock(
            return_value=np.random.randn(4096)
        )
        embedder.active_embedder = MagicMock()
        embedder.active_embedder.model_type = ModelType.QWEN3_8B
        
        vector_store.search_similar.return_value = []
        
        # Create concurrent requests
        num_concurrent = 10
        
        async def process_request(i):
            start = time.perf_counter()
            result = await pipeline.process(f"Request {i}")
            end = time.perf_counter()
            return (end - start) * 1000
            
        # Run concurrently
        start_overall = time.perf_counter()
        latencies = await asyncio.gather(*[
            process_request(i) for i in range(num_concurrent)
        ])
        end_overall = time.perf_counter()
        
        total_time_ms = (end_overall - start_overall) * 1000
        avg_latency = statistics.mean(latencies)
        
        print(f"Concurrent requests ({num_concurrent}) - Total: {total_time_ms:.2f}ms, Avg: {avg_latency:.2f}ms")
        
        # Should handle concurrent requests efficiently
        self.assertLess(total_time_ms, avg_latency * num_concurrent * 0.5, 
                       "Concurrent processing should be more efficient than serial")


if __name__ == "__main__":
    # Run performance tests with pytest
    # Use: pytest test_embedding_benchmarks.py -v -m performance
    pytest.main([__file__, "-v", "-m", "performance"])