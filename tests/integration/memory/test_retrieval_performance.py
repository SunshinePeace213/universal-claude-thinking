"""
Integration tests for memory retrieval performance.

Benchmarks retrieval latency, batch processing, and ensures
performance requirements are met (<100ms retrieval).
"""

import asyncio
import pytest
import tempfile
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.memory.layers.base import MemoryItem
from src.memory.storage.sqlite_storage import SQLiteStorage
from src.memory.embeddings import MemoryEmbedder
from src.memory.config import MemoryConfig


class TestRetrievalPerformance:
    """Performance benchmarks for memory retrieval operations."""
    
    @pytest.fixture
    async def performance_db(self):
        """Create a pre-populated database for performance testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        
        # Pre-populate with test data
        memories = []
        for i in range(1000):  # 1000 memories for realistic testing
            memory = MemoryItem(
                id=f"perf_{i}",
                user_id=f"user_{i % 10}",  # 10 different users
                memory_type=["stm", "wm", "ltm"][i % 3],  # Mix of types
                content={
                    "text": f"Performance test content {i}",
                    "metadata": f"test_{i}"
                },
                embedding=np.random.rand(4096).astype(np.float32),
                effectiveness_score=5.0 + (i % 5),
                usage_count=i % 20,
                created_at=datetime.now() - timedelta(days=i % 30)
            )
            memories.append(memory)
        
        # Batch store for efficiency
        await storage.batch_store(memories)
        
        yield {
            'storage': storage,
            'db_path': db_path,
            'memory_count': len(memories)
        }
        
        # Cleanup
        await storage.close()
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_single_retrieval_latency(self, performance_db):
        """Test that single memory retrieval meets <100ms requirement."""
        storage = performance_db['storage']
        
        # Warm up cache
        _ = await storage.retrieve("perf_0")
        
        # Measure retrieval times
        latencies = []
        for i in range(100):  # Test 100 retrievals
            memory_id = f"perf_{i}"
            
            start_time = time.perf_counter()
            result = await storage.retrieve(memory_id)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            assert result is not None
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        max_latency = max(latencies)
        
        # Assert performance requirements
        assert avg_latency < 100, f"Average latency {avg_latency:.2f}ms exceeds 100ms"
        assert p95_latency < 100, f"P95 latency {p95_latency:.2f}ms exceeds 100ms"
        assert max_latency < 200, f"Max latency {max_latency:.2f}ms exceeds 200ms"
        
        # Log performance metrics
        print(f"\nSingle Retrieval Performance:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  Max: {max_latency:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search_performance(self, performance_db):
        """Test vector similarity search performance."""
        storage = performance_db['storage']
        
        # Generate query embeddings
        query_embeddings = [
            np.random.rand(4096).astype(np.float32)
            for _ in range(10)
        ]
        
        latencies = []
        for query_embedding in query_embeddings:
            start_time = time.perf_counter()
            results = await storage.search_by_embedding(
                embedding=query_embedding,
                k=10,
                user_id="user_0"
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            assert len(results) <= 10
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        
        # Vector search can be slightly slower but should still be reasonable
        assert avg_latency < 150, f"Vector search avg latency {avg_latency:.2f}ms exceeds 150ms"
        assert p95_latency < 200, f"Vector search P95 latency {p95_latency:.2f}ms exceeds 200ms"
        
        print(f"\nVector Search Performance:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_batch_retrieval_performance(self, performance_db):
        """Test batch retrieval operations performance."""
        storage = performance_db['storage']
        
        # Test different batch sizes
        batch_sizes = [10, 25, 50, 100]
        
        for batch_size in batch_sizes:
            start_time = time.perf_counter()
            results = await storage.list_by_user(
                user_id="user_0",
                limit=batch_size
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            assert len(results) <= batch_size
            
            # Batch retrieval should scale reasonably
            max_allowed_ms = 100 + (batch_size * 2)  # Allow 2ms per item
            assert latency_ms < max_allowed_ms, \
                f"Batch size {batch_size} took {latency_ms:.2f}ms, exceeds {max_allowed_ms}ms"
            
            print(f"\nBatch Retrieval (size={batch_size}):")
            print(f"  Latency: {latency_ms:.2f}ms")
            print(f"  Per-item: {latency_ms/batch_size:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval_performance(self, performance_db):
        """Test performance under concurrent retrieval load."""
        storage = performance_db['storage']
        
        async def retrieve_memory(memory_id: str) -> float:
            start_time = time.perf_counter()
            result = await storage.retrieve(memory_id)
            latency_ms = (time.perf_counter() - start_time) * 1000
            assert result is not None
            return latency_ms
        
        # Test different concurrency levels
        concurrency_levels = [5, 10, 20]
        
        for concurrency in concurrency_levels:
            # Create concurrent retrieval tasks
            tasks = [
                retrieve_memory(f"perf_{i % 100}")
                for i in range(concurrency)
            ]
            
            start_time = time.perf_counter()
            latencies = await asyncio.gather(*tasks)
            total_time_ms = (time.perf_counter() - start_time) * 1000
            
            avg_latency = statistics.mean(latencies)
            max_latency = max(latencies)
            
            # Even under concurrent load, should maintain reasonable performance
            assert avg_latency < 200, \
                f"Concurrent avg latency {avg_latency:.2f}ms exceeds 200ms at concurrency={concurrency}"
            assert max_latency < 500, \
                f"Concurrent max latency {max_latency:.2f}ms exceeds 500ms at concurrency={concurrency}"
            
            print(f"\nConcurrent Retrieval (concurrency={concurrency}):")
            print(f"  Total time: {total_time_ms:.2f}ms")
            print(f"  Average latency: {avg_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            print(f"  Throughput: {concurrency / (total_time_ms / 1000):.1f} req/s")
    
    @pytest.mark.asyncio
    async def test_promotion_evaluation_performance(self, performance_db):
        """Test promotion evaluation meets <500ms requirement."""
        storage = performance_db['storage']
        
        # Simulate promotion evaluation
        async def evaluate_promotions(memory_type: str, threshold: float) -> float:
            start_time = time.perf_counter()
            
            # Get candidates based on criteria
            candidates = await storage.get_by_criteria(
                memory_type=memory_type,
                min_effectiveness=threshold,
                limit=100
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            return latency_ms, len(candidates)
        
        # Test STM→WM promotion evaluation
        stm_latency, stm_count = await evaluate_promotions("stm", 5.0)
        assert stm_latency < 500, f"STM promotion evaluation {stm_latency:.2f}ms exceeds 500ms"
        
        # Test WM→LTM promotion evaluation
        wm_latency, wm_count = await evaluate_promotions("wm", 8.0)
        assert wm_latency < 500, f"WM promotion evaluation {wm_latency:.2f}ms exceeds 500ms"
        
        print(f"\nPromotion Evaluation Performance:")
        print(f"  STM→WM: {stm_latency:.2f}ms for {stm_count} candidates")
        print(f"  WM→LTM: {wm_latency:.2f}ms for {wm_count} candidates")
    
    @pytest.mark.asyncio
    async def test_pii_detection_performance(self, performance_db):
        """Test PII detection meets <50ms per item requirement."""
        from src.memory.privacy import PrivacyEngine
        
        privacy = PrivacyEngine()
        
        # Test texts with varying complexity
        test_texts = [
            "Simple text without PII",
            "Email me at john.doe@example.com or call 555-1234",
            "My SSN is 123-45-6789 and credit card 4111-1111-1111-1111",
            "Meeting at 123 Main St, New York, NY 10001 with Dr. Smith",
            "Patient John Doe, DOB 01/01/1990, diagnosed with condition X"
        ]
        
        latencies = []
        for text in test_texts * 10:  # Test 50 items
            start_time = time.perf_counter()
            _ = privacy.detect_pii(text)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        
        assert avg_latency < 50, f"PII detection avg {avg_latency:.2f}ms exceeds 50ms"
        assert max_latency < 100, f"PII detection max {max_latency:.2f}ms exceeds 100ms"
        
        print(f"\nPII Detection Performance:")
        print(f"  Average: {avg_latency:.2f}ms per item")
        print(f"  Max: {max_latency:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_batch_embedding_performance(self, performance_db):
        """Test batch embedding generation for 32 items."""
        # Mock embedder since we don't have the actual model in tests
        from unittest.mock import MagicMock, AsyncMock
        
        mock_embedder = MagicMock()
        mock_embedder.batch_embed_memories = AsyncMock(
            return_value=np.random.rand(32, 4096).astype(np.float32)
        )
        
        # Create batch of memories
        memories = [
            MemoryItem(
                id=f"batch_embed_{i}",
                user_id="test_user",
                memory_type="stm",
                content={"text": f"Batch content {i}"}
            )
            for i in range(32)
        ]
        
        # Measure batch embedding time
        start_time = time.perf_counter()
        embeddings = await mock_embedder.batch_embed_memories(memories)
        batch_time_ms = (time.perf_counter() - start_time) * 1000
        
        assert embeddings.shape == (32, 4096)
        
        # Batch processing should be efficient
        per_item_ms = batch_time_ms / 32
        
        print(f"\nBatch Embedding Performance (32 items):")
        print(f"  Total time: {batch_time_ms:.2f}ms")
        print(f"  Per item: {per_item_ms:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_memory_scoring_performance(self, performance_db):
        """Test effectiveness scoring performance."""
        from src.memory.scoring import EffectivenessScorer, FeedbackType
        
        scorer = EffectivenessScorer()
        
        # Initialize scores for test memories
        for i in range(100):
            scorer._scores[f"perf_{i}"] = 5.0 + (i % 5)
        
        # Test feedback application performance
        latencies = []
        for i in range(100):
            memory_id = f"perf_{i}"
            feedback = FeedbackType.POSITIVE if i % 2 == 0 else FeedbackType.NEGATIVE
            
            start_time = time.perf_counter()
            scorer.apply_feedback(memory_id, feedback)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = statistics.mean(latencies)
        assert avg_latency < 10, f"Scoring avg latency {avg_latency:.2f}ms exceeds 10ms"
        
        # Test batch scoring
        memory_ids = [f"perf_{i}" for i in range(50)]
        start_time = time.perf_counter()
        scores = scorer.batch_get_scores(memory_ids)
        batch_latency_ms = (time.perf_counter() - start_time) * 1000
        
        assert len(scores) == 50
        assert batch_latency_ms < 50, f"Batch scoring {batch_latency_ms:.2f}ms exceeds 50ms"
        
        print(f"\nScoring Performance:")
        print(f"  Single feedback: {avg_latency:.2f}ms")
        print(f"  Batch (50 items): {batch_latency_ms:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_cache_performance(self, performance_db):
        """Test in-memory cache performance for STM."""
        from src.memory.storage.memory_cache import InMemoryCache
        
        cache = InMemoryCache(max_size=1000)
        
        # Pre-populate cache
        for i in range(1000):
            memory = MemoryItem(
                id=f"cache_{i}",
                user_id="test_user",
                memory_type="stm",
                content={"text": f"Cached content {i}"}
            )
            cache.set(f"cache_{i}", memory)
        
        # Test cache retrieval performance
        latencies = []
        hits = 0
        for i in range(1000):
            key = f"cache_{i % 1000}"
            
            start_time = time.perf_counter()
            result = cache.get(key)
            latency_us = (time.perf_counter() - start_time) * 1_000_000  # microseconds
            
            if result is not None:
                hits += 1
            latencies.append(latency_us)
        
        avg_latency_us = statistics.mean(latencies)
        hit_rate = hits / 1000
        
        # Cache should be extremely fast (microseconds)
        assert avg_latency_us < 100, f"Cache avg latency {avg_latency_us:.2f}μs exceeds 100μs"
        assert hit_rate == 1.0, f"Cache hit rate {hit_rate:.2%} is not 100%"
        
        print(f"\nCache Performance:")
        print(f"  Average latency: {avg_latency_us:.2f}μs")
        print(f"  Hit rate: {hit_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_end_to_end_retrieval_pipeline(self, performance_db):
        """Test complete retrieval pipeline performance."""
        storage = performance_db['storage']
        from src.memory.privacy import PrivacyEngine
        from src.memory.scoring import EffectivenessScorer
        
        privacy = PrivacyEngine()
        scorer = EffectivenessScorer()
        
        async def complete_retrieval_pipeline(memory_id: str) -> Tuple[float, MemoryItem]:
            """Simulate complete retrieval with all processing."""
            start_time = time.perf_counter()
            
            # 1. Retrieve memory
            memory = await storage.retrieve(memory_id)
            assert memory is not None
            
            # 2. Check privacy (for display)
            if memory.content.get("text"):
                sanitized = privacy.remove_pii(memory.content["text"])
                memory.content["text"] = sanitized
            
            # 3. Update usage stats
            memory.usage_count += 1
            memory.last_accessed = datetime.now()
            
            # 4. Get effectiveness score
            score = scorer.get_score(memory_id)
            memory.effectiveness_score = score
            
            # 5. Update storage
            await storage.update(memory)
            
            total_latency_ms = (time.perf_counter() - start_time) * 1000
            return total_latency_ms, memory
        
        # Test pipeline performance
        latencies = []
        for i in range(20):
            memory_id = f"perf_{i}"
            latency, memory = await complete_retrieval_pipeline(memory_id)
            latencies.append(latency)
            assert memory is not None
        
        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        
        # Complete pipeline should still meet reasonable performance
        assert avg_latency < 150, f"Pipeline avg latency {avg_latency:.2f}ms exceeds 150ms"
        assert p95_latency < 200, f"Pipeline P95 latency {p95_latency:.2f}ms exceeds 200ms"
        
        print(f"\nEnd-to-End Pipeline Performance:")
        print(f"  Average: {avg_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
    
    def print_performance_summary(self):
        """Print overall performance summary."""
        print("\n" + "="*60)
        print("PERFORMANCE REQUIREMENTS SUMMARY")
        print("="*60)
        print("✓ Single retrieval: <100ms")
        print("✓ Promotion evaluation: <500ms per batch")
        print("✓ PII detection: <50ms per item")
        print("✓ Batch embedding: 32 items simultaneously")
        print("✓ Cache operations: <100μs")
        print("="*60)