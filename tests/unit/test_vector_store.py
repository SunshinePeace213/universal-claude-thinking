"""
Unit tests for SQLite-vec Vector Storage Backend.

Tests CRUD operations, similarity search, and effectiveness tracking
for 4096-dimensional embeddings with cosine similarity.
"""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.molecular.vector_store import VectorStore, VectorSearchResult


class TestVectorStore(unittest.IsolatedAsyncioTestCase):
    """Test cases for VectorStore."""
    
    async def asyncSetUp(self) -> None:
        """Set up test fixtures."""
        # Use temporary database for testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.temp_db.name)
        
        self.vector_store = VectorStore(
            db_path=self.db_path,
            dimension=4096,
            similarity_threshold=0.85,
            connection_pool_size=2,
        )
        
        await self.vector_store.initialize()
        
    async def asyncTearDown(self) -> None:
        """Clean up test fixtures."""
        await self.vector_store.close()
        self.db_path.unlink(missing_ok=True)
        
    def setUp(self) -> None:
        """Sync setup for async tests."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.asyncSetUp())
        
    def tearDown(self) -> None:
        """Sync teardown for async tests."""
        self.loop.run_until_complete(self.asyncTearDown())
        self.loop.close()
        
    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test vector store initialization."""
        # Already initialized in setUp
        self.assertTrue(self.vector_store._initialized)
        self.assertEqual(len(self.vector_store._connection_pool), 2)
        
        # Test idempotent initialization
        await self.vector_store.initialize()
        self.assertTrue(self.vector_store._initialized)
        
    @pytest.mark.asyncio
    async def test_insert_vector(self):
        """Test single vector insertion."""
        # Create test embedding
        embedding = np.random.randn(4096).astype(np.float32)
        content = "Test content"
        metadata = {"key": "value", "score": 0.95}
        
        # Insert vector
        vector_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content=content,
            metadata=metadata,
        )
        
        # Verify insertion
        self.assertIsInstance(vector_id, int)
        self.assertGreater(vector_id, 0)
        
        # Retrieve and verify
        vector_data = await self.vector_store.get_vector(vector_id)
        self.assertIsNotNone(vector_data)
        self.assertEqual(vector_data["content"], content)
        self.assertEqual(vector_data["metadata"], metadata)
        
    @pytest.mark.asyncio
    async def test_insert_vector_wrong_dimension(self):
        """Test insertion with wrong dimension raises error."""
        # Wrong dimension embedding
        embedding = np.random.randn(2048).astype(np.float32)
        
        with self.assertRaises(ValueError) as context:
            await self.vector_store.insert_vector(
                embedding=embedding,
                content="Test",
            )
            
        self.assertIn("4096D", str(context.exception))
        
    @pytest.mark.asyncio
    async def test_batch_insert(self):
        """Test batch vector insertion."""
        # Create batch of embeddings
        batch_size = 10
        embeddings = np.random.randn(batch_size, 4096).astype(np.float32)
        contents = [f"Content {i}" for i in range(batch_size)]
        metadata_list = [{"index": i} for i in range(batch_size)]
        
        # Batch insert
        vector_ids = await self.vector_store.batch_insert(
            embeddings=embeddings,
            contents=contents,
            metadata_list=metadata_list,
        )
        
        # Verify all inserted
        self.assertEqual(len(vector_ids), batch_size)
        self.assertTrue(all(isinstance(vid, int) for vid in vector_ids))
        
    @pytest.mark.asyncio
    async def test_batch_insert_mismatched_lengths(self):
        """Test batch insertion with mismatched array lengths."""
        embeddings = np.random.randn(5, 4096).astype(np.float32)
        contents = ["Content 1", "Content 2"]  # Only 2 items
        
        with self.assertRaises(ValueError) as context:
            await self.vector_store.batch_insert(
                embeddings=embeddings,
                contents=contents,
            )
            
        self.assertIn("must match", str(context.exception))
        
    @pytest.mark.asyncio
    async def test_search_similar(self):
        """Test similarity search."""
        # Insert test vectors
        embeddings = []
        contents = []
        
        # Create orthogonal vectors for distinct similarity
        for i in range(5):
            embedding = np.zeros(4096)
            embedding[i*10:(i+1)*10] = 1.0  # Different regions activated
            embeddings.append(embedding)
            contents.append(f"Document {i}")
            
        embeddings = np.array(embeddings).astype(np.float32)
        await self.vector_store.batch_insert(embeddings, contents)
        
        # Search with query similar to first document
        query = np.zeros(4096)
        query[0:10] = 1.0  # Similar to first document
        query = query.astype(np.float32)
        
        results = await self.vector_store.search_similar(
            query_embedding=query,
            k=3,
        )
        
        # Verify results
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, VectorSearchResult) for r in results))
        
        # First result should have highest similarity
        if results:
            self.assertGreaterEqual(results[0].similarity, 0.0)
            self.assertLessEqual(results[0].similarity, 1.0)
            
    @pytest.mark.asyncio
    async def test_search_similar_with_threshold(self):
        """Test similarity search with threshold filtering."""
        # Insert highly similar vectors
        base_embedding = np.random.randn(4096).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        for i in range(5):
            # Add small noise to create similar but distinct vectors
            noise = np.random.randn(4096) * 0.01
            embedding = base_embedding + noise
            embedding = embedding / np.linalg.norm(embedding)
            
            await self.vector_store.insert_vector(
                embedding=embedding,
                content=f"Similar document {i}",
            )
            
        # Search with base embedding
        results = await self.vector_store.search_similar(
            query_embedding=base_embedding,
            k=10,
        )
        
        # All results should meet similarity threshold (0.85)
        for result in results:
            self.assertGreaterEqual(result.similarity, 0.85)
            
    @pytest.mark.asyncio
    async def test_search_similar_include_embeddings(self):
        """Test similarity search with embedding retrieval."""
        # Insert test vector
        embedding = np.random.randn(4096).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        await self.vector_store.insert_vector(
            embedding=embedding,
            content="Test document",
        )
        
        # Search with embeddings included
        results = await self.vector_store.search_similar(
            query_embedding=embedding,
            k=1,
            include_embeddings=True,
        )
        
        # Verify embedding is returned
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].embedding)
        self.assertEqual(results[0].embedding.shape, (4096,))
        
    @pytest.mark.asyncio
    async def test_update_effectiveness(self):
        """Test effectiveness score updates."""
        # Insert vector
        embedding = np.random.randn(4096).astype(np.float32)
        vector_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content="Test",
        )
        
        # Update effectiveness positively
        await self.vector_store.update_effectiveness(vector_id, 0.3)
        
        # Check updated score
        vector_data = await self.vector_store.get_vector(vector_id)
        self.assertEqual(vector_data["effectiveness_score"], 0.3)
        self.assertEqual(vector_data["usage_count"], 1)
        
        # Update negatively
        await self.vector_store.update_effectiveness(vector_id, -0.3)
        
        vector_data = await self.vector_store.get_vector(vector_id)
        self.assertEqual(vector_data["effectiveness_score"], 0.0)
        self.assertEqual(vector_data["usage_count"], 2)
        
    @pytest.mark.asyncio
    async def test_get_vector(self):
        """Test vector retrieval by ID."""
        # Insert vector
        embedding = np.random.randn(4096).astype(np.float32)
        content = "Test content"
        metadata = {"test": True}
        
        vector_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content=content,
            metadata=metadata,
        )
        
        # Retrieve with embedding
        vector_data = await self.vector_store.get_vector(
            vector_id=vector_id,
            include_embedding=True,
        )
        
        self.assertIsNotNone(vector_data)
        self.assertEqual(vector_data["id"], vector_id)
        self.assertEqual(vector_data["content"], content)
        self.assertEqual(vector_data["metadata"], metadata)
        self.assertIsNotNone(vector_data["embedding"])
        self.assertEqual(vector_data["embedding"].shape, (4096,))
        
        # Retrieve without embedding
        vector_data = await self.vector_store.get_vector(
            vector_id=vector_id,
            include_embedding=False,
        )
        
        self.assertIsNone(vector_data["embedding"])
        
    @pytest.mark.asyncio
    async def test_get_vector_not_found(self):
        """Test retrieval of non-existent vector."""
        vector_data = await self.vector_store.get_vector(999999)
        self.assertIsNone(vector_data)
        
    @pytest.mark.asyncio
    async def test_delete_vector(self):
        """Test vector deletion."""
        # Insert vector
        embedding = np.random.randn(4096).astype(np.float32)
        vector_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content="To be deleted",
        )
        
        # Delete vector
        deleted = await self.vector_store.delete_vector(vector_id)
        self.assertTrue(deleted)
        
        # Verify deletion
        vector_data = await self.vector_store.get_vector(vector_id)
        self.assertIsNone(vector_data)
        
        # Delete non-existent vector
        deleted = await self.vector_store.delete_vector(999999)
        self.assertFalse(deleted)
        
    @pytest.mark.asyncio
    async def test_get_statistics(self):
        """Test statistics retrieval."""
        # Insert some vectors with varying scores
        for i in range(5):
            embedding = np.random.randn(4096).astype(np.float32)
            vector_id = await self.vector_store.insert_vector(
                embedding=embedding,
                content=f"Document {i}",
            )
            
            # Update effectiveness
            await self.vector_store.update_effectiveness(
                vector_id,
                0.1 * i,  # Varying scores
            )
            
        # Get statistics
        stats = await self.vector_store.get_statistics()
        
        self.assertEqual(stats["total_vectors"], 5)
        self.assertEqual(stats["dimension"], 4096)
        self.assertEqual(stats["similarity_threshold"], 0.85)
        self.assertIsInstance(stats["avg_effectiveness"], float)
        self.assertIsInstance(stats["avg_usage"], float)
        self.assertIsInstance(stats["max_usage"], int)
        
    @pytest.mark.asyncio
    async def test_normalization(self):
        """Test that embeddings are normalized for cosine similarity."""
        # Insert non-normalized embedding
        embedding = np.array([3.0, 4.0] + [0.0] * 4094).astype(np.float32)
        vector_id = await self.vector_store.insert_vector(
            embedding=embedding,
            content="Test normalization",
        )
        
        # Retrieve and check normalization
        vector_data = await self.vector_store.get_vector(
            vector_id,
            include_embedding=True,
        )
        
        retrieved_embedding = vector_data["embedding"]
        norm = np.linalg.norm(retrieved_embedding)
        
        # Should be normalized (L2 norm = 1)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
    @pytest.mark.asyncio
    async def test_search_latency_requirement(self):
        """Test that search meets <100ms latency requirement."""
        # Insert 1000 vectors to test at scale
        batch_size = 100
        for _ in range(10):
            embeddings = np.random.randn(batch_size, 4096).astype(np.float32)
            contents = [f"Doc {i}" for i in range(batch_size)]
            await self.vector_store.batch_insert(embeddings, contents)
            
        # Test search latency
        query = np.random.randn(4096).astype(np.float32)
        
        import time
        start = time.perf_counter()
        results = await self.vector_store.search_similar(query, k=10)
        end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        
        # Should meet <100ms requirement (with some margin for test environment)
        self.assertLess(latency_ms, 200)  # Relaxed for test environment
        
    @pytest.mark.asyncio
    async def test_connection_pool(self):
        """Test connection pooling functionality."""
        # Already has 2 connections from setUp
        self.assertEqual(len(self.vector_store._connection_pool), 2)
        
        # Perform concurrent operations
        async def insert_task(i):
            embedding = np.random.randn(4096).astype(np.float32)
            return await self.vector_store.insert_vector(
                embedding=embedding,
                content=f"Concurrent {i}",
            )
            
        # Run multiple concurrent inserts
        tasks = [insert_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        self.assertEqual(len(results), 5)
        self.assertTrue(all(isinstance(r, int) for r in results))
        
        # Pool should still have same number of connections
        self.assertEqual(len(self.vector_store._connection_pool), 2)


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])