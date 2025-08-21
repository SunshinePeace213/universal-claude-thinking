"""
Unit tests for memory embedding integration.

Tests the MemoryEmbedder class that wraps Qwen3-Embedding-8B
for memory-specific vector generation and retrieval.
"""

import asyncio
import numpy as np
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path

from src.memory.layers.base import MemoryItem


class TestMemoryEmbedder:
    """Test suite for MemoryEmbedder class."""
    
    @pytest.fixture
    def mock_qwen_embedder(self):
        """Create a mock Qwen8BEmbedder."""
        mock = MagicMock()
        mock.initialize = AsyncMock()
        mock.generate_embedding = AsyncMock(
            return_value=np.random.rand(1, 4096).astype(np.float32)
        )
        mock.batch_generate = AsyncMock(
            return_value=np.random.rand(32, 4096).astype(np.float32)
        )
        mock.close = AsyncMock()
        mock.embedding_dim = 4096
        mock.batch_size = 32
        mock.device = "mps"
        return mock
    
    @pytest.fixture
    def sample_memory_items(self):
        """Create sample memory items for testing."""
        items = []
        for i in range(5):
            items.append(MemoryItem(
                id=f"mem_{i}",
                user_id="test_user",
                memory_type="stm" if i < 2 else "wm",  # Use string instead of enum
                content={"text": f"Memory content {i}"},
                metadata={"source": "test"},
                effectiveness_score=5.0 + i * 0.5,
                usage_count=i
            ))
        return items
    
    @pytest.mark.asyncio
    async def test_embedder_initialization(self, mock_qwen_embedder):
        """Test MemoryEmbedder initialization with model loading."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(
                model_path="embedding/Qwen3-Embedding-8B",
                cache_size=100
            )
            
            await embedder.initialize()
            
            assert embedder.model is not None
            assert embedder.cache_size == 100
            assert embedder.embedding_dim == 4096
            mock_qwen_embedder.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_single_memory_embedding(self, mock_qwen_embedder, sample_memory_items):
        """Test embedding generation for a single memory item."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder()
            await embedder.initialize()
            
            memory = sample_memory_items[0]
            embedding = await embedder.generate_memory_embedding(memory)
            
            # Check embedding shape and type
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (4096,)
            
            # Verify instruction prefix was used
            mock_qwen_embedder.generate_embedding.assert_called_once()
            call_args = mock_qwen_embedder.generate_embedding.call_args
            assert "Recent interaction:" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_memory_instruction_prefixes(self, mock_qwen_embedder):
        """Test that correct instruction prefixes are used for different memory types."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder()
            await embedder.initialize()
            
            # Test STM prefix
            stm_memory = MemoryItem(
                id="stm_1", user_id="test", memory_type="stm",  # Use string instead of enum
                content={"text": "STM content"}
            )
            await embedder.generate_memory_embedding(stm_memory)
            assert "Recent interaction:" in mock_qwen_embedder.generate_embedding.call_args[0][0]
            
            # Test WM prefix
            wm_memory = MemoryItem(
                id="wm_1", user_id="test", memory_type="wm",  # Use string instead of enum
                content={"text": "WM content"}
            )
            await embedder.generate_memory_embedding(wm_memory)
            assert "Working context:" in mock_qwen_embedder.generate_embedding.call_args[0][0]
            
            # Test LTM prefix
            ltm_memory = MemoryItem(
                id="ltm_1", user_id="test", memory_type="ltm",  # Use string instead of enum
                content={"text": "LTM content"}
            )
            await embedder.generate_memory_embedding(ltm_memory)
            assert "User preference:" in mock_qwen_embedder.generate_embedding.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self, mock_qwen_embedder, sample_memory_items):
        """Test batch processing of up to 32 memory items."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            # Create 32 memory items
            memories = []
            for i in range(32):
                memories.append(MemoryItem(
                    id=f"mem_{i}", user_id="test", memory_type="stm",  # Use string instead of enum
                    content={"text": f"Content {i}"}
                ))
            
            embedder = MemoryEmbedder(batch_size=32)
            await embedder.initialize()
            
            embeddings = await embedder.batch_embed_memories(memories)
            
            assert embeddings.shape == (32, 4096)
            mock_qwen_embedder.batch_generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_embedding_cache_functionality(self, mock_qwen_embedder):
        """Test that embedding cache reduces redundant computations."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(cache_size=10)
            await embedder.initialize()
            
            memory = MemoryItem(
                id="cached_mem", user_id="test", memory_type="stm",  # Use string instead of enum
                content={"text": "Cached content"}
            )
            
            # First call - should generate embedding
            embedding1 = await embedder.generate_memory_embedding(memory)
            assert mock_qwen_embedder.generate_embedding.call_count == 1
            
            # Second call - should use cache
            embedding2 = await embedder.generate_memory_embedding(memory)
            assert mock_qwen_embedder.generate_embedding.call_count == 1  # Still 1
            
            # Embeddings should be identical
            np.testing.assert_array_equal(embedding1, embedding2)
    
    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, mock_qwen_embedder):
        """Test cosine similarity search for memory retrieval."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder()
            await embedder.initialize()
            
            # Create query and candidate embeddings
            query_embedding = np.random.rand(4096).astype(np.float32)
            candidate_embeddings = np.random.rand(10, 4096).astype(np.float32)
            
            # Normalize for cosine similarity
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            candidate_embeddings = candidate_embeddings / np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True
            )
            
            # Calculate similarities
            similarities = embedder.calculate_similarities(query_embedding, candidate_embeddings)
            
            assert len(similarities) == 10
            assert all(0 <= sim <= 1 for sim in similarities)
    
    @pytest.mark.asyncio
    async def test_retrieval_performance(self, mock_qwen_embedder):
        """Test that retrieval meets <100ms latency requirement."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(cache_size=100)
            await embedder.initialize()
            
            # Pre-cache some embeddings
            for i in range(10):
                memory = MemoryItem(
                    id=f"perf_mem_{i}", user_id="test", memory_type="stm",  # Use string instead of enum
                    content={"text": f"Performance test {i}"}
                )
                await embedder.generate_memory_embedding(memory)
            
            # Test retrieval from cache
            memory = MemoryItem(
                id="perf_mem_5", user_id="test", memory_type="stm",  # Use string instead of enum
                content={"text": "Performance test 5"}
            )
            
            start_time = time.time()
            embedding = await embedder.generate_memory_embedding(memory)
            retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
            
            assert retrieval_time < 100, f"Retrieval took {retrieval_time}ms, exceeds 100ms limit"
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_model(self):
        """Test proper error handling when model files are missing."""
        with patch('src.memory.embeddings.Qwen8BEmbedder') as mock_class:
            mock_instance = MagicMock()
            mock_instance.initialize = AsyncMock(
                side_effect=FileNotFoundError("Model not found")
            )
            mock_class.return_value = mock_instance
            
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(model_path="nonexistent/path")
            
            with pytest.raises(FileNotFoundError, match="Model not found"):
                await embedder.initialize()
    
    @pytest.mark.asyncio
    async def test_mac_m3_mps_optimization(self, mock_qwen_embedder):
        """Test that Mac M3 MPS acceleration is properly utilized."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(device="mps")
            await embedder.initialize()
            
            assert embedder.device == "mps"
            assert embedder.batch_size <= 32  # Mac M3 optimization
    
    @pytest.mark.asyncio
    async def test_memory_content_extraction(self, mock_qwen_embedder):
        """Test extraction of text content from memory items."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder()
            await embedder.initialize()
            
            # Test with different content structures
            memory_with_text = MemoryItem(
                id="m1", user_id="test", memory_type="stm",  # Use string instead of enum
                content={"text": "Direct text content"}
            )
            
            memory_with_nested = MemoryItem(
                id="m2", user_id="test", memory_type="wm",  # Use string instead of enum
                content={"data": {"message": "Nested content"}}
            )
            
            text1 = embedder._extract_text_content(memory_with_text)
            text2 = embedder._extract_text_content(memory_with_nested)
            
            assert text1 == "Direct text content"
            assert "Nested content" in text2
    
    @pytest.mark.asyncio
    async def test_embedding_normalization(self, mock_qwen_embedder):
        """Test that embeddings are properly normalized for cosine similarity."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder(normalize_embeddings=True)
            await embedder.initialize()
            
            memory = MemoryItem(
                id="norm_test", user_id="test", memory_type="stm",  # Use string instead of enum
                content={"text": "Normalization test"}
            )
            
            embedding = await embedder.generate_memory_embedding(memory)
            
            # Check that embedding is normalized (L2 norm should be 1)
            norm = np.linalg.norm(embedding)
            np.testing.assert_almost_equal(norm, 1.0, decimal=5)
    
    @pytest.mark.asyncio
    async def test_concurrent_embedding_generation(self, mock_qwen_embedder):
        """Test thread-safe concurrent embedding generation."""
        with patch('src.memory.embeddings.Qwen8BEmbedder', return_value=mock_qwen_embedder):
            from src.memory.embeddings import MemoryEmbedder
            
            embedder = MemoryEmbedder()
            await embedder.initialize()
            
            # Create multiple memories
            memories = [
                MemoryItem(
                    id=f"concurrent_{i}", user_id="test", memory_type="stm",  # Use string instead of enum
                    content={"text": f"Concurrent test {i}"}
                )
                for i in range(10)
            ]
            
            # Generate embeddings concurrently
            tasks = [embedder.generate_memory_embedding(m) for m in memories]
            embeddings = await asyncio.gather(*tasks)
            
            assert len(embeddings) == 10
            assert all(e.shape == (4096,) for e in embeddings)