"""
Integration tests for memory embedding system.

Tests the complete integration of embedding generation with
memory storage, retrieval, and promotion pipelines.
"""

import asyncio
import numpy as np
import pytest
import time
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.memory.layers.base import MemoryItem
from src.memory.layers.stm import ShortTermMemory
from src.memory.layers.wm import WorkingMemory
from src.memory.layers.ltm import LongTermMemory
from src.memory.storage.sqlite_storage import SQLiteStorage
from src.memory.promotion import PromotionPipeline
from src.memory.privacy import PrivacyEngine
from src.memory.scoring import EffectivenessScorer, FeedbackType
from src.memory.config import MemoryConfig, load_config


class TestEmbeddingIntegration:
    """Integration tests for embedding with memory system components."""
    
    @pytest.fixture
    async def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name
        
        # Initialize database schema
        storage = SQLiteStorage(db_path)
        await storage.initialize()
        
        yield db_path
        
        # Cleanup
        await storage.close()
        Path(db_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_embedder(self):
        """Create a mock MemoryEmbedder for integration tests."""
        mock = MagicMock()
        mock.initialize = AsyncMock()
        mock.generate_memory_embedding = AsyncMock(
            return_value=np.random.rand(4096).astype(np.float32)
        )
        mock.batch_embed_memories = AsyncMock(
            return_value=np.random.rand(10, 4096).astype(np.float32)
        )
        mock.calculate_similarities = Mock(
            return_value=np.array([0.95, 0.85, 0.75, 0.65, 0.55])
        )
        mock.close = AsyncMock()
        return mock
    
    @pytest.fixture
    def memory_config(self):
        """Create test memory configuration."""
        return MemoryConfig(
            stm={'ttl_hours': 2, 'cache_size': 100},
            wm={'ttl_days': 7, 'promotion_threshold': 5.0},
            ltm={'promotion_score': 8.0, 'promotion_uses': 5},
            embedding={
                'model_path': 'embedding/Qwen3-Embedding-8B',
                'dimension': 4096,
                'batch_size': 32,
                'cache_size': 100
            },
            performance={
                'max_retrieval_latency_ms': 100,
                'max_promotion_latency_ms': 500,
                'max_pii_detection_latency_ms': 50
            }
        )
    
    @pytest.mark.asyncio
    async def test_memory_lifecycle_with_embeddings(self, temp_db, mock_embedder, memory_config):
        """Test complete memory lifecycle from creation to retrieval with embeddings."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            # Initialize components
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            stm = ShortTermMemory(
                cache_size=memory_config.stm['cache_size'],
                ttl_hours=memory_config.stm['ttl_hours']
            )
            await stm.initialize()
            
            # Create and store memory with embedding
            memory = MemoryItem(
                id="lifecycle_test",
                user_id="test_user",
                memory_type="stm",  # Use string instead of enum
                content={"text": "Important meeting notes"},
                metadata={"source": "meeting"}
            )
            
            # Generate embedding
            embedding = await mock_embedder.generate_memory_embedding(memory)
            memory.embedding = embedding
            
            # Store memory directly in storage (since STM uses internal cache)
            await storage.store(memory)
            
            # Retrieve by embedding similarity
            query_embedding = np.random.rand(4096).astype(np.float32)
            similar_memories = await storage.search_by_embedding(
                embedding=query_embedding,
                k=5,
                user_id="test_user"
            )
            
            assert len(similar_memories) > 0
            retrieved_memory, similarity = similar_memories[0]
            assert retrieved_memory.id == "lifecycle_test"
    
    @pytest.mark.asyncio
    async def test_promotion_pipeline_with_embeddings(self, temp_db, mock_embedder, memory_config):
        """Test memory promotion from STM to WM with embedding updates."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            # Initialize components
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            scorer = EffectivenessScorer()
            promotion = PromotionPipeline(
                storage=storage,
                scorer=scorer,
                config=memory_config
            )
            
            # Create high-scoring STM memory
            memory = MemoryItem(
                id="promotion_test",
                user_id="test_user",
                memory_type="stm",  # Use string instead of enum
                content={"text": "Valuable information"},
                effectiveness_score=6.0,  # Above WM threshold
                usage_count=3
            )
            
            # Generate and attach embedding
            embedding = await mock_embedder.generate_memory_embedding(memory)
            memory.embedding = embedding
            
            await storage.store(memory)
            
            # Apply positive feedback to increase score
            scorer.apply_feedback(memory.id, FeedbackType.POSITIVE)
            
            # Run promotion evaluation
            promoted = await promotion.evaluate_stm_to_wm()
            
            assert len(promoted) > 0
            assert promoted[0].id == "promotion_test"
            
            # Verify memory type changed
            retrieved = await storage.retrieve("promotion_test")
            assert retrieved.memory_type == "wm"
    
    @pytest.mark.asyncio
    async def test_retrieval_performance_benchmark(self, temp_db, mock_embedder):
        """Test that retrieval with embeddings meets <100ms latency requirement."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            # Pre-populate with memories and embeddings
            memories = []
            for i in range(100):
                memory = MemoryItem(
                    id=f"perf_{i}",
                    user_id="test_user",
                    memory_type="wm",  # Use string instead of enum
                    content={"text": f"Performance test content {i}"},
                    embedding=np.random.rand(4096).astype(np.float32)
                )
                memories.append(memory)
            
            await storage.batch_store(memories)
            
            # Measure retrieval time
            query_embedding = np.random.rand(4096).astype(np.float32)
            
            start_time = time.time()
            results = await storage.search_by_embedding(
                embedding=query_embedding,
                k=10,
                user_id="test_user"
            )
            retrieval_time = (time.time() - start_time) * 1000  # ms
            
            assert retrieval_time < 100, f"Retrieval took {retrieval_time}ms"
            assert len(results) <= 10
    
    @pytest.mark.asyncio
    async def test_privacy_preserved_embeddings(self, mock_embedder):
        """Test that PII is removed before embedding generation."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            privacy = PrivacyEngine()
            
            # Memory with PII
            memory = MemoryItem(
                id="privacy_test",
                user_id="test_user",
                memory_type="stm",  # Use string instead of enum
                content={
                    "text": "Meeting with john.doe@example.com at 555-1234"
                }
            )
            
            # Apply privacy filtering
            sanitized_content = privacy.remove_pii(memory.content["text"])
            memory.content["text"] = sanitized_content
            
            # Generate embedding on sanitized content
            embedding = await mock_embedder.generate_memory_embedding(memory)
            
            # Verify PII was removed from the content used for embedding
            call_args = mock_embedder.generate_memory_embedding.call_args
            memory_arg = call_args[0][0]
            assert "@example.com" not in memory_arg.content["text"]
            assert "555-1234" not in memory_arg.content["text"]
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_storage(self, temp_db, mock_embedder):
        """Test batch embedding generation and storage."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            # Create batch of memories
            memories = []
            for i in range(32):  # Max batch size
                memory = MemoryItem(
                    id=f"batch_{i}",
                    user_id="test_user",
                    memory_type="stm",  # Use string instead of enum
                    content={"text": f"Batch content {i}"}
                )
                memories.append(memory)
            
            # Generate embeddings in batch
            embeddings = await mock_embedder.batch_embed_memories(memories)
            
            # Attach embeddings to memories
            for memory, embedding in zip(memories, embeddings):
                memory.embedding = embedding
            
            # Store in batch
            await storage.batch_store(memories)
            
            # Verify all stored
            for memory in memories:
                retrieved = await storage.retrieve(memory.id)
                assert retrieved is not None
                assert retrieved.embedding is not None
    
    @pytest.mark.asyncio
    async def test_cross_layer_memory_search(self, temp_db, mock_embedder, memory_config):
        """Test searching across STM, WM, and LTM with embeddings."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            # Create memories in different layers
            stm_memory = MemoryItem(
                id="stm_search",
                user_id="test_user",
                memory_type="stm",  # Use string instead of enum
                content={"text": "Recent event"},
                embedding=np.random.rand(4096).astype(np.float32)
            )
            
            wm_memory = MemoryItem(
                id="wm_search",
                user_id="test_user",
                memory_type="wm",  # Use string instead of enum
                content={"text": "Working context"},
                embedding=np.random.rand(4096).astype(np.float32)
            )
            
            ltm_memory = MemoryItem(
                id="ltm_search",
                user_id="test_user",
                memory_type="ltm",  # Use string instead of enum
                content={"text": "Long-term knowledge"},
                embedding=np.random.rand(4096).astype(np.float32)
            )
            
            await storage.store(stm_memory)
            await storage.store(wm_memory)
            await storage.store(ltm_memory)
            
            # Search across all layers
            query_embedding = np.random.rand(4096).astype(np.float32)
            results = await storage.search_by_embedding(
                embedding=query_embedding,
                k=10,
                user_id="test_user"
            )
            
            # Should find memories from all layers
            memory_types = {r[0].memory_type for r in results}
            assert "stm" in memory_types
            assert "wm" in memory_types
            assert "ltm" in memory_types
    
    @pytest.mark.asyncio
    async def test_embedding_update_on_promotion(self, temp_db, mock_embedder, memory_config):
        """Test that embeddings are updated with new instruction prefixes on promotion."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            # Create STM memory
            memory = MemoryItem(
                id="update_test",
                user_id="test_user",
                memory_type="stm",  # Use string instead of enum
                content={"text": "Content to promote"},
                effectiveness_score=7.0
            )
            
            # Generate initial STM embedding
            stm_embedding = await mock_embedder.generate_memory_embedding(memory)
            memory.embedding = stm_embedding
            await storage.store(memory)
            
            # Promote to WM
            memory.memory_type = "wm"
            
            # Generate new WM embedding with different instruction prefix
            wm_embedding = await mock_embedder.generate_memory_embedding(memory)
            memory.embedding = wm_embedding
            
            await storage.update(memory)
            
            # Verify embedding was updated
            assert mock_embedder.generate_memory_embedding.call_count == 2
            
            # Check different instruction prefixes were used
            call_args_list = mock_embedder.generate_memory_embedding.call_args_list
            # First call should have STM memory, second should have WM memory
            assert call_args_list[0][0][0].memory_type == "stm"
            assert call_args_list[1][0][0].memory_type == "wm"
    
    @pytest.mark.asyncio
    async def test_sqlite_vec_integration(self, temp_db):
        """Test sqlite-vec vector storage and retrieval."""
        # This test verifies sqlite-vec is properly integrated
        conn = sqlite3.connect(temp_db)
        
        # Check if vec0 extension is available (would be loaded by storage init)
        cursor = conn.cursor()
        
        # Create vector table
        try:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS test_vectors 
                USING vec0(id TEXT PRIMARY KEY, embedding FLOAT[4096])
            """)
            
            # Insert test vector
            test_id = "vec_test"
            test_vector = np.random.rand(4096).astype(np.float32)
            cursor.execute(
                "INSERT INTO test_vectors (id, embedding) VALUES (?, ?)",
                (test_id, test_vector.tobytes())
            )
            
            conn.commit()
            
            # Retrieve and verify
            cursor.execute("SELECT id FROM test_vectors WHERE id = ?", (test_id,))
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == test_id
            
        except sqlite3.OperationalError as e:
            # sqlite-vec not available in test environment
            pytest.skip(f"sqlite-vec extension not available: {e}")
        finally:
            conn.close()
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval_operations(self, temp_db, mock_embedder):
        """Test thread-safe concurrent retrieval operations."""
        with patch('src.memory.embeddings.MemoryEmbedder', return_value=mock_embedder):
            storage = SQLiteStorage(temp_db)
            await storage.initialize()
            
            # Pre-populate memories
            for i in range(20):
                memory = MemoryItem(
                    id=f"concurrent_{i}",
                    user_id=f"user_{i % 3}",  # Multiple users
                    memory_type="wm",  # Use string instead of enum
                    content={"text": f"Concurrent test {i}"},
                    embedding=np.random.rand(4096).astype(np.float32)
                )
                await storage.store(memory)
            
            # Concurrent retrieval tasks
            async def retrieve_memories(user_id: str):
                query_embedding = np.random.rand(4096).astype(np.float32)
                return await storage.search_by_embedding(
                    embedding=query_embedding,
                    k=5,
                    user_id=user_id
                )
            
            # Run concurrent retrievals
            tasks = [retrieve_memories(f"user_{i % 3}") for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Verify all completed successfully
            assert len(results) == 10
            for result in results:
                assert isinstance(result, list)