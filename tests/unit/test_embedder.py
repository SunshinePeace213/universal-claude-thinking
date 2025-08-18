"""
Unit tests for Qwen3 Embedding Model.

Tests the Qwen8BEmbedder class with proper mocking that matches
the real model interface (4096-dimensional embeddings).
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from pathlib import Path

import numpy as np
import pytest
import torch

from src.rag.embedder import (
    ModelType,
    Qwen8BEmbedder,
)


class TestQwen8BEmbedder(unittest.IsolatedAsyncioTestCase):
    """Test cases for Qwen8BEmbedder."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedder = Qwen8BEmbedder(batch_size=16)
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_initialize(self, mock_tokenizer, mock_model):
        """Test model initialization."""
        # Mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model instance with eval method and to() for device mapping
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        # Verify initialization
        self.assertIsNotNone(self.embedder.model)
        self.assertIsNotNone(self.embedder.tokenizer)
        mock_model_instance.eval.assert_called_once()
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_generate_embedding_single(self, mock_tokenizer, mock_model):
        """Test single text embedding generation."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock tokenizer output
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer_instance.return_value = mock_inputs
        
        # Mock model instance with eval method and to() for device mapping
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        
        # Create actual tensor for hidden states
        hidden_states = torch.randn(1, 10, 4096)  # batch=1, seq_len=10, dim=4096
        mean_pooled = hidden_states.mean(dim=1)  # Shape: (1, 4096)
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state.mean.return_value = mean_pooled
        mock_model_instance.__call__ = MagicMock(return_value=mock_output)
        
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize and test
        await self.embedder.initialize()
        
        text = "Test text for embedding"
        embedding = await self.embedder.generate_embedding(text)
        
        # Verify output shape
        self.assertEqual(embedding.shape, (1, 4096))
        # Accept both float16 (for MPS/CUDA) and float32 (for CPU)
        self.assertIn(embedding.dtype, [np.float16, np.float32])
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_generate_embedding_batch(self, mock_tokenizer, mock_model):
        """Test batch embedding generation."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer_instance.return_value = mock_inputs
        
        # Mock model instance with eval method and to() for device mapping
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        
        # Create actual tensor for hidden states
        batch_size = 3
        hidden_states = torch.randn(batch_size, 10, 4096)
        mean_pooled = hidden_states.mean(dim=1)  # Shape: (3, 4096)
        
        # Mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state.mean.return_value = mean_pooled
        mock_model_instance.__call__ = MagicMock(return_value=mock_output)
        
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await self.embedder.generate_embedding(texts)
        
        # Verify batch output
        self.assertEqual(embeddings.shape, (batch_size, 4096))
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_batch_generate(self, mock_tokenizer, mock_model):
        """Test batch_generate method with multiple batches."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer_instance.return_value = mock_inputs
        
        # Mock model instance with eval method and to() for device mapping
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        
        # Mock model to return different sizes based on call
        def generate_output(*args, **kwargs):
            # For 50 texts with batch_size=16, we'll have batches of 16, 16, 16, 2
            if not hasattr(generate_output, 'call_count'):
                generate_output.call_count = 0
            
            batch_sizes = [16, 16, 16, 2]  # 50 texts split into batches
            current_batch_size = batch_sizes[generate_output.call_count % len(batch_sizes)]
            generate_output.call_count += 1
            
            # Create actual tensor for this batch
            hidden_states = torch.randn(current_batch_size, 10, 4096)
            mean_pooled = hidden_states.mean(dim=1)  # Shape: (current_batch_size, 4096)
            
            mock_output = MagicMock()
            mock_output.last_hidden_state.mean.return_value = mean_pooled
            return mock_output
            
        mock_model_instance.__call__ = MagicMock(side_effect=generate_output)
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        # Test with 50 texts (should create multiple batches)
        texts = [f"Text {i}" for i in range(50)]
        embeddings = await self.embedder.batch_generate(texts)
        
        # Verify final shape
        self.assertEqual(embeddings.shape[0], 50)
        self.assertEqual(embeddings.shape[1], 4096)
        
    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        # Create test embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])
        
        normalized = self.embedder.normalize_embeddings(embeddings)
        
        # Check L2 norm is 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3))
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_close(self, mock_tokenizer, mock_model):
        """Test resource cleanup."""
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        # Verify initialized
        self.assertIsNotNone(self.embedder.model)
        self.assertIsNotNone(self.embedder.tokenizer)
        
        # Close and verify cleanup
        await self.embedder.close()
        
        self.assertIsNone(self.embedder.model)
        self.assertIsNone(self.embedder.tokenizer)


if __name__ == "__main__":
    # Run with pytest for better async support
    pytest.main([__file__, "-v"])