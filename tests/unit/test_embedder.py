"""
Unit tests for Qwen3 Embedding Models.

Tests both Qwen8BEmbedder and Qwen8B4BitEmbedder classes, as well as
the AdaptiveEmbedder for model selection logic.
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
    Qwen8B4BitEmbedder,
    AdaptiveEmbedder,
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
        
        # Mock model instance with eval method
        mock_model_instance = MagicMock()
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
        
        # Mock model instance with eval method
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        
        # Mock model output - need to properly mock the __call__ method
        mock_output = MagicMock()
        mock_hidden_states = torch.randn(1, 10, 4096)  # batch=1, seq_len=10, dim=4096
        mock_output.last_hidden_state = mock_hidden_states
        mock_model_instance.__call__ = MagicMock(return_value=mock_output)
        
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize and test
        await self.embedder.initialize()
        
        text = "Test text for embedding"
        embedding = await self.embedder.generate_embedding(text)
        
        # Verify output shape
        self.assertEqual(embedding.shape, (1, 4096))
        self.assertEqual(embedding.dtype, np.float32)
        
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
        
        # Mock model instance with eval method
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        
        # Mock model output for batch
        mock_output = MagicMock()
        batch_size = 3
        mock_hidden_states = torch.randn(batch_size, 10, 4096)
        mock_output.last_hidden_state = mock_hidden_states
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
        
        # Mock model instance with eval method
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        
        # Mock model to return different sizes based on call
        def generate_output(*args, **kwargs):
            mock_output = MagicMock()
            # Return appropriate batch size - this should match the actual batch size being processed
            # For 50 texts with batch_size=16, we'll have batches of 16, 16, 16, 2
            # We need to track which batch we're processing
            if not hasattr(generate_output, 'call_count'):
                generate_output.call_count = 0
            
            batch_sizes = [16, 16, 16, 2]  # 50 texts split into batches
            current_batch_size = batch_sizes[generate_output.call_count % len(batch_sizes)]
            generate_output.call_count += 1
            
            mock_hidden_states = torch.randn(current_batch_size, 10, 4096)
            mock_output.last_hidden_state = mock_hidden_states
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
        
        # Check L2 norm is 1 for each embedding
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3))
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_close(self, mock_tokenizer, mock_model):
        """Test resource cleanup."""
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        
        await self.embedder.initialize()
        await self.embedder.close()
        
        self.assertIsNone(self.embedder.model)
        self.assertIsNone(self.embedder.tokenizer)


class TestQwen8B4BitEmbedder(unittest.IsolatedAsyncioTestCase):
    """Test cases for Qwen8B4BitEmbedder."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedder = Qwen8B4BitEmbedder(batch_size=16)
        
    @pytest.mark.asyncio
    @patch("transformers.BitsAndBytesConfig")
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_initialize_quantized(self, mock_tokenizer, mock_model, mock_bnb_config):
        """Test quantized model initialization."""
        # Mock quantization config
        mock_config_instance = MagicMock()
        mock_bnb_config.return_value = mock_config_instance
        
        # Mock model and tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        # Verify quantization config was created
        mock_bnb_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        # Verify model loaded
        self.assertIsNotNone(self.embedder.model)
        self.assertIsNotNone(self.embedder.tokenizer)
        mock_model_instance.eval.assert_called_once()
        
    @pytest.mark.asyncio
    @patch("transformers.BitsAndBytesConfig")
    @patch("src.rag.embedder.AutoModel")
    @patch("src.rag.embedder.AutoTokenizer")
    async def test_generate_embedding_quantized(self, mock_tokenizer, mock_model, mock_bnb_config):
        """Test embedding generation with quantized model."""
        # Setup mocks
        mock_bnb_config.return_value = MagicMock()
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer_instance.return_value = mock_inputs
        
        # Mock model instance with eval method
        mock_model_instance = MagicMock()
        mock_model_instance.eval = MagicMock()
        
        # Mock model output
        mock_output = MagicMock()
        mock_hidden_states = torch.randn(1, 10, 4096)
        mock_output.last_hidden_state = mock_hidden_states
        mock_model_instance.__call__ = MagicMock(return_value=mock_output)
        
        mock_model.from_pretrained.return_value = mock_model_instance
        
        await self.embedder.initialize()
        
        text = "Test text for quantized model"
        embedding = await self.embedder.generate_embedding(text)
        
        # Verify output (should be same dimensions despite quantization)
        self.assertEqual(embedding.shape, (1, 4096))
        self.assertEqual(embedding.dtype, np.float32)


class TestAdaptiveEmbedder(unittest.IsolatedAsyncioTestCase):
    """Test cases for AdaptiveEmbedder."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.embedder = AdaptiveEmbedder(
            prefer_quality=True,
            max_memory_gb=20.0,
            target_latency_ms=500.0,
        )
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.Qwen8B4BitEmbedder")
    @patch("src.rag.embedder.Qwen8BEmbedder")
    async def test_initialize_primary_model(self, mock_primary, mock_fallback):
        """Test initialization with primary model."""
        # Mock successful primary model init
        mock_primary_instance = AsyncMock()
        mock_primary.return_value = mock_primary_instance
        
        await self.embedder.initialize()
        
        # Should use primary model
        self.assertEqual(self.embedder.active_embedder, mock_primary_instance)
        mock_primary_instance.initialize.assert_called_once()
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.Qwen8B4BitEmbedder")
    @patch("src.rag.embedder.Qwen8BEmbedder")
    async def test_initialize_fallback_on_memory_error(self, mock_primary, mock_fallback):
        """Test fallback to quantized model on memory constraints."""
        # Mock primary model exceeding memory
        mock_primary_instance = AsyncMock()
        mock_primary.return_value = mock_primary_instance
        mock_primary_instance.initialize.side_effect = MemoryError("Not enough memory")
        
        # Mock fallback model
        mock_fallback_instance = AsyncMock()
        mock_fallback.return_value = mock_fallback_instance
        
        # Set low memory limit to trigger fallback
        self.embedder.max_memory_gb = 5.0
        
        await self.embedder.initialize()
        
        # Should use fallback model
        self.assertEqual(self.embedder.active_embedder, mock_fallback_instance)
        mock_fallback_instance.initialize.assert_called_once()
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.time")
    @patch("src.rag.embedder.Qwen8B4BitEmbedder")
    @patch("src.rag.embedder.Qwen8BEmbedder")
    async def test_adaptive_switching_on_latency(self, mock_primary, mock_fallback, mock_time):
        """Test switching models based on latency."""
        # Setup primary model
        mock_primary_instance = AsyncMock()
        mock_primary.return_value = mock_primary_instance
        mock_primary_instance.generate_embedding.return_value = np.zeros((1, 4096))
        
        # Setup fallback model
        mock_fallback_instance = AsyncMock()
        mock_fallback.return_value = mock_fallback_instance
        mock_fallback_instance.generate_embedding.return_value = np.zeros((1, 4096))
        
        # Mock time to simulate high latency
        mock_time.time.side_effect = [0.0, 1.0]  # 1000ms latency
        
        await self.embedder.initialize()
        self.embedder.target_latency_ms = 500.0  # Set low target
        
        # Generate embedding (should trigger switch)
        embedding = await self.embedder.generate_embedding("test")
        
        # Should initialize and switch to fallback
        self.assertIsNotNone(self.embedder.fallback_embedder)
        
    def test_estimate_memory_usage(self):
        """Test memory usage estimation."""
        # Test primary model estimation
        primary = Qwen8BEmbedder()
        memory_primary = self.embedder._estimate_memory_usage(primary)
        self.assertAlmostEqual(memory_primary, 18.0, delta=1.0)
        
        # Test quantized model estimation
        quantized = Qwen8B4BitEmbedder()
        memory_quantized = self.embedder._estimate_memory_usage(quantized)
        self.assertAlmostEqual(memory_quantized, 5.0, delta=1.0)
        
    @pytest.mark.asyncio
    @patch("src.rag.embedder.Qwen8B4BitEmbedder")
    @patch("src.rag.embedder.Qwen8BEmbedder")
    async def test_close(self, mock_primary, mock_fallback):
        """Test resource cleanup for adaptive embedder."""
        # Setup both models
        mock_primary_instance = AsyncMock()
        mock_primary.return_value = mock_primary_instance
        
        mock_fallback_instance = AsyncMock()
        mock_fallback.return_value = mock_fallback_instance
        
        await self.embedder.initialize()
        self.embedder.fallback_embedder = mock_fallback_instance
        
        await self.embedder.close()
        
        # Both models should be closed
        mock_primary_instance.close.assert_called_once()
        mock_fallback_instance.close.assert_called_once()


if __name__ == "__main__":
    # Run async tests with pytest
    pytest.main([__file__, "-v"])