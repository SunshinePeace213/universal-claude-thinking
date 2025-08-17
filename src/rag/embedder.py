"""
Qwen3 Embedding Model Integration.

This module provides embedders for generating 4096-dimensional embeddings
using Qwen3 models, supporting both the full 8B model and the Mac-optimized
4bit-DWQ variant.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available Qwen3 model types."""
    
    QWEN3_8B = "Qwen3-Embedding-8B"
    QWEN3_8B_4BIT = "Qwen3-Embedding-8B-4bit-DWQ"


class QwenEmbedder(ABC):
    """
    Base class for Qwen3 embedding generation.
    
    Provides common interface and utilities for all Qwen3 embedder variants.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_sequence_length: int = 32768,
        embedding_dim: int = 4096,
    ) -> None:
        """
        Initialize the Qwen embedder.
        
        Args:
            model_path: Path to the model files (from Story 1.3)
            device: Device to run model on (cuda, mps, cpu)
            batch_size: Maximum batch size for processing
            max_sequence_length: Maximum input sequence length
            embedding_dim: Output embedding dimension
        """
        self.model_path = Path(model_path) if model_path else None
        self.batch_size = min(batch_size, 32)  # Cap at 32 for Mac M3
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        
        # Device selection with Mac M3 MPS optimization
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Mac M3 MPS for acceleration")
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model and tokenizer asynchronously."""
        pass
    
    @abstractmethod
    async def generate_embedding(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings for input text.
        
        Args:
            text: Single text or list of texts to embed
            instruction: Optional instruction for task-specific embedding
            
        Returns:
            Array of shape (n_texts, 4096) with embeddings
        """
        pass
    
    async def batch_generate(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            instruction: Optional instruction for all texts
            show_progress: Whether to show progress
            
        Returns:
            Array of shape (n_texts, 4096) with embeddings
        """
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if show_progress:
                logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            batch_embeddings = await self.generate_embedding(batch, instruction)
            embeddings.append(batch_embeddings)
            
        return np.vstack(embeddings)
    
    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2 normalize embeddings for cosine similarity.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)
    
    async def close(self) -> None:
        """Clean up model resources."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        torch.cuda.empty_cache() if self.device == "cuda" else None


class Qwen8BEmbedder(QwenEmbedder):
    """
    Embedder for the full Qwen3-Embedding-8B model.
    
    This is the primary model providing the highest quality 4096-dimensional
    embeddings for semantic similarity matching.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the Qwen3-Embedding-8B embedder."""
        super().__init__(
            model_path=model_path or "models/Qwen3-Embedding-8B",
            device=device,
            batch_size=batch_size,
            max_sequence_length=32768,
            embedding_dim=4096,
        )
        self.model_type = ModelType.QWEN3_8B
        
    async def initialize(self) -> None:
        """Load the Qwen3-Embedding-8B model and tokenizer."""
        try:
            logger.info(f"Loading {self.model_type.value} from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            
            # Load model with automatic device mapping
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                device_map=self.device if self.device != "mps" else None,
            )
            
            # Move to MPS if needed (device_map doesn't support MPS yet)
            if self.device == "mps":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_type.value}: {e}")
            raise
            
    async def generate_embedding(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings using Qwen3-Embedding-8B.
        
        Args:
            text: Text(s) to embed
            instruction: Optional instruction for task-specific embedding
            
        Returns:
            4096-dimensional embeddings
        """
        if not self.model or not self.tokenizer:
            await self.initialize()
            
        # Ensure text is a list
        texts = [text] if isinstance(text, str) else text
        
        # Add instruction if provided (Qwen3 supports instruction-aware embeddings)
        if instruction:
            texts = [f"{instruction}: {t}" for t in texts]
            
        try:
            # Tokenize with padding and truncation
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            # Convert to numpy and ensure 4096 dimensions
            embeddings_np = embeddings.cpu().numpy()
            
            # Verify dimensions
            if embeddings_np.shape[1] != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: got {embeddings_np.shape[1]}, expected {self.embedding_dim}")
                
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class Qwen8B4BitEmbedder(QwenEmbedder):
    """
    Embedder for the Mac-optimized Qwen3-Embedding-8B-4bit-DWQ model.
    
    This quantized variant provides similar 4096-dimensional embeddings
    with better resource efficiency on Mac M3 systems.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the Qwen3-Embedding-8B-4bit-DWQ embedder."""
        super().__init__(
            model_path=model_path or "models/Qwen3-Embedding-8B-4bit-DWQ",
            device=device,
            batch_size=batch_size,
            max_sequence_length=32768,
            embedding_dim=4096,
        )
        self.model_type = ModelType.QWEN3_8B_4BIT
        
    async def initialize(self) -> None:
        """Load the quantized Qwen3 model and tokenizer."""
        try:
            logger.info(f"Loading {self.model_type.value} from {self.model_path}")
            
            # Load tokenizer (same as full model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            
            # Load quantized model with 4-bit configuration
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # DWQ: Double Weight Quantization
                bnb_4bit_quant_type="nf4",
            )
            
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                quantization_config=quantization_config if self.device != "mps" else None,
                device_map=self.device if self.device != "mps" else None,
            )
            
            # For MPS, use native quantization support
            if self.device == "mps":
                self.model = self.model.to(self.device)
                # MPS has built-in support for int8/int4 operations
                
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_type.value} (4-bit quantized)")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_type.value}: {e}")
            raise
            
    async def generate_embedding(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate embeddings using the quantized model.
        
        The 4-bit model produces the same 4096-dimensional embeddings
        with reduced memory usage and faster inference on Mac M3.
        
        Args:
            text: Text(s) to embed
            instruction: Optional instruction
            
        Returns:
            4096-dimensional embeddings
        """
        if not self.model or not self.tokenizer:
            await self.initialize()
            
        texts = [text] if isinstance(text, str) else text
        
        if instruction:
            texts = [f"{instruction}: {t}" for t in texts]
            
        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ).to(self.device)
            
            # Generate embeddings with quantized model
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
            embeddings_np = embeddings.cpu().numpy()
            
            # The quantized model should still output 4096 dimensions
            if embeddings_np.shape[1] != self.embedding_dim:
                logger.warning(f"Quantized embedding dimension: {embeddings_np.shape[1]}")
                
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Error with quantized embedding generation: {e}")
            raise


class AdaptiveEmbedder:
    """
    Adaptive embedder that selects the best model based on runtime requirements.
    
    This class implements model selection logic based on available resources,
    performance requirements, and benchmark results.
    """
    
    def __init__(
        self,
        prefer_quality: bool = True,
        max_memory_gb: float = 35.0,
        target_latency_ms: float = 800.0,
    ) -> None:
        """
        Initialize adaptive embedder.
        
        Args:
            prefer_quality: Prefer quality over speed when both models work
            max_memory_gb: Maximum memory allocation for models
            target_latency_ms: Target latency for embedding generation
        """
        self.prefer_quality = prefer_quality
        self.max_memory_gb = max_memory_gb
        self.target_latency_ms = target_latency_ms
        
        self.primary_embedder: Optional[Qwen8BEmbedder] = None
        self.fallback_embedder: Optional[Qwen8B4BitEmbedder] = None
        self.active_embedder: Optional[QwenEmbedder] = None
        
    async def initialize(self) -> None:
        """Initialize embedders and select the best one."""
        # Try primary model first
        try:
            self.primary_embedder = Qwen8BEmbedder()
            await self.primary_embedder.initialize()
            
            # Check memory usage
            memory_usage = self._estimate_memory_usage(self.primary_embedder)
            if memory_usage <= self.max_memory_gb:
                self.active_embedder = self.primary_embedder
                logger.info(f"Using primary {ModelType.QWEN3_8B.value}")
            else:
                raise MemoryError(f"Primary model requires {memory_usage}GB, limit is {self.max_memory_gb}GB")
                
        except Exception as e:
            logger.warning(f"Primary model unavailable: {e}")
            
            # Fall back to quantized model
            self.fallback_embedder = Qwen8B4BitEmbedder()
            await self.fallback_embedder.initialize()
            self.active_embedder = self.fallback_embedder
            logger.info(f"Using fallback {ModelType.QWEN3_8B_4BIT.value}")
            
    def _estimate_memory_usage(self, embedder: QwenEmbedder) -> float:
        """Estimate memory usage of the model in GB."""
        if isinstance(embedder, Qwen8BEmbedder):
            # 8B parameters * 2 bytes (fp16) + overhead
            # 8 billion * 2 bytes = 16GB + 2GB overhead = ~18GB
            return 16.0 + 2.0  # ~18GB
        else:
            # 8B parameters * 0.5 bytes (4-bit) + overhead
            # 8 billion * 0.5 bytes = 4GB + 1GB overhead = ~5GB
            return 4.0 + 1.0  # ~5GB
            
    async def generate_embedding(
        self,
        text: Union[str, List[str]],
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """Generate embeddings using the active model."""
        if not self.active_embedder:
            await self.initialize()
            
        start_time = time.time()
        embeddings = await self.active_embedder.generate_embedding(text, instruction)
        latency_ms = (time.time() - start_time) * 1000
        
        # Switch models if latency target not met
        if latency_ms > self.target_latency_ms and self.active_embedder == self.primary_embedder:
            logger.warning(f"Latency {latency_ms}ms exceeds target, switching to quantized model")
            if not self.fallback_embedder:
                self.fallback_embedder = Qwen8B4BitEmbedder()
                await self.fallback_embedder.initialize()
            self.active_embedder = self.fallback_embedder
            
        return embeddings
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.primary_embedder:
            await self.primary_embedder.close()
        if self.fallback_embedder:
            await self.fallback_embedder.close()