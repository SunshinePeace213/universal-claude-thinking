"""
Qwen3 Embedding Model Integration.

This module provides an embedder for generating 4096-dimensional embeddings
using the Qwen3-Embedding-8B model loaded from local path.

The model is expected to be pre-downloaded at: embedding/Qwen3-Embedding-8B
"""

import asyncio
import logging
import os
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
    """Qwen3 model type."""
    
    QWEN3_8B = "Qwen3-Embedding-8B"  # Single model: 4096-dimensional embeddings


class QwenEmbedder(ABC):
    """
    Base class for Qwen3 embedding generation.
    
    Provides common interface and utilities for the Qwen3 embedder.
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
            model_path: Path to the local model files
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
    Embedder for the Qwen3-Embedding-8B model.
    
    Provides 4096-dimensional embeddings for semantic similarity matching
    using the locally stored Qwen3-Embedding-8B model.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize the Qwen3-Embedding-8B embedder.
        
        Args:
            model_path: Path to local model. Defaults to "embedding/Qwen3-Embedding-8B"
            device: Device to run model on (cuda, mps, cpu). Auto-detected if None.
            batch_size: Maximum batch size for processing (capped at 32 for Mac M3)
        """
        # Use environment variable override if available
        default_path = os.environ.get("QWEN_MODEL_PATH", "embedding/Qwen3-Embedding-8B")
        
        super().__init__(
            model_path=model_path or default_path,
            device=device,
            batch_size=batch_size,
            max_sequence_length=32768,
            embedding_dim=4096,
        )
        self.model_type = ModelType.QWEN3_8B
        
    async def initialize(self) -> None:
        """Load the Qwen3-Embedding-8B model and tokenizer from local path."""
        try:
            # Verify model path exists
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    f"Please ensure Qwen3-Embedding-8B model is downloaded to this location."
                )
            
            logger.info(f"Loading {self.model_type.value} from {self.model_path}")
            
            # Check for required model files
            required_files = ["config.json", "tokenizer_config.json"]
            missing_files = [f for f in required_files if not (self.model_path / f).exists()]
            if missing_files:
                logger.warning(f"Missing expected files: {missing_files}")
            
            # Load tokenizer with local_files_only to ensure no downloads
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,  # Ensure we only use local files
            )
            logger.info("Tokenizer loaded successfully")
            
            # Load model with local_files_only
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
                local_files_only=True,  # Ensure we only use local files
                device_map=self.device if self.device != "mps" else None,
                torch_dtype=torch.float16 if self.device in ["cuda", "mps"] else torch.float32,
            )
            
            # Move to MPS if needed (device_map doesn't support MPS yet)
            if self.device == "mps":
                self.model = self.model.to(self.device)
                
            self.model.eval()
            logger.info(f"Successfully loaded {self.model_type.value} on {self.device}")
            
            # Log model info
            if hasattr(self.model.config, 'hidden_size'):
                logger.info(f"Model hidden size: {self.model.config.hidden_size}")
            
        except FileNotFoundError as e:
            logger.error(f"Model files not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load {self.model_type.value}: {e}")
            logger.error(f"Ensure model files are present at: {self.model_path}")
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
