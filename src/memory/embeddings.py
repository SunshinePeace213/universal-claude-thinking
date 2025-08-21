"""
Memory embedding integration module.

Provides memory-specific embedding generation using Qwen3-Embedding-8B model
with caching, batch processing, and optimized retrieval for <100ms latency.
"""

import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np

from src.memory.layers.base import MemoryItem
from src.rag.embedder import Qwen8BEmbedder

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for stored embeddings."""
    embedding: np.ndarray
    timestamp: float
    access_count: int = 1


class LRUCache:
    """
    Thread-safe LRU cache for embeddings.
    
    Provides fast retrieval of frequently accessed embeddings
    to meet <100ms latency requirements.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, EmbeddingCacheEntry] = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        """
        Retrieve embedding from cache.
        
        Args:
            key: Cache key (memory ID)
            
        Returns:
            Cached embedding or None if not found
        """
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.access_count += 1
                entry.timestamp = time.time()
                self.cache[key] = entry
                self.hits += 1
                return entry.embedding.copy()
            self.misses += 1
            return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            key: Cache key (memory ID)
            embedding: Embedding vector to cache
        """
        with self._lock:
            # Remove if already exists
            if key in self.cache:
                del self.cache[key]

            # Add to end
            self.cache[key] = EmbeddingCacheEntry(
                embedding=embedding.copy(),
                timestamp=time.time()
            )

            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class MemoryEmbedder:
    """
    Memory-specific embedding generator.
    
    Wraps Qwen3-Embedding-8B model with memory-optimized features including
    instruction prefixes, caching, and batch processing for efficient retrieval.
    """

    # Memory-specific instruction prefixes
    INSTRUCTION_PREFIXES = {
        "stm": "Recent interaction: ",
        "wm": "Working context: ",
        "ltm": "User preference: ",
        "swarm": "Community pattern: "
    }

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
        cache_size: int = 1000,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the memory embedder.
        
        Args:
            model_path: Path to Qwen3-Embedding-8B model
            device: Device to run on (cuda, mps, cpu)
            batch_size: Maximum batch size for processing
            cache_size: Size of embedding cache
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        self.model_path = model_path or "embedding/Qwen3-Embedding-8B"
        self.device = device
        self.batch_size = min(batch_size, 32)  # Cap at 32 for Mac M3
        self.normalize = normalize_embeddings

        # Initialize base embedder
        self.model = Qwen8BEmbedder(
            model_path=self.model_path,
            device=device,
            batch_size=self.batch_size
        )

        # Initialize cache
        self.cache = LRUCache(max_size=cache_size)
        self.cache_size = cache_size

        # Copy model attributes for compatibility
        self.embedding_dim = 4096
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the model and tokenizer.
        
        Raises:
            FileNotFoundError: If model files not found
        """
        try:
            await self.model.initialize()
            self._initialized = True

            # Update device from model
            self.device = self.model.device

            logger.info(
                f"MemoryEmbedder initialized with device={self.device}, "
                f"batch_size={self.batch_size}, cache_size={self.cache_size}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize MemoryEmbedder: {e}")
            raise

    def _extract_text_content(self, memory: MemoryItem) -> str:
        """
        Extract text content from memory item.
        
        Args:
            memory: Memory item to extract text from
            
        Returns:
            Extracted text content
        """
        content = memory.content

        # Handle different content structures
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Look for common text keys
            for key in ['text', 'message', 'content', 'data']:
                if key in content:
                    value = content[key]
                    if isinstance(value, str):
                        return value
                    elif isinstance(value, dict):
                        # Nested dict - convert to JSON
                        return json.dumps(value, ensure_ascii=False)
            # Fallback to JSON representation
            return json.dumps(content, ensure_ascii=False)
        else:
            return str(content)

    def _get_instruction_prefix(self, memory_type: str) -> str:
        """
        Get instruction prefix for memory type.
        
        Args:
            memory_type: Type of memory (string)
            
        Returns:
            Instruction prefix string
        """
        return self.INSTRUCTION_PREFIXES.get(
            memory_type,
            "Memory context: "  # Default prefix
        )

    async def generate_memory_embedding(
        self,
        memory: MemoryItem,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Generate embedding for a single memory item.
        
        Args:
            memory: Memory item to embed
            use_cache: Whether to use cache
            
        Returns:
            4096-dimensional embedding vector
        """
        if not self._initialized:
            await self.initialize()

        # Check cache first
        if use_cache:
            cached = self.cache.get(memory.id)
            if cached is not None:
                logger.debug(f"Cache hit for memory {memory.id}")
                return cached

        # Extract text content
        text = self._extract_text_content(memory)

        # Add memory-specific instruction prefix
        prefix = self._get_instruction_prefix(memory.memory_type)
        prefixed_text = f"{prefix}{text}"

        # Generate embedding
        embedding = await self.model.generate_embedding(prefixed_text)

        # Ensure correct shape (flatten if needed)
        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Cache the result
        if use_cache:
            self.cache.put(memory.id, embedding)

        return embedding

    async def encode(
        self,
        text: str | list[str] | MemoryItem | list[MemoryItem],
        use_cache: bool = True
    ) -> np.ndarray | list[np.ndarray]:
        """
        Encode text or memory items into embeddings.
        
        This method provides compatibility with different interfaces.
        
        Args:
            text: String, list of strings, MemoryItem, or list of MemoryItems
            use_cache: Whether to use cache for memory items
            
        Returns:
            Single embedding or list of embeddings
        """
        if not self._initialized:
            await self.initialize()

        # Handle single string
        if isinstance(text, str):
            embedding = await self.model.generate_embedding(text)
            if self.normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            return embedding

        # Handle list of strings
        elif isinstance(text, list) and all(isinstance(t, str) for t in text):
            embeddings = []
            for t in text:
                embedding = await self.model.generate_embedding(t)
                if self.normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                embeddings.append(embedding)
            return np.array(embeddings)

        # Handle single MemoryItem
        elif isinstance(text, MemoryItem):
            return await self.generate_memory_embedding(text, use_cache)

        # Handle list of MemoryItems
        elif isinstance(text, list) and all(isinstance(m, MemoryItem) for m in text):
            return await self.batch_embed_memories(text)

        # Handle generic object with text content
        elif hasattr(text, 'content'):
            # Create a temporary MemoryItem
            temp_memory = MemoryItem(
                id="temp_encode",
                user_id="system",
                memory_type="temp",
                content=text.content if hasattr(text, 'content') else str(text),
                metadata={}
            )
            return await self.generate_memory_embedding(temp_memory, use_cache=False)

        else:
            raise ValueError(f"Unsupported input type: {type(text)}")

    async def batch_embed_memories(
        self,
        memories: list[MemoryItem],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for multiple memories in batch.
        
        Args:
            memories: List of memory items
            show_progress: Whether to show progress
            
        Returns:
            Array of shape (n_memories, 4096)
        """
        if not self._initialized:
            await self.initialize()

        embeddings = []

        # Process in batches
        for i in range(0, len(memories), self.batch_size):
            batch = memories[i:i + self.batch_size]

            if show_progress:
                logger.info(
                    f"Processing batch {i//self.batch_size + 1}/"
                    f"{(len(memories)-1)//self.batch_size + 1}"
                )

            # Extract and prefix texts
            texts = []
            for memory in batch:
                text = self._extract_text_content(memory)
                prefix = self._get_instruction_prefix(memory.memory_type)
                texts.append(f"{prefix}{text}")

            # Generate batch embeddings
            batch_embeddings = await self.model.batch_generate(
                texts,
                show_progress=False
            )

            # Normalize if requested
            if self.normalize:
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / np.maximum(norms, 1e-10)

            # Cache individual embeddings
            for memory, embedding in zip(batch, batch_embeddings, strict=False):
                self.cache.put(memory.id, embedding)

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    def calculate_similarities(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarities between query and candidates.
        
        Args:
            query_embedding: Query vector (1D or 2D)
            candidate_embeddings: Candidate vectors (2D)
            
        Returns:
            Array of similarity scores
        """
        # Ensure correct shapes
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if not already normalized
        if not self.normalize:
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

            candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            candidate_embeddings = candidate_embeddings / np.maximum(candidate_norms, 1e-10)

        # Calculate cosine similarity
        similarities = np.dot(candidate_embeddings, query_embedding.T).squeeze()

        # Clip to [0, 1] range (handle numerical errors)
        similarities = np.clip(similarities, 0.0, 1.0)

        return similarities

    async def search_similar_memories(
        self,
        query_memory: MemoryItem,
        candidate_memories: list[MemoryItem],
        k: int = 10,
        min_similarity: float = 0.5
    ) -> list[tuple[MemoryItem, float]]:
        """
        Search for similar memories using embedding similarity.
        
        Args:
            query_memory: Query memory
            candidate_memories: Candidate memories to search
            k: Number of results to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (memory, similarity) tuples sorted by similarity
        """
        if not candidate_memories:
            return []

        # Generate query embedding
        query_embedding = await self.generate_memory_embedding(query_memory)

        # Generate candidate embeddings (use cache where possible)
        candidate_embeddings = []
        for memory in candidate_memories:
            embedding = await self.generate_memory_embedding(memory)
            candidate_embeddings.append(embedding)

        candidate_embeddings = np.array(candidate_embeddings)

        # Calculate similarities
        similarities = self.calculate_similarities(query_embedding, candidate_embeddings)

        # Filter by minimum similarity and sort
        results = []
        for memory, similarity in zip(candidate_memories, similarities, strict=False):
            if similarity >= min_similarity:
                results.append((memory, float(similarity)))

        # Sort by similarity (descending) and limit to k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.cache.get_stats()

    async def close(self) -> None:
        """Clean up resources."""
        if self.model:
            await self.model.close()
        self.clear_cache()
        self._initialized = False
