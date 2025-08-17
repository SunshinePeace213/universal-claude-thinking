"""
RAG Pipeline Orchestrator.

Main orchestration module for the Retrieval-Augmented Generation pipeline,
coordinating embedding generation, vector search, and context assembly.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core.molecular.context_builder import (
    MoleculeContext,
    MoleculeContextBuilder,
    TokenAllocation,
)
from src.core.molecular.example_selector import (
    ExampleSelector,
    SelectionResult,
    SelectionStrategy,
)
from src.core.molecular.vector_store import VectorStore
from src.rag.embedder import AdaptiveEmbedder, ModelType

logger = logging.getLogger(__name__)


class PipelineMode(Enum):
    """RAG pipeline operation modes."""
    
    RETRIEVAL = "retrieval"  # Pure retrieval mode
    GENERATION = "generation"  # Context generation mode
    HYBRID = "hybrid"  # Combined retrieval and generation


@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline."""
    
    mode: PipelineMode = PipelineMode.HYBRID
    max_examples: int = 10
    similarity_threshold: float = 0.85
    chunk_size: int = 1024
    overlap_ratio: float = 0.15
    selection_strategy: SelectionStrategy = SelectionStrategy.HYBRID
    enable_caching: bool = True
    target_latency_ms: float = 800.0
    batch_size: int = 32  # Max for Mac M3


@dataclass
class PipelineResult:
    """Result from RAG pipeline processing."""
    
    context: MoleculeContext
    examples_retrieved: int
    embeddings_generated: int
    total_latency_ms: float
    breakdown: Dict[str, float]  # Timing breakdown
    model_used: ModelType
    cache_hits: int = 0
    warnings: List[str] = None


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.
    
    Coordinates embedding generation, vector search, example selection,
    and context assembly to achieve <800ms total latency.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Optional[AdaptiveEmbedder] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Vector storage backend
            embedder: Adaptive embedder (creates default if None)
            config: Pipeline configuration
        """
        self.vector_store = vector_store
        self.embedder = embedder or AdaptiveEmbedder()
        self.config = config or PipelineConfig()
        
        # Initialize components
        self.context_builder = MoleculeContextBuilder(
            chunk_size=self.config.chunk_size,
            overlap_ratio=self.config.overlap_ratio,
            max_examples=self.config.max_examples,
        )
        
        self.example_selector = ExampleSelector(
            vector_store=vector_store,
            context_builder=self.context_builder,
            default_strategy=self.config.selection_strategy,
        )
        
        # Performance tracking
        self._timing_history: List[Dict[str, float]] = []
        self._cache: Dict[str, PipelineResult] = {} if self.config.enable_caching else None
        
        # Initialize flag
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return
            
        logger.info("Initializing RAG pipeline...")
        
        # Initialize vector store
        await self.vector_store.initialize()
        
        # Initialize embedder
        await self.embedder.initialize()
        
        self._initialized = True
        logger.info(f"RAG pipeline initialized with {self.config.mode.value} mode")
        
    async def process(
        self,
        input_text: str,
        instruction: Optional[str] = None,
        context: Optional[str] = None,
        force_model: Optional[ModelType] = None,
    ) -> PipelineResult:
        """
        Process input through the RAG pipeline.
        
        Args:
            input_text: Input text to process
            instruction: Optional instruction for the task
            context: Optional additional context
            force_model: Force specific model type
            
        Returns:
            Pipeline result with assembled context
        """
        if not self._initialized:
            await self.initialize()
            
        start_time = time.time()
        timings = {}
        warnings = []
        
        # Check cache if enabled
        cache_key = self._get_cache_key(input_text, instruction, context)
        if self._cache is not None and cache_key in self._cache:
            logger.debug("Cache hit for pipeline input")
            cached_result = self._cache[cache_key]
            cached_result.cache_hits += 1
            return cached_result
            
        try:
            # Step 1: Generate embedding for input
            embed_start = time.time()
            
            # Prepare text for embedding
            embed_text = input_text
            if instruction:
                embed_text = f"{instruction}: {embed_text}"
                
            input_embedding = await self.embedder.generate_embedding(embed_text)
            timings["embedding_generation"] = (time.time() - embed_start) * 1000
            
            # Verify embedding dimensions
            if input_embedding.shape[-1] != 4096:
                warnings.append(f"Embedding dimension mismatch: {input_embedding.shape[-1]} != 4096")
                
            # Handle batch or single embedding
            if len(input_embedding.shape) > 1:
                input_embedding = input_embedding[0]  # Take first if batch
                
            # Step 2: Retrieve similar examples
            retrieval_start = time.time()
            selection_result = await self.example_selector.select_examples(
                query_embedding=input_embedding,
                k=self.config.max_examples,
                strategy=self.config.selection_strategy,
                min_similarity=self.config.similarity_threshold,
            )
            timings["example_retrieval"] = (time.time() - retrieval_start) * 1000
            
            if not selection_result.examples:
                warnings.append("No examples found above similarity threshold")
                
            # Step 3: Build MOLECULE context
            build_start = time.time()
            
            # Convert examples to dict format
            example_dicts = [ex.to_dict() for ex in selection_result.examples]
            similarity_scores = [ex.similarity_score for ex in selection_result.examples]
            
            # Default instruction if not provided
            if not instruction:
                instruction = "Process the following input based on the provided examples:"
                
            # Build context
            molecule_context = self.context_builder.build_context(
                instruction=instruction or "",
                examples=example_dicts,
                context=context or "",
                new_input=input_text,
                similarity_scores=similarity_scores,
            )
            timings["context_building"] = (time.time() - build_start) * 1000
            
            # Calculate total latency
            total_latency = (time.time() - start_time) * 1000
            timings["total"] = total_latency
            
            # Check latency target
            if total_latency > self.config.target_latency_ms:
                warnings.append(
                    f"Pipeline latency {total_latency:.2f}ms exceeds "
                    f"target {self.config.target_latency_ms}ms"
                )
                
            # Track timing for optimization
            self._timing_history.append(timings)
            
            # Create result
            result = PipelineResult(
                context=molecule_context,
                examples_retrieved=len(selection_result.examples),
                embeddings_generated=1,
                total_latency_ms=total_latency,
                breakdown=timings,
                model_used=self.embedder.active_embedder.model_type,
                cache_hits=0,
                warnings=warnings if warnings else None,
            )
            
            # Cache result if enabled
            if self._cache is not None:
                self._cache[cache_key] = result
                
            logger.info(
                f"Pipeline processed in {total_latency:.2f}ms "
                f"(embed: {timings['embedding_generation']:.2f}ms, "
                f"retrieve: {timings['example_retrieval']:.2f}ms, "
                f"build: {timings['context_building']:.2f}ms)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            raise
            
    async def batch_process(
        self,
        inputs: List[str],
        instruction: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[PipelineResult]:
        """
        Process multiple inputs in batch.
        
        Args:
            inputs: List of input texts
            instruction: Common instruction for all inputs
            context: Common context for all inputs
            
        Returns:
            List of pipeline results
        """
        if not self._initialized:
            await self.initialize()
            
        results = []
        
        # Process in batches based on config
        for i in range(0, len(inputs), self.config.batch_size):
            batch = inputs[i:i + self.config.batch_size]
            
            # Generate embeddings for batch
            embed_text_batch = [
                f"{instruction}: {text}" if instruction else text
                for text in batch
            ]
            
            embeddings = await self.embedder.generate_embedding(embed_text_batch)
            
            # Process each with its embedding
            for j, (input_text, embedding) in enumerate(zip(batch, embeddings)):
                # Retrieve examples for this input
                selection_result = await self.example_selector.select_examples(
                    query_embedding=embedding,
                    k=self.config.max_examples,
                    strategy=self.config.selection_strategy,
                    min_similarity=self.config.similarity_threshold,
                )
                
                # Build context
                example_dicts = [ex.to_dict() for ex in selection_result.examples]
                similarity_scores = [ex.similarity_score for ex in selection_result.examples]
                
                molecule_context = self.context_builder.build_context(
                    instruction=instruction or "",
                    examples=example_dicts,
                    context=context or "",
                    new_input=input_text,
                    similarity_scores=similarity_scores,
                )
                
                # Create result (simplified timing for batch)
                result = PipelineResult(
                    context=molecule_context,
                    examples_retrieved=len(selection_result.examples),
                    embeddings_generated=1,
                    total_latency_ms=0,  # Not tracked for batch
                    breakdown={},
                    model_used=self.embedder.active_embedder.model_type,
                )
                
                results.append(result)
                
            logger.info(f"Processed batch {i//self.config.batch_size + 1}")
            
        return results
        
    def _get_cache_key(
        self,
        input_text: str,
        instruction: Optional[str],
        context: Optional[str],
    ) -> str:
        """Generate cache key for input."""
        parts = [input_text]
        if instruction:
            parts.append(instruction)
        if context:
            parts.append(context)
        return "|".join(parts)
        
    async def update_effectiveness(
        self,
        result: PipelineResult,
        feedback: float,
    ) -> None:
        """
        Update effectiveness scores based on feedback.
        
        Args:
            result: Pipeline result to update
            feedback: Feedback value (+0.3/-0.3)
        """
        # Update effectiveness for all examples used
        for example in result.context.examples:
            # Find example ID from content (this is simplified)
            # In production, we'd track IDs through the pipeline
            await self.example_selector.update_effectiveness(
                example_id=hash(str(example)),  # Placeholder
                feedback=feedback,
            )
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        if not self._timing_history:
            return {}
            
        # Calculate averages
        avg_timings = {}
        for key in self._timing_history[0].keys():
            values = [t[key] for t in self._timing_history if key in t]
            avg_timings[f"avg_{key}_ms"] = np.mean(values) if values else 0
            
        # Add cache statistics
        cache_stats = {
            "cache_enabled": self.config.enable_caching,
            "cache_size": len(self._cache) if self._cache else 0,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }
        
        # Add model info
        model_info = {
            "active_model": self.embedder.active_embedder.model_type.value if self.embedder.active_embedder else None,
            "batch_size": self.config.batch_size,
        }
        
        return {
            **avg_timings,
            **cache_stats,
            **model_info,
            "total_requests": len(self._timing_history),
        }
        
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self._cache:
            return 0.0
            
        total_hits = sum(r.cache_hits for r in self._cache.values())
        total_requests = len(self._timing_history) + total_hits
        
        return total_hits / total_requests if total_requests > 0 else 0.0
        
    def clear_cache(self) -> None:
        """Clear the result cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Pipeline cache cleared")
            
    async def optimize_for_latency(self) -> None:
        """
        Optimize pipeline for latency based on performance history.
        
        Adjusts batch sizes, switches models, and tunes parameters.
        """
        if len(self._timing_history) < 10:
            logger.info("Not enough history for optimization")
            return
            
        # Analyze recent performance
        recent_timings = self._timing_history[-10:]
        avg_total = np.mean([t["total"] for t in recent_timings])
        
        if avg_total > self.config.target_latency_ms:
            logger.info(f"Average latency {avg_total:.2f}ms exceeds target, optimizing...")
            
            # Try switching to 4bit model if using 8B
            if self.embedder.active_embedder.model_type == ModelType.QWEN3_8B:
                logger.info("Switching to 4bit model for better latency")
                if not self.embedder.fallback_embedder:
                    from src.rag.embedder import Qwen8B4BitEmbedder
                    self.embedder.fallback_embedder = Qwen8B4BitEmbedder()
                    await self.embedder.fallback_embedder.initialize()
                self.embedder.active_embedder = self.embedder.fallback_embedder
                
            # Reduce batch size
            elif self.config.batch_size > 16:
                self.config.batch_size = max(16, self.config.batch_size // 2)
                logger.info(f"Reduced batch size to {self.config.batch_size}")
                
            # Reduce max examples
            elif self.config.max_examples > 5:
                self.config.max_examples = max(5, self.config.max_examples - 2)
                logger.info(f"Reduced max examples to {self.config.max_examples}")
                
    async def close(self) -> None:
        """Clean up resources."""
        await self.embedder.close()
        await self.vector_store.close()
        self.clear_cache()
        logger.info("RAG pipeline closed")