"""
MOLECULE Context Builder.

Implements the MOLECULE structure for few-shot learning:
[INSTRUCTION] + [EXAMPLES] + [CONTEXT] + [NEW INPUT]
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class MoleculeSection(Enum):
    """Sections of the MOLECULE structure."""
    
    INSTRUCTION = "INSTRUCTION"
    EXAMPLES = "EXAMPLES"
    CONTEXT = "CONTEXT"
    NEW_INPUT = "NEW_INPUT"


@dataclass
class TokenAllocation:
    """Token allocation for each MOLECULE section."""
    
    instruction: int = 256
    examples: int = 512
    context: int = 256
    new_input: int = 0  # Calculated based on input
    
    @property
    def total_allocated(self) -> int:
        """Total tokens allocated."""
        return self.instruction + self.examples + self.context + self.new_input
        
    def adjust_for_limit(self, limit: int = 1024) -> None:
        """Adjust allocations to fit within limit."""
        if self.total_allocated <= limit:
            return
            
        # Proportionally reduce each section
        scale = limit / self.total_allocated
        self.instruction = int(self.instruction * scale)
        self.examples = int(self.examples * scale)
        self.context = int(self.context * scale)
        self.new_input = int(self.new_input * scale)


@dataclass
class MoleculeContext:
    """Complete MOLECULE context structure."""
    
    instruction: str
    examples: List[Dict[str, str]]
    context: str
    new_input: str
    token_count: int
    similarity_scores: List[float] = field(default_factory=list)
    
    def format(self) -> str:
        """Format as complete MOLECULE structure."""
        parts = []
        
        # INSTRUCTION section
        if self.instruction:
            parts.append(f"## INSTRUCTION\n{self.instruction}\n")
            
        # EXAMPLES section
        if self.examples:
            parts.append("## EXAMPLES")
            for i, example in enumerate(self.examples, 1):
                parts.append(f"\n### Example {i}")
                if "input" in example:
                    parts.append(f"Input: {example['input']}")
                if "output" in example:
                    parts.append(f"Output: {example['output']}")
                    
        # CONTEXT section
        if self.context:
            parts.append(f"\n## CONTEXT\n{self.context}\n")
            
        # NEW INPUT section
        if self.new_input:
            parts.append(f"\n## NEW INPUT\n{self.new_input}")
            
        return "\n".join(parts)


class MoleculeContextBuilder:
    """
    Builds context following the MOLECULE structure for few-shot learning.
    
    Manages token allocation, chunk overlap, and similarity-based prioritization
    to achieve 10-30% accuracy improvement.
    """
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        chunk_size: int = 1024,
        overlap_ratio: float = 0.15,
        max_examples: int = 10,
    ) -> None:
        """
        Initialize the context builder.
        
        Args:
            tokenizer_name: Name of tokenizer for token counting
            chunk_size: Size of context chunks in tokens
            overlap_ratio: Overlap ratio between chunks (0.15 = 15%)
            max_examples: Maximum number of examples to include
        """
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_tokens = int(chunk_size * overlap_ratio)
        self.max_examples = max_examples
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: estimate ~4 characters per token
            return len(text) // 4
            
    def build_context(
        self,
        instruction: str,
        examples: List[Dict[str, Any]],
        context: str,
        new_input: str,
        similarity_scores: Optional[List[float]] = None,
    ) -> MoleculeContext:
        """
        Build MOLECULE-structured context.
        
        Args:
            instruction: Task instruction
            examples: List of example dictionaries with 'input'/'output' keys
            context: Additional context information
            new_input: The new input to process
            similarity_scores: Similarity scores for example prioritization
            
        Returns:
            Formatted MOLECULE context
        """
        start_time = time.time()
        
        # Initialize token allocation
        allocation = TokenAllocation()
        allocation.new_input = self.count_tokens(new_input)
        allocation.adjust_for_limit(self.chunk_size)
        
        # Process instruction
        processed_instruction = self._truncate_to_tokens(
            instruction,
            allocation.instruction
        )
        
        # Process and prioritize examples
        processed_examples = self._select_examples(
            examples,
            allocation.examples,
            similarity_scores
        )
        
        # Process context
        processed_context = self._truncate_to_tokens(
            context,
            allocation.context
        )
        
        # Build final context
        molecule_context = MoleculeContext(
            instruction=processed_instruction,
            examples=processed_examples,
            context=processed_context,
            new_input=new_input,
            token_count=allocation.total_allocated,
            similarity_scores=similarity_scores or [],
        )
        
        # Verify timing constraint (<800ms for total context construction)
        construction_time_ms = (time.time() - start_time) * 1000
        if construction_time_ms > 800:
            logger.warning(f"Context construction took {construction_time_ms:.2f}ms (>800ms target)")
        else:
            logger.debug(f"Context built in {construction_time_ms:.2f}ms")
            
        return molecule_context
        
    def _select_examples(
        self,
        examples: List[Dict[str, Any]],
        token_budget: int,
        similarity_scores: Optional[List[float]] = None,
    ) -> List[Dict[str, str]]:
        """
        Select examples based on similarity scores and token budget.
        
        Args:
            examples: Available examples
            token_budget: Token budget for examples section
            similarity_scores: Similarity scores for prioritization
            
        Returns:
            Selected and formatted examples
        """
        if not examples:
            return []
            
        # Sort by similarity if scores provided
        if similarity_scores and len(similarity_scores) == len(examples):
            sorted_pairs = sorted(
                zip(examples, similarity_scores),
                key=lambda x: x[1],
                reverse=True
            )
            examples = [ex for ex, _ in sorted_pairs]
            
        selected = []
        tokens_used = 0
        
        for example in examples[:self.max_examples]:
            # Format example
            formatted = {}
            if "input" in example:
                formatted["input"] = str(example["input"])
            if "output" in example:
                formatted["output"] = str(example["output"])
                
            # Check token budget
            example_tokens = self.count_tokens(str(formatted))
            if tokens_used + example_tokens > token_budget:
                break
                
            selected.append(formatted)
            tokens_used += example_tokens
            
        logger.debug(f"Selected {len(selected)} examples using {tokens_used} tokens")
        return selected
        
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if not text:
            return ""
            
        tokens = self.count_tokens(text)
        if tokens <= max_tokens:
            return text
            
        # Binary search for right truncation point
        left, right = 0, len(text)
        while left < right:
            mid = (left + right + 1) // 2
            if self.count_tokens(text[:mid]) <= max_tokens:
                left = mid
            else:
                right = mid - 1
                
        return text[:left]
        
    def create_chunks(
        self,
        contexts: List[MoleculeContext],
    ) -> List[str]:
        """
        Create overlapping chunks from multiple contexts.
        
        Args:
            contexts: List of MOLECULE contexts
            
        Returns:
            List of formatted chunks with overlap
        """
        chunks = []
        full_text = "\n\n---\n\n".join([c.format() for c in contexts])
        
        # Split into chunks with overlap
        tokens = self.tokenizer.encode(full_text) if self.tokenizer else None
        
        if tokens:
            # Token-based chunking
            for i in range(0, len(tokens), self.chunk_size - self.overlap_tokens):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if i + self.chunk_size >= len(tokens):
                    break
        else:
            # Character-based chunking (fallback)
            chunk_chars = self.chunk_size * 4  # Estimate
            overlap_chars = self.overlap_tokens * 4
            
            for i in range(0, len(full_text), chunk_chars - overlap_chars):
                chunk = full_text[i:i + chunk_chars]
                chunks.append(chunk)
                
                if i + chunk_chars >= len(full_text):
                    break
                    
        logger.info(f"Created {len(chunks)} chunks with {self.overlap_ratio*100}% overlap")
        return chunks
        
    def calculate_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score [0, 1]
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure in [0, 1] range
        return float(np.clip(similarity, 0, 1))
        
    def prioritize_by_similarity(
        self,
        items: List[Any],
        similarities: List[float],
        threshold: float = 0.85,
    ) -> List[Tuple[Any, float]]:
        """
        Prioritize items by similarity scores.
        
        Args:
            items: Items to prioritize
            similarities: Similarity scores
            threshold: Minimum similarity threshold
            
        Returns:
            Sorted list of (item, similarity) tuples above threshold
        """
        if len(items) != len(similarities):
            raise ValueError("Items and similarities must have same length")
            
        # Filter by threshold
        filtered = [
            (item, sim)
            for item, sim in zip(items, similarities)
            if sim >= threshold
        ]
        
        # Sort by similarity (highest first)
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Prioritized {len(filtered)}/{len(items)} items above {threshold} threshold")
        return filtered