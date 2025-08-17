"""
Molecular Layer - Context Engineering Layer 2.

This layer implements intelligent context assembly through:
- Dynamic example selection using semantic similarity
- MOLECULE structure formatting
- Vector storage and retrieval
- Effectiveness tracking and feedback
"""

__all__ = [
    "VectorStore",
    "MoleculeContextBuilder",
    "ExampleSelector",
]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vector_store import VectorStore
    from .context_builder import MoleculeContextBuilder
    from .example_selector import ExampleSelector