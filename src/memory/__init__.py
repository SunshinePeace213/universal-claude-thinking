"""Memory system integration for cognitive architecture."""

from .config import MemoryConfig
from .embeddings import MemoryEmbedder

# Memory layers
from .layers.base import MemoryItem, MemoryLayer, MemoryType
from .layers.ltm import LongTermMemory
from .layers.stm import ShortTermMemory
from .layers.swarm import SwarmMemory
from .layers.wm import WorkingMemory
from .privacy import AnonymizationMethod, PIIType, PrivacyEngine
from .promotion import PromotionPipeline
from .scoring import EffectivenessScorer, MemoryScore

# Storage backends
from .storage.base import StorageBackend
from .storage.memory_cache import InMemoryCache
from .storage.sqlite_storage import SQLiteStorage

__all__ = [
    # Configuration
    "MemoryConfig",

    # Core classes
    "MemoryLayer",
    "MemoryItem",
    "MemoryType",

    # Memory layers
    "ShortTermMemory",
    "WorkingMemory",
    "LongTermMemory",
    "SwarmMemory",

    # Storage
    "StorageBackend",
    "SQLiteStorage",
    "InMemoryCache",

    # Supporting systems
    "MemoryEmbedder",
    "PrivacyEngine",
    "PIIType",
    "AnonymizationMethod",
    "PromotionPipeline",
    "EffectivenessScorer",
    "MemoryScore",
]
