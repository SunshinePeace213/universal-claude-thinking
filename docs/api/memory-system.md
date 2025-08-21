# Memory System API Documentation

## Overview

The Universal Claude Thinking v2 Memory System provides a hierarchical 5-layer memory architecture for managing context and learning from user interactions. The system includes Short-Term Memory (STM), Working Memory (WM), Long-Term Memory (LTM), SWARM community memory preparation, and a comprehensive privacy engine.

## Core Components

### Memory Layers

#### ShortTermMemory (STM)

**Purpose**: Manages immediate context with a 2-hour TTL.

```python
from src.memory.layers.stm import ShortTermMemory

stm = ShortTermMemory(
    cache_size: int = 1000,  # Maximum cache entries
    ttl_hours: float = 2.0    # Time-to-live in hours
)

# Initialize the memory layer
await stm.initialize()

# Store a memory
memory_id = await stm.store(memory: MemoryItem) -> str

# Retrieve a memory
memory = await stm.retrieve(memory_id: str) -> Optional[MemoryItem]

# Get promotion candidates
candidates = await stm.get_promotion_candidates(
    min_score: float = 5.0
) -> List[MemoryItem]

# Clean up expired memories
removed_count = await stm.cleanup_expired() -> int
```

#### WorkingMemory (WM)

**Purpose**: Maintains recent patterns with a 7-day TTL.

```python
from src.memory.layers.wm import WorkingMemory

wm = WorkingMemory(
    db_path: str = "data/memories/working_memory.db",
    ttl_days: float = 7.0,
    max_size: int = 10000
)

await wm.initialize()

# Store memory with automatic TTL
memory_id = await wm.store(memory: MemoryItem) -> str

# Retrieve with usage tracking
memory = await wm.retrieve(
    memory_id: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 10
) -> Union[MemoryItem, List[MemoryItem], None]

# Vector similarity search
results = await wm.search(
    query_embedding: np.ndarray,
    k: int = 10,
    threshold: float = 0.85
) -> List[MemoryItem]

# Get LTM promotion candidates
candidates = await wm.get_candidates_for_promotion(
    min_effectiveness: float = 8.0,
    min_usage: int = 5
) -> List[MemoryItem]
```

#### LongTermMemory (LTM)

**Purpose**: Permanent storage for high-value patterns.

```python
from src.memory.layers.ltm import LongTermMemory

ltm = LongTermMemory(
    db_path: str = "data/memories/long_term_memory.db",
    max_size: int = 100000
)

await ltm.initialize()

# Store permanent memory (no expiration)
memory_id = await ltm.store(memory: MemoryItem) -> str

# Retrieve by category
memories = await ltm.get_by_category(
    category: str,
    user_id: str,
    limit: int = 20
) -> List[MemoryItem]

# Search by tags
memories = await ltm.search_by_tags(
    tags: List[str],
    user_id: str,
    match_all: bool = False
) -> List[MemoryItem]
```

### Memory Item

```python
from src.memory.layers.base import MemoryItem, MemoryType

memory = MemoryItem(
    id: str,                           # Unique identifier
    user_id: str,                      # User association
    memory_type: str,                  # "stm", "wm", "ltm", "swarm"
    content: Dict[str, Any],          # Memory content
    embedding: Optional[np.ndarray],   # 4096-dim vector
    metadata: Dict[str, Any] = {},    # Additional metadata
    effectiveness_score: float = 5.0,  # Effectiveness (0-10)
    usage_count: int = 0,              # Access count
    created_at: datetime,              # Creation timestamp
    last_accessed: datetime,           # Last access time
    expires_at: Optional[datetime],    # Expiration (None for LTM)
    promoted_from: Optional[str],      # Source layer
    promoted_at: Optional[datetime],   # Promotion time
    promotion_reason: Optional[str]    # Promotion rationale
)

# Methods
memory.is_expired() -> bool
memory.set_ttl(ttl_hours: float = None, ttl_days: float = None)
memory.to_dict() -> Dict[str, Any]
```

### Privacy Engine

**Purpose**: Detects and removes PII before storage and sharing.

```python
from src.memory.privacy import PrivacyEngine, PIIType

privacy = PrivacyEngine(
    enable_ner: bool = True,           # Use spaCy NER
    confidence_threshold: float = 0.8  # Detection confidence
)

# Detect PII in text
detections = privacy.detect_pii(text: str) -> List[PIIDetection]

# Remove PII from text
sanitized = privacy.remove_pii(
    text: str,
    method: str = "token"  # "token", "mask", or "hash"
) -> str

# Anonymize memory item
anonymized = privacy.anonymize_memory(
    memory: MemoryItem,
    method: str = "token"
) -> MemoryItem

# Prepare for SWARM sharing
swarm_ready = privacy.prepare_for_swarm(
    memory: MemoryItem
) -> Optional[MemoryItem]

# Verify k-anonymity
is_anonymous = privacy.verify_k_anonymity(
    memories: List[MemoryItem],
    k: int = 3
) -> bool
```

### Promotion Pipeline

**Purpose**: Automates memory promotion through layers.

```python
from src.memory.promotion import PromotionPipeline

pipeline = PromotionPipeline(
    storage: StorageBackend,
    scorer: EffectivenessScorer,
    config: MemoryConfig
)

# Start automatic promotion scheduler
await pipeline.start()

# Manual promotion evaluation
promoted_to_wm = await pipeline.evaluate_stm_to_wm() -> List[MemoryItem]
promoted_to_ltm = await pipeline.evaluate_wm_to_ltm() -> List[MemoryItem]

# Manual promotion
success = await pipeline.promote_memory(
    memory: MemoryItem,
    to_layer: str  # "wm" or "ltm"
) -> bool

# Get promotion statistics
stats = pipeline.get_statistics() -> Dict[str, Any]

# Stop scheduler
await pipeline.stop()
```

### Effectiveness Scorer

**Purpose**: Manages memory effectiveness scoring and feedback.

```python
from src.memory.scoring import EffectivenessScorer, FeedbackType

scorer = EffectivenessScorer(
    positive_delta: float = 0.3,  # Positive feedback boost
    negative_delta: float = -0.3, # Negative feedback penalty
    usage_boost: float = 0.1,     # Per-usage increase
    decay_rate: float = 0.01      # Daily decay rate
)

# Apply feedback
scorer.apply_feedback(
    memory_id: str,
    feedback: FeedbackType  # POSITIVE or NEGATIVE
)

# Get current score
score = scorer.get_score(memory_id: str) -> float

# Track usage
scorer.track_usage(memory_id: str)

# Apply time decay
scorer.apply_decay(memory_id: str, days_elapsed: float)

# Batch operations
scores = scorer.batch_get_scores(
    memory_ids: List[str]
) -> Dict[str, float]

# Get top memories
top_memories = scorer.get_top_memories(
    n: int = 10,
    min_score: float = 5.0
) -> List[Tuple[str, float]]
```

### Memory Embedder

**Purpose**: Generates embeddings for semantic search.

```python
from src.memory.embeddings import MemoryEmbedder

embedder = MemoryEmbedder(
    model_path: str = "embedding/Qwen3-Embedding-8B",
    dimension: int = 4096,
    batch_size: int = 32,
    cache_size: int = 1000,
    use_mps: bool = True  # Mac M3 acceleration
)

await embedder.initialize()

# Single memory embedding
embedding = await embedder.generate_memory_embedding(
    memory: MemoryItem,
    instruction_prefix: Optional[str] = None
) -> np.ndarray

# Batch embedding
embeddings = await embedder.batch_embed_memories(
    memories: List[MemoryItem],
    show_progress: bool = True
) -> np.ndarray

# Calculate similarities
similarities = embedder.calculate_similarities(
    query_embedding: np.ndarray,
    memory_embeddings: np.ndarray
) -> np.ndarray

# Vector search
results = await embedder.search_similar(
    query: str,
    memory_embeddings: np.ndarray,
    memories: List[MemoryItem],
    k: int = 10,
    threshold: float = 0.7
) -> List[Tuple[MemoryItem, float]]
```

### Storage Backend

**Purpose**: Persistent storage abstraction.

```python
from src.memory.storage import SQLiteStorage

storage = SQLiteStorage(
    db_path: str = "data/memories/memories.db",
    connection_pool_size: int = 10
)

await storage.initialize()

# CRUD operations
await storage.store(memory: MemoryItem) -> str
await storage.retrieve(memory_id: str) -> Optional[MemoryItem]
await storage.update(memory: MemoryItem) -> bool
await storage.delete(memory_id: str) -> bool

# Batch operations
await storage.batch_store(memories: List[MemoryItem])
await storage.batch_retrieve(memory_ids: List[str]) -> List[MemoryItem]

# Query operations
await storage.list_by_user(
    user_id: str,
    memory_type: Optional[str] = None,
    limit: int = 100
) -> List[MemoryItem]

await storage.get_by_criteria(
    memory_type: str,
    min_effectiveness: float,
    min_usage: int = 0,
    limit: int = 100
) -> List[MemoryItem]

# Vector search
await storage.search_by_embedding(
    embedding: np.ndarray,
    k: int = 10,
    user_id: Optional[str] = None,
    memory_type: Optional[str] = None
) -> List[Tuple[MemoryItem, float]]

# Cleanup
await storage.cleanup_expired() -> int
await storage.close()
```

### Configuration

```python
from src.memory.config import MemoryConfig, load_config

# Load from YAML
config = load_config("config/memory_config.yaml")

# Or create programmatically
config = MemoryConfig(
    stm={
        'ttl_hours': 2,
        'cache_size': 1000
    },
    wm={
        'ttl_days': 7,
        'promotion_threshold': 5.0
    },
    ltm={
        'promotion_score': 8.0,
        'promotion_uses': 5
    },
    swarm={
        'enabled': False,
        'anonymization_required': True
    },
    privacy={
        'pii_detection': True,
        'anonymization': True,
        'confidence_threshold': 0.8
    },
    promotion={
        'stm_check_interval': 3600,  # 1 hour
        'wm_check_interval': 86400   # 1 day
    },
    performance={
        'max_retrieval_latency_ms': 100,
        'max_promotion_latency_ms': 500,
        'max_pii_detection_latency_ms': 50
    }
)

# Access configuration
config.get('stm.ttl_hours') -> float
config.update('wm.promotion_threshold', 6.0)
config.validate() -> bool
```

## Error Handling

All memory operations can raise the following exceptions:

```python
from src.memory.exceptions import (
    MemoryError,           # Base exception
    StorageError,          # Storage operation failed
    PromotionError,        # Promotion failed
    PrivacyError,          # Privacy validation failed
    ConfigurationError     # Invalid configuration
)

try:
    memory_id = await stm.store(memory)
except StorageError as e:
    logger.error(f"Failed to store memory: {e}")
except MemoryError as e:
    logger.error(f"Memory operation failed: {e}")
```

## Performance Requirements

| Operation | Requirement | Notes |
|-----------|------------|-------|
| Single Retrieval | <100ms | From any layer |
| Vector Search | <150ms | Top-10 results |
| Batch Store (32 items) | <500ms | With embeddings |
| Promotion Evaluation | <500ms | Per batch |
| PII Detection | <50ms | Per item |
| Cache Retrieval | <100μs | In-memory |

## Integration with RAG Pipeline

The memory system integrates with the RAG pipeline to provide context-aware responses:

```python
from src.memory import MemorySystem
from src.rag.pipeline import RAGPipeline

# Initialize memory system
memory_system = MemorySystem(config)
await memory_system.initialize()

# RAG pipeline with memory context
rag = RAGPipeline(
    memory_system=memory_system,
    embedder=embedder
)

# Process query with memory context
response = await rag.process(
    query="What are my coding preferences?",
    user_id="user_123",
    use_memory=True,
    memory_layers=["stm", "wm", "ltm"]
)
```

## Best Practices

1. **TTL Management**: Set appropriate TTLs based on content volatility
2. **Promotion Thresholds**: Tune thresholds based on usage patterns
3. **Privacy First**: Always sanitize PII before storage
4. **Batch Operations**: Use batch methods for multiple items
5. **Embedding Cache**: Cache embeddings to avoid recomputation
6. **Connection Pooling**: Use connection pools for database access
7. **Async Operations**: Use async/await for all I/O operations
8. **Error Recovery**: Implement retry logic for transient failures
9. **Monitoring**: Track performance metrics and memory usage
10. **Testing**: Include integration tests for memory workflows