# Data Architecture & Storage Systems

## Database Schema Design

```sql
-- Core Context Engineering Tables
CREATE TABLE atomic_analyses (
    id UUID PRIMARY KEY,
    prompt_hash TEXT UNIQUE,
    structure JSONB NOT NULL,
    quality_score DECIMAL(3,1) CHECK (quality_score >= 1 AND quality_score <= 10),
    gaps JSONB,
    suggestions JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_count INTEGER DEFAULT 0
);

CREATE TABLE molecular_contexts (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    task_type VARCHAR(50),
    instruction TEXT,
    examples JSONB,
    context JSONB,
    effectiveness_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_task (user_id, task_type)
);

-- 5-Layer Memory System Tables
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    memory_type VARCHAR(20) CHECK (memory_type IN ('stm', 'wm', 'ltm', 'swarm')),
    content JSONB NOT NULL,
    embedding BLOB,  -- Binary storage for vector
    metadata JSONB,
    effectiveness_score DECIMAL(3,1) DEFAULT 5.0,
    usage_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,  -- NULL for ltm/swarm
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Promotion tracking
    promoted_from VARCHAR(20),
    promoted_at TIMESTAMP,
    promotion_reason TEXT,
    INDEX idx_user_memory (user_id, memory_type),
    INDEX idx_effectiveness (effectiveness_score DESC),
    INDEX idx_expiration (expires_at)
);

-- Vector storage with sqlite-vec
CREATE VIRTUAL TABLE memory_vectors USING vec0(
    id TEXT PRIMARY KEY,
    embedding FLOAT[1536]
);

-- Memory promotion tracking
CREATE TABLE memory_promotions (
    id UUID PRIMARY KEY,
    memory_id UUID REFERENCES memories(id),
    from_type VARCHAR(20),
    to_type VARCHAR(20),
    promotion_score DECIMAL(3,1),
    promoted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT
);

-- Organ Orchestration Tables
CREATE TABLE orchestration_workflows (
    id UUID PRIMARY KEY,
    workflow_name VARCHAR(100),
    pattern_type VARCHAR(50),
    task_decomposition JSONB,
    agent_assignments JSONB,
    execution_plan JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE specialist_coordination (
    id UUID PRIMARY KEY,
    workflow_id UUID REFERENCES orchestration_workflows(id),
    source_agent VARCHAR(50),
    target_agent VARCHAR(50),
    message_type VARCHAR(50),
    payload JSONB,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_workflow_agents (workflow_id, source_agent, target_agent)
);

-- Cognitive Tools Tables
CREATE TABLE cognitive_tools (
    id UUID PRIMARY KEY,
    tool_name VARCHAR(100) UNIQUE,
    tool_type VARCHAR(50),
    category VARCHAR(50),
    template JSONB NOT NULL,
    parameters JSONB,
    effectiveness_metrics JSONB,
    usage_statistics JSONB,
    version VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cognitive_functions (
    id UUID PRIMARY KEY,
    function_name VARCHAR(100) UNIQUE,
    signature JSONB NOT NULL,
    parameters JSONB NOT NULL,
    function_body TEXT NOT NULL,
    return_type VARCHAR(50),
    metadata JSONB,
    performance_stats JSONB,
    version VARCHAR(20),
    created_by TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_function_search (function_name, created_by)
);

-- SWARM Community Tables
CREATE TABLE swarm_patterns (
    id UUID PRIMARY KEY,
    pattern_type VARCHAR(50),
    pattern_data JSONB NOT NULL,
    effectiveness_score DECIMAL(3,2),
    usage_count INTEGER DEFAULT 0,
    anonymized_metrics JSONB,
    community_rating DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_pattern_effectiveness (pattern_type, effectiveness_score DESC)
);
```

## Memory System Architecture

```python
class FiveLayerMemorySystem:
    """
    Implements 5-layer memory architecture with intelligent promotion pipeline.
    Provides persistent, cross-session intelligence with community learning.
    """
    
    def __init__(self, storage: StorageBackend, embedder: Qwen3Embedding8B):
        self.storage = storage
        self.embedder = embedder
        self.privacy_engine = PrivacyEngine()
        
        # Memory layers with TTL
        self.layers = {
            'stm': ShortTermMemory(ttl_hours=2),
            'wm': WorkingMemory(ttl_days=7),
            'ltm': LongTermMemory(ttl=None),  # Permanent
            'swarm': SwarmMemory(privacy_engine=self.privacy_engine)
        }
        
    async def store_memory(
        self,
        user_id: str,
        memory_type: str,
        content: Any,
        metadata: Dict[str, Any]
    ) -> MemoryRecord:
        """Store memory with appropriate processing for each layer."""
        # Generate embedding using Qwen3
        instruction = "Represent this memory for semantic search:"
        embedding = await self.embedder.encode(f"{instruction} {content}")
        
        # Apply layer-specific processing
        if memory_type == 'swarm':
            # Privacy processing for community sharing
            content = await self.privacy_engine.anonymize(content)
            metadata['anonymized'] = True
            
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            metadata=metadata,
            effectiveness_score=5.0,
            created_at=datetime.utcnow(),
            expires_at=self._calculate_expiry(memory_type)
        )
        
        await self.storage.save(record)
        await self._store_vector(record.id, embedding)
        
        return record
    
    async def promote_memories(self):
        """Automated memory promotion pipeline."""
        # STM → WM promotion (score > 5.0)
        stm_candidates = await self.storage.find(
            memory_type='stm',
            effectiveness_score={'$gte': 5.0},
            expires_at={'$lte': datetime.utcnow() + timedelta(hours=1)}
        )
        
        for memory in stm_candidates:
            await self._promote_memory(memory, 'stm', 'wm')
        
        # WM → LTM promotion (score > 8.0, uses > 5)
        wm_candidates = await self.storage.find(
            memory_type='wm',
            effectiveness_score={'$gte': 8.0},
            usage_count={'$gte': 5}
        )
        
        for memory in wm_candidates:
            await self._promote_memory(memory, 'wm', 'ltm')
        
        # LTM → SWARM promotion (community value)
        ltm_candidates = await self.storage.find(
            memory_type='ltm',
            effectiveness_score={'$gte': 9.0},
            metadata__general_applicability=True
        )
        
        for memory in ltm_candidates:
            if await self.privacy_engine.validate_for_sharing(memory):
                await self._promote_memory(memory, 'ltm', 'swarm')
```

## Local Vector Database Architecture

### **Privacy-First Vector Operations**

```python
class LocalVectorStore:
    """
    Local-only vector database with complete privacy preservation.
    Integrates Qwen3-Embedding-8B for high-quality embeddings.
    """
    
    def __init__(self, db_path: str = "data/thinking_v2.db"):
        self.db_path = db_path
        # Initialize Qwen3-Embedding-8B for local embeddings
        self.embedder = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
        self.embedder.to('mps')  # Mac M3 GPU acceleration
        
        # Configuration for Mac M3 Max 128GB
        self.batch_size = 32  # Optimal for M3
        self.embedding_dim = 1536  # Reduced from 4096 for efficiency
        self.privacy_config = PrivacyConfig()
        
    async def embed_and_store(
        self,
        content: str,
        metadata: Dict[str, Any],
        user_id: str
    ) -> VectorRecord:
        """Generate embeddings locally and store in user's database."""
        # Add instruction for better embedding quality
        instruction = "Represent this text for retrieval:"
        full_text = f"{instruction} {content}"
        
        # Generate embeddings on local M3 GPU
        with torch.no_grad():
            embedding = self.embedder.encode(
                full_text,
                normalize_embeddings=True,
                show_progress_bar=False,
                device='mps'
            )
        
        # Reduce dimension if needed
        if self.embedding_dim < 4096:
            embedding = self._reduce_dimension(embedding, self.embedding_dim)
        
        # Store in user's local database with sqlite-vec
        record = VectorRecord(
            content=content,
            embedding=embedding,
            metadata=metadata,
            user_id=user_id,
            created_at=datetime.utcnow()
        )
        
        await self._store_with_sqlite_vec(record)
        return record
        
    async def hybrid_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 100,
        rerank_top_k: int = 10
    ) -> List[RankedResult]:
        """Perform hybrid search with embedding retrieval + reranking."""
        # Stage 1: Embedding-based retrieval
        query_embedding = await self._encode_query(query)
        candidates = await self._vector_similarity_search(
            query_embedding=query_embedding,
            user_id=user_id,
            limit=top_k
        )
        
        # Stage 2: Would integrate with Qwen3-Reranker-8B here
        # (Reranking implementation handled by RAG pipeline)
        
        return candidates
```

### **Community Sharing Architecture (Optional)**

```yaml
community_sharing:
  enabled: false  # Disabled by default
  privacy_mode: "differential_privacy"
  
  opt_in_only:
    description: "Users must explicitly enable community sharing"
    data_shared: "anonymized_patterns_only"
    personal_data: "never_shared"
    
  anonymization:
    techniques:
      - differential_privacy: "Add statistical noise to prevent re-identification"
      - k_anonymity: "Ensure patterns represent minimum k users"
      - data_minimization: "Share only essential pattern information"
      
  local_control:
    user_controls:
      - "opt_out_anytime": "Complete withdrawal from community sharing"
      - "selective_sharing": "Choose which patterns to share"
      - "data_deletion": "Request complete data removal"
      
  technical_implementation:
    sharing_method: "federated_learning"
    data_residence: "always_local"
    aggregation: "secure_multiparty_computation"
```

### **Installation Methods for Vector Database**

**Method 1: SQLite with FTS5 (Development)**
```bash