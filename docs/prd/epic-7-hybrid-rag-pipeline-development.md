# Epic 7: Hybrid RAG Pipeline Development

**Epic Goal**: Implement the two-stage retrieval system combining Qwen3-Embedding-8B for fast candidate retrieval and Qwen3-Reranker-8B for precise reranking, with custom scoring and Mac M3 optimization. This epic transforms memory search from simple keyword matching to intelligent semantic understanding with dramatically improved accuracy and performance.

**Business Value**:
- **20-70x Performance Improvement**: Leverage prompt caching and GPU acceleration
- **49% Reduction in Failed Retrievals**: Based on similar hybrid search implementations
- **Sub-200ms Total Latency**: Instant responses for interactive use
- **Improved Answer Quality**: Better context leads to more accurate responses

**Technical Scope**:
- Bi-encoder (Qwen3-Embedding-8B) for fast initial retrieval
- Cross-encoder (Qwen3-Reranker-8B) for accurate reranking
- Hybrid recursive-semantic chunking with 1024-token segments
- Custom scoring combining semantic, keyword, recency, and effectiveness
- Mac M3 MPS optimization for batch processing

## Story 7.1: Embedding-Based Retrieval Implementation
As a **RAG system requiring fast initial retrieval**,  
I want **to quickly find relevant candidates using vector similarity**,  
so that **the system can process large document collections efficiently**.

**Detailed Description**: The bi-encoder approach enables pre-computation of document embeddings, allowing fast similarity search at query time. Using Qwen3-Embedding-8B, the system generates high-quality 4096-dimensional embeddings (reduced to 1536 for efficiency) that capture semantic meaning for accurate initial retrieval.

### Acceptance Criteria
1. Integrate Qwen3-Embedding-8B with sentence-transformers framework
2. Generate embeddings with instruction prefixes for better performance
3. Implement dimension reduction from 4096 to 1536 using PCA
4. Achieve <50ms embedding generation for single queries
5. Support batch processing of 32 texts simultaneously on M3
6. Store embeddings in sqlite-vec for local vector operations
7. Maintain 95% recall@100 for test query sets

## Story 7.2: Hybrid Chunking Strategy
As a **system processing chat histories**,  
I want **intelligent document chunking that preserves context**,  
so that **retrieved chunks contain complete, meaningful information**.

**Detailed Description**: The hybrid recursive-semantic approach combines structural awareness (respecting conversation boundaries) with semantic validation (ensuring coherence). This strategy is specifically optimized for chat histories, maintaining conversation flow while enabling precise retrieval.

### Acceptance Criteria
1. Implement recursive splitting with conversation-aware boundaries
2. Target 1024-token chunks with 15% (154-token) overlap
3. Validate semantic coherence with 0.85 similarity threshold
4. Preserve conversation turn integrity (never split Q&A pairs)
5. Add comprehensive metadata (speakers, timestamps, turn numbers)
6. Handle edge cases (code blocks, long responses) gracefully
7. Achieve 90% user satisfaction on chunk completeness

## Story 7.3: Cross-Encoder Reranking Implementation
As a **RAG pipeline requiring precise relevance assessment**,  
I want **to rerank initial candidates using joint query-document encoding**,  
so that **the most semantically relevant results are prioritized**.

**Detailed Description**: While embeddings excel at finding broadly similar content, they miss nuanced query-document relationships. The cross-encoder processes query-document pairs jointly, understanding contextual relationships, implicit requirements, and semantic nuances that bi-encoders miss. This dramatically improves result quality, especially for complex technical queries.

### Acceptance Criteria
1. Integrate Qwen3-Reranker-8B with transformers library
2. Process top-100 candidates from embedding retrieval
3. Generate relevance scores using cross-attention mechanisms
4. Achieve 85%+ precision@10 on benchmark datasets
5. Maintain <200ms latency for reranking 100 documents
6. Support dynamic batch sizing (8 pairs/batch) for M3 optimization
7. Provide confidence scores and explanation capabilities

## Story 7.4: Custom Scoring Layer
As a **system with domain-specific requirements**,  
I want **multi-factor scoring beyond pure semantic similarity**,  
so that **results are optimized for actual user needs**.

**Detailed Description**: Pure semantic similarity isn't always optimal for practical use. The custom scoring layer combines multiple signals including keyword matches (for exact terms), recency (for temporal relevance), and historical effectiveness (learning from usage) to provide results that match real-world needs.

### Acceptance Criteria
1. Implement weighted scoring: semantic (50%), keyword (20%), recency (15%), effectiveness (15%)
2. Support BM25 keyword scoring for exact match requirements
3. Calculate time-based decay with configurable half-life
4. Track and apply historical effectiveness scores
5. Enable dynamic weight adjustment based on query type
6. Provide score breakdown for transparency
7. Support A/B testing of scoring strategies

## Story 7.5: Mac M3 Performance Optimization
As a **system running on Apple Silicon**,  
I want **hardware-specific optimizations for maximum performance**,  
so that **users experience fast, responsive retrieval**.

**Detailed Description**: Mac M3's unified memory architecture and Metal Performance Shaders provide unique optimization opportunities. This story implements M3-specific configurations, batch sizes, and memory management strategies to achieve optimal performance on Apple Silicon.

### Acceptance Criteria
1. Configure PyTorch with MPS backend and optimization flags
2. Implement optimal batch sizes (32 for embeddings, 8 for reranking)
3. Utilize unified memory for zero-copy tensor operations
4. Achieve 80% GPU utilization during batch processing
5. Implement memory allocation strategy for 128GB systems
6. Support graceful degradation on lower-memory systems
7. Provide performance monitoring and auto-tuning

## Story 7.6: User Feedback Integration
As a **system that learns from usage**,  
I want **to incorporate user feedback into retrieval rankings**,  
so that **search quality improves over time**.

**Detailed Description**: User interactions provide valuable signals about retrieval quality. This feedback system adjusts memory effectiveness scores based on implicit (dwell time, selection) and explicit (thumbs up/down) feedback, creating a learning loop that continuously improves retrieval accuracy.

### Acceptance Criteria
1. Capture implicit feedback (clicks, dwell time, copy actions)
2. Support explicit feedback with simple UI (helpful/not helpful)
3. Adjust effectiveness scores (+0.3 positive, -0.3 negative)
4. Implement feedback decay to prevent old signals from dominating
5. Track feedback attribution to specific retrieval strategies
6. Generate feedback analytics for system optimization
7. Support feedback export for model retraining
