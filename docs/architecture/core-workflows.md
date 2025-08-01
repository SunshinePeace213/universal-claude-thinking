# Core Workflows

This section provides a comprehensive view of the system's operational workflows, organized hierarchically from high-level system flows to detailed technical implementations. Each subsection focuses on a specific aspect of the architecture, with diagrams that build upon each other to create a complete picture.

## 1. Primary System Flows

These are the main operational workflows that define how the system processes requests from end to end.

### User Interaction Workflow
The primary entry point for all user interactions, showing how requests flow through the enhancement, processing, and response pipeline.

```mermaid
sequenceDiagram
    participant User
    participant CLI as Claude Code CLI
    participant Hook as Prompt Enhancer Hook
    participant Atomic as Atomic Layer
    participant Molecular as Molecular Layer
    participant Organ as Orchestrator
    participant Agents as Sub-Agents
    participant Memory as Memory System
    
    User->>CLI: Submit prompt
    CLI->>Hook: UserPromptSubmit event
    Hook->>Atomic: Analyze prompt quality
    
    alt Quality Score < 7
        Atomic->>Hook: Enhancement suggestions
        Hook->>User: "Do you mean X, Y, or Z?"
        User->>Hook: Select enhancement
        Hook->>Atomic: Enhanced prompt
    end
    
    Atomic->>Molecular: Structure + Quality score
    Molecular->>Memory: Retrieve relevant context
    Memory->>Molecular: Past patterns + preferences
    Molecular->>Organ: Enhanced context
    
    Organ->>Organ: Task decomposition
    Organ->>Agents: Delegate to specialists
    
    par Parallel Processing
        Agents->>Agents: Researcher gathers info
    and
        Agents->>Agents: Reasoner analyzes
    and
        Agents->>Agents: Evaluator validates
    end
    
    Agents->>Organ: Specialist results
    Organ->>Memory: Store patterns
    Organ->>CLI: Synthesized response
    CLI->>User: Final output
```

### Multi-Agent Coordination Workflow
Demonstrates how the orchestrator manages multiple sub-agents for complex tasks, including task classification, parallel execution, and result synthesis.

```mermaid
sequenceDiagram
    participant Orchestrator
    participant Task as Task Analyzer
    participant WF as Workflow Engine
    participant PE as Prompt Enhancer
    participant R as Researcher
    participant Re as Reasoner
    participant E as Evaluator
    participant W as Writer
    participant Synth as Result Synthesizer
    
    Orchestrator->>Task: Analyze task complexity
    Task->>Task: Classify (A/B/C/D/E)
    Task->>WF: Task decomposition
    
    alt Type C (Research)
        WF->>R: Parallel research tasks
        R->>R: Tavily search
        R->>R: Context7 docs
        R->>R: GitHub search
        R->>WF: Research results
        
        WF->>Re: Analyze findings
        Re->>Re: SAGE framework
        Re->>Re: Mental models
        Re->>WF: Analysis results
    else Type B (Complex)
        WF->>PE: Check prompts
        PE->>WF: Enhanced prompts
        
        par Parallel Execution
            WF->>R: Gather context
        and
            WF->>Re: Initial analysis
        end
        
        WF->>E: Validate approach
        E->>WF: Quality metrics
    end
    
    WF->>W: Generate output
    W->>W: Style adaptation
    W->>E: Final validation
    E->>WF: Approved output
    
    WF->>Synth: All results
    Synth->>Orchestrator: Final response
```

### Cognitive Function Contribution Workflow
Shows the complete lifecycle of community-contributed cognitive functions, from development through validation to deployment.

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant Local as Local Testing
    participant Val as Validator
    participant GH as GitHub
    participant Reg as Function Registry
    participant Users as Community
    
    Dev->>Dev: Create cognitive function
    Dev->>Local: Test locally
    Local->>Val: Validate function
    
    Val->>Val: Syntax check
    Val->>Val: Type validation
    Val->>Val: Security scan
    Val->>Val: Performance test
    
    alt Validation passed
        Dev->>GH: Submit PR
        GH->>GH: Automated checks
        GH->>Users: Peer review
        
        alt Approved
            GH->>Reg: Merge to registry
            Reg->>Users: Available for use
            Users->>Users: Rate & feedback
        else Changes needed
            Users->>Dev: Feedback
            Dev->>Dev: Revise function
        end
    else Validation failed
        Val->>Dev: Improvement suggestions
        Dev->>Dev: Fix issues
    end
```

## 2. Delegation System

The delegation system intelligently routes user requests to the appropriate sub-agents using a hybrid approach that combines fast keyword matching with sophisticated semantic understanding.

### Hybrid Delegation Flow
```mermaid
flowchart TD
    subgraph "User Input Processing"
        UI[User Input] --> DE[Delegation Engine]
    end
    
    subgraph "Stage 1: Keyword Matching"
        DE --> KW[Keyword Analysis]
        KW --> KP{Pattern<br/>Match?}
        KP -->|Yes| KR[Return Agent<br/>Confidence: 0.9]
        KP -->|No| ST2[Go to Stage 2]
    end
    
    subgraph "Stage 2: Semantic Matching"
        ST2 --> EMB[Encode Input<br/>Qwen3-Embedding]
        EMB --> COMP[Compare with<br/>Agent Embeddings]
        COMP --> SIM[Calculate<br/>Similarities]
        SIM --> CONF{Confidence<br/>> 0.7?}
        CONF -->|Yes| SR[Return Best Agent]
        CONF -->|No| ST3[Go to Stage 3]
    end
    
    subgraph "Stage 3: PE Fallback"
        ST3 --> PE[Route to PE<br/>Prompt Enhancer]
        PE --> ENH[Enhancement<br/>Required]
        ENH --> CLAR[User<br/>Clarification]
    end
    
    KR --> EXEC[Execute with Agent]
    SR --> EXEC
    CLAR --> DE
```

### Agent Selection Logic
Combines agent capabilities, task mapping, and confidence scoring to determine the optimal agent for each request.

```mermaid
graph TB
    subgraph "Input Analysis"
        UI[User Input] --> KW[Keyword Check]
        UI --> SEM[Semantic Analysis]
        KW --> CONF[Confidence Score]
        SEM --> CONF
    end
    
    subgraph "Confidence-Based Routing"
        CONF --> HIGH{Score > 0.7?}
        HIGH -->|Yes| DIRECT[Direct Agent Selection]
        HIGH -->|No| MED{Score > 0.4?}
        MED -->|Yes| PE_ENH[PE Enhancement]
        MED -->|No| CLARIFY[User Clarification]
    end
    
    subgraph "Agent Capabilities & Task Mapping"
        DIRECT --> AGENTS{Select Agent}
        PE_ENH --> AGENTS
        
        AGENTS --> PE[PE 🔧<br/>Unclear/Ambiguous]
        AGENTS --> R1[R1 🔍<br/>Research/Search]
        AGENTS --> A1[A1 🧠<br/>Analysis/Reasoning]
        AGENTS --> E1[E1 📊<br/>Validation/QA]
        AGENTS --> T1[T1 🛠️<br/>Tool Execution]
        AGENTS --> W1[W1 🖋️<br/>Content Creation]
        AGENTS --> I1[I1 🗣️<br/>User Interface]
    end
    
    subgraph "Auto-Invoke Rules"
        AUTO[Always Active:<br/>PE, E1, I1]
        MANUAL[On-Demand:<br/>R1, A1, T1, W1]
    end
    
    PE --> |Enhancement| ALL[All Agents]
    E1 --> |Validation| ALL
    I1 --> |Communication| ALL
    
    style HIGH fill:#90ee90
    style MED fill:#ffeb3b
    style CLARIFY fill:#ff9800
```

### Advanced Delegation Patterns
Demonstrates how the system handles edge cases with low-confidence inputs and coordinates multiple agents for complex multi-step tasks.

```mermaid
graph TB
    subgraph "Edge Case: Ambiguous Input"
        AMB["Do the thing with the stuff"] --> LOWCONF[Low Confidence<br/>Score: 0.35]
        LOWCONF --> PEROUTE[Route to PE]
        PEROUTE --> CLARIFY[User Clarification:<br/>1. Process data?<br/>2. Execute command?<br/>3. Generate docs?]
        CLARIFY --> ENHANCED["Process CSV files"]
        ENHANCED --> HIGHCONF[High Confidence<br/>Score: 0.9]
        HIGHCONF --> T1ROUTE[Route to T1<br/>Tool User]
    end
    
    subgraph "Complex Task Coordination"
        COMPLEX[Research, analyze,<br/>and write report] --> DECOMP[Task Decomposition]
        
        DECOMP --> PHASE1[Phase 1: Research]
        DECOMP --> PHASE2[Phase 2: Analysis]
        DECOMP --> PHASE3[Phase 3: Writing]
        
        PHASE1 --> |Parallel| R1A[R1: Web Search]
        PHASE1 --> |Parallel| R1B[R1: Doc Search]
        PHASE1 --> |Parallel| R1C[R1: GitHub Search]
        
        R1A --> GATHER[Gather Results]
        R1B --> GATHER
        R1C --> GATHER
        
        GATHER --> A1[A1: Analyze Data]
        A1 --> W1[W1: Write Report]
        W1 --> E1[E1: Validate]
        E1 --> I1[I1: Present]
    end
    
    style LOWCONF fill:#ff9800
    style HIGHCONF fill:#90ee90
    style PHASE1 fill:#e3f2fd
    style PHASE2 fill:#f3e5f5
    style PHASE3 fill:#e8f5e9
```

## 3. Memory System

The memory system provides intelligent context persistence across sessions, with a multi-tiered architecture that promotes valuable patterns while maintaining user privacy.

### Memory Persistence Workflow
Shows the primary memory flow from short-term storage through working memory to long-term persistence and optional community sharing.

```mermaid
sequenceDiagram
    participant Session
    participant STM as Short-Term Memory
    participant WM as Working Memory
    participant LTM as Long-Term Memory
    participant SWARM
    participant Privacy as Privacy Engine
    
    Session->>STM: Store interaction
    STM->>STM: 2-hour TTL
    
    alt Pattern detected
        STM->>WM: Promote pattern
        WM->>WM: 7-day TTL
        
        alt High value pattern
            WM->>LTM: Persist pattern
            LTM->>Privacy: Anonymize
            Privacy->>SWARM: Share pattern
            SWARM->>SWARM: Community learning
        end
    end
    
    Note over Session: Session ends
    Session->>STM: Consolidate
    STM->>WM: Transfer relevant
    WM->>LTM: Update user profile
```

### Memory Architecture & Session Lifecycle
Provides a comprehensive view of memory types, promotion criteria, and how sessions interact with the memory system.

```mermaid
graph TB
    subgraph "Session Flow"
        START[Session Start] --> INIT[Initialize Context]
        INIT --> ACTIVE[Active Session]
        ACTIVE --> END[Session End]
        END --> CONSOLIDATE[Consolidate Patterns]
    end
    
    subgraph "Memory Hierarchy"
        STM[Short-Term Memory<br/>2h TTL<br/>Current Context]
        WM[Working Memory<br/>7d TTL<br/>Recent Patterns]
        LTM[Long-Term Memory<br/>Permanent<br/>User Profile]
        SWARM[SWARM<br/>Community Patterns<br/>Opt-in Shared]
    end
    
    subgraph "Vector Storage"
        VDB[(Vector Database)]
        EMB[Embeddings Index]
        VDB --> EMB
    end
    
    subgraph "Privacy Controls"
        PRIV[Privacy Engine]
        ANON[Anonymization]
        PII[PII Detection]
        PRIV --> ANON
        PRIV --> PII
    end
    
    START --> |Load Profile| LTM
    START --> |Recent Context| WM
    ACTIVE --> |Store| STM
    ACTIVE --> |Search| VDB
    
    STM --> |Score > 5| WM
    WM --> |Score > 8 & Uses > 5| LTM
    LTM --> |Community Value| PRIV
    PRIV --> |Sanitized| SWARM
    
    CONSOLIDATE --> |Patterns| WM
    CONSOLIDATE --> |Profile Update| LTM
    CONSOLIDATE --> |Index Update| VDB
    
    style STM fill:#ffeb3b
    style WM fill:#90ee90
    style LTM fill:#64b5f6
    style SWARM fill:#e1bee7
```

### Memory Technical Implementation
Shows the promotion pipeline criteria and vector embedding process for semantic search capabilities.

```mermaid
flowchart TD
    subgraph "Promotion Pipeline"
        NEW[New Memory] --> SCORE{Quality Score}
        SCORE -->|< 5| EXPIRE1[Expire in 2h]
        SCORE -->|5-8| STM_WM[STM → WM<br/>7d retention]
        SCORE -->|> 8| CHECK_USES{Uses > 5?}
        CHECK_USES -->|No| EXPIRE2[Expire in 7d]
        CHECK_USES -->|Yes| WM_LTM[WM → LTM<br/>Permanent]
        
        WM_LTM --> COMMUNITY{Community<br/>Value?}
        COMMUNITY -->|No| PRIVATE[Keep Private]
        COMMUNITY -->|Yes| PII_CHECK{PII Free?}
        PII_CHECK -->|No| PRIVATE
        PII_CHECK -->|Yes| SWARM_SHARE[Share to SWARM]
    end
    
    subgraph "Vector Embedding Process"
        TEXT[Memory Text] --> INSTRUCT[Add Instruction<br/>Template]
        INSTRUCT --> QWEN[Qwen3-Embedding-8B]
        QWEN --> VEC4K[4096-dim Vector]
        VEC4K --> REDUCE[PCA Reduction]
        REDUCE --> VEC1K[1536-dim Vector]
        VEC1K --> NORM[L2 Normalize]
    end
    
    subgraph "Storage & Retrieval"
        NORM --> STORE{Store}
        STORE --> BLOB[SQLite Blob<br/>Binary Storage]
        STORE --> INDEX[sqlite-vec<br/>Vector Index]
        
        QUERY[Search Query] --> QEMBED[Query Embedding]
        QEMBED --> SEARCH[Similarity Search]
        INDEX --> SEARCH
        SEARCH --> TOPK[Top-K Results<br/>+ Scores]
    end
    
    style EXPIRE1 fill:#ffcdd2
    style EXPIRE2 fill:#ffcdd2
    style PRIVATE fill:#fff9c4
    style SWARM_SHARE fill:#c8e6c9
```

## 4. RAG Pipeline

The Retrieval-Augmented Generation (RAG) pipeline enables intelligent memory search using a two-stage approach: fast embedding retrieval followed by precise reranking.

### Complete RAG Pipeline
Illustrates the full retrieval process from query processing through embedding search, reranking, and final output generation.

```mermaid
graph TB
    subgraph "Input Processing"
        Q[User Query] --> QP[Query Processing]
        QP --> INS[Add Instruction]
    end
    
    subgraph "Stage 1: Embedding Retrieval"
        INS --> EMB[Qwen3-Embedding-8B<br/>Bi-Encoder]
        EMB --> QV[Query Vector]
        
        DC[Document Corpus] --> PE[Pre-computed<br/>Embeddings]
        PE --> VDB[(Vector Database)]
        
        QV --> VS[Vector Search]
        VDB --> VS
        VS --> TOP100[Top 100 Candidates]
    end
    
    subgraph "Stage 2: Reranking"
        TOP100 --> PAIRS[Query-Doc Pairs]
        PAIRS --> RR[Qwen3-Reranker-8B<br/>Cross-Encoder]
        RR --> SCORES[Relevance Scores]
        SCORES --> TOP10[Top 10 Results]
    end
    
    subgraph "Stage 3: Custom Scoring"
        TOP10 --> CS[Custom Scorer]
        CS --> SEM[Semantic Score 50%]
        CS --> KEY[Keyword Score 20%]
        CS --> REC[Recency Score 15%]
        CS --> EFF[Effectiveness 15%]
        CS --> FINAL[Final Ranking]
    end
    
    subgraph "Output"
        FINAL --> REF[Memory References]
        REF --> RES[Results + Context]
        RES --> USER[User Display]
    end
```

### RAG Technical Architecture
Explains the difference between bi-encoder embeddings for fast retrieval and cross-encoder reranking for precision, plus the document chunking strategy.

```mermaid
graph TB
    subgraph "Two-Stage Architecture"
        subgraph "Stage 1: Bi-Encoder (Fast)"
            Q1[Query] --> E1[Query Encoder]
            D1[Documents] --> E2[Doc Encoder]
            E1 --> V1[Query Vector]
            E2 --> V2[Doc Vectors]
            V1 --> COS[Cosine Similarity]
            V2 --> COS
            COS --> TOP100[Top 100 Candidates]
        end
        
        subgraph "Stage 2: Cross-Encoder (Precise)"
            TOP100 --> PAIRS[Query-Doc Pairs]
            PAIRS --> CONCAT[Concatenate]
            CONCAT --> CE[Cross-Encoder<br/>Model]
            CE --> SCORES[Relevance Scores]
            SCORES --> TOP10[Top 10 Results]
        end
    end
    
    subgraph "Document Chunking Strategy"
        DOC[Raw Document] --> BOUND[Detect Boundaries]
        BOUND --> SEMANTIC[Semantic Units:<br/>• Paragraphs<br/>• Sections<br/>• Sentences]
        SEMANTIC --> SIZE{Size Check}
        SIZE -->|< 1024 tokens| CHUNK[Valid Chunk]
        SIZE -->|> 1024 tokens| SPLIT[Recursive Split]
        SPLIT --> SIZE
        CHUNK --> OVERLAP[Add 15% Overlap]
        OVERLAP --> META[Add Metadata:<br/>• Source<br/>• Position<br/>• Type]
        META --> STORE[Store in Vector DB]
    end
    
    style E1 fill:#e1f5fe
    style E2 fill:#e1f5fe
    style CE fill:#ffebee
    style TOP100 fill:#fff9c4
    style TOP10 fill:#c8e6c9
```

### RAG Performance & Output
Shows how the system optimizes for Mac M3 hardware and displays results with memory references and feedback integration.

```mermaid
graph TB
    subgraph "Hardware Optimization"
        subgraph "Mac M3 Processing"
            QUERY[Query Batch] --> GPU1[MPS GPU:<br/>Embeddings]
            GPU1 --> BATCH1[32 texts/batch]
            BATCH1 --> CAND[100 Candidates]
            CAND --> GPU2[MPS GPU:<br/>Reranking]
            GPU2 --> BATCH2[8 pairs/batch]
            BATCH2 --> FINAL[Top Results]
        end
        
        subgraph "Memory Allocation"
            RAM[128GB Unified Memory]
            RAM --> MODELS[Models: ~35GB]
            RAM --> VECTORS[Vectors: ~65GB]
            RAM --> WORKING[Working: ~25GB]
            RAM --> SYSTEM[System: ~3GB]
        end
    end
    
    subgraph "Result Display & Feedback"
        FINAL --> FORMAT[Format Results]
        FORMAT --> REFS[Memory References:<br/>STM Recent 0.92<br/>WM Pattern 0.87<br/>LTM Knowledge 0.75]
        REFS --> DISPLAY[User Display]
        
        DISPLAY --> FEEDBACK{User Feedback}
        FEEDBACK -->|Positive| BOOST[Score +0.3]
        FEEDBACK -->|Negative| REDUCE[Score -0.3]
        FEEDBACK -->|Neutral| MAINTAIN[No Change]
        
        BOOST --> UPDATE[Update Memory<br/>Effectiveness]
        REDUCE --> UPDATE
        MAINTAIN --> UPDATE
    end
    
    style GPU1 fill:#90ee90
    style GPU2 fill:#90ee90
    style BOOST fill:#c8e6c9
    style REDUCE fill:#ffcdd2
```

## 5. Chunking Strategy

The RAG pipeline implements a sophisticated chunking strategy optimized for chat history storage and retrieval. This strategy ensures optimal context preservation while maintaining efficient retrieval performance.

### Chosen Approach: Hybrid Recursive-Semantic Chunking

After comprehensive analysis of various chunking methods (fixed-size, recursive, document-based, semantic, and agentic), we've selected a **Hybrid Recursive-Semantic** approach that combines the structural awareness of recursive chunking with the meaning preservation of semantic validation.

**Key Configuration**:
```yaml
chunking_configuration:
  strategy: "hybrid_recursive_semantic"
  chunk_size: 1024  # tokens - optimal for chat history
  chunk_overlap: 154  # tokens (15% of chunk_size)
  min_chunk_size: 256  # tokens (prevent fragments)
  max_chunk_size: 1536  # tokens (handle edge cases)
  
  rationale:
    - "1024 tokens captures 2-3 complete conversation turns"
    - "Balances retrieval precision with context completeness"
    - "Optimal for Qwen3-Embedding-8B processing"
    - "Aligns with memory promotion thresholds"
    - "Enables effective pattern recognition"
```

**Why 1024 Tokens?**
- **Perfect Balance**: Captures complete conversation exchanges without overwhelming context
- **Model Efficiency**: 25% of Qwen3's 4096 token limit, leaving room for query expansion
- **Memory Alignment**: Works seamlessly with STM→WM→LTM promotion pipeline
- **Performance**: Optimal batch processing on Mac M3 (32 texts/batch)

**Implementation Features**:
- Preserves conversation boundaries (user/assistant turns)
- Maintains semantic coherence with 0.85 threshold validation
- Tracks metadata: timestamps, speakers, effectiveness scores
- Supports pattern detection across chat histories

> **Note**: For detailed chunking strategy documentation, see [Chunking Strategy Architecture](architecture/chunking-strategy.md).
