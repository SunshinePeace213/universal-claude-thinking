# Epic 2: Multi-Agent Specialist Coordination

**Epic Goal**: Implement the complete Enhanced Sub-Agent Architecture with all 7 specialized sub-agents operating through individual context windows. Enable true parallel processing, sophisticated specialist coordination, and seamless integration with MCP tools. This epic transforms the system from basic cognitive processing to advanced multi-agent intelligence with measurable performance improvements through parallel specialist operation.

## Story 2.1: Prompt-Enhancer Sub-Agent with Delegation Fallback
As a **user providing prompts to the system**,  
I want **proactive prompt enhancement and delegation fallback handling**,  
so that **ambiguous requests are clarified before processing**.

### Acceptance Criteria
1. PE operates as Stage 3 fallback when delegation confidence <0.7
2. Automatic quality assessment with CoVe validation for consistency
3. Generate 3-5 specific clarification options for ambiguous inputs
4. Enhancement includes examples for each clarification path
5. Multi-turn clarification support with context preservation
6. Auto-invoke rule enables PE without explicit user request
7. Track clarification success rates for delegation improvement
8. Integration with Qwen3-Embedding-8B for semantic enhancement analysis

## Story 2.2: Researcher Sub-Agent with Delegation Mapping
As a **user requiring information gathering**,  
I want **expert research with intelligent task routing**,  
so that **research requests are automatically delegated with high confidence**.

### Acceptance Criteria
1. R1 capability embeddings include ["web search", "data gathering", "source verification"]
2. Keyword triggers: ["find", "search", "gather", "investigate", "lookup", "explore"]
3. Delegation confidence >0.9 for clear research requests
4. Utilizes Tavily MCP (20-70x performance with caching), Context7, GitHub tools
5. SEIQF framework with structured credibility scoring
6. Map-Reduce pattern for parallel research across multiple sources
7. Integration with RAG pipeline for memory-augmented search
8. Batch processing: 32 documents/batch for embedding generation

## Story 2.3: Reasoner Sub-Agent with Advanced Techniques
As a **user with complex analytical needs**,  
I want **systematic reasoning with CoVe and ReAct patterns**,  
so that **I receive validated insights with reduced hallucination**.

### Acceptance Criteria
1. A1 capability embeddings: ["logical analysis", "problem solving", "reasoning chains"]
2. Implements CoVe for 30-50% hallucination reduction through verification
3. ReAct pattern for interleaved reasoning and action steps
4. Clear-thought MCP integration with self-consistency validation
5. SAGE framework with quantified bias scores
6. Mental models selection based on problem type
7. Delegation triggers: ["solve", "analyze", "deduce", "evaluate", "think"]
8. Processing on Mac M3 with MPS optimization for reasoning chains

## Story 2.4: Evaluator Sub-Agent with Auto-Invoke
As a **system user expecting high-quality outputs**,  
I want **automatic quality validation with performance tracking**,  
so that **all outputs meet standards without manual checking**.

### Acceptance Criteria
1. E1 auto-invoke rule: activates for all agent outputs without request
2. Capability embeddings: ["quality assessment", "validation", "error detection"]
3. Multi-dimensional scoring with weighted metrics for task types
4. Integration with memory effectiveness scoring (+0.3/-0.3 adjustments)
5. Quality gates: minimum 7/10 with automatic enhancement triggers
6. Cross-specialist validation for parallel processing consistency
7. Performance metrics feed back to delegation confidence scores
8. Batch validation: 8 outputs simultaneously on Mac M3

## Story 2.5: Tool-User Sub-Agent Automation
As a **user requiring external actions and integrations**,  
I want **systematic tool orchestration and execution**,  
so that **complex automation tasks are performed safely and efficiently**.

### Acceptance Criteria
1. Tool-User sub-agent manages all MCP tools, Bash operations, and file manipulations
2. Systematic planning includes safety validation and execution strategies
3. Error handling and recovery procedures manage tool failures gracefully
4. Tool orchestration patterns support sequential, parallel, and pipeline execution
5. Results validation confirms successful execution with quality assessment
6. Integration with other sub-agents provides tool capabilities for research, analysis, and content creation

## Story 2.6: Writer Sub-Agent Content Creation
As a **user needing content generation**,  
I want **expert content creation with style adaptation**,  
so that **I receive polished, audience-appropriate content that meets professional standards**.

### Acceptance Criteria
1. Writer sub-agent implements structured content generation with iterative refinement
2. Style adaptation framework adjusts tone, format, and complexity for target audiences
3. Content types include professional documents, creative content, and technical documentation
4. Quality standards ensure clarity, coherence, completeness, conciseness, and correctness
5. Integration with Researcher and Reasoner sub-agents enables fact-based, well-reasoned content
6. Content output includes quality metrics and refinement history tracking

## Story 2.7: Interface Sub-Agent User Communication
As a **user interacting with the multi-agent system**,  
I want **personalized communication and context translation**,  
so that **complex technical outputs are presented in accessible, user-friendly formats**.

### Acceptance Criteria
1. Interface sub-agent maintains user profile with communication preferences and expertise levels
2. Context translation converts technical specialist outputs into user-friendly explanations
3. Personalization adapts communication style, complexity, and format to user preferences
4. Feedback integration incorporates user responses for continuous communication improvement
5. Coordination with all specialists ensures consistent user experience across different processing types
6. Communication quality metrics track clarity, accessibility, and user satisfaction

## Story 2.8: Parallel Processing with Mac M3 Optimization
As a **system coordinator managing multiple sub-agents**,  
I want **optimized parallel processing on Apple Silicon**,  
so that **complex tasks leverage full hardware capabilities**.

### Acceptance Criteria
1. Configure PyTorch with MPS backend: PYTORCH_ENABLE_MPS_FALLBACK=1
2. Optimal batch sizes: 32 texts/batch (embeddings), 8 pairs/batch (reranking)
3. Memory allocation for 128GB: ~35GB models, ~65GB vectors, ~25GB working
4. Map-Reduce with unified memory for zero-copy tensor operations
5. Achieve 80% GPU utilization during batch processing
6. 3x performance improvement with <5% quality variance
7. Graceful degradation for systems with less memory
8. MLX framework integration for native M3 acceleration
