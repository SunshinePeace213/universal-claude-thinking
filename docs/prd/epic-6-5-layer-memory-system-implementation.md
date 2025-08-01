# Epic 6: 5-Layer Memory System Implementation

**Epic Goal**: Build the complete hierarchical memory system with STM→WM→LTM→SWARM→Privacy layers, including automatic promotion pipelines, vector-based retrieval, and privacy-preserving community learning. This epic delivers persistent intelligence that learns from every interaction while maintaining user privacy and enabling community-driven improvement through anonymized pattern sharing.

**Business Value**:
- **95% Cross-Session Continuity**: Users never need to re-explain context or preferences
- **Community Learning Acceleration**: Solutions discovered by one user benefit all users
- **Privacy-First Trust**: Complete data sovereignty with zero external dependencies
- **Intelligent Context Retrieval**: Sub-100ms semantic search across all memory layers

**Technical Scope**:
- SQLite with sqlite-vec for complete local vector storage
- Qwen3-Embedding-8B integration for 1536-dim semantic embeddings
- Automated promotion pipeline with configurable effectiveness thresholds
- Privacy engine for comprehensive PII stripping and anonymization
- SWARM integration for opt-in community intelligence sharing

## Story 6.1: Short-Term Memory (STM) Implementation
As a **system maintaining session context**,  
I want **to capture and store immediate interactions with 2-hour TTL**,  
so that **current conversation context is preserved during active work sessions**.

**Detailed Description**: STM acts as the system's working notepad, storing everything from the current session including user requests, system responses, intermediate calculations, and temporary state. Unlike traditional stateless interactions, STM enables the system to reference earlier parts of the conversation, track what solutions have been attempted, and maintain context for complex multi-step tasks. The 2-hour TTL balances memory efficiency with practical session duration, automatically expiring old memories to prevent unbounded growth.

### Acceptance Criteria
1. Store all session interactions with <50ms write latency using in-memory cache
2. Maintain conversation thread integrity with proper turn sequencing and timestamps
3. Track attempted solutions and their outcomes to avoid repetition
4. Calculate real-time effectiveness scores based on user interactions and feedback
5. Identify high-value patterns for potential WM promotion (score >5)
6. Handle session recovery after unexpected disconnections with context restoration
7. Support concurrent session isolation for multiple parallel tasks

## Story 6.2: Working Memory (WM) Bridge Implementation
As a **system bridging multiple sessions**,  
I want **to preserve recent patterns and insights for 7 days**,  
so that **users experience continuity across separate work sessions**.

**Detailed Description**: WM serves as the bridge between ephemeral session memory and permanent knowledge, capturing patterns that prove useful across multiple interactions. This includes discovered user preferences, successful debugging approaches, frequently used code patterns, and project-specific context. The 7-day retention allows the system to maintain relevance while automatically pruning outdated information.

### Acceptance Criteria
1. Automatically promote STM patterns with effectiveness score >5.0
2. Store patterns with semantic embeddings for similarity search
3. Maintain usage counters and last-accessed timestamps for all patterns
4. Support pattern merging when similar memories are promoted
5. Enable manual pattern pinning for important temporary knowledge
6. Implement age-based decay with configurable retention policies
7. Provide pattern effectiveness analytics for optimization

## Story 6.3: Long-Term Memory (LTM) Persistence
As a **user with established workflows and preferences**,  
I want **permanent storage of proven patterns and knowledge**,  
so that **the system becomes increasingly personalized and effective over time**.

**Detailed Description**: LTM represents the system's core knowledge about the user, including expertise profile, coding preferences, project patterns, and highly effective solutions. Only the most valuable patterns (effectiveness >8.0, used >5 times) are promoted to LTM, ensuring quality over quantity. This permanent knowledge base enables truly personalized interactions.

### Acceptance Criteria
1. Promote WM patterns meeting strict quality criteria (>8.0 score, >5 uses)
2. Build comprehensive user expertise profile from interaction patterns
3. Store project-specific knowledge with contextual metadata
4. Implement version control for evolving patterns and preferences
5. Support knowledge export for backup and portability
6. Enable selective memory deletion for privacy control
7. Maintain knowledge graph relationships between related memories

## Story 6.4: Privacy Engine Development
As a **privacy-conscious user**,  
I want **complete control over my data with anonymization capabilities**,  
so that **I can benefit from community learning without compromising privacy**.

**Detailed Description**: The Privacy Engine acts as the gatekeeper between personal memories and community sharing, implementing comprehensive PII detection, data anonymization, and consent management. It ensures that patterns shared with the community contain no identifying information while preserving their learning value.

### Acceptance Criteria
1. Implement comprehensive PII detection using pattern matching and NLP
2. Strip all identifiable information including names, paths, URLs, and IDs
3. Generalize specific patterns to remove user-specific context
4. Require explicit opt-in consent for any community sharing
5. Provide transparency reports showing what data would be shared
6. Support selective sharing with granular control options
7. Maintain audit logs of all privacy-related operations

## Story 6.5: SWARM Community Intelligence
As a **member of the developer community**,  
I want **to benefit from collective learning while maintaining privacy**,  
so that **common problems are solved once and shared by all**.

**Detailed Description**: SWARM represents the collective intelligence layer where anonymized patterns from multiple users are aggregated to identify universally valuable solutions. This opt-in system accelerates problem-solving by learning from the community's collective experience while maintaining strict privacy boundaries.

### Acceptance Criteria
1. Aggregate anonymized patterns from opted-in users
2. Identify patterns used successfully by >3 independent users
3. Calculate community effectiveness scores for shared patterns
4. Distribute high-value patterns back to all users
5. Implement pattern attribution and contribution tracking
6. Support pattern voting and quality feedback mechanisms
7. Enable selective SWARM participation by pattern category

## Story 6.6: Memory Retrieval & Reference System
As a **user seeking relevant context**,  
I want **transparent memory retrieval with clear attribution**,  
so that **I understand what information the system is using and why**.

**Detailed Description**: The retrieval system provides fast, relevant access to memories across all layers using semantic search. It displays clear references showing which memories influenced responses, including their type (STM/WM/LTM/SWARM), relevance scores, and temporal context, building trust through transparency.

### Acceptance Criteria
1. Implement semantic search using Qwen3-Embedding-8B with <100ms latency
2. Display memory references with type, age, and relevance scores
3. Support explicit memory queries for debugging and exploration
4. Provide memory usage statistics and effectiveness metrics
5. Enable memory pinning to ensure specific memories are always considered
6. Implement relevance feedback to improve retrieval accuracy
7. Support memory search filters by type, time range, and effectiveness
