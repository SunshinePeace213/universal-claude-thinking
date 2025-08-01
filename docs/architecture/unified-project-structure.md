# Unified Project Structure

**Updated Structure**: Includes new components for memory system, RAG pipeline, and delegation engine

```
universal-claude-thinking-v2/
├── .claude/                           # Claude Code configuration directory
│   ├── agents/                        # Sub-agent specifications
│   │   ├── prompt-enhancer.md        # PE 🔧 Prompt enhancement specialist
│   │   ├── researcher.md             # R1 🔍 Research specialist
│   │   ├── reasoner.md               # A1 🧠 Reasoning specialist
│   │   ├── evaluator.md              # E1 📊 Quality evaluation specialist
│   │   ├── tool-user.md              # T1 🛠️ Tool orchestration specialist
│   │   ├── writer.md                 # W1 🖋️ Content creation specialist
│   │   └── interface.md              # I1 🗣️ User interface specialist
│   ├── hooks/                         # Hook scripts directory
│   │   ├── prompt_enhancer.py        # UserPromptSubmit hook + CoVe
│   │   ├── atomic_validator.py       # PreToolUse validation hook
│   │   ├── pattern_learner.py        # PostToolUse learning hook
│   │   ├── memory_persist.py         # Stop hook for memory consolidation
│   │   ├── agent_coordinator.py      # SubagentStop coordination
│   │   └── delegation_router.py      # New: 3-stage delegation routing
│   ├── commands/                      # Command templates directory
│   │   ├── monitor/                  # System monitoring and health checks
│   │   ├── setup/                    # Installation and configuration utilities  
│   │   ├── debug/                    # Troubleshooting and diagnostic tools
│   │   ├── maintain/                 # Cleanup and optimization operations
│   │   └── report/                   # Analytics and data reporting
│   └── settings.json                 # Claude Code hooks configuration
│
├── src/                               # Core source code
│   ├── __init__.py
│   ├── core/                          # Context Engineering layers
│   │   ├── __init__.py
│   │   ├── atomic/                    # Layer 1: Atomic Foundation
│   │   │   ├── __init__.py
│   │   │   ├── analyzer.py           # Prompt structure analysis
│   │   │   ├── scorer.py             # Quality scoring system
│   │   │   └── enhancer.py           # Enhancement suggestions
│   │   ├── molecular/                 # Layer 2: Molecular Enhancement
│   │   │   ├── __init__.py
│   │   │   ├── context_builder.py    # Context assembly
│   │   │   ├── example_selector.py   # Dynamic example selection
│   │   │   └── vector_store.py       # Semantic similarity
│   │   ├── cellular/                  # Layer 3: Cellular Memory
│   │   │   ├── __init__.py
│   │   │   ├── memory_orchestrator.py # Memory coordination
│   │   │   ├── short_term.py         # 2-hour memory window
│   │   │   ├── working.py            # 7-day memory window
│   │   │   └── long_term.py          # Persistent memory
│   │   ├── organ/                     # Layer 4: Organ Orchestration
│   │   │   ├── __init__.py
│   │   │   ├── orchestrator.py       # Multi-agent coordination
│   │   │   ├── task_analyzer.py      # Task decomposition
│   │   │   ├── workflow_engine.py    # Workflow patterns
│   │   │   └── result_synthesizer.py # Result combination
│   │   ├── cognitive/                 # Layer 5: Cognitive Tools
│   │   │   ├── __init__.py
│   │   │   ├── prompt_programs.py    # Structured reasoning
│   │   │   ├── context_schemas.py    # Information organization
│   │   │   ├── recursive_engine.py   # Self-improvement
│   │   │   └── protocol_shells.py    # Communication templates
│   │   └── programming/               # Layer 6: Prompt Programming
│   │       ├── __init__.py
│   │       ├── function_registry.py   # Cognitive functions
│   │       ├── composer.py            # Function composition
│   │       ├── meta_engine.py         # Meta-programming
│   │       └── executor.py            # Function execution
│   │
│   ├── agents/                        # Sub-agent implementations
│   │   ├── __init__.py
│   │   ├── base.py                   # Base sub-agent class
│   │   ├── manager.py                # Sub-agent manager
│   │   └── implementations/          # Specific sub-agents
│   │       ├── __init__.py
│   │       └── [agent implementations]
│   │
│   ├── memory/                        # 5-Layer memory system implementations
│   │   ├── __init__.py
│   │   ├── layers/                    # Memory layer implementations
│   │   │   ├── stm.py                # Short-term memory (2h)
│   │   │   ├── wm.py                 # Working memory (7d)
│   │   │   ├── ltm.py                # Long-term memory (∞)
│   │   │   ├── swarm.py              # SWARM community memory
│   │   │   └── privacy.py            # Privacy engine
│   │   ├── promotion.py              # Memory promotion pipeline
│   │   ├── consolidation.py          # Memory consolidation engine
│   │   ├── storage/                   # Storage backends
│   │   │   ├── __init__.py
│   │   │   ├── sqlite_vec.py         # SQLite + sqlite-vec backend
│   │   │   └── postgres_pgvector.py  # PostgreSQL + pgvector backend
│   │   └── embeddings.py             # Qwen3 embedding integration
│   │
│   ├── hooks/                         # Hook implementations
│   │   ├── __init__.py
│   │   ├── base.py                   # Base hook class
│   │   └── [hook modules]            # Hook implementations
│   │
│   ├── delegation/                    # Hybrid delegation engine
│   │   ├── __init__.py
│   │   ├── engine.py                 # 3-stage delegation engine
│   │   ├── keyword_matcher.py        # Stage 1: Fast keyword matching
│   │   ├── semantic_matcher.py       # Stage 2: Semantic embedding
│   │   ├── pe_fallback.py            # Stage 3: PE enhancement
│   │   └── confidence_scorer.py      # Delegation confidence scoring
│   │
│   ├── rag/                           # Hybrid RAG pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py               # Main RAG orchestrator
│   │   ├── embedder.py               # Qwen3-Embedding-8B integration
│   │   ├── reranker.py               # Qwen3-Reranker-8B integration
│   │   ├── custom_scorer.py          # Hybrid scoring algorithm
│   │   ├── chunker.py                # Semantic document chunking
│   │   └── reference_display.py      # Memory reference formatting
│   │
│   ├── integrations/                  # External integrations
│   │   ├── __init__.py
│   │   ├── mcp/                      # MCP ecosystem
│   │   │   ├── __init__.py
│   │   │   ├── client.py             # MCP client
│   │   │   └── adapter.py            # Tool adaptation
│   │   └── github.py                 # GitHub integration
│   │
│   └── utils/                         # Utility modules
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── logging.py                 # Structured logging
│       ├── metrics.py                 # Performance metrics
│       └── types.py                   # Type definitions
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── unit/                          # Unit tests
│   │   ├── test_atomic.py
│   │   ├── test_molecular.py
│   │   └── [other unit tests]
│   ├── integration/                   # Integration tests
│   │   ├── test_workflows.py
│   │   ├── test_memory.py
│   │   └── [other integration tests]
│   ├── cognitive/                     # Cognitive tests
│   │   ├── test_reasoning.py
│   │   ├── test_functions.py
│   │   └── [cognitive quality tests]
│   └── fixtures/                      # Test fixtures
│       └── [test data and mocks]
│
├── docs/                              # Documentation
│   ├── architecture.md                # This document
│   ├── api/                           # API documentation
│   ├── guides/                        # User guides
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── advanced.md
│   └── cognitive/                     # Cognitive patterns docs
│       └── [pattern documentation]
│
├── examples/                          # Usage examples
│   ├── basic/                         # Basic examples
│   │   ├── simple_prompt.py
│   │   └── memory_usage.py
│   ├── advanced/                      # Advanced examples
│   │   ├── multi_agent.py
│   │   ├── cognitive_functions.py
│   │   └── custom_hooks.py
│   └── patterns/                      # Cognitive patterns
│       └── [pattern examples]
│
├── scripts/                           # Setup and utility scripts
│   ├── install.py                     # Installation script
│   ├── setup_hooks.py                 # Hook configuration
│   ├── memory/                        # Memory management scripts
│   │   ├── consolidate.py            # Memory consolidation
│   │   ├── promote.py                # Memory promotion
│   │   ├── export.py                 # Export memories
│   │   └── import.py                 # Import memories
│   ├── rag/                           # RAG maintenance scripts
│   │   ├── index_documents.py        # Build vector indices
│   │   ├── optimize_embeddings.py    # Optimize embeddings
│   │   └── benchmark_retrieval.py    # Performance testing
│   ├── migrate_memory.py              # Memory migration
│   └── validate_cognitive.py          # Cognitive validation
│
├── data/                              # Data directory
│   ├── vectors/                       # Local vector storage
│   │   ├── embeddings.db             # sqlite-vec embeddings
│   │   └── indices/                  # Vector indices
│   ├── memories/                      # Memory database
│   │   ├── thinking_v2.db            # Main memory storage
│   │   └── backups/                  # Memory backups
│   ├── cache/                         # Prompt caching
│   │   ├── attention_states/         # Cached attention states
│   │   └── prompts/                  # Cached prompt segments
│   ├── cognitive_tools/               # Cognitive tool templates
│   ├── examples/                      # Example database
│   └── patterns/                      # Pattern library
│
├── .github/                           # GitHub configuration
│   ├── workflows/                     # CI/CD workflows
│   │   ├── test.yml
│   │   ├── deploy.yml
│   │   └── cognitive_validation.yml
│   └── ISSUE_TEMPLATE/                # Issue templates
│
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Python dependencies
├── README.md                          # Project README
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
└── .gitignore                         # Git ignore rules
```

## Directory Structure Explanation

### Core Directories

1. **`.claude/`** - Claude Code specific configuration
   - Contains sub-agent specifications as markdown files
   - Hook scripts for Claude Code integration
   - Settings configuration for hook registration

2. **`src/`** - Main source code directory
   - **`core/`** - Implementation of 7-layer Context Engineering
   - **`agents/`** - Sub-agent management and implementations
   - **`memory/`** - Memory systems including SWARM integration
   - **`hooks/`** - Hook implementations for Claude Code events
   - **`integrations/`** - External service integrations (MCP, GitHub)
   - **`utils/`** - Shared utilities and helpers

3. **`tests/`** - Comprehensive test suite
   - **`unit/`** - Unit tests for individual components
   - **`integration/`** - Integration tests for workflows
   - **`cognitive/`** - Specialized tests for cognitive quality

4. **`docs/`** - Documentation
   - Architecture documentation (this file)
   - API reference documentation
   - User guides and tutorials
   - Cognitive pattern documentation

5. **`examples/`** - Working examples
   - Basic usage examples
   - Advanced multi-agent workflows
   - Cognitive pattern demonstrations

6. **`scripts/`** - Utility scripts
   - Installation and setup automation
   - Hook configuration helpers
   - Migration and validation tools

---
