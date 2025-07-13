# Backend Architecture

## Service Architecture

### Module Organization

```
.claude/
├── CLAUDE.md                    # Core orchestrator (~500 tokens)
├── thinking-modules/
│   ├── SAGE.md                  # Bias detection (~2K tokens)
│   ├── SEIQF.md                 # Info quality (~3K tokens)
│   ├── SIA.md                   # Intent analysis (~2K tokens)
│   └── response-formats.md      # Output templates (~1K tokens)
├── cognitive-tools/
│   ├── understanding.md         # Comprehension ops (~500 tokens)
│   ├── reasoning.md             # Analytical ops (~500 tokens)
│   └── verification.md          # Validation ops (~500 tokens)
├── config/
│   ├── metadata.yaml            # Module registry
│   ├── triggers.yaml            # Classification rules
│   └── agents.yaml              # Agent definitions
└── shared/
    └── protocol-state.json      # Runtime state
```

### Orchestrator Template (CLAUDE.md)

```markdown
---
version: 1.0.0
tokens: 500
---
```
