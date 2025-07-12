# Unified Project Structure

```
universal-thinking-claude/
├── .github/                    # CI/CD workflows
│   └── workflows/
│       ├── test.yaml           # Module validation tests
│       └── security.yaml       # Security scanning
├── .claude/                    # Claude Code modules
│   ├── CLAUDE.md              # Core orchestrator
│   ├── thinking-modules/       # Protocol modules
│   │   ├── SAGE.md
│   │   ├── SEIQF.md
│   │   ├── SIA.md
│   │   └── response-formats.md
│   ├── cognitive-tools/        # Thinking operations
│   │   ├── understanding.md
│   │   ├── reasoning.md
│   │   └── verification.md
│   ├── config/                 # Configuration
│   │   ├── metadata.yaml
│   │   ├── triggers.yaml
│   │   └── agents.yaml
│   └── shared/                 # Shared state
│       └── protocol-state.json
├── scripts/                    # Utility scripts
│   ├── validate-modules.sh
│   ├── calculate-tokens.py
│   └── migrate-from-v3.sh
├── tests/                      # Test suites
│   ├── unit/
│   ├── integration/
│   └── performance/
├── docs/                       # Documentation
│   ├── prd.md
│   ├── architecture.md
│   └── migration-guide.md
├── .env.example               # Environment template
├── package.json               # Node.js config
└── README.md                  # Project overview
```
