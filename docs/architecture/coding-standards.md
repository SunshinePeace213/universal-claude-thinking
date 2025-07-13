# Coding Standards

## Critical Fullstack Rules

- **Module Imports:** Always use @import with relative paths from .claude/
- **State Updates:** Never mutate protocol state directly - use state manager
- **Security Validation:** All modules must have SHA-256 hash in metadata.yaml
- **Token Limits:** Each module must declare and respect its token budget
- **Error Handling:** All MCP failures must gracefully degrade to basic operation
- **Thinking Logs:** Always include emoji indicators for visual parsing
- **Module Dependencies:** Declare all dependencies in module header
- **Classification Confidence:** Require >0.8 confidence or fallback to safe defaults
- **Hook Security:** All hook scripts must validate paths and sanitize inputs
- **Hook Performance:** Critical hooks (security/validation) must complete in <5s
- **Hook Failures:** Security hooks block operations, others warn and continue
- **Hook Testing:** Every hook must have corresponding test coverage

## Naming Conventions

| Element      | Frontend   | Backend | Example            |
| ------------ | ---------- | ------- | ------------------ |
| Modules      | kebab-case | -       | `sage-protocol.md` |
| Config Files | kebab-case | -       | `metadata.yaml`    |
| State Keys   | camelCase  | -       | `protocolState`    |
| Agent IDs    | kebab-case | -       | `research-agent`   |
