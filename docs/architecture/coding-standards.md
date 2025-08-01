# Coding Standards

## Critical Cognitive Architecture Rules

- **Cognitive Function Purity**: All cognitive functions must be pure functions with no side effects - *Ensures reproducibility and testability*
- **Memory Access Pattern**: Always access memory through the orchestrator, never directly - *Prevents memory corruption and ensures consistency*
- **Sub-Agent Communication**: Use structured message schemas for all inter-agent communication - *Ensures type safety and debugging capability*
- **Error Propagation**: Never silence errors in cognitive layers, always propagate with context - *Enables proper error handling and recovery*
- **Hook Return Values**: All hooks must return standardized response format with status - *Ensures consistent Claude Code integration*
- **Async-First Design**: All I/O operations must be async with proper timeout handling - *Prevents blocking and ensures responsiveness*
- **Type Annotations**: Every function must have complete type annotations including returns - *Enables static analysis and IDE support*
- **Context Limits**: Check context window limits before sub-agent delegation - *Prevents context overflow errors*

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Cognitive Functions | snake_case with domain prefix | `reasoning_analyze_claims()` |
| Sub-Agent Classes | PascalCase with Agent suffix | `ResearcherAgent` |
| Hook Functions | snake_case with hook suffix | `prompt_enhancer_hook()` |
| Memory Keys | SCREAMING_SNAKE_CASE | `USER_PREFERENCE_CACHE` |
| Event Names | PascalCase | `UserPromptSubmit` |
| Config Variables | UPPER_SNAKE_CASE | `MAX_CONTEXT_SIZE` |
| Test Functions | test_ prefix with description | `test_atomic_analysis_quality()` |
| Async Functions | async_ prefix or _async suffix | `async_process_request()` |

## Project-Specific Standards

```python