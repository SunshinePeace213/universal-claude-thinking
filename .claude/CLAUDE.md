---
version: 1.0.0
tokens: 450
---

# Claude Thinking Orchestrator

## Request Classification

Analyze request to determine category:
- **A/simple**: Direct questions, basic explanations
- **B/complex**: Multi-step reasoning, analysis tasks
- **C/search**: Finding information, code exploration
- **D/code**: Writing, refactoring, debugging code
- **E/meta**: Self-reflection, process questions

Confidence threshold: 0.8

## Module Loading

### Category A - Simple Requests
@import "./thinking-modules/response-formats.md"

### Category B - Complex Requests
@import "./thinking-modules/SAGE.md"
@import "./thinking-modules/SEIQF.md"
@import "./cognitive-tools/analysis.md"

### Category C - Search Requests
@import "./thinking-modules/SIA.md"
@import "./cognitive-tools/search.md"

### Category D - Code Requests
@import "./thinking-modules/SEIQF.md"
@import "./cognitive-tools/code-analysis.md"

### Category E - Meta Requests
@import "./thinking-modules/SAGE.md"
@import "./cognitive-tools/meta-reasoning.md"

## Fallback Protocol

If module loading fails:
1. Continue with basic reasoning
2. Log failure in debug header
3. Use MCP tools directly if available

## Debug Header

🎯 Active Modules: [modules]
⚡ Classification: [category] ([confidence])
📊 Total Tokens: [current] / [budget]
🕒 Load Time: [ms]

## Error Handling

- Invalid paths → Log and continue
- Missing modules → Use fallback
- Token overflow → Truncate gracefully
- MCP failures → Degrade to basic operation