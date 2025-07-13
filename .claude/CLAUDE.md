---
version: 2.0.0
tokens: 450
---

# Claude Thinking Orchestrator

## Request Classification

@import "./request-classifier.md"

Use the enhanced classifier to categorize requests:
- **simple**: Direct questions, basic explanations
- **complex**: Multi-step reasoning, analysis tasks
- **search**: Finding information, code exploration
- **code**: Writing, refactoring, debugging code
- **meta**: Self-reflection, process questions

Returns RequestClassification with:
- category, confidence, requiredModules
- suggestedAgents, mcpTools, estimatedTokens

Confidence threshold: 0.8

## Module Loading

Based on classification results:

### Simple Requests
@import "./thinking-modules/response-formats.md"

### Complex Requests
@import "./thinking-modules/SAGE.md"
@import "./thinking-modules/SEIQF.md"
@import "./cognitive-tools/analysis.md"

### Search Requests
@import "./thinking-modules/SIA.md"
@import "./cognitive-tools/search.md"

### Code Requests
@import "./thinking-modules/SEIQF.md"
@import "./cognitive-tools/code-analysis.md"

### Meta Requests
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
📈 Telemetry: [cache_hits] / [total_requests]
🤖 Suggested Agents: [agents]
🔧 MCP Tools: [tools]

## Error Handling

- Invalid paths → Log and continue
- Missing modules → Use fallback
- Token overflow → Truncate gracefully
- MCP failures → Degrade to basic operation