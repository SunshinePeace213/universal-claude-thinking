# Frontend Architecture

_Note: This system has no traditional frontend - it operates within Claude Code's context_

## Context Interface Architecture

### Module Display Format

```text
🎯 THINKING STATUS: Active
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Loaded Modules: SAGE (2K) + SEIQF (3K) + response-formats (1K)
🤖 Active Agents: Research → Analysis → Synthesis
🧠 MCP Tools: [sequential-thinking] → [tavily-search] ×3
⚡ Token Usage: 6,234 / 38,221 (84% reduction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Thinking Visibility Logs

```text
[🔍 Research Phase]
├─ 🧠 Sequential thinking initiated
├─ 📊 Information gap detected: "latest ML research"
├─ 🔍 Invoking nested search...
│  └─ [🔍×3] Parallel searches: arxiv, papers, blogs
├─ ✅ Quality validation passed (SEIQF: 0.92)
└─ 📝 Research complete

[🎯 Analysis Phase]
├─ 🧠 Mental model: First Principles
├─ ⚠️ SAGE Alert: Potential bias in sources
├─ 🔄 Applying bias mitigation...
└─ ✅ Analysis validated
```
