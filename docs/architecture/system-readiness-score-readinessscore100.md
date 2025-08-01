# System Readiness Score: {{readiness_score}}/100

{{#if validation_passed}}
🎉 **Installation Complete!** Universal Claude Thinking v2 is ready to use.

Try running:
- `/monitor-cognitive` - Check system health
- Create your first memory with an agent interaction
{{else}}
⚠️ **Issues Detected** - Please address the following:
{{#each issues}}
- {{this}}
{{/each}}
{{/if}}
```

## Category 3: Debug Commands (`/debug`)

**Purpose**: Troubleshooting, tracing, and diagnostic analysis

### `/debug-agent-trace`
```markdown
---
description: Debug agent coordination and trace workflow execution
allowed-tools: Bash, Read, Grep
---
