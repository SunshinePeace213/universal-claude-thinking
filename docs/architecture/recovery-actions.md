# Recovery Actions
{{#if corruption_detected}}
⚠️ **Database corruption detected** - Running automatic repair...
```bash
sqlite3 .claude/memory/cognitive.db ".recover" > recovered.db
mv recovered.db .claude/memory/cognitive.db
```
{{/if}}
```

## Category 4: Report Commands (`/report`)

**Purpose**: Analytics, metrics, and comprehensive reporting

### `/report-usage-stats`
```markdown
---
description: Generate comprehensive system usage and performance report
allowed-tools: Bash, Read
---
