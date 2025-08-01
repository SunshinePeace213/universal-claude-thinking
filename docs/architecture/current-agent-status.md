# Current Agent Status
{{#each agents}}
## {{name}} ({{nickname}})
- **Status**: {{status}}
- **Last Activity**: {{last_activity}}
- **Current Task**: {{current_task}}
- **Memory Load**: {{memory_usage}}MB
- **Tools Used**: {{tools_used}}

{{#if errors}}
**Recent Errors**:
{{#each errors}}
- {{timestamp}}: {{message}}
{{/each}}
{{/if}}
{{/each}}
