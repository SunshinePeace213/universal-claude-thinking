# Coordination Flow Analysis
```mermaid
graph TD
    {{#each workflow_steps}}
    {{from}} --> {{to}}[{{action}}]
    {{/each}}
```
