# Agent Performance Metrics

## Individual Agent Statistics
| Agent | Nickname | Invocations | Success Rate | Avg Duration | User Rating |
|-------|----------|-------------|--------------|--------------|-------------|
{{#each agent_stats}}
| {{name}} | {{nickname}} | {{invocations}} | {{success_rate}}% | {{avg_duration}}s | {{rating}}/10 |
{{/each}}

## Top Performing Agents
1. **{{top_agent.name}}** ({{top_agent.nickname}}) - {{top_agent.metric}}
2. **{{second_agent.name}}** ({{second_agent.nickname}}) - {{second_agent.metric}}
3. **{{third_agent.name}}** ({{third_agent.nickname}}) - {{third_agent.metric}}
