# Command Architecture: 5-Category Structure

## Command Organization Strategy

Commands are organized into 5 primary categories, each serving distinct operational purposes:

```
.claude/commands/
├── monitor/          # System monitoring and health checks
├── setup/            # Installation and configuration utilities
├── debug/            # Troubleshooting and diagnostic tools
├── report/           # Analytics and data reporting
└── maintain/         # Cleanup and optimization operations
```

## Category 1: Monitor Commands (`/monitor`)

**Purpose**: Real-time system monitoring and health assessments

```yaml
monitor_commands:
  cognitive:
    description: "Monitor cognitive architecture performance and health"
    tools: [Bash, Read]
    monitors: ["memory usage", "agent performance", "database health"]
    
  agents:
    description: "Monitor individual agent performance and coordination"
    tools: [Bash, Read]
    tracks: ["context usage", "response times", "error rates"]
    
  memory:
    description: "Monitor memory system performance and optimization"
    tools: [Bash, sqlite3]
    analyzes: ["memory utilization", "retrieval performance", "pattern effectiveness"]
```

**Example Command**:
```markdown