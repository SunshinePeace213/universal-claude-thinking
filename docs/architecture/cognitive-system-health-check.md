# Cognitive System Health Check

Current system status:
- Memory usage: !`ps aux | grep thinking_v2`
- Database size: !`du -sh data/thinking_v2.db`
- Active agents: !`ls -la .claude/agents/`
- Hook status: @.claude/settings.json

Show cognitive layer performance and any issues.
```

## Category 2: Setup Commands (`/setup`)

**Purpose**: Project initialization, configuration, and validation

```yaml
setup_commands:
  init-project:
    description: "Initialize Universal Claude Thinking v2 in new project"
    tools: [Bash, Write]
    creates: ["agent directories", "hook configurations", "memory database"]
    
  configure-hooks:
    description: "Configure Claude Code hooks for cognitive processing"
    tools: [Bash, Write, Read]
    configures: ["hook scripts", "settings.json", "permissions"]
    
  validate:
    description: "Validate complete system installation and configuration"
    tools: [Bash, Read]
    validates: ["agent specs", "database", "hook integration"]
```

## Category 3: Debug Commands (`/debug`)

**Purpose**: Troubleshooting, tracing, and diagnostic analysis

```yaml
debug_commands:
  agent-trace:
    description: "Debug agent coordination and trace workflow execution"
    tools: [Bash, Read]
    traces: ["agent interactions", "context flows", "error patterns"]
    
  memory-debug:
    description: "Debug memory system issues and inconsistencies"
    tools: [Bash, sqlite3, Read]
    debugs: ["memory corruption", "retrieval failures", "pattern conflicts"]
    
  workflow-analysis:
    description: "Analyze workflow execution and identify bottlenecks"
    tools: [Bash, Read]
    analyzes: ["execution paths", "performance bottlenecks", "coordination issues"]
```

## Category 4: Report Commands (`/report`)

**Purpose**: Analytics, metrics, and comprehensive reporting

```yaml
report_commands:
  memory-stats:
    description: "Generate comprehensive memory system usage report"
    tools: [Bash, sqlite3, Read]
    reports: ["usage patterns", "effectiveness metrics", "optimization opportunities"]
    
  performance:
    description: "Generate system performance and efficiency report"
    tools: [Bash, Read]
    metrics: ["response times", "token efficiency", "parallel processing gains"]
    
  usage-summary:
    description: "Generate usage summary with insights and recommendations"
    tools: [Bash, sqlite3, Read]
    summarizes: ["user patterns", "agent utilization", "feature adoption"]
```

## Category 5: Maintain Commands (`/maintain`)

**Purpose**: Cleanup, optimization, and system maintenance

```yaml
maintain_commands:
  cleanup:
    description: "Clean up cognitive architecture data and optimize performance"
    tools: [Bash, sqlite3]
    cleans: ["expired memories", "unused contexts", "old logs"]
    
  optimize:
    description: "Optimize database and system performance"
    tools: [Bash, sqlite3]
    optimizes: ["database indices", "memory allocation", "cache performance"]
    
  backup:
    description: "Backup critical system data and configurations"
    tools: [Bash, Read]
    backs_up: ["memory database", "agent configurations", "user patterns"]
```

## Command Design Principles

### **Lightweight & Focused**
- Each command serves a single, specific purpose
- Minimal token usage for fast execution
- Clear argument patterns and help text

### **Safe Operations**
- Dry-run modes for destructive operations
- Confirmation prompts for critical changes
- Comprehensive error handling and recovery

### **Integration Aware**
- Commands can trigger agent workflows when needed
- Agents can execute commands for operational tasks
- Seamless data flow between command categories

### **User Experience Optimized**
- Consistent naming conventions across categories
- Clear, actionable output formatting
- Progressive disclosure of complex information

---
