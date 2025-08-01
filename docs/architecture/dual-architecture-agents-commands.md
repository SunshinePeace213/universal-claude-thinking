# Dual Architecture: Agents + Commands

## Core Architectural Decision: Separation of Concerns

Universal Claude Thinking v2 implements a **dual architecture** combining both **Sub-Agents** and **Slash Commands** to achieve optimal separation of concerns:

### **Agents (.claude/agents/)**: Intelligence Layer
- **Purpose**: Complex reasoning, analysis, and multi-step cognitive workflows
- **Characteristics**: 
  - Persistent specialists with individual context windows
  - Automatic delegation based on intelligent task analysis
  - Stateful processing with memory and cross-session continuity
  - No CLAUDE.md inheritance (pure context isolation)
  - Multi-turn interactions with sophisticated reasoning

### **Commands (.claude/commands/)**: Operations Layer  
- **Purpose**: Utility operations, monitoring, setup, and maintenance tasks
- **Characteristics**:
  - Stateless prompt templates for quick execution
  - Manual invocation by user typing `/command`
  - Single-turn execution with immediate results
  - Can access project context when needed
  - Simple operational tasks without complex reasoning

## Why Both Are Essential

**Agents handle "THINKING"** - Complex reasoning requiring specialist expertise:
- Research and analysis requiring multiple sources and synthesis
- Quality evaluation with sophisticated metrics and validation
- Content creation with iterative refinement and style adaptation
- Tool orchestration requiring safety validation and error recovery

**Commands handle "DOING"** - Operational tasks requiring quick execution:
- System monitoring and health checks
- Project setup and configuration
- Debugging and troubleshooting utilities  
- Data reporting and analytics
- Maintenance and optimization operations

## Integration Patterns

**Hybrid Workflows**: Agents can trigger commands for operational tasks:
```python