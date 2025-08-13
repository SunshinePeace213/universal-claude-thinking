# Story 1.33221132: Enhanced Sub-Agent Architecture Framework

## Status
Draft

## Story
**As a** system administrator,
**I want** native sub-agent infrastructure with simplified management,
**so that** I can coordinate multiple specialists without complex orchestration code.

## Acceptance Criteria
1. Native `/agents` command provides sub-agent management interface
2. Individual context windows are created for each specialist sub-agent
3. Sub-agent isolation prevents context pollution between specialists
4. Basic coordination protocols enable communication between sub-agents
5. Error handling ensures isolated failures don't cascade across specialists
6. Performance monitoring tracks sub-agent utilization and coordination efficiency
7. Sub-agent configurations are stored in version-controlled `.claude/agents/` files

## Change Log
| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-08-11 | 1.0 | Initial story creation | Bob (Scrum Master) |
| 2025-08-11 | 1.1 | Enhanced with native sub-agent specs, parallel patterns, Further Reading | Bob (Scrum Master) |
| 2025-08-11 | 1.2 | Added production patterns from GitHub repos, orchestration limits, Task 10 | Bob (Scrum Master) |

## Tasks / Subtasks

### Task 1: Create `.claude/agents/` Directory Structure with Enhanced Native Specifications (AC: 7)
- [ ] Create agent specification markdown files with enhanced YAML frontmatter for 7 sub-agents:
  - [ ] `prompt-enhancer.md` - PE 🔧 Prompt enhancement specialist
    - [ ] YAML: `name: prompt-enhancer`, `nickname: PE`, `text_face: 🔧`, `color: green`
    - [ ] Description: Multi-line with 3-4 examples including context and commentary
    - [ ] Tools: `Read, Write, Grep, Glob` (lightweight <3k tokens)
    - [ ] System prompt: 500+ words with clear sections
  - [ ] `researcher.md` - R1 🔍 Research specialist
    - [ ] YAML: `name: researcher`, `nickname: R1`, `text_face: 🔍`, `color: blue`
    - [ ] Description: Multi-line with research scenario examples
    - [ ] Tools: `mcp__tavily__*, mcp__context7__*, WebFetch, Read`
    - [ ] System prompt: Detailed research methodologies and frameworks
  - [ ] `reasoner.md` - A1 🧠 Reasoning specialist
    - [ ] YAML: `name: reasoner`, `nickname: A1`, `text_face: 🧠`, `color: purple`
    - [ ] Description: Multi-line with complex reasoning examples
    - [ ] Tools: `mcp__waldzellai-clear-thought__clear_thought, Read, Write`
    - [ ] Model: Consider `opus` for complex reasoning tasks
  - [ ] `evaluator.md` - E1 📊 Quality evaluation specialist
    - [ ] YAML: `name: evaluator`, `nickname: E1`, `text_face: 📊`, `color: orange`
    - [ ] Description: Multi-line with quality assessment examples
    - [ ] Tools: `Read, Grep, Glob, Bash` (validation focused)
  - [ ] `tool-user.md` - T1 🛠️ Tool orchestration specialist
    - [ ] YAML: `name: tool-user`, `nickname: T1`, `text_face: 🛠️`, `color: yellow`
    - [ ] Description: Multi-line with automation scenario examples
    - [ ] Tools: `Bash, Read, Write, Edit, MultiEdit, mcp__github__*, mcp__playwright__*`
  - [ ] `writer.md` - W1 🖋️ Content creation specialist
    - [ ] YAML: `name: writer`, `nickname: W1`, `text_face: 🖋️`, `color: indigo`
    - [ ] Description: Multi-line with content creation examples
    - [ ] Tools: `Write, Edit, MultiEdit, Read` (content focused)
  - [ ] `interface.md` - I1 🗣️ User interface specialist
    - [ ] YAML: `name: interface`, `nickname: I1`, `text_face: 🗣️`, `color: cyan`
    - [ ] Description: Multi-line with user interaction examples
    - [ ] Tools: `Read, Write` (minimal for efficiency)
- [ ] Define structured message schemas for inter-agent communication
- [ ] Create agent capability matrices with cognitive tools per specialist
- [ ] Ensure all descriptions include "use PROACTIVELY" or "use IMMEDIATELY" triggers

### Task 2: Implement Base Sub-Agent Infrastructure (AC: 1, 2)
- [ ] Create `/src/agents/base.py` with BaseAgent abstract class
  - [ ] Define agent lifecycle methods (init, start, stop, reset)
  - [ ] Implement context window isolation mechanism
  - [ ] Add message passing interface
- [ ] Implement context window management
  - [ ] Create individual context tracking per agent
  - [ ] Add context overflow prevention
  - [ ] Implement context reset capabilities
- [ ] Add agent state management (idle, active, processing, error)

### Task 3: Build SubAgentManager with Orchestration Patterns (AC: 1, 4)
- [ ] Create `/src/agents/manager.py` with SubAgentManager class
  - [ ] Implement agent registration and discovery
  - [ ] Add agent lifecycle management
  - [ ] Create agent pool management with max 2 parallel execution limit
  - [ ] Implement orchestrator pattern (delegates but never implements)
- [ ] Implement coordination protocols
  - [ ] Define structured task assignment format
  - [ ] Add agent selection hierarchy (specific > generic)
  - [ ] Implement message routing between agents
  - [ ] Add request/response patterns
  - [ ] Implement broadcast messaging
- [ ] Add task distribution logic
  - [ ] Create work queue management with parallel execution limits
  - [ ] Implement load balancing (max 2 concurrent agents)
  - [ ] Add priority handling with framework-specific preference
  - [ ] Track token consumption per orchestration

### Task 4: Implement `/agents` Command Interface (AC: 1)
- [ ] Create command handler in `.claude/commands/agents.py`
  - [ ] Implement `list` subcommand to show available agents
  - [ ] Add `status` subcommand for agent health monitoring
  - [ ] Create `reset` subcommand for agent context clearing
- [ ] Add agent discovery from `.claude/agents/` directory
- [ ] Implement agent activation/deactivation controls

### Task 5: Add Error Isolation (AC: 5)
- [ ] Implement error boundaries per agent
  - [ ] Create agent-specific error handlers
  - [ ] Add error containment mechanisms
  - [ ] Implement graceful degradation
- [ ] Add circuit breaker patterns
  - [ ] Create failure threshold monitoring
  - [ ] Implement automatic circuit breaking
  - [ ] Add recovery mechanisms
- [ ] Create error recovery and retry logic
  - [ ] Define retry strategies per agent type
  - [ ] Implement exponential backoff
  - [ ] Add failure logging and alerting

### Task 6: Integrate Performance Monitoring (AC: 6)
- [ ] Add utilization tracking
  - [ ] Track agent token usage (target: <3k for lightweight, <15k for medium)
  - [ ] Monitor context window utilization per agent
  - [ ] Record initialization times (target: <500ms per agent)
  - [ ] Track agent CPU/memory usage
- [ ] Implement coordination efficiency metrics
  - [ ] Track message passing latency (target: <50ms)
  - [ ] Monitor task completion rates by agent type
  - [ ] Measure agent idle time and chaining efficiency
  - [ ] Track parallel execution performance gains
- [ ] Create monitoring dashboard queries
  - [ ] Build real-time agent status views
  - [ ] Add token usage analytics per agent
  - [ ] Create agent weight classification reports
  - [ ] Monitor proactive delegation success rates

### Task 7: Write Comprehensive Tests
- [ ] Unit tests for BaseAgent and Manager
  - [ ] Test agent lifecycle management
  - [ ] Verify context isolation
  - [ ] Test error handling
- [ ] Integration tests for coordination flows
  - [ ] Test multi-agent task execution
  - [ ] Verify message passing
  - [ ] Test failure cascade prevention
- [ ] Performance benchmark tests
  - [ ] Measure agent startup times
  - [ ] Test concurrent agent execution
  - [ ] Benchmark message passing overhead

### Task 8: Update Documentation
- [ ] Document agent specification format
  - [ ] Create YAML frontmatter template with name, nickname, text_face, description, tools
  - [ ] Define capability declaration structure
  - [ ] Document message schema requirements
  - [ ] Include proactive delegation keywords guide
- [ ] Create usage examples
  - [ ] Basic agent activation examples with nicknames (e.g., "ask R1 to research")
  - [ ] Multi-agent coordination examples
  - [ ] Parallel processing workflow patterns
  - [ ] Error recovery scenarios
- [ ] Add troubleshooting guide
  - [ ] Common issues and solutions
  - [ ] Debug commands and `/agents` interface usage
  - [ ] Performance tuning for token optimization

### Task 9: Update Architecture Documentation (AC: 1, 7)
- [ ] Update `/docs/architecture/7-layer-context-engineering-architecture.md`
  - [ ] Add Layer 7: Enhanced Sub-Agent Management section
  - [ ] Document native `/agents` command integration
  - [ ] Include parallel processing capabilities
  - [ ] Add context isolation benefits
- [ ] Update `/docs/architecture/unified-project-structure.md`
  - [ ] Ensure `.claude/agents/` directory is properly documented
  - [ ] Add sub-agent file format specifications with YAML frontmatter
  - [ ] Document nickname and text_face conventions
- [ ] Create `/docs/architecture/sub-agent-architecture.md`
  - [ ] Document complete sub-agent implementation from project-brief-resources
  - [ ] Include parallel workflow patterns (Map-Reduce, Specialist Coordination)
  - [ ] Add performance optimization guidelines (token usage, initialization)
  - [ ] Document cognitive tools per specialist

### Task 10: Implement Agent Organization Structure (AC: 7)
- [ ] Create department-based organization in `.claude/agents/`
  - [ ] `orchestrators/` - Task coordination and delegation agents
    - [ ] Tech lead orchestrator (delegates but never implements)
    - [ ] Project analyst for stack detection
  - [ ] `engineering/` - Development and implementation specialists
    - [ ] Framework-specific agents (Django, Rails, React, Vue)
    - [ ] Universal fallback agents
  - [ ] `testing/` - Quality assurance and validation agents
    - [ ] Test writer/fixer
    - [ ] Performance benchmarker
  - [ ] `core/` - Essential cross-cutting agents
    - [ ] Code reviewer (always included)
    - [ ] Performance optimizer (always included)
    - [ ] Code archaeologist for legacy exploration
- [ ] Implement agent discovery by department
  - [ ] Scan department directories for available agents
  - [ ] Build capability matrix by department
  - [ ] Support both project and user-level agents
- [ ] Add visual identification support
  - [ ] Implement color field rendering
  - [ ] Support text_face emoji display
  - [ ] Create consistent visual hierarchy

## Dev Notes

### Previous Story Insights
**From Story 1.2**: Successfully implemented RequestClassifier and HybridDelegationEngine with 3-stage routing (keyword → semantic → PE fallback). Achieved <10ms keyword matching and 100% classification accuracy. The delegation infrastructure is ready to route requests to the sub-agents that will be created in this story.

### Architecture Context

#### Native Sub-Agent Configuration Format
**YAML Frontmatter Structure** [Source: docs.anthropic.com/claude-code/sub-agents]:
```yaml
---
name: agent-name          # Unique identifier using lowercase and hyphens
nickname: A1              # Short nickname for quick invocation
text_face: 🧠            # Visual personality indicator
description: Expert description. Use PROACTIVELY for automatic delegation.
tools: tool1, tool2      # Comma-separated list (omit to inherit all)
model: sonnet           # Optional: sonnet, opus, or haiku
color: blue             # Quick Example of colour
---
System prompt defining role, capabilities, and specialized approaches...
```

#### Project Structure
**Sub-Agent Locations** [Source: architecture/unified-project-structure.md]:
- Agent specifications: `.claude/agents/` directory containing markdown with YAML frontmatter
- Implementation classes: `/src/agents/` directory with Python implementations
- Base infrastructure: `/src/agents/base.py` - Abstract base class for all agents
- Manager component: `/src/agents/manager.py` - Central coordination system
- Specific implementations: `/src/agents/implementations/` - Individual agent logic

#### Component Specifications
**SubAgentManager Architecture** [Source: architecture/7-layer-context-engineering-architecture.md#layer-4]:
- Central orchestration system managing specialist sub-agent coordination
- Implements task decomposition, routing, and result synthesis
- Enhanced with ReAct (Reasoning + Acting) pattern for transparent decision-making
- Orchestration patterns: Sequential Pipeline, Parallel Map-Reduce, Feedback Loops

**Agent Communication** [Source: architecture/coding-standards.md#critical-cognitive-architecture-rules]:
- Use structured message schemas for all inter-agent communication
- Check context window limits before sub-agent delegation
- All cognitive functions must be pure functions with no side effects
- Never silence errors in cognitive layers, always propagate with context

#### Error Handling Requirements
**Error Isolation Architecture** [Source: architecture/error-handling-strategy.md]:
```python
@dataclass
class CognitiveError(Exception):
    error_code: str
    message: str
    layer: str
    details: Optional[Dict[str, Any]]
    recovery_suggestions: Optional[List[str]]
    correlation_id: Optional[str]
```

**Sub-Agent Error Patterns**:
- Each agent must have isolated error boundaries
- Failures should not cascade between agents
- Circuit breaker patterns for repeated failures
- Error recovery with exponential backoff

#### Claude Code Integration
**Hook Integration Points** [Source: architecture/api-integration-architecture.md#claude-code-hooks]:
- SubagentStop hook for coordination between agents
- Agent status monitoring through hook callbacks
- Memory persistence on agent context changes
- Performance metrics collection via hooks

#### Testing Standards
**Test Requirements** [Source: architecture/coding-standards.md#naming-conventions]:
- Test file locations: `/tests/unit/test_agents.py`, `/tests/integration/test_agent_coordination.py`
- Test naming convention: `test_` prefix with descriptive names
- Async test support for agent coordination
- Performance benchmarks for message passing latency

### Parallel Processing Workflow Patterns
**Map-Reduce Pattern** [Source: project-brief-resources/parallel-processing-workflows.md]:
- Use for research tasks requiring multiple information sources
- 4x efficiency through parallel research contexts
- No context interference between parallel agents
- Orchestrator synthesizes results in fresh context

**Specialist Coordination Pattern**:
- Multiple specialists analyze problem simultaneously
- Tool-User, Reasoner, Evaluator work in parallel
- Each maintains individual context window
- Results coordinated by orchestrator

### Performance Optimization Guidelines
**Agent Weight Classifications** [Source: claudelog.com/mechanics/agent-engineering]:
- **Lightweight agents** (<3k tokens): Minimal tools, fast initialization, high composability
- **Medium agents** (10-15k tokens): Balanced capability and performance
- **Heavy agents** (25k+ tokens): Complex analysis, higher initialization cost
- **Orchestrators** (10-50k tokens): ⚠️ Token-intensive coordination agents

**Token Usage by Tool Count** (Empirical):
- 0 tools: ~640 tokens (best case with empty CLAUDE.md)
- 1-3 tools: 2.6k-3.2k tokens (lightweight)
- 4-6 tools: 3.4k-4.1k tokens (still efficient)
- 7-10 tools: 5k-8k tokens (medium weight)
- 15+ tools: 13.9k-25k tokens (heavy, use sparingly)
- Orchestrators: 10-50k tokens per complex workflow (monitor carefully!)

### Production Agent Patterns
**Example-Driven Descriptions** [Source: github.com/contains-studio/agents]:
```yaml
description: |
  Use this agent when [scenario]. This agent specializes in [expertise]. 
  Examples:
  
  <example>
  Context: [situation]
  user: "[user request]"
  assistant: "[response approach]"
  <commentary>
  [why this example matters]
  </commentary>
  </example>
  
  [3 more examples with context and commentary...]
```

**Orchestrator Delegation Pattern** [Source: github.com/vijaythecoder/awesome-claude-agents]:
- Orchestrators analyze and delegate but NEVER implement code
- Use structured format for task assignments
- Enforce maximum 2 parallel agents to prevent token explosion
- Prefer framework-specific agents over universal ones
- Selection hierarchy: specific > generic (e.g., django-backend-expert > backend-developer)

**Department Organization Structure**:
- `orchestrators/` - Task coordination and delegation specialists
- `engineering/` - Implementation and development specialists
- `product/` - Feature planning and prioritization agents
- `testing/` - Quality assurance and validation agents
- `core/` - Essential agents (reviewer, optimizer, archaeologist)

### Technical Constraints & Warnings
- Context window isolation: Each agent has independent context (no pollution)
- Message passing latency target: <50ms between agents
- Agent initialization time target: <500ms per agent (varies by token count)
- Memory isolation: Each agent maintains separate memory space
- Concurrent execution: Maximum 2 agents in parallel (prevent token explosion)
- Proactive delegation: Use "PROACTIVELY" and "IMMEDIATELY" keywords for auto-activation

⚠️ **Token Usage Warnings**:
- Multi-agent orchestration can consume 10-50k tokens per complex feature
- Orchestrator agents with `model: opus` are particularly token-intensive
- Monitor usage carefully when using parallel execution
- Consider token limits when designing agent chains
- Lightweight agents (<3k tokens) are preferred for composability
- Heavy agents (25k+ tokens) should be used sparingly

### Integration Points
- Connects with Story 1.2's delegation engine for request routing
- Provides foundation for Story 1.4's molecular context assembly
- Enables Story 1.5's memory system to track per-agent interactions
- Required for Epic 2's advanced cognitive capabilities

## Testing

### Testing Standards
- Test framework: pytest with async support
- Coverage requirement: >90% for critical paths
- Performance tests must verify latency constraints
- Integration tests should simulate multi-agent workflows
- Mock external dependencies for unit tests
- Use fixtures for common agent setups

## Dev Agent Record

### Agent Model Used
_To be populated by development agent_

### Debug Log References
_To be populated during development_

### Completion Notes
_To be populated upon task completion_

### File List
_To be populated with created/modified files_

## QA Results
_To be populated by QA agent review_

## Further Reading (Optional)

### Core Concepts
- **Sub-Agent Architecture**: [Official Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- **Task/Agent Tools**: [Task Delegation Patterns](https://claudelog.com/mechanics/task-agent-tools/)
- **Agent-First Design**: [Designing for AI Agents](https://claudelog.com/mechanics/agent-first-design/)
- **Split Role Sub-Agents**: [Multi-Perspective Analysis](https://claudelog.com/mechanics/split-role-sub-agents/)

### Implementation References
- **Custom Agents**: [Agent Configuration Guide](https://claudelog.com/mechanics/custom-agents/)
- **Agent Engineering**: [Performance Optimization](https://claudelog.com/mechanics/agent-engineering/)
- **Humanising Agents**: [Personality and Engagement](https://claudelog.com/mechanics/humanising-agents/)
- **Parallel Processing**: [Context Isolation Benefits](https://claudelog.com/mechanics/task-agent-tools/)

### Community Agent Collections
- **Production-Ready Agents**: [Contains Studio Agents](https://github.com/contains-studio/agents) - 40+ specialized agents with detailed examples, department organization, and 6-day sprint methodology
- **Framework Specialists**: [Awesome Claude Agents](https://github.com/vijaythecoder/awesome-claude-agents) - Laravel, Django, Rails, React, Vue specialists with orchestrator patterns
- **Agent Organization Patterns**: Department-based structure for large agent collections (orchestrators, engineering, testing, core)
- **Token-Intensive Warning**: Community agents report 10-50k token usage for complex multi-agent workflows

### Best Practices
- **Token Optimization**: Keep agents lightweight (<3k tokens) for composability
- **Tool Selection**: Grant only necessary tools to minimize initialization cost
- **Proactive Delegation**: Use "use PROACTIVELY" and "IMMEDIATELY" in descriptions
- **Nickname Efficiency**: Implement short nicknames (R1, A1, E1) for quick invocation
- **Model Selection**: Match model to agent weight (Haiku for lightweight, Sonnet for medium, Opus for orchestrators)
- **Context Isolation**: Each agent operates in separate context window for quality
- **Parallel Limits**: Maximum 2 agents in parallel to prevent token explosion
- **Performance Monitoring**: Track token usage, initialization time, and delegation success
- **Example-Driven Descriptions**: Include 3-4 examples with context and commentary for better activation
- **Orchestrator Pattern**: Orchestrators delegate but never implement code themselves