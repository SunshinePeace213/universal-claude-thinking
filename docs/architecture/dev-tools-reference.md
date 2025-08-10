# Development Tools Reference

## Tool Selection Quick Reference

### Request Classification → Tool Requirements

#### Type A (Simple/Direct)
- **No mandatory tools**
- Optional: Offer enhanced analysis
- Check: Avoid unnecessary complexity

#### Type B (Complex/Multi-step) 
- **MANDATORY**: `first_principles` (foundation)
- **MANDATORY**: `sequentialthinking` (reasoning)
- **MANDATORY**: `decisionframework` (options)
- **AUTO-TRIGGER**: Based on language patterns
- **CONDITIONAL**: Research tools for information gaps

#### Type C (Research Required)
- **MANDATORY**: `tavily-mcp` OR `context7`
- **MANDATORY**: `time` (if temporal context needed)
- **CONDITIONAL**: Analysis tools if needed

#### Type D (Web/Testing)
- **MANDATORY**: `playwright`
- **MANDATORY**: `sequentialthinking`
- **CONDITIONAL**: Research for methodologies

#### Type E (Debugging)
- **MANDATORY**: `debuggingapproach`
- **MANDATORY**: `sequentialthinking`
- **CONDITIONAL**: Research for error documentation

## Tool Categories

### Clear-Thought Tools (Mental Models)

#### Core Reasoning
- **`first_principles`**: Break down to fundamental truths
- **`sequentialthinking`**: Step-by-step logical reasoning
- **`decisionframework`**: Structured decision making
- **`debuggingapproach`**: Systematic debugging methods

#### System Analysis
- **`systemsthinking`**: Complex interdependencies
- **`error_propagation`**: Reliability and failure analysis
- **`metacognitivemonitoring`**: Monitor thinking effectiveness

#### Optimization & Trade-offs
- **`opportunity_cost`**: Trade-off analysis
- **`pareto_principle`**: 80/20 optimization
- **`occams_razor`**: Simplification

#### Communication & Inquiry
- **`rubber_duck`**: Teaching and explanation
- **`socraticmethod`**: Systematic questioning
- **`collaborativereasoning`**: Multiple perspectives

#### Creative & Scientific
- **`creativethinking`**: Creative approaches
- **`scientificmethod`**: Hypothesis testing
- **`structuredargumentation`**: Logical arguments

### Research & Information Tools

#### Web Research
- **`tavily-mcp`**: 
  - Current information, trends, best practices
  - Recent developments, market analysis
  - Cross-validation of claims

#### Technical Documentation  
- **`context7`**:
  - API documentation, libraries, frameworks
  - Technical specifications
  - Programming language references

#### Temporal Context
- **`time`**:
  - Current time in timezones
  - Time conversion
  - Temporal validation for research

### Development Tools

#### Web Testing
- **`playwright`**:
  - Browser automation
  - UI testing
  - User journey validation

#### Analysis & Computation
- **`repl`**:
  - Complex calculations
  - Data analysis
  - Code execution and testing

### Collaboration Tools

#### Version Control
- **`github`**:
  - Repository management
  - Issue tracking
  - Code collaboration

## Auto-Trigger Patterns

### Language Pattern → Tool
- "choose between", "vs", "alternatives" → `opportunity_cost`
- "optimize", "improve", "maximize" → `pareto_principle`
- "simplify", "streamline", "clean" → `occams_razor`
- "explain to", "teach", "document for" → `rubber_duck`
- "what could go wrong", "failure", "reliability" → `error_propagation`
- "system design", "architecture", "ecosystem" → `systemsthinking`
- "comprehensive", "thorough", "deep dive" → `metacognitivemonitoring`

### Context Pattern → Tool
- "current", "latest", "recent", "today" → `tavily-mcp` + `time`
- "documentation", "API", "library" → `context7`
- "test", "validate", "UI", "browser" → `playwright`
- "error", "bug", "debug", "fix" → `debuggingapproach`
- "not working", "broken", "fails" → `debuggingapproach`

## Tool Combination Patterns

### Common Workflows

#### Technical Decision Making
```
context7 → first_principles → opportunity_cost → decisionframework
```

#### System Architecture Design
```
first_principles → systemsthinking → error_propagation → decisionframework
```

#### Current Practice Research
```
time → tavily-mcp → sequentialthinking → analysis
```

#### Complex Problem Solving
```
first_principles → [auto-triggered models] → sequentialthinking → decisionframework
```

#### Debugging Workflow
```
debuggingapproach → sequentialthinking → [optional: context7 for docs]
```

#### Web Testing Workflow
```
sequentialthinking → playwright → validation
```

## Tool Selection Principles

### Simplicity First
1. Can this be answered directly without tools?
2. What's the minimum tool set needed?
3. Will tools serve user benefit or demonstrate capability?

### User Benefit Focus
1. Which tools solve the actual problem?
2. What specific value does each tool provide?
3. How do tools contribute to the final solution?

### Appropriate Complexity
1. Does tool complexity match problem complexity?
2. Are we using tools because they're available or because they're needed?
3. Would a simpler approach better serve the user?

## Information Quality Standards

### When Using Research Tools
- **Source Credibility**: Validate authority and expertise
- **Cross-Validation**: Confirm claims through multiple sources
- **Bias Detection**: Monitor for confirmation and selection bias
- **Temporal Validity**: Ensure information currency when relevant

### Research Tool Selection
- **`tavily-mcp`**: Current trends, practices, general web research
- **`context7`**: Technical documentation, specific library/framework info
- **`time`**: Temporal context, current time validation

## Debug Protocol Integration

### Bug Management with Tools
1. **`review-bugs`**: Check existing issues
2. **`debuggingapproach`**: Select systematic method
3. **`sequentialthinking`**: Document process
4. **Optional research**: Use `context7` for error documentation
5. **`mark-bug-resolved`**: Complete with validation

### Debug Approach Selection
- **Binary Search**: Large codebase issue isolation
- **Backtracking**: Trace when issue started
- **Root Cause Analysis**: Find fundamental cause
- **Log Analysis**: Systematic log examination
- **Static Analysis**: Code analysis without execution

## Protocol Compliance

### Every Response Must Include
1. **Protocol Status Header**: Classification and tool status
2. **Tool Justification**: Why each tool serves user benefit
3. **Process Documentation**: How tools contributed to solution
4. **Quality Validation**: Information quality when research used
5. **Completion Verification**: All requirements met

### Quality Gates
- **Before Tool Use**: Justify necessity and user benefit
- **During Tool Use**: Monitor for bias and scope creep
- **After Tool Use**: Validate output serves user needs
- **Final Check**: Confirm appropriate complexity and quality
