# 6. Epic 2: Thinking Module Migration

Decompose the monolithic CLAUDE-v3.md into standalone modules (SAGE, SEIQF, SIA, etc.), optimize each for token efficiency, and create consistent module interfaces.

## Story 2.1: Extract SAGE Protocol Module
As a developer,
I want to extract SAGE (Self-Aware Guidance Engine) into a standalone module,
so that bias detection can be loaded only when needed.

### Acceptance Criteria
1: Create SAGE.md in .claude/thinking-modules/ directory
2: Module size must not exceed 2,000 tokens
3: Preserve all SAGE functionality from original CLAUDE-v3.md
4: Include module header with metadata (version, dependencies, description)
5: Implement SAGE status monitoring and reporting
6: Create unit tests for SAGE bias detection scenarios

## Story 2.2: Extract SEIQF Protocol Module
As a developer,
I want to extract SEIQF (Information Quality Framework) into its own module,
so that information quality assessment is available on demand.

### Acceptance Criteria
1: Create SEIQF.md in thinking-modules/ with size under 3,000 tokens
2: Maintain all search bias prevention and quality assessment features
3: Include CRAAP+ methodology and source credibility checks
4: Support integration with search tools (WebSearch, tavily-mcp)
5: Provide quality scoring interface for other modules
6: Document SEIQF activation triggers and use cases

## Story 2.3: Extract SIA Protocol Module
As a developer,
I want to extract SIA (Semantic Intent Analysis) as a separate module,
so that query understanding can be enhanced when needed.

### Acceptance Criteria
1: Create SIA.md module under 2,000 tokens
2: Preserve all intent classification categories
3: Support semantic query expansion without bias
4: Integrate with tavily-mcp parameter optimization
5: Provide intent confidence scores to other modules
6: Include examples of each intent type for clarity

## Story 2.4: Create Module Interface Standards
As a system architect,
I want consistent interfaces across all thinking modules,
so that modules can interact predictably and efficiently.

### Acceptance Criteria
1: Define standard module header format (YAML frontmatter)
2: Establish input/output contracts for module communication
3: Create module lifecycle hooks (init, activate, deactivate)
4: Specify inter-module communication protocols
5: Document module versioning and compatibility rules
6: Provide module template for future additions

## Story 2.5: Extract Response Format Standards Module
As a developer,
I want to modularize all response format requirements from CLAUDE-v3.md,
so that consistent formatting is maintained across all interactions.

### Acceptance Criteria
1: Create response-formats.md module containing all headers, footers, and logging templates
2: Include mandatory protocol status header with all emoji indicators
3: Extract tool usage documentation format and completion verification footers
4: Include exception handling formats (SAGE alerts, SEIQF quality warnings)
5: Provide self-check questions and compliance verification templates
6: Ensure module size under 1,000 tokens by using compact format definitions

## Story 2.6: Create Auto-Trigger Keywords Module
As the system,
I want a comprehensive keyword mapping for automatic protocol activation,
so that appropriate thinking tools activate without explicit commands.

### Acceptance Criteria
1: Extract all auto-trigger keywords from CLAUDE-v3.md into triggers.yaml
2: Map keywords to specific protocols (SAGE, SEIQF, SIA) and mental models
3: Support multi-word triggers and contextual patterns
4: Include priority scoring for conflicting triggers
5: Enable runtime updates to trigger mappings
6: Provide debugging mode to show why specific protocols activated

## Story 2.7: Module Test Data Generation
As a developer,
I want comprehensive test data for all thinking modules,
so that I can validate module behavior across diverse scenarios.

### Acceptance Criteria
1: Generate 50+ test scenarios per module (SAGE, SEIQF, SIA)
2: Edge cases: empty input, max tokens, special characters
3: Performance test data: 100 requests of varying complexity
4: Integration test scenarios combining multiple modules
5: Regression test suite from CLAUDE-v3.md behavior
6: Test data versioning aligned with module versions

### Technical Notes
- Store in __tests__/data/ with JSON format
- Use faker.js for realistic test data
- Tag scenarios by complexity level
- Dependencies: ["2.1", "2.2", "2.3"]
- Estimated hours: 12
- Priority: high
