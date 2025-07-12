# 3. Technical Assumptions

## Repository Structure: Monorepo
The project will use a monorepo structure to maintain all thinking modules, cognitive tools, and configuration in a single repository for easier version control and deployment.

## Service Architecture: Modular Monolith
The system will be implemented as a modular monolith within Claude Code's context system, using @import directives for dynamic module loading rather than microservices.

## Testing Requirements: Comprehensive Testing Strategy
- Unit tests for individual thinking modules and classifiers
- Integration tests for module loading and MCP integration
- Performance tests to validate token usage reduction
- Manual testing convenience methods for module activation patterns
- Security tests for module validation and sandboxing
- End-to-end tests comparing outputs with CLAUDE-v3.md baseline
- Regression tests for each protocol (SAGE, SEIQF, SIA)
- Load tests for parallel MCP execution scenarios
- User acceptance tests for thinking visibility and accuracy

## Additional Technical Assumptions and Requests
- Claude Code's @import syntax will be used for dynamic module loading
- Markdown format (.md) for all thinking modules to ensure compatibility
- YAML format for configuration and routing rules
- Clear-thought MCP server must be installed and configured separately
- No external dependencies beyond Claude Code and configured MCP servers
- Module files will use consistent header format for metadata (version, dependencies, token count)
- Request classification will use pattern matching rather than ML models for simplicity
- Fallback to essential modules if dynamic loading fails
- Git versioning for individual module updates
- Documentation generation from module headers
- Thinking visibility logs will follow existing emoji format from CLAUDE-v3.md
- Log format: header status → active modules → MCP tools → thinking summary
- Real-time logging without blocking main response generation
- SAGE, SEIQF, and SIA protocols must maintain their integrated design from CLAUDE-v3.md
- Virtual agents are organizational units, not isolation boundaries for protocols
- Protocol state must be shared across virtual agent phases
- Universal Dynamic Information Gathering from CLAUDE-v3.md must be preserved
- Thinking tools must be able to invoke other MCP tools during execution
- Maximum recursion depth of 3 for nested tool invocations to prevent infinite loops
