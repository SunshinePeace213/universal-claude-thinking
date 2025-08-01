# API & Integration Architecture

## MCP Ecosystem Integration

```python
class MCPIntegrationLayer:
    """
    Provides seamless integration with Anthropic's MCP ecosystem.
    Manages tool discovery, registration, and invocation.
    """
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.tool_registry = ToolRegistry()
        self.adapter = MCPAdapter()
        
    async def discover_tools(self) -> List[MCPTool]:
        """Discover available MCP tools from ecosystem."""
        tools = await self.mcp_client.list_tools()
        
        for tool in tools:
            adapted_tool = await self.adapter.adapt_tool(tool)
            await self.tool_registry.register(adapted_tool)
            
        return tools
        
    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> ToolResult:
        """Invoke MCP tool with proper context and error handling."""
        tool = await self.tool_registry.get(tool_name)
        
        # Validate parameters
        validated_params = await tool.validate_parameters(parameters)
        
        # Execute with circuit breaker
        async with self.circuit_breaker(tool_name):
            result = await self.mcp_client.invoke(
                tool=tool,
                parameters=validated_params,
                timeout=context.timeout
            )
            
        return self.adapter.adapt_result(result)
```

## Claude Code Hooks Integration

Universal Claude Thinking v2 leverages Claude Code's hooks system to integrate cognitive processing seamlessly into the Claude workflow. Hooks are user-defined shell commands that execute at specific points, providing deterministic control over the system's behavior.

### Hook Types and Usage

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.prompt_enhancer"
        }]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Write|Edit|MultiEdit",
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.atomic_validator"
        }]
      },
      {
        "matcher": "Bash",
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.command_validator"
        }]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.pattern_learner"
        }]
      }
    ],
    "Stop": [
      {
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.memory_persist"
        }]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [{
          "type": "command",
          "command": "python -m thinking_v2.hooks.agent_coordinator"
        }]
      }
    ]
  }
}
```

### Hook Implementation Examples

```python