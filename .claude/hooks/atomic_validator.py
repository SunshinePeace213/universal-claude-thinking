#!/usr/bin/env python3
"""PreToolUse validation hook for atomic prompt quality."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.atomic import AtomicFoundation

# Tools that should have quality-checked prompts
QUALITY_CHECK_TOOLS = [
    "Write", "Edit", "MultiEdit", "Create", "Generate",
    "Task", "Agent", "Command"
]


async def main():
    """Validate prompt quality before tool execution."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract tool name and arguments
        tool_name = input_data.get("tool", "")
        tool_args = input_data.get("arguments", {})
        
        # Check if this tool needs quality validation
        if tool_name not in QUALITY_CHECK_TOOLS:
            # Not a tool that needs validation
            sys.exit(0)
        
        # Extract prompt-like content from tool arguments
        prompt_content = ""
        if "prompt" in tool_args:
            prompt_content = tool_args["prompt"]
        elif "content" in tool_args:
            prompt_content = tool_args["content"]
        elif "description" in tool_args:
            prompt_content = tool_args["description"]
        elif "task" in tool_args:
            prompt_content = tool_args["task"]
        
        if not prompt_content:
            # No prompt content to validate
            sys.exit(0)
        
        # Initialize atomic foundation
        atomic = AtomicFoundation()
        
        # Analyze the prompt
        analysis = await atomic.analyze_prompt(prompt_content)
        
        # Check if quality is too low
        if analysis.quality_score < 4.0:
            # Block execution with quality warning
            print(
                f"⛔ Tool execution blocked: Prompt quality too low "
                f"({analysis.quality_score}/10.0)\n\n"
                f"The prompt lacks clarity in these areas: {', '.join(analysis.gaps)}\n\n"
                f"Please enhance your prompt with:\n"
            )
            
            for suggestion in analysis.enhancement_suggestions[:2]:
                print(f"  • {suggestion}")
            
            # Exit with error to block tool execution
            sys.exit(1)
        
        # Log warning for moderate quality
        elif analysis.quality_score < 7.0:
            print(
                f"⚠️ Prompt quality is moderate ({analysis.quality_score}/10.0). "
                f"Consider enhancing for better results."
            )
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        # Log error but don't block execution
        print(f"Validation error: {str(e)}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())