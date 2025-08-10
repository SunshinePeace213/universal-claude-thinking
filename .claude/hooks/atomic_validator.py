#!/usr/bin/env python3
"""PreToolUse validation hook with pattern validation and adaptive thresholds."""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.atomic import AtomicFoundation
from src.core.storage.db import DatabaseConnection
from src.core.storage.atomic_repository import AtomicAnalysisRepository
from src.delegation.keyword_matcher import KeywordMatcher
from src.core.atomic.pattern_library import PatternLibrary

# Tools that should have quality-checked prompts
QUALITY_CHECK_TOOLS = [
    "Write", "Edit", "MultiEdit", "Create", "Generate",
    "Task", "Agent", "Command"
]

# Adaptive threshold configuration
ADAPTIVE_THRESHOLDS = {
    "default": {"block": 4.0, "warn": 7.0},
    "Task": {"block": 5.0, "warn": 8.0},  # Higher standards for sub-agents
    "Agent": {"block": 5.0, "warn": 8.0},
    "Write": {"block": 3.5, "warn": 6.5},  # Slightly lower for file writes
    "Edit": {"block": 3.5, "warn": 6.5},
}


async def main():
    """Validate prompt quality with pattern awareness and adaptive thresholds."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract tool name, arguments, and metadata
        tool_name = input_data.get("tool_name", "") or input_data.get("tool", "")
        tool_args = input_data.get("tool_input", {}) or input_data.get("arguments", {})
        session_id = input_data.get("session_id", "")
        
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
        
        # Get adaptive thresholds for this tool
        thresholds = ADAPTIVE_THRESHOLDS.get(tool_name, ADAPTIVE_THRESHOLDS["default"])
        block_threshold = thresholds["block"]
        warn_threshold = thresholds["warn"]
        
        # Initialize database and components
        db = None
        repository = None
        pattern_library = None
        keyword_matcher = None
        
        try:
            db = DatabaseConnection()
            await db.initialize_schema()
            repository = AtomicAnalysisRepository(db)
            
            # Initialize pattern components for validation
            pattern_library = PatternLibrary()
            keyword_matcher = KeywordMatcher()
        except Exception:
            # Continue without advanced features if initialization fails
            pass
        
        try:
            atomic = AtomicFoundation(repository=repository)
            
            # Analyze the prompt
            analysis = await atomic.analyze_prompt(prompt_content)
            
            # Check pattern matching if available
            pattern_validation = None
            if keyword_matcher:
                # Check if prompt matches known patterns
                match_result = await keyword_matcher.match(prompt_content, None)
                if not match_result.matched:
                    pattern_validation = {
                        "status": "unmatched",
                        "message": "Prompt doesn't match known patterns - learning opportunity"
                    }
                    
                    # Log unmatched pattern for learning
                    if repository:
                        await repository.save_pattern_validation(
                            prompt=prompt_content,
                            tool_name=tool_name,
                            quality_score=analysis.quality_score,
                            matched=False,
                            session_id=session_id
                        )
            
            # Prepare JSON output for advanced control
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse"
                }
            }
            
            # Check if quality is too low (using adaptive threshold)
            if analysis.quality_score < block_threshold:
                # Block execution with detailed feedback
                output["hookSpecificOutput"]["permissionDecision"] = "deny"
                
                reasons = []
                reasons.append(f"⛔ Prompt quality too low: {analysis.quality_score:.1f}/10.0 (minimum: {block_threshold})")
                
                if analysis.gaps:
                    reasons.append(f"Missing: {', '.join(analysis.gaps)}")
                
                if analysis.enhancement_suggestions:
                    reasons.append("Suggestions: " + " | ".join(analysis.enhancement_suggestions[:2]))
                
                if pattern_validation and pattern_validation["status"] == "unmatched":
                    reasons.append("🎯 " + pattern_validation["message"])
                
                output["hookSpecificOutput"]["permissionDecisionReason"] = "\n".join(reasons)
                
                # Output JSON and block
                print(json.dumps(output))
                sys.exit(0)  # Exit 0 but decision=deny blocks execution
            
            # Check for warning threshold
            elif analysis.quality_score < warn_threshold:
                # Allow but with warning
                output["hookSpecificOutput"]["permissionDecision"] = "allow"
                
                warnings = []
                warnings.append(f"⚠️ Quality: {analysis.quality_score:.1f}/10.0 (recommended: {warn_threshold}+)")
                
                if pattern_validation:
                    warnings.append(pattern_validation["message"])
                
                output["hookSpecificOutput"]["permissionDecisionReason"] = " | ".join(warnings)
                
                # Output JSON and allow
                print(json.dumps(output))
                sys.exit(0)
            
            # High quality - allow silently
            output["hookSpecificOutput"]["permissionDecision"] = "allow"
            output["suppressOutput"] = True  # Don't show in transcript for high quality
            
            # Track successful pattern match if applicable
            if keyword_matcher and repository:
                match_result = await keyword_matcher.match(prompt_content, None)
                if match_result.matched:
                    await repository.save_pattern_validation(
                        prompt=prompt_content,
                        tool_name=tool_name,
                        quality_score=analysis.quality_score,
                        matched=True,
                        agent=match_result.agent,
                        confidence=match_result.confidence,
                        session_id=session_id
                    )
            
            print(json.dumps(output))
            sys.exit(0)
            
        finally:
            # Cleanup database connection
            if db:
                await db.close()
        
    except Exception as e:
        # Return error in JSON format but don't block
        error_output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": f"Validation error (allowing): {str(e)}"
            },
            "suppressOutput": True
        }
        print(json.dumps(error_output))
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())