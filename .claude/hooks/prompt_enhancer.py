#!/usr/bin/env python3
"""UserPromptSubmit hook for prompt enhancement with CoVe and pattern learning."""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.atomic import AtomicFoundation, ChainOfVerification
from src.core.storage.db import DatabaseConnection
from src.core.storage.atomic_repository import AtomicAnalysisRepository
from src.delegation.engine import HybridDelegationEngine
from src.core.atomic.pattern_library import PatternLibrary


async def main():
    """Process user prompt with enhanced delegation awareness and pattern learning."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract prompt and metadata from the expected structure
        prompt = input_data.get("prompt", "")
        session_id = input_data.get("session_id", "")
        cwd = input_data.get("cwd", "")
        
        if not prompt:
            # No prompt to enhance
            sys.exit(0)
        
        # Initialize database and components
        db = None
        repository = None
        delegation_engine = None
        pattern_library = None
        
        try:
            db = DatabaseConnection()
            await db.initialize_schema()
            repository = AtomicAnalysisRepository(db)
            
            # Initialize delegation engine to check pattern matching
            delegation_engine = HybridDelegationEngine()
            await delegation_engine.initialize()
            
            # Initialize pattern library for analysis
            pattern_library = PatternLibrary()
        except Exception as e:
            # Continue without advanced features if initialization fails
            pass
        
        try:
            # Initialize atomic foundation and CoVe
            atomic = AtomicFoundation(repository=repository)
            cove = ChainOfVerification()
            
            # Analyze the prompt
            analysis = await atomic.analyze_prompt(prompt)
            
            # Apply CoVe enhancement if needed
            enhanced_analysis = await cove.enhance_if_needed(analysis, prompt)
            
            # Check delegation path if engine available
            delegation_info = None
            pattern_match_info = None
            
            if delegation_engine:
                # Simulate delegation to understand routing
                delegation_result = await delegation_engine.delegate(prompt)
                delegation_info = {
                    "method": delegation_result.delegation_method,
                    "agent": delegation_result.selected_agent,
                    "confidence": delegation_result.confidence_score.overall_score
                }
                
                # Check if this went to fallback (unmatched patterns)
                if delegation_result.delegation_method == "fallback":
                    # Track this as a pattern learning opportunity
                    pattern_match_info = {
                        "status": "unmatched",
                        "prompt_snippet": prompt[:100],
                        "timestamp": datetime.now().isoformat(),
                        "session": session_id
                    }
                    
                    # Log to pattern learning database if available
                    if repository:
                        await repository.save_pattern_learning_opportunity(
                            prompt=prompt,
                            delegation_method="fallback",
                            session_id=session_id
                        )
            
            # Prepare JSON output for Claude Code hook system
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": None
                },
                "suppressOutput": False,
                "continue": True
            }
            
            # Build context message
            context_parts = []
            
            # Add quality analysis
            context_parts.append(f"📊 Prompt Quality: {enhanced_analysis.quality_score}/10.0")
            
            # Add delegation info if available
            if delegation_info:
                context_parts.append(
                    f"🚦 Delegation: {delegation_info['method']} → {delegation_info['agent']} "
                    f"(confidence: {delegation_info['confidence']:.2f})"
                )
            
            # Add pattern learning notice for unmatched patterns
            if pattern_match_info and pattern_match_info["status"] == "unmatched":
                context_parts.append(
                    "🎯 Pattern Learning: This prompt doesn't match existing patterns. "
                    "The system will learn from this interaction."
                )
            
            # Add gaps if any
            if enhanced_analysis.gaps:
                context_parts.append(f"🔍 Missing: {', '.join(enhanced_analysis.gaps)}")
            
            # Add enhancement suggestions if quality is low
            if enhanced_analysis.quality_score < 7.0:
                suggestions = enhanced_analysis.enhancement_suggestions[:2]
                if suggestions:
                    context_parts.append("💡 Suggestions: " + " | ".join(suggestions))
            
            # Set additional context if we have insights
            if context_parts:
                output["hookSpecificOutput"]["additionalContext"] = "\n".join(context_parts)
            
            # Performance warning if slow
            if enhanced_analysis.processing_time_ms > 500:
                output["hookSpecificOutput"]["performanceWarning"] = (
                    f"Analysis took {enhanced_analysis.processing_time_ms:.0f}ms (target: <500ms)"
                )
            
            # Output JSON for Claude Code to process
            print(json.dumps(output))
            
            # Exit successfully
            sys.exit(0)
            
        finally:
            # Cleanup database connection
            if db:
                await db.close()
        
    except Exception as e:
        # Return error in JSON format for Claude Code
        error_output = {
            "continue": True,  # Don't block on errors
            "suppressOutput": True,
            "error": f"Enhancement error: {str(e)}"
        }
        print(json.dumps(error_output))
        sys.exit(0)  # Don't block the user


if __name__ == "__main__":
    asyncio.run(main())