#!/usr/bin/env python3
"""UserPromptSubmit hook for prompt enhancement with CoVe."""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.atomic import AtomicFoundation, ChainOfVerification


async def main():
    """Process user prompt and provide enhancement suggestions."""
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
        
        # Extract prompt from the expected structure
        prompt = input_data.get("prompt", "")
        
        if not prompt:
            # No prompt to enhance
            sys.exit(0)
        
        # Initialize atomic foundation and CoVe
        atomic = AtomicFoundation()
        cove = ChainOfVerification()
        
        # Analyze the prompt
        analysis = await atomic.analyze_prompt(prompt)
        
        # Apply CoVe enhancement if needed
        enhanced_analysis = await cove.enhance_if_needed(analysis, prompt)
        
        # Prepare output message
        output_lines = []
        
        # Show quality score
        output_lines.append(
            f"📊 Prompt Quality Score: {enhanced_analysis.quality_score}/10.0"
        )
        
        # Show rationale if score is low
        if enhanced_analysis.rationale:
            output_lines.append(f"\n💡 {enhanced_analysis.rationale}")
        
        # Show gaps if any
        if enhanced_analysis.gaps:
            output_lines.append(
                f"\n🔍 Missing components: {', '.join(enhanced_analysis.gaps)}"
            )
        
        # Show enhancement suggestions
        if enhanced_analysis.quality_score < 7.0:
            output_lines.append("\n✨ Enhancement Suggestions:")
            for i, suggestion in enumerate(enhanced_analysis.enhancement_suggestions[:3], 1):
                output_lines.append(f"   {i}. {suggestion}")
        
        # Performance check
        if enhanced_analysis.processing_time_ms > 500:
            output_lines.append(
                f"\n⚠️ Analysis took {enhanced_analysis.processing_time_ms:.0f}ms "
                "(target: <500ms)"
            )
        
        # Output the message
        if output_lines:
            print("\n".join(output_lines))
        
        # Exit successfully
        sys.exit(0)
        
    except Exception as e:
        # Log error and exit gracefully
        print(f"Error in prompt enhancer: {str(e)}", file=sys.stderr)
        sys.exit(0)  # Don't block the user


if __name__ == "__main__":
    asyncio.run(main())