"""Atomic Foundation analyzer for prompt structure and gap detection."""

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

from .safety import SafetyValidator
from .scorer import QualityScorer


@dataclass
class AtomicAnalysis:
    """Result of atomic prompt analysis."""

    structure: dict[str, str | None]
    quality_score: float
    gaps: list[str]
    enhancement_suggestions: list[str]
    processing_time_ms: float
    prompt_hash: str
    rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "structure": self.structure,
            "quality_score": self.quality_score,
            "gaps": self.gaps,
            "enhancement_suggestions": self.enhancement_suggestions,
            "processing_time_ms": self.processing_time_ms,
            "prompt_hash": self.prompt_hash,
            "rationale": self.rationale,
        }


class AtomicFoundation:
    """Core atomic prompt analysis and enhancement."""

    def __init__(self) -> None:
        self.gap_analyzer = GapAnalyzer()
        self.quality_scorer = QualityScorer()
        self.safety_validator = SafetyValidator()

    async def analyze_prompt(self, prompt: str) -> AtomicAnalysis:
        """Analyze prompt structure and return quality assessment.

        Args:
            prompt: User input prompt to analyze

        Returns:
            AtomicAnalysis with structure, score, gaps, and suggestions
        """
        start_time = time.time()

        # Validate prompt safety first
        is_safe, safety_error = self.safety_validator.validate_prompt(prompt)
        if not is_safe:
            # Return low-quality analysis for unsafe prompts
            return AtomicAnalysis(
                structure={"task": None, "constraints": None, "output_format": None},
                quality_score=1.0,
                gaps=["task", "constraints", "output_format"],
                enhancement_suggestions=[f"Safety validation failed: {safety_error}"],
                processing_time_ms=(time.time() - start_time) * 1000,
                prompt_hash="unsafe",
                rationale=f"Prompt rejected due to safety concerns: {safety_error}",
            )

        # Sanitize prompt
        prompt = self.safety_validator.sanitize_prompt(prompt)

        # Generate prompt hash for caching
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

        # Extract atomic structure components
        structure = self._extract_structure(prompt)

        # Detect gaps in the structure
        gaps = self.gap_analyzer.detect_gaps(structure)

        # Calculate quality score with rationale
        quality_score, rationale = self.quality_scorer.calculate_score(
            structure, gaps, prompt
        )

        # Generate enhancement suggestions
        suggestions = self._generate_suggestions(structure, gaps)

        processing_time_ms = (time.time() - start_time) * 1000

        return AtomicAnalysis(
            structure=structure,
            quality_score=quality_score,
            gaps=gaps,
            enhancement_suggestions=suggestions,
            processing_time_ms=processing_time_ms,
            prompt_hash=prompt_hash,
            rationale=rationale,
        )

    def _extract_structure(self, prompt: str) -> dict[str, str | None]:
        """Extract Task, Constraints, and Output Format from prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Dictionary with task, constraints, and output_format keys
        """
        structure: dict[str, str | None] = {
            "task": None,
            "constraints": None,
            "output_format": None,
        }

        # Normalize prompt
        prompt_lower = prompt.lower().strip()

        # Extract components
        structure["task"] = self._extract_task(prompt, prompt_lower)
        structure["constraints"] = self._extract_constraints(prompt, prompt_lower)
        structure["output_format"] = self._extract_output_format(
            prompt, prompt_lower, structure.get("task") or ""
        )

        return structure

    def _extract_task(self, prompt: str, prompt_lower: str) -> str | None:
        """Extract the main task from the prompt.

        Args:
            prompt: Original prompt
            prompt_lower: Lowercase version of prompt

        Returns:
            Extracted task or None
        """
        # Task patterns - usually the main instruction/verb
        task_patterns = [
            r"^(.*?)(?:with|using|given|considering|ensuring|following)",
            r"^(.*?)(?:\.|\?|!|$)",
            r"^(please\s+)?(\w+.*?)(?:\s+and\s+|\s+but\s+|\s+with\s+|$)",
        ]

        for pattern in task_patterns:
            match = re.search(pattern, prompt_lower, re.IGNORECASE | re.DOTALL)
            if match:
                task_text = match.group(1).strip()
                if task_text and len(task_text) > 3:
                    return prompt[: len(task_text)].strip()

        return None

    def _extract_constraints(self, prompt: str, prompt_lower: str) -> str | None:
        """Extract constraints from the prompt.

        Args:
            prompt: Original prompt
            prompt_lower: Lowercase version of prompt

        Returns:
            Extracted constraints joined by semicolons or None
        """
        constraint_indicators = [
            "using",
            "with",
            "ensuring",
            "must",
            "should",
            "cannot",
            "don't",
            "avoid",
            "only",
            "within",
            "less than",
            "more than",
            "at least",
            "at most",
            "between",
            "without",
            "except",
            "focusing on",
            "limited to",
            "restricted to",
            "based on",
            "from",
            "keeping",
            "under",
            "suitable for",
            "appropriate for",
        ]

        constraints = []
        for indicator in constraint_indicators:
            pattern = rf"\b{indicator}\s+([^,.;]+)"
            matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
            for match in matches:
                # Find original case version
                start_idx = prompt_lower.find(indicator + " " + match)
                if start_idx != -1:
                    end_idx = start_idx + len(indicator) + 1 + len(match)
                    constraints.append(prompt[start_idx:end_idx].strip())

        return "; ".join(constraints) if constraints else None

    def _extract_output_format(
        self, prompt: str, prompt_lower: str, task: str
    ) -> str | None:
        """Extract output format from the prompt.

        Args:
            prompt: Original prompt
            prompt_lower: Lowercase version of prompt
            task: Extracted task for implicit format detection

        Returns:
            Extracted output format or None
        """
        # Check for implicit formats in creative tasks
        implicit_formats = [
            "haiku",
            "sonnet",
            "limerick",
            "poem",
            "story",
            "essay",
            "letter",
            "email",
            "explanation",
            "summary",
            "review",
            "report",
            "analysis",
        ]
        task_lower = task.lower()
        for fmt in implicit_formats:
            if fmt in task_lower:
                return fmt

        # Format indicators
        format_indicators = [
            "format",
            "as a",
            "as an",
            "in the form of",
            "structured as",
            "return",
            "output",
            "provide",
            "give me",
            "show me",
            "list",
            "table",
            "json",
            "xml",
            "markdown",
            "bullet",
            "numbered",
            "present",
            "deliver",
            "report",
        ]

        # First check for "in a ... format" pattern
        format_match = re.search(r"in\s+a\s+([^,.;]+)\s+format", prompt_lower)
        if format_match:
            start_idx = prompt_lower.find(format_match.group(0))
            end_idx = start_idx + len(format_match.group(0))
            return prompt[start_idx:end_idx].strip()

        # Check other format indicators
        for indicator in format_indicators:
            pattern = rf"\b{indicator}\s+([^,.;]+)"
            match = re.search(pattern, prompt_lower, re.IGNORECASE)
            if match:
                # Find original case version
                start_idx = prompt_lower.find(indicator)
                if start_idx != -1:
                    end_idx = start_idx + len(indicator) + 1 + len(match.group(1))
                    return prompt[start_idx:end_idx].strip()

        return None

    def _generate_suggestions(
        self, structure: dict[str, str | None], gaps: list[str]
    ) -> list[str]:
        """Generate 3-5 enhancement suggestions based on gaps.

        Args:
            structure: Extracted prompt structure
            gaps: Detected gaps in the prompt

        Returns:
            List of enhancement suggestions
        """
        suggestions = []

        if "task" in gaps:
            suggestions.append(
                "Start with a clear action verb (e.g., 'Write', 'Analyze', 'Create') "
                "to define what you want to accomplish."
            )
            suggestions.append(
                "Be specific about your objective. Instead of 'help me', "
                "try 'explain how to...' or 'create a plan for...'"
            )

        if "constraints" in gaps:
            suggestions.append(
                "Add specific constraints or requirements using words like "
                "'must', 'should', 'within', 'using only', etc."
            )
            suggestions.append(
                "Consider adding scope limitations (e.g., word count, time frame, "
                "resource constraints) to get more focused results."
            )

        if "output_format" in gaps:
            suggestions.append(
                "Specify the desired output format (e.g., 'as a bulleted list', "
                "'in JSON format', 'as a step-by-step guide')."
            )
            suggestions.append(
                "Clarify how you want the information structured "
                "(e.g., 'provide 3 examples', 'organize by priority')."
            )

        # General enhancement suggestions
        if not gaps:
            suggestions.append(
                "Your prompt has a clear structure! Consider adding more specific "
                "examples or context for even better results."
            )
            suggestions.append(
                "Well-structured prompt! You might enhance it by specifying "
                "the level of detail or expertise you're looking for."
            )

        # Ensure we return 3-5 suggestions
        if len(suggestions) < 3:
            suggestions.append(
                "Consider adding context about your use case or end goal "
                "to get more tailored results."
            )

        return suggestions[:5]  # Return at most 5 suggestions


class GapAnalyzer:
    """Analyzes prompts for missing atomic components."""

    def detect_gaps(self, structure: dict[str, str | None]) -> list[str]:
        """Detect which atomic components are missing.

        Args:
            structure: Extracted prompt structure

        Returns:
            List of missing component names
        """
        gaps = []

        # Check for missing or weak task definition
        task = structure.get("task", "")
        weak_tasks = ["help", "help me", "do something", "please help", "assist"]
        if not task or len(task) < 5 or task.lower().strip() in weak_tasks:
            gaps.append("task")

        # Check for missing or weak constraints
        constraints = structure.get("constraints", "")
        weak_constraints = ["with user data", "with data", "with information"]
        if (
            not constraints
            or len(constraints) < 10
            or constraints.lower().strip() in weak_constraints
        ):
            gaps.append("constraints")

        # Check for missing output format
        if not structure.get("output_format"):
            gaps.append("output_format")

        return gaps

    def generate_clarifications(self, prompt: str, gaps: list[str]) -> list[str]:
        """Generate 'Do you mean X?' clarification options.

        Args:
            prompt: Original prompt
            gaps: Detected gaps

        Returns:
            List of clarification questions
        """
        clarifications = []

        if "task" in gaps:
            # Analyze prompt for potential task intent
            if "help" in prompt.lower():
                clarifications.append("Do you mean: 'Explain how to...'?")
                clarifications.append("Do you mean: 'Create a guide for...'?")
            elif "need" in prompt.lower():
                clarifications.append("Do you mean: 'Provide recommendations for...'?")
                clarifications.append("Do you mean: 'Generate a solution for...'?")
            else:
                clarifications.append("Do you mean: 'Analyze and summarize...'?")
                clarifications.append("Do you mean: 'Create a detailed plan for...'?")

        if "output_format" in gaps:
            clarifications.append("Do you want the response as a bulleted list?")
            clarifications.append("Do you want a step-by-step explanation?")
            clarifications.append("Do you want a structured summary?")

        return clarifications[:3]  # Return at most 3 clarifications
