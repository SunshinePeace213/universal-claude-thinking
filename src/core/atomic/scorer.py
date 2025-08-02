"""Quality scoring system for atomic prompt analysis."""

import time


class QualityScorer:
    """Scores prompt quality on a 1-10 scale with detailed rationale."""

    # Scoring weights for different components
    WEIGHTS = {
        "task_clarity": 0.35,
        "constraints_specificity": 0.25,
        "output_format_clarity": 0.20,
        "overall_coherence": 0.20,
    }

    # Minimum score threshold that triggers detailed rationale
    RATIONALE_THRESHOLD = 7.0

    # Score adjustments for various criteria
    SCORE_ADJUSTMENTS = {
        # Task clarity adjustments
        "task_base": 5.0,
        "task_good_length": 1.0,
        "task_too_long": -1.0,
        "task_action_verb": 2.0,
        "task_generic_term": -1.5,
        "task_question_mark": 0.5,
        # Constraints adjustments
        "constraints_base_missing": 3.0,
        "constraints_base_present": 5.0,
        "constraints_multiple": 2.0,
        "constraints_single": 1.0,
        "constraints_specific_term": 0.5,
        "constraints_measurable": 1.0,
        # Output format adjustments
        "format_base_missing": 4.0,
        "format_base_present": 6.0,
        "format_specific_term": 1.0,
        "format_has_example": 1.0,
        # Coherence adjustments
        "coherence_base": 5.0,
        "coherence_all_components": 2.0,
        "coherence_good_length": 1.0,
        "coherence_too_long": -1.0,
        "coherence_grammar_bonus": 0.5,
        "coherence_related_parts": 1.0,
    }

    # Thresholds for various checks
    THRESHOLDS = {
        "min_score": 1.0,
        "max_score": 10.0,
        "min_task_length": 5,
        "task_word_count_min": 3,
        "task_word_count_max": 20,
        "task_word_count_too_long": 30,
        "min_constraints_length": 10,
        "prompt_word_count_min": 10,
        "prompt_word_count_max": 100,
        "prompt_word_count_too_long": 200,
        "performance_warning_ms": 100,
        "max_format_terms": 3,
        "max_specific_terms": 2,
    }

    def calculate_score(
        self, structure: dict[str, str | None], gaps: list[str], prompt: str
    ) -> tuple[float, str | None]:
        """Calculate quality score with rationale for low scores.

        Args:
            structure: Extracted prompt structure
            gaps: Detected gaps in the prompt
            prompt: Original prompt text

        Returns:
            Tuple of (score, rationale) where rationale is provided for scores < 7
        """
        start_time = time.time()

        # Component scores
        task_score = self._score_task_clarity(structure.get("task"), prompt)
        constraints_score = self._score_constraints(structure.get("constraints"))
        format_score = self._score_output_format(structure.get("output_format"))
        coherence_score = self._score_coherence(structure, prompt)

        # Weighted average
        total_score = (
            task_score * self.WEIGHTS["task_clarity"]
            + constraints_score * self.WEIGHTS["constraints_specificity"]
            + format_score * self.WEIGHTS["output_format_clarity"]
            + coherence_score * self.WEIGHTS["overall_coherence"]
        )

        # Ensure score is within range
        total_score = max(self.THRESHOLDS["min_score"], min(self.THRESHOLDS["max_score"], total_score))

        # Generate rationale for low scores
        rationale = None
        if total_score < self.RATIONALE_THRESHOLD:
            rationale = self._generate_rationale(
                total_score,
                task_score,
                constraints_score,
                format_score,
                coherence_score,
                gaps,
            )

        # Ensure we stay within performance constraint
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > self.THRESHOLDS["performance_warning_ms"]:
            print(f"Warning: Scoring took {elapsed_ms:.1f}ms")

        return (round(total_score, 1), rationale)

    def _score_task_clarity(self, task: str | None, prompt: str) -> float:
        """Score task clarity (1-10).

        Args:
            task: Extracted task component
            prompt: Original prompt for context

        Returns:
            Score from 1-10
        """
        if not task:
            return self.THRESHOLDS["min_score"]

        score = self.SCORE_ADJUSTMENTS["task_base"]

        # Length check - too short or too long reduces clarity
        task_length = len(task.split())
        if self.THRESHOLDS["task_word_count_min"] <= task_length <= self.THRESHOLDS["task_word_count_max"]:
            score += self.SCORE_ADJUSTMENTS["task_good_length"]
        elif task_length > self.THRESHOLDS["task_word_count_too_long"]:
            score += self.SCORE_ADJUSTMENTS["task_too_long"]

        # Action verb check
        action_verbs = [
            "create",
            "write",
            "analyze",
            "explain",
            "generate",
            "design",
            "implement",
            "evaluate",
            "compare",
            "summarize",
            "develop",
            "calculate",
            "define",
            "describe",
            "identify",
            "list",
        ]
        if any(verb in task.lower() for verb in action_verbs):
            score += self.SCORE_ADJUSTMENTS["task_action_verb"]

        # Specificity check - generic words reduce score
        generic_terms = ["help", "something", "stuff", "thing", "whatever"]
        if any(term in task.lower() for term in generic_terms):
            score += self.SCORE_ADJUSTMENTS["task_generic_term"]

        # Question mark handling
        if "?" in task and task.strip().endswith("?"):
            score += self.SCORE_ADJUSTMENTS["task_question_mark"]

        return min(self.THRESHOLDS["max_score"], max(self.THRESHOLDS["min_score"], score))

    def _score_constraints(self, constraints: str | None) -> float:
        """Score constraint specificity (1-10).

        Args:
            constraints: Extracted constraints

        Returns:
            Score from 1-10
        """
        if not constraints:
            return self.SCORE_ADJUSTMENTS["constraints_base_missing"]

        score = self.SCORE_ADJUSTMENTS["constraints_base_present"]

        # Count distinct constraints
        constraint_count = len(constraints.split(";"))
        if constraint_count >= 2:
            score += self.SCORE_ADJUSTMENTS["constraints_multiple"]
        elif constraint_count == 1:
            score += self.SCORE_ADJUSTMENTS["constraints_single"]

        # Specificity indicators
        specific_terms = [
            "must",
            "should",
            "cannot",
            "within",
            "exactly",
            "only",
            "at least",
            "at most",
            "between",
            "maximum",
            "minimum",
        ]
        specific_count = sum(
            1 for term in specific_terms if term in constraints.lower()
        )
        score += min(self.THRESHOLDS["max_specific_terms"], specific_count * self.SCORE_ADJUSTMENTS["constraints_specific_term"])

        # Measurable constraints bonus
        if any(char.isdigit() for char in constraints):
            score += self.SCORE_ADJUSTMENTS["constraints_measurable"]

        return min(self.THRESHOLDS["max_score"], max(self.THRESHOLDS["min_score"], score))

    def _score_output_format(self, output_format: str | None) -> float:
        """Score output format clarity (1-10).

        Args:
            output_format: Extracted output format specification

        Returns:
            Score from 1-10
        """
        if not output_format:
            return self.SCORE_ADJUSTMENTS["format_base_missing"]

        score = self.SCORE_ADJUSTMENTS["format_base_present"]

        # Specific format indicators
        format_terms = [
            "list",
            "table",
            "json",
            "xml",
            "markdown",
            "code",
            "paragraph",
            "bullet",
            "numbered",
            "step-by-step",
            "summary",
            "detailed",
            "brief",
            "comprehensive",
        ]

        format_count = sum(1 for term in format_terms if term in output_format.lower())
        score += min(self.THRESHOLDS["max_format_terms"], format_count * self.SCORE_ADJUSTMENTS["format_specific_term"])

        # Structure specificity
        if "example" in output_format.lower():
            score += self.SCORE_ADJUSTMENTS["format_has_example"]

        return min(self.THRESHOLDS["max_score"], max(self.THRESHOLDS["min_score"], score))

    def _score_coherence(self, structure: dict[str, str | None], prompt: str) -> float:
        """Score overall prompt coherence (1-10).

        Args:
            structure: Extracted structure
            prompt: Original prompt

        Returns:
            Score from 1-10
        """
        score = self.SCORE_ADJUSTMENTS["coherence_base"]

        # All components present bonus
        if all(structure.values()):
            score += self.SCORE_ADJUSTMENTS["coherence_all_components"]

        # Length appropriateness
        word_count = len(prompt.split())
        if self.THRESHOLDS["prompt_word_count_min"] <= word_count <= self.THRESHOLDS["prompt_word_count_max"]:
            score += self.SCORE_ADJUSTMENTS["coherence_good_length"]
        elif word_count > self.THRESHOLDS["prompt_word_count_too_long"]:
            score += self.SCORE_ADJUSTMENTS["coherence_too_long"]

        # Grammar indicators (simple check)
        if prompt.strip() and prompt[0].isupper():
            score += self.SCORE_ADJUSTMENTS["coherence_grammar_bonus"]
        if prompt.strip().endswith(".") or prompt.strip().endswith("?"):
            score += self.SCORE_ADJUSTMENTS["coherence_grammar_bonus"]

        # Logical flow check - components should relate
        task = structure.get("task")
        constraints = structure.get("constraints")
        if task and constraints:
            # Check if constraints relate to task
            task_words = set(task.lower().split())
            constraint_words = set(constraints.lower().split())
            if task_words & constraint_words:  # Some overlap
                score += self.SCORE_ADJUSTMENTS["coherence_related_parts"]

        return min(self.THRESHOLDS["max_score"], max(self.THRESHOLDS["min_score"], score))

    def _generate_rationale(
        self,
        total_score: float,
        task_score: float,
        constraints_score: float,
        format_score: float,
        coherence_score: float,
        gaps: list[str],
    ) -> str:
        """Generate detailed rationale for scores below 7.

        Args:
            total_score: Overall quality score
            task_score: Task clarity subscore
            constraints_score: Constraints subscore
            format_score: Output format subscore
            coherence_score: Coherence subscore
            gaps: Detected gaps

        Returns:
            Detailed rationale explaining the score
        """
        rationale_parts = []

        rationale_parts.append(
            f"Quality Score: {total_score:.1f}/10.0 - "
            "Your prompt could be enhanced for better results."
        )

        # Explain component scores
        if task_score < 7:
            rationale_parts.append(
                f"Task Clarity ({task_score:.1f}/10): "
                "The main objective could be more specific. "
                "Try starting with a clear action verb."
            )

        if constraints_score < 7:
            rationale_parts.append(
                f"Constraints ({constraints_score:.1f}/10): "
                "Adding specific requirements or limitations would help "
                "narrow down the response to your needs."
            )

        if format_score < 7:
            rationale_parts.append(
                f"Output Format ({format_score:.1f}/10): "
                "Specifying how you want the information structured "
                "(e.g., list, paragraphs, examples) would improve results."
            )

        if coherence_score < 7:
            rationale_parts.append(
                f"Overall Structure ({coherence_score:.1f}/10): "
                "The prompt could benefit from better organization "
                "of its components."
            )

        # Mention specific gaps
        if gaps:
            gap_text = "Missing components: " + ", ".join(gaps)
            rationale_parts.append(gap_text)

        # Add encouragement
        rationale_parts.append(
            "Consider using the enhancement suggestions to improve your prompt."
        )

        return " ".join(rationale_parts)
