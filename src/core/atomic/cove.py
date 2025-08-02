"""Chain of Verification (CoVe) implementation for prompt enhancement."""

from .analyzer import AtomicAnalysis


class ChainOfVerification:
    """Implements 4-step Chain of Verification for prompt enhancement.

    CoVe Process:
    1. Generate baseline response (initial analysis)
    2. Plan verification questions based on gaps
    3. Execute verification to check each question
    4. Generate final verified response with enhancements
    """

    # Quality threshold below which CoVe is triggered
    COVE_THRESHOLD = 7.0

    async def enhance_if_needed(
        self, analysis: AtomicAnalysis, prompt: str
    ) -> AtomicAnalysis:
        """Apply CoVe enhancement if quality score is below threshold.

        Args:
            analysis: Initial atomic analysis result
            prompt: Original user prompt

        Returns:
            Enhanced AtomicAnalysis if needed, otherwise original
        """
        if analysis.quality_score >= self.COVE_THRESHOLD:
            return analysis

        # Step 1: We already have baseline (the analysis)

        # Step 2: Generate verification questions
        verification_questions = self._generate_verification_questions(analysis)

        # Step 3: Check each question against the prompt
        verification_results = await self._execute_verification(
            prompt, analysis, verification_questions
        )

        # Step 4: Generate enhanced prompt based on verification
        enhanced_prompt = self._generate_enhanced_prompt(
            prompt, analysis, verification_results
        )

        # Update analysis with enhancement
        analysis.enhancement_suggestions.insert(
            0, f"CoVe Enhanced Prompt: {enhanced_prompt}"
        )

        return analysis

    def _generate_verification_questions(self, analysis: AtomicAnalysis) -> list[str]:
        """Generate verification questions based on gaps and low scores.

        Args:
            analysis: Initial analysis with gaps

        Returns:
            List of verification questions
        """
        questions = []

        # Core verification questions for atomic components
        if "task" in analysis.gaps or not analysis.structure.get("task"):
            questions.extend(
                [
                    "Is the main task/objective clearly defined?",
                    "Does the prompt start with a specific action verb?",
                    "Can the task be completed without additional clarification?",
                ]
            )

        if "constraints" in analysis.gaps or not analysis.structure.get("constraints"):
            questions.extend(
                [
                    "Are all constraints and requirements explicitly stated?",
                    "Are there any implicit limitations that should be made explicit?",
                    "Do the constraints include measurable criteria?",
                ]
            )

        if "output_format" in analysis.gaps or not analysis.structure.get(
            "output_format"
        ):
            questions.extend(
                [
                    "Is the desired output format clearly specified?",
                    "Would adding structure specifications improve the response?",
                    "Should examples be included to clarify expectations?",
                ]
            )

        # Additional quality-focused questions
        if analysis.quality_score < 5.0:
            questions.extend(
                [
                    "Is the prompt self-contained without requiring external context?",
                    "Are all technical terms properly defined or clear from context?",
                    "Would breaking this into multiple specific requests be clearer?",
                ]
            )

        return questions

    async def _execute_verification(
        self, prompt: str, analysis: AtomicAnalysis, questions: list[str]
    ) -> dict[str, bool]:
        """Execute verification by checking each question.

        Args:
            prompt: Original prompt
            analysis: Initial analysis
            questions: Verification questions to check

        Returns:
            Dictionary mapping questions to True/False answers
        """
        results = {}

        for question in questions:
            # Simulate verification logic (in real implementation,
            # this could use more sophisticated NLP)
            answer = await self._verify_single_question(prompt, analysis, question)
            results[question] = answer

        return results

    async def _verify_single_question(
        self, prompt: str, analysis: AtomicAnalysis, question: str
    ) -> bool:
        """Verify a single question against the prompt.

        Args:
            prompt: Original prompt
            analysis: Initial analysis
            question: Verification question

        Returns:
            True if verification passes, False otherwise
        """
        prompt_lower = prompt.lower()

        # Simple heuristic-based verification
        if "clearly defined" in question and "task" in question:
            task = analysis.structure.get("task")
            return bool(task) and len(task.split()) > 2 if task else False

        elif "action verb" in question:
            action_verbs = [
                "create",
                "write",
                "analyze",
                "explain",
                "generate",
                "implement",
                "evaluate",
                "design",
                "develop",
            ]
            return any(verb in prompt_lower for verb in action_verbs)

        elif "constraints" in question and "explicitly" in question:
            return bool(analysis.structure.get("constraints"))

        elif "measurable criteria" in question:
            constraints = analysis.structure.get("constraints") or ""
            return any(char.isdigit() for char in constraints)

        elif "output format" in question and "clearly" in question:
            return bool(analysis.structure.get("output_format"))

        elif "self-contained" in question:
            # Check for references to external context
            external_refs = ["this", "that", "it", "the above", "mentioned"]
            return not any(ref in prompt_lower for ref in external_refs)

        # Default to False for unmatched questions
        return False

    def _generate_enhanced_prompt(
        self,
        prompt: str,
        analysis: AtomicAnalysis,
        verification_results: dict[str, bool],
    ) -> str:
        """Generate enhanced prompt based on verification results.

        Args:
            prompt: Original prompt
            analysis: Initial analysis
            verification_results: Results of verification questions

        Returns:
            Enhanced prompt with atomic structure
        """
        # Start with original prompt
        enhanced_parts = []

        # Check what needs enhancement based on verification
        needs_task = any(
            not result
            for q, result in verification_results.items()
            if "task" in q or "objective" in q
        )
        needs_constraints = any(
            not result
            for q, result in verification_results.items()
            if "constraints" in q
        )
        needs_format = any(
            not result
            for q, result in verification_results.items()
            if "output format" in q
        )

        # Build enhanced prompt with clear structure
        if needs_task:
            if analysis.structure.get("task"):
                # Clarify existing task
                enhanced_parts.append(f"Task: {analysis.structure['task']}")
            else:
                # Suggest task structure
                enhanced_parts.append("Task: [Specify your main objective here]")
        else:
            # Keep original task
            task = analysis.structure.get("task")
            if task:
                enhanced_parts.append(task)

        # Add constraints section if needed
        if needs_constraints:
            enhanced_parts.append("\n\nConstraints:")
            constraints = analysis.structure.get("constraints")
            if constraints:
                enhanced_parts.append(f"- {constraints}")
            enhanced_parts.append(
                "- [Add specific requirements, limitations, or criteria]"
            )
        else:
            constraints = analysis.structure.get("constraints")
            if constraints:
                enhanced_parts.append(f"\n\nWith constraints: {constraints}")

        # Add output format section if needed
        if needs_format:
            enhanced_parts.append("\n\nOutput Format:")
            enhanced_parts.append(
                "- [Specify desired structure: list, paragraphs, JSON, etc.]"
            )
            enhanced_parts.append("- [Include any formatting requirements]")
        else:
            output_format = analysis.structure.get("output_format")
            if output_format:
                enhanced_parts.append(f"\n\nFormat: {output_format}")

        # Join parts into coherent enhanced prompt
        enhanced_prompt = " ".join(enhanced_parts).strip()

        # If enhancement is too similar to original, provide template
        if len(enhanced_prompt) < len(prompt) * 1.2:
            enhanced_prompt = self._generate_template_prompt(analysis)

        return enhanced_prompt

    def _generate_template_prompt(self, analysis: AtomicAnalysis) -> str:
        """Generate a template prompt when enhancement is minimal.

        Args:
            analysis: Initial analysis

        Returns:
            Template prompt with clear atomic structure
        """
        template = """[Atomic Prompt Template]

Task: [Main objective - start with an action verb]

Constraints:
- [Specific requirement 1]
- [Specific requirement 2]
- [Any limitations or boundaries]

Output Format:
- [Desired structure: list, paragraphs, code, etc.]
- [Length or detail requirements]
- [Any specific formatting needs]

Example Enhancement:
Original: "Help me write something about AI"
Enhanced: "Task: Write an introductory explanation of artificial intelligence

Constraints:
- Target audience: beginners with no technical background
- Length: 300-500 words
- Include at least 2 real-world examples

Output Format:
- 3-4 paragraphs
- Start with a simple definition
- End with future implications"
"""
        return template
