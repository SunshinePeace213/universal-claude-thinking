"""
Stage 1: Fast Keyword Matching
Achieves <10ms performance through pre-compiled patterns
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import logging
import re
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class KeywordMatchResult:
    """Result from keyword matching"""
    matched: bool
    agent: str | None
    confidence: float
    patterns_matched: list[str]
    processing_time_ms: float


class KeywordMatcher:
    """
    Stage 1: Fast keyword matching for high-confidence direct routing.
    Pre-compiled patterns ensure <10ms performance.
    """

    def __init__(self):
        """Initialize with pre-compiled agent keyword patterns"""
        self._compile_agent_patterns()

    def _compile_agent_patterns(self):
        """
        Pre-compile regex patterns for each agent.
        Patterns are ordered by specificity and frequency.
        """
        # PE (Prompt Enhancer) patterns
        self.pe_patterns = [
            (re.compile(r'\b(enhance|improve|clarify|refine) (?:my |the )?(?:prompt|request|query)\b', re.I), 0.95),
            (re.compile(r'\b(make .+ clearer|help me ask|rephrase|reword)\b', re.I), 0.90),
            (re.compile(r'\b(quality|validation|assessment) (?:of |for )?(?:prompt|input)\b', re.I), 0.85),
        ]

        # R1 (Researcher) patterns
        self.r1_patterns = [
            (re.compile(r'\b(research|search|find|lookup|investigate)\b', re.I), 0.95),
            (re.compile(r'\b(gather|collect|compile) (?:information|data|sources)\b', re.I), 0.90),
            (re.compile(r'\b(web search|online research|internet lookup|google)\b', re.I), 0.95),
            (re.compile(r'\b(verify|validate|fact.check|source)\b', re.I), 0.85),
            (re.compile(r'\b(latest|current|recent|best practices?|documentation)\b', re.I), 0.90),
        ]

        # A1 (Reasoner) patterns
        self.a1_patterns = [
            (re.compile(r'\b(reason|analyze|solve|think through|figure out)\b', re.I), 0.90),
            (re.compile(r'\b(logic|reasoning|analysis|problem.solving)\b', re.I), 0.85),
            (re.compile(r'\b(debug|troubleshoot|diagnose|investigate).*(error|issue|problem|bug)\b', re.I), 0.95),
            (re.compile(r'\b(step.by.step|systematic|methodical) (?:approach|solution)\b', re.I), 0.90),
        ]

        # E1 (Evaluator) patterns
        self.e1_patterns = [
            (re.compile(r'\b(evaluate|assess|review|check|validate) (?:quality|code|solution)\b', re.I), 0.95),
            (re.compile(r'\b(quality assurance|QA|testing|validation)\b', re.I), 0.90),
            (re.compile(r'\b(error detection|find (?:bugs|issues|problems))\b', re.I), 0.85),
        ]

        # T1 (Tool User) patterns
        self.t1_patterns = [
            (re.compile(r'\b(execute|run|use|invoke) (?:tool|command|script|automation)\b', re.I), 0.95),
            (re.compile(r'\b(browser|selenium|playwright|web automation)\b', re.I), 0.95),
            (re.compile(r'\b(API|integration|system interaction|external service)\b', re.I), 0.85),
            (re.compile(r'\b(automate|automation|scripting)\b', re.I), 0.90),
            (re.compile(r'\b(test|testing).*(UI|form|login|web|browser)\b', re.I), 0.95),
        ]

        # W1 (Writer) patterns
        self.w1_patterns = [
            (re.compile(r'\b(write|create|draft|compose) (?:content|document|article|text)\b', re.I), 0.95),
            (re.compile(r'\b(documentation|blog|report|essay|paper)\b', re.I), 0.85),
            (re.compile(r'\b(content creation|copywriting|technical writing)\b', re.I), 0.90),
        ]

        # I1 (Interface) patterns
        self.i1_patterns = [
            (re.compile(r'\b(interact|communicate|interface|user interaction)\b', re.I), 0.85),
            (re.compile(r'\b(clarif(?:y|ication)|explain|help me understand)\b', re.I), 0.90),
            (re.compile(r'\b(user experience|UX|interface design)\b', re.I), 0.85),
        ]

        # Compile all patterns into a single structure for efficient matching
        self.agent_patterns = {
            'PE': self.pe_patterns,
            'R1': self.r1_patterns,
            'A1': self.a1_patterns,
            'E1': self.e1_patterns,
            'T1': self.t1_patterns,
            'W1': self.w1_patterns,
            'I1': self.i1_patterns,
        }

    async def match(self, user_input: str, task_type: str | None = None) -> KeywordMatchResult:
        """
        Perform fast keyword matching to identify agent.

        Args:
            user_input: The user's request text
            task_type: Optional task type from classification

        Returns:
            KeywordMatchResult with agent and confidence
        """
        start_time = time.perf_counter()

        # Track all matches
        all_matches = []

        # Check each agent's patterns
        for agent, patterns in self.agent_patterns.items():
            agent_confidence, matched_patterns = self._check_agent_patterns(user_input, patterns)
            if agent_confidence > 0:
                all_matches.append((agent, agent_confidence, matched_patterns))

        # Select best match
        if all_matches:
            # Sort by confidence
            all_matches.sort(key=lambda x: x[1], reverse=True)
            best_agent, best_confidence, best_patterns = all_matches[0]

            # Apply task type boost if available
            if task_type:
                best_confidence = self._apply_task_type_boost(best_agent, task_type, best_confidence)

            # Check if confidence meets threshold for keyword matching
            if best_confidence >= 0.9:
                processing_time = (time.perf_counter() - start_time) * 1000
                return KeywordMatchResult(
                    matched=True,
                    agent=best_agent,
                    confidence=best_confidence,
                    patterns_matched=best_patterns,
                    processing_time_ms=processing_time
                )

        # No high-confidence match found
        processing_time = (time.perf_counter() - start_time) * 1000
        return KeywordMatchResult(
            matched=False,
            agent=None,
            confidence=0.0,
            patterns_matched=[],
            processing_time_ms=processing_time
        )

    def _check_agent_patterns(self, text: str, patterns: list[tuple[re.Pattern, float]]) -> tuple[float, list[str]]:
        """
        Check patterns for a specific agent.

        Returns:
            Tuple of (confidence, matched_patterns)
        """
        matched_patterns = []
        max_confidence = 0.0

        for pattern, confidence in patterns:
            if pattern.search(text):
                matched_patterns.append(pattern.pattern[:50] + "...")
                max_confidence = max(max_confidence, confidence)

        # Boost confidence if multiple patterns match
        if len(matched_patterns) > 1:
            max_confidence = min(max_confidence * (1 + len(matched_patterns) * 0.05), 1.0)

        return max_confidence, matched_patterns

    def _apply_task_type_boost(self, agent: str, task_type: str, confidence: float) -> float:
        """
        Apply confidence boost based on task type alignment.

        Args:
            agent: Matched agent
            task_type: Task type from classification
            confidence: Current confidence

        Returns:
            Boosted confidence
        """
        # Task type to preferred agent mapping
        task_agent_map = {
            'simple_direct': 'PE',
            'complex_multi_step': 'A1',
            'research_required': 'R1',
            'web_testing': 'T1',
            'debugging_error': 'A1',
        }

        # Apply boost if agent matches task type preference
        if task_type in task_agent_map and task_agent_map[task_type] == agent:
            return min(confidence * 1.1, 1.0)

        return confidence

    def get_agent_capabilities(self) -> dict[str, list[str]]:
        """Get capability descriptions for each agent"""
        return {
            'PE': ['prompt enhancement', 'input validation', 'quality assessment', 'clarification'],
            'R1': ['web search', 'data gathering', 'source verification', 'research compilation'],
            'A1': ['logical analysis', 'problem solving', 'reasoning chains', 'debugging'],
            'E1': ['quality assessment', 'validation', 'error detection', 'code review'],
            'T1': ['tool execution', 'automation', 'system interaction', 'browser control'],
            'W1': ['content creation', 'writing', 'documentation', 'copywriting'],
            'I1': ['user interaction', 'clarification', 'communication', 'interface design'],
        }
