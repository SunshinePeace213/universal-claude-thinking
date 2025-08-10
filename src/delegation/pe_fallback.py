"""
Stage 3: PE (Prompt Enhancer) Fallback
Routes low-confidence requests for enhancement
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PEFallbackResult:
    """Result from PE fallback routing"""
    agent: str  # Always 'PE' for fallback
    confidence: float  # Always 1.0 after PE enhancement
    reason: str
    original_confidence: float
    processing_time_ms: float
    enhancement_needed: bool


class PEFallback:
    """
    Stage 3: Fallback mechanism that routes to Prompt Enhancer.
    Used when confidence is below threshold in previous stages.
    """

    def __init__(self):
        """Initialize PE fallback handler"""
        self.fallback_reasons = []

    async def route(self, user_input: str,
                   classification_result: Any | None = None,
                   keyword_result: Any | None = None,
                   semantic_result: Any | None = None) -> PEFallbackResult:
        """
        Route to PE for enhancement when confidence is low.

        Args:
            user_input: The user's request text
            classification_result: Result from classification engine
            keyword_result: Result from keyword matching (Stage 1)
            semantic_result: Result from semantic matching (Stage 2)

        Returns:
            PEFallbackResult with PE routing and reasoning
        """
        start_time = time.perf_counter()

        # Determine why we're falling back to PE
        reason = self._determine_fallback_reason(
            classification_result,
            keyword_result,
            semantic_result
        )

        # Calculate original confidence (best from previous stages)
        original_confidence = self._get_original_confidence(
            classification_result,
            keyword_result,
            semantic_result
        )

        # Determine if enhancement is truly needed
        enhancement_needed = self._check_enhancement_needed(
            user_input,
            original_confidence,
            classification_result
        )

        processing_time = (time.perf_counter() - start_time) * 1000

        return PEFallbackResult(
            agent='PE',
            confidence=1.0,  # PE always returns with full confidence after enhancement
            reason=reason,
            original_confidence=original_confidence,
            processing_time_ms=processing_time,
            enhancement_needed=enhancement_needed
        )

    def _determine_fallback_reason(self,
                                  classification_result: Any | None,
                                  keyword_result: Any | None,
                                  semantic_result: Any | None) -> str:
        """
        Determine the reason for falling back to PE.

        Returns:
            Human-readable reason for PE fallback
        """
        reasons = []

        # Check classification confidence
        if classification_result and hasattr(classification_result, 'confidence'):
            if classification_result.confidence < 0.7:
                reasons.append(f"Low classification confidence ({classification_result.confidence:.2f})")

        # Check keyword matching
        if keyword_result:
            if hasattr(keyword_result, 'matched') and not keyword_result.matched:
                reasons.append("No keyword patterns matched")
            elif hasattr(keyword_result, 'confidence') and keyword_result.confidence < 0.9:
                reasons.append(f"Insufficient keyword confidence ({keyword_result.confidence:.2f})")

        # Check semantic matching
        if semantic_result:
            if hasattr(semantic_result, 'matched') and not semantic_result.matched:
                reasons.append("No semantic match found")
            elif hasattr(semantic_result, 'confidence') and semantic_result.confidence < 0.7:
                reasons.append(f"Low semantic similarity ({semantic_result.confidence:.2f})")

        if not reasons:
            reasons.append("General ambiguity requiring clarification")

        return "PE fallback triggered: " + "; ".join(reasons)

    def _get_original_confidence(self,
                                classification_result: Any | None,
                                keyword_result: Any | None,
                                semantic_result: Any | None) -> float:
        """
        Get the best confidence from previous stages.

        Returns:
            Highest confidence score from previous stages
        """
        confidences = []

        if classification_result and hasattr(classification_result, 'confidence'):
            confidences.append(classification_result.confidence)

        if keyword_result and hasattr(keyword_result, 'confidence'):
            confidences.append(keyword_result.confidence)

        if semantic_result and hasattr(semantic_result, 'confidence'):
            confidences.append(semantic_result.confidence)

        return max(confidences) if confidences else 0.0

    def _check_enhancement_needed(self,
                                 user_input: str,
                                 original_confidence: float,
                                 classification_result: Any | None) -> bool:
        """
        Determine if enhancement is actually needed.

        Args:
            user_input: The user's request
            original_confidence: Best confidence from previous stages
            classification_result: Classification result

        Returns:
            True if enhancement would be beneficial
        """
        # Always enhance if confidence is very low
        if original_confidence < 0.5:
            return True

        # Check input length - very short inputs often need enhancement
        if len(user_input.split()) < 5:
            return True

        # Check for ambiguous task types
        if classification_result and hasattr(classification_result, 'task_type'):
            task_type = str(classification_result.task_type)
            if 'complex' in task_type.lower():
                return True

        # Check for question marks without clear intent
        if '?' in user_input and original_confidence < 0.7:
            return True

        # Default to enhancement if confidence is below 0.7
        return original_confidence < 0.7

    def get_enhancement_prompt(self, user_input: str, context: dict[str, Any]) -> str:
        """
        Generate a prompt for the PE agent to enhance the user's request.

        Args:
            user_input: Original user input
            context: Context about why enhancement is needed

        Returns:
            Prompt for PE agent
        """
        prompt = f"""Please enhance and clarify the following user request:

Original Request: {user_input}

Enhancement Context:
- Classification Confidence: {context.get('classification_confidence', 'N/A')}
- Detected Task Type: {context.get('task_type', 'Unclear')}
- Ambiguity Reason: {context.get('reason', 'General ambiguity')}

Please provide:
1. A clearer, more specific version of the request
2. Identification of the user's actual intent
3. Any missing information that should be requested from the user
4. Suggested task type and appropriate agent for routing
"""
        return prompt
