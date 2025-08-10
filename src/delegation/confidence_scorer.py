"""
Confidence Scoring System
Multi-factor confidence calculation for delegation decisions
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceScore:
    """Comprehensive confidence score with factors"""
    overall_score: float  # 0.0-1.0
    factors: dict[str, float]
    weights: dict[str, float]
    explanation: str
    recommendation: str


class ConfidenceScorer:
    """
    Multi-factor confidence scoring system for delegation decisions.
    Combines multiple signals to determine routing confidence.
    """

    def __init__(self):
        """Initialize confidence scorer with default weights"""
        self.default_weights = {
            'classification': 0.30,  # Task type classification confidence
            'keyword_match': 0.25,   # Keyword pattern matching
            'semantic_similarity': 0.25,  # Embedding similarity
            'context_quality': 0.10,  # Context and quality scores
            'input_clarity': 0.10,    # Input clarity and completeness
        }

    def calculate_confidence(self,
                            classification_result: Any | None = None,
                            keyword_result: Any | None = None,
                            semantic_result: Any | None = None,
                            user_input: str = "",
                            context: dict[str, Any] | None = None) -> ConfidenceScore:
        """
        Calculate multi-factor confidence score.

        Args:
            classification_result: Result from classification engine
            keyword_result: Result from keyword matching
            semantic_result: Result from semantic matching
            user_input: Original user input
            context: Additional context (quality scores, etc.)

        Returns:
            ConfidenceScore with overall score and factor breakdown
        """
        factors = {}
        weights = self.default_weights.copy()

        # Factor 1: Classification confidence
        if classification_result and hasattr(classification_result, 'confidence'):
            factors['classification'] = classification_result.confidence
        else:
            factors['classification'] = 0.5  # Default middle confidence

        # Factor 2: Keyword match confidence
        if keyword_result and hasattr(keyword_result, 'confidence'):
            factors['keyword_match'] = keyword_result.confidence
        elif keyword_result and hasattr(keyword_result, 'matched'):
            factors['keyword_match'] = 1.0 if keyword_result.matched else 0.0
        else:
            factors['keyword_match'] = 0.0

        # Factor 3: Semantic similarity
        if semantic_result and hasattr(semantic_result, 'similarity_score'):
            factors['semantic_similarity'] = semantic_result.similarity_score
        elif semantic_result and hasattr(semantic_result, 'confidence'):
            factors['semantic_similarity'] = semantic_result.confidence
        else:
            factors['semantic_similarity'] = 0.0

        # Factor 4: Context quality
        context_score = self._calculate_context_score(context)
        factors['context_quality'] = context_score

        # Factor 5: Input clarity
        clarity_score = self._calculate_input_clarity(user_input)
        factors['input_clarity'] = clarity_score

        # Adjust weights if some factors are missing
        weights = self._adjust_weights(factors, weights)

        # Calculate weighted average
        overall_score = self._calculate_weighted_average(factors, weights)

        # Generate explanation and recommendation
        explanation = self._generate_explanation(factors, overall_score)
        recommendation = self._generate_recommendation(overall_score, factors)

        return ConfidenceScore(
            overall_score=overall_score,
            factors=factors,
            weights=weights,
            explanation=explanation,
            recommendation=recommendation
        )

    def _calculate_context_score(self, context: dict[str, Any] | None) -> float:
        """
        Calculate confidence based on context quality.

        Args:
            context: Context dictionary with quality scores, etc.

        Returns:
            Context quality score (0.0-1.0)
        """
        if not context:
            return 0.5  # Neutral score if no context

        score = 0.5  # Base score

        # Check quality score from atomic analysis
        if 'quality_score' in context:
            quality = context['quality_score']
            # Normalize quality score (1-10) to confidence (0-1)
            score = quality / 10.0

        # Boost if clear intent is detected
        if context.get('has_clear_intent', False):
            score = min(score * 1.2, 1.0)

        # Reduce if ambiguous
        if context.get('is_ambiguous', False):
            score *= 0.8

        return score

    def _calculate_input_clarity(self, user_input: str) -> float:
        """
        Calculate clarity score based on input characteristics.

        Args:
            user_input: The user's request text

        Returns:
            Input clarity score (0.0-1.0)
        """
        if not user_input:
            return 0.0

        score = 0.5  # Base score

        # Check length (very short or very long inputs are less clear)
        word_count = len(user_input.split())
        if 5 <= word_count <= 50:
            score += 0.2
        elif word_count < 3 or word_count > 100:
            score -= 0.2

        # Check for specific indicators
        if '?' in user_input:
            score += 0.1  # Questions are usually clearer

        # Check for structured input (numbered lists, bullet points)
        if any(marker in user_input for marker in ['1.', '2.', '•', '-', '*']):
            score += 0.2

        # Check for code blocks or technical terms
        if '```' in user_input or 'function' in user_input.lower():
            score += 0.1

        # Ensure score stays within bounds
        return min(max(score, 0.0), 1.0)

    def _adjust_weights(self, factors: dict[str, float],
                       weights: dict[str, float]) -> dict[str, float]:
        """
        Adjust weights when some factors are missing or zero.

        Args:
            factors: Calculated factor scores
            weights: Initial weights

        Returns:
            Adjusted weights that sum to 1.0
        """
        # Identify non-zero factors
        active_factors = {k: v for k, v in factors.items() if v > 0}

        if not active_factors:
            # If no factors are active, use equal weights
            return {k: 1.0 / len(weights) for k in weights}

        # Redistribute weights from zero factors to active ones
        total_active_weight = sum(weights[k] for k in active_factors if k in weights)

        if total_active_weight > 0:
            adjusted_weights = {}
            for k in weights:
                if k in active_factors:
                    # Scale up active factor weights
                    adjusted_weights[k] = weights[k] / total_active_weight
                else:
                    adjusted_weights[k] = 0.0
            return adjusted_weights

        return weights

    def _calculate_weighted_average(self, factors: dict[str, float],
                                   weights: dict[str, float]) -> float:
        """
        Calculate weighted average of factors.

        Args:
            factors: Factor scores
            weights: Factor weights

        Returns:
            Weighted average score (0.0-1.0)
        """
        weighted_sum = 0.0
        weight_sum = 0.0

        for factor, score in factors.items():
            if factor in weights:
                weighted_sum += score * weights[factor]
                weight_sum += weights[factor]

        if weight_sum > 0:
            return weighted_sum / weight_sum
        else:
            return 0.5  # Default middle confidence

    def _generate_explanation(self, factors: dict[str, float],
                            overall_score: float) -> str:
        """
        Generate human-readable explanation of confidence score.

        Args:
            factors: Individual factor scores
            overall_score: Combined confidence score

        Returns:
            Explanation string
        """
        # Determine confidence level
        if overall_score >= 0.9:
            level = "Very High"
        elif overall_score >= 0.7:
            level = "High"
        elif overall_score >= 0.5:
            level = "Medium"
        elif overall_score >= 0.3:
            level = "Low"
        else:
            level = "Very Low"

        # Find strongest and weakest factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_factors[0] if sorted_factors else None
        weakest = sorted_factors[-1] if sorted_factors else None

        explanation = f"Confidence Level: {level} ({overall_score:.2%}). "

        if strongest:
            explanation += f"Strongest signal: {strongest[0].replace('_', ' ')} ({strongest[1]:.2f}). "

        if weakest and weakest[1] < 0.5:
            explanation += f"Weakest signal: {weakest[0].replace('_', ' ')} ({weakest[1]:.2f}). "

        return explanation

    def _generate_recommendation(self, overall_score: float,
                                factors: dict[str, float]) -> str:
        """
        Generate routing recommendation based on confidence.

        Args:
            overall_score: Combined confidence score
            factors: Individual factor scores

        Returns:
            Recommendation string
        """
        if overall_score >= 0.9:
            return "Direct routing with high confidence"
        elif overall_score >= 0.7:
            return "Proceed with routing, confidence acceptable"
        elif overall_score >= 0.5:
            return "Consider semantic verification before routing"
        else:
            return "Recommend PE enhancement for clarification"

    def get_threshold_for_method(self, method: str) -> float:
        """
        Get confidence threshold for delegation method.

        Args:
            method: Delegation method ('keyword', 'semantic', 'fallback')

        Returns:
            Minimum confidence threshold
        """
        thresholds = {
            'keyword': 0.9,   # High confidence for direct keyword routing
            'semantic': 0.7,  # Medium confidence for semantic routing
            'fallback': 0.0,  # Always accept fallback
        }
        return thresholds.get(method, 0.5)
