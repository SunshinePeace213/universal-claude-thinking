"""
Hybrid Delegation Engine
Main orchestrator for 3-stage delegation: keyword → semantic → PE fallback
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

from .confidence_scorer import ConfidenceScore, ConfidenceScorer
from .keyword_matcher import KeywordMatcher, KeywordMatchResult
from .pe_fallback import PEFallback, PEFallbackResult
from .semantic_matcher import SemanticMatcher, SemanticMatchResult

# Import SubAgentManager for integration
try:
    from ..agents.manager import SubAgentManager
    SUB_AGENT_MANAGER_AVAILABLE = True
except ImportError:
    SUB_AGENT_MANAGER_AVAILABLE = False
    SubAgentManager = None

logger = logging.getLogger(__name__)


@dataclass
class DelegationResult:
    """Complete result from delegation engine"""
    success: bool
    selected_agent: str
    delegation_method: str  # "keyword", "semantic", or "fallback"
    confidence_score: ConfidenceScore
    stage_results: Dict[str, Any] = field(default_factory=dict)
    total_processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'success': self.success,
            'selected_agent': self.selected_agent,
            'delegation_method': self.delegation_method,
            'confidence': self.confidence_score.overall_score,
            'factors': self.confidence_score.factors,
            'total_time_ms': self.total_processing_time_ms,
            'timestamp': self.timestamp.isoformat(),
        }


class HybridDelegationEngine:
    """
    3-Stage Hybrid Delegation Engine.
    Routes requests through keyword → semantic → PE fallback.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize delegation engine with components.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize stages
        self.keyword_matcher = KeywordMatcher()
        self.semantic_matcher = SemanticMatcher(
            model_name=self.config.get('embedding_model', 'all-MiniLM-L6-v2'),
            use_mlx=self.config.get('use_mlx', None)
        )
        self.pe_fallback = PEFallback()
        self.confidence_scorer = ConfidenceScorer()

        # Initialize SubAgentManager if available
        self.sub_agent_manager = None
        if SUB_AGENT_MANAGER_AVAILABLE:
            try:
                self.sub_agent_manager = SubAgentManager()
                logger.info("SubAgentManager initialized for delegation engine")
            except Exception as e:
                logger.warning(f"Failed to initialize SubAgentManager: {e}")
                self.sub_agent_manager = None

        # Performance tracking
        self.metrics = {
            'total_delegations': 0,
            'keyword_matches': 0,
            'semantic_matches': 0,
            'fallback_routes': 0,
            'avg_processing_time': 0.0,
        }

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Hybrid Delegation Engine")
        await self.semantic_matcher.initialize()
        logger.info("Delegation Engine initialized successfully")

    async def delegate(self, user_input: str,
                      classification_result: Any | None = None,
                      context: Dict[str, Any] | None = None) -> DelegationResult:
        """
        Main delegation method using 3-stage system.

        Args:
            user_input: The user's request text
            classification_result: Result from classification engine
            context: Additional context (quality scores, etc.)

        Returns:
            DelegationResult with selected agent and confidence
        """
        start_time = time.perf_counter()
        stage_results = {}

        # Extract task type from classification
        task_type = None
        if classification_result and hasattr(classification_result, 'task_type'):
            task_type = str(classification_result.task_type.value)

        # Stage 1: Fast keyword matching (<10ms)
        logger.debug("Stage 1: Keyword matching")
        keyword_result = await self._keyword_match(user_input, task_type)
        stage_results['keyword'] = keyword_result

        # Check if keyword match is sufficient
        if keyword_result.matched and keyword_result.confidence >= 0.9:
            # High-confidence keyword match - use it
            logger.info(f"Stage 1 match: Agent {keyword_result.agent} with confidence {keyword_result.confidence:.2f}")

            # Calculate final confidence
            confidence_score = self.confidence_scorer.calculate_confidence(
                classification_result=classification_result,
                keyword_result=keyword_result,
                user_input=user_input,
                context=context
            )

            # Update metrics
            self.metrics['keyword_matches'] += 1
            self.metrics['total_delegations'] += 1

            total_time = (time.perf_counter() - start_time) * 1000
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_delegations'] - 1) + total_time) /
                self.metrics['total_delegations']
            )

            return DelegationResult(
                success=True,
                selected_agent=keyword_result.agent,
                delegation_method="keyword",
                confidence_score=confidence_score,
                stage_results=stage_results,
                total_processing_time_ms=total_time
            )

        # Stage 2: Semantic matching with embeddings
        logger.debug("Stage 2: Semantic matching")
        semantic_result = await self._semantic_match(user_input, task_type, keyword_result)
        stage_results['semantic'] = semantic_result

        # Check if semantic match is sufficient
        if semantic_result.matched and semantic_result.confidence >= 0.7:
            # Good semantic match - use it
            logger.info(f"Stage 2 match: Agent {semantic_result.agent} with confidence {semantic_result.confidence:.2f}")

            # Calculate final confidence
            confidence_score = self.confidence_scorer.calculate_confidence(
                classification_result=classification_result,
                keyword_result=keyword_result,
                semantic_result=semantic_result,
                user_input=user_input,
                context=context
            )

            # Update metrics
            self.metrics['semantic_matches'] += 1
            self.metrics['total_delegations'] += 1

            total_time = (time.perf_counter() - start_time) * 1000
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_delegations'] - 1) + total_time) /
                self.metrics['total_delegations']
            )

            return DelegationResult(
                success=True,
                selected_agent=semantic_result.agent,
                delegation_method="semantic",
                confidence_score=confidence_score,
                stage_results=stage_results,
                total_processing_time_ms=total_time
            )

        # Stage 3: PE enhancement fallback
        logger.debug("Stage 3: PE fallback")
        fallback_result = await self._pe_enhancement_fallback(
            user_input,
            classification_result,
            keyword_result,
            semantic_result
        )
        stage_results['fallback'] = fallback_result

        logger.info("Stage 3 fallback: Routing to PE for enhancement")

        # Calculate final confidence (will be 1.0 after PE enhancement)
        confidence_score = self.confidence_scorer.calculate_confidence(
            classification_result=classification_result,
            keyword_result=keyword_result,
            semantic_result=semantic_result,
            user_input=user_input,
            context=context
        )

        # Update metrics
        self.metrics['fallback_routes'] += 1

        total_time = (time.perf_counter() - start_time) * 1000

        # Update average processing time
        self.metrics['total_delegations'] += 1
        self.metrics['avg_processing_time'] = (
            (self.metrics['avg_processing_time'] * (self.metrics['total_delegations'] - 1) + total_time) /
            self.metrics['total_delegations']
        )

        return DelegationResult(
            success=True,
            selected_agent=fallback_result.agent,
            delegation_method="fallback",
            confidence_score=confidence_score,
            stage_results=stage_results,
            total_processing_time_ms=total_time
        )

    async def _keyword_match(self, user_input: str, task_type: str | None) -> KeywordMatchResult:
        """Stage 1: Fast keyword matching (<10ms)"""
        try:
            return await self.keyword_matcher.match(user_input, task_type)
        except Exception as e:
            logger.error(f"Error in keyword matching: {e}")
            return KeywordMatchResult(
                matched=False,
                agent=None,
                confidence=0.0,
                patterns_matched=[],
                processing_time_ms=0.0
            )

    async def _semantic_match(self, user_input: str, task_type: str | None,
                             keyword_result: KeywordMatchResult) -> SemanticMatchResult:
        """Stage 2: Semantic similarity using embeddings"""
        try:
            return await self.semantic_matcher.match(user_input, task_type, keyword_result)
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return SemanticMatchResult(
                matched=False,
                agent=None,
                confidence=0.0,
                similarity_score=0.0,
                processing_time_ms=0.0,
                method="error"
            )

    async def _pe_enhancement_fallback(self, user_input: str,
                                      classification_result: Any | None,
                                      keyword_result: KeywordMatchResult,
                                      semantic_result: SemanticMatchResult) -> PEFallbackResult:
        """Stage 3: Route to PE for clarification"""
        try:
            return await self.pe_fallback.route(
                user_input,
                classification_result,
                keyword_result,
                semantic_result
            )
        except Exception as e:
            logger.error(f"Error in PE fallback: {e}")
            return PEFallbackResult(
                agent='PE',
                confidence=1.0,
                reason="Fallback due to error in delegation",
                original_confidence=0.0,
                processing_time_ms=0.0,
                enhancement_needed=True
            )

    async def execute_agent(self, agent_name: str, user_input: str,
                           context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Execute selected agent through SubAgentManager.

        Args:
            agent_name: Name of agent to execute
            user_input: User's request
            context: Additional context for agent

        Returns:
            Agent execution result
        """
        if not self.sub_agent_manager:
            logger.warning("SubAgentManager not available, returning placeholder result")
            return {
                'agent': agent_name,
                'status': 'unavailable',
                'message': 'SubAgentManager not initialized'
            }

        try:
            # Activate agent if not already active
            if agent_name not in self.sub_agent_manager.active_agents:
                await self.sub_agent_manager.activate_agent(agent_name)

            # Execute agent with isolated context
            result = await self.sub_agent_manager.execute_agent(
                agent_name=agent_name,
                input_data={'prompt': user_input, 'context': context or {}}
            )

            return result
        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")
            return {
                'agent': agent_name,
                'status': 'error',
                'error': str(e)
            }

    def get_metrics(self) -> Dict[str, Any]:
        """Get delegation metrics"""
        metrics = {
            **self.metrics,
            'keyword_percentage': (
                self.metrics['keyword_matches'] / max(self.metrics['total_delegations'], 1) * 100
            ),
            'semantic_percentage': (
                self.metrics['semantic_matches'] / max(self.metrics['total_delegations'], 1) * 100
            ),
            'fallback_percentage': (
                self.metrics['fallback_routes'] / max(self.metrics['total_delegations'], 1) * 100
            ),
        }

        # Add SubAgentManager metrics if available
        if self.sub_agent_manager:
            agent_stats = self.sub_agent_manager.get_all_agents_stats()
            metrics['sub_agent_stats'] = agent_stats

        return metrics

    def get_delegation_header(self, result: DelegationResult) -> str:
        """Generate formatted delegation header for display"""
        # Determine stage icon
        stage_icons = {
            'keyword': '⚡',
            'semantic': '🧠',
            'fallback': '🔄'
        }
        stage_icon = stage_icons.get(result.delegation_method, '❓')

        # Format confidence bar
        confidence = result.confidence_score.overall_score
        confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))

        # Determine performance status
        perf_status = "✅" if result.total_processing_time_ms < 100 else "⚠️"

        header = f"""
🚦 HYBRID DELEGATION ENGINE v1.2
=====================================
{stage_icon} Delegation Method: {result.delegation_method.upper()}
🤖 Selected Agent: {result.selected_agent}
🔢 Confidence: {confidence_bar} {confidence:.2f}/1.0

📊 Confidence Factors:
{self._format_confidence_factors(result.confidence_score.factors)}

⚡ Performance: {result.total_processing_time_ms:.2f}ms {perf_status}
   ├─ Stage 1 (Keyword): {result.stage_results.get('keyword', {}).get('processing_time_ms', 0):.2f}ms
   ├─ Stage 2 (Semantic): {result.stage_results.get('semantic', {}).get('processing_time_ms', 0):.2f}ms
   └─ Stage 3 (Fallback): {result.stage_results.get('fallback', {}).get('processing_time_ms', 0):.2f}ms

💡 {result.confidence_score.recommendation}
=====================================
"""
        return header

    def _format_confidence_factors(self, factors: Dict[str, float]) -> str:
        """Format confidence factors for display"""
        lines = []
        for factor, score in factors.items():
            bar = "▪" * int(score * 5)
            lines.append(f"   ├─ {factor.replace('_', ' ').title()}: {bar} {score:.2f}")
        return "\n".join(lines)
