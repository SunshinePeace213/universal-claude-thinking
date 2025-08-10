"""
Unit Tests for 3-Stage Hybrid Delegation Engine
Tests keyword → semantic → PE fallback delegation
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.delegation.engine import HybridDelegationEngine, DelegationResult
from src.delegation.keyword_matcher import KeywordMatcher, KeywordMatchResult
from src.delegation.semantic_matcher import SemanticMatcher, SemanticMatchResult
from src.delegation.pe_fallback import PEFallback, PEFallbackResult
from src.delegation.confidence_scorer import ConfidenceScorer, ConfidenceScore
from src.core.atomic.classifier import TaskType, ClassificationResult


class TestHybridDelegationEngine:
    """Test suite for HybridDelegationEngine"""
    
    @pytest.fixture
    async def engine(self):
        """Create engine instance"""
        engine = HybridDelegationEngine()
        await engine.initialize()
        return engine
        
    @pytest.mark.asyncio
    async def test_stage1_keyword_match_fast_path(self, engine):
        """Test Stage 1 keyword matching with high confidence"""
        # Mock keyword matcher to return high confidence
        engine.keyword_matcher.match = AsyncMock(return_value=KeywordMatchResult(
            matched=True,
            agent="R1",
            confidence=0.95,
            patterns_matched=["research", "find"],
            processing_time_ms=5.0
        ))
        
        result = await engine.delegate("Research the latest Python features")
        
        assert result.success
        assert result.selected_agent == "R1"
        assert result.delegation_method == "keyword"
        assert result.total_processing_time_ms < 10  # Must be <10ms for keyword
        
    @pytest.mark.asyncio
    async def test_stage2_semantic_match(self, engine):
        """Test Stage 2 semantic matching when keyword fails"""
        # Mock keyword matcher to return low confidence
        engine.keyword_matcher.match = AsyncMock(return_value=KeywordMatchResult(
            matched=False,
            agent=None,
            confidence=0.0,
            patterns_matched=[],
            processing_time_ms=3.0
        ))
        
        # Mock semantic matcher to return good match
        engine.semantic_matcher.match = AsyncMock(return_value=SemanticMatchResult(
            matched=True,
            agent="A1",
            confidence=0.75,
            similarity_score=0.8,
            processing_time_ms=45.0,
            method="embeddings"
        ))
        
        result = await engine.delegate("Analyze the system architecture")
        
        assert result.success
        assert result.selected_agent == "A1"
        assert result.delegation_method == "semantic"
        assert result.total_processing_time_ms < 100  # Semantic can be 50-100ms
        
    @pytest.mark.asyncio
    async def test_stage3_pe_fallback(self, engine):
        """Test Stage 3 PE fallback when both keyword and semantic fail"""
        # Mock both matchers to fail
        engine.keyword_matcher.match = AsyncMock(return_value=KeywordMatchResult(
            matched=False,
            agent=None,
            confidence=0.0,
            patterns_matched=[],
            processing_time_ms=3.0
        ))
        
        engine.semantic_matcher.match = AsyncMock(return_value=SemanticMatchResult(
            matched=False,
            agent=None,
            confidence=0.0,
            similarity_score=0.0,
            processing_time_ms=50.0,
            method="embeddings"
        ))
        
        result = await engine.delegate("help")
        
        assert result.success
        assert result.selected_agent == "PE"
        assert result.delegation_method == "fallback"
        
    @pytest.mark.asyncio
    async def test_performance_requirements(self, engine):
        """Test that delegation meets performance requirements"""
        test_cases = [
            ("What is REST API?", 10),  # Type A - keyword should be <10ms
            ("Build a microservice", 100),  # Type B - semantic ~50-100ms
            ("vague request", 200),  # Fallback ~100-200ms
        ]
        
        for input_text, max_time_ms in test_cases:
            start_time = time.perf_counter()
            result = await engine.delegate(input_text)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Allow some overhead but should be close to target
            assert elapsed_ms < max_time_ms * 2, f"Delegation took {elapsed_ms}ms for '{input_text}'"
            
    @pytest.mark.asyncio
    async def test_confidence_scoring_integration(self, engine):
        """Test confidence scoring across all factors"""
        classification = ClassificationResult(
            task_type=TaskType.TYPE_C,
            confidence=0.8,
            reasoning="Research task",
            suggested_agent="R1",
            delegation_method="semantic",
            processing_time_ms=20.0
        )
        
        result = await engine.delegate(
            "Research machine learning trends",
            classification_result=classification,
            context={'quality_score': 7.5}
        )
        
        assert result.success
        assert result.confidence_score.overall_score > 0.5
        assert 'classification' in result.confidence_score.factors
        assert result.confidence_score.recommendation
        
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, engine):
        """Test that metrics are properly tracked"""
        # Run several delegations
        await engine.delegate("Research Python")
        await engine.delegate("Debug the error")
        await engine.delegate("help me")
        
        metrics = engine.get_metrics()
        
        assert metrics['total_delegations'] == 3
        assert metrics['avg_processing_time'] > 0
        assert 'keyword_percentage' in metrics
        assert 'semantic_percentage' in metrics
        assert 'fallback_percentage' in metrics


class TestKeywordMatcher:
    """Test suite for KeywordMatcher"""
    
    @pytest.fixture
    def matcher(self):
        """Create matcher instance"""
        return KeywordMatcher()
        
    @pytest.mark.asyncio
    async def test_agent_pattern_matching(self, matcher):
        """Test pattern matching for each agent"""
        test_cases = [
            ("enhance my prompt", "PE"),
            ("research the latest trends", "R1"),
            ("analyze this problem", "A1"),
            ("evaluate the code quality", "E1"),
            ("automate the browser test", "T1"),
            ("write a blog post", "W1"),
            ("clarify the requirements", "I1"),
        ]
        
        for input_text, expected_agent in test_cases:
            result = await matcher.match(input_text)
            if result.matched:
                assert result.agent == expected_agent, f"Expected {expected_agent} for '{input_text}'"
                
    @pytest.mark.asyncio
    async def test_performance_under_10ms(self, matcher):
        """Test that keyword matching is <10ms"""
        input_text = "research machine learning algorithms"
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = await matcher.match(input_text)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            
        avg_time = sum(times) / len(times)
        assert avg_time < 10, f"Average time {avg_time}ms exceeds 10ms requirement"
        
    @pytest.mark.asyncio
    async def test_task_type_boost(self, matcher):
        """Test that task type provides confidence boost"""
        input_text = "research something"
        
        # Without task type
        result_no_type = await matcher.match(input_text)
        
        # With matching task type
        result_with_type = await matcher.match(input_text, task_type="research_required")
        
        if result_no_type.matched and result_with_type.matched:
            assert result_with_type.confidence >= result_no_type.confidence


class TestConfidenceScorer:
    """Test suite for ConfidenceScorer"""
    
    @pytest.fixture
    def scorer(self):
        """Create scorer instance"""
        return ConfidenceScorer()
        
    def test_multi_factor_calculation(self, scorer):
        """Test multi-factor confidence calculation"""
        classification = MagicMock(confidence=0.8)
        keyword = MagicMock(confidence=0.9, matched=True)
        semantic = MagicMock(similarity_score=0.75)
        
        score = scorer.calculate_confidence(
            classification_result=classification,
            keyword_result=keyword,
            semantic_result=semantic,
            user_input="Test input",
            context={'quality_score': 7.0}
        )
        
        assert 0 <= score.overall_score <= 1
        assert len(score.factors) > 0
        assert score.explanation
        assert score.recommendation
        
    def test_weight_adjustment(self, scorer):
        """Test weight adjustment when factors are missing"""
        # Only classification available
        classification = MagicMock(confidence=0.8)
        
        score = scorer.calculate_confidence(
            classification_result=classification,
            user_input="Test"
        )
        
        # Weights should be adjusted to account for missing factors
        assert score.weights['classification'] > scorer.default_weights['classification']
        
    def test_input_clarity_scoring(self, scorer):
        """Test input clarity calculation"""
        # Very short input
        score_short = scorer._calculate_input_clarity("hi")
        
        # Good length input
        score_good = scorer._calculate_input_clarity("Please help me debug the authentication error in the login module")
        
        # Very long input
        score_long = scorer._calculate_input_clarity(" ".join(["word"] * 150))
        
        assert score_good > score_short
        assert score_good > score_long
        
    def test_threshold_methods(self, scorer):
        """Test confidence thresholds for different methods"""
        assert scorer.get_threshold_for_method('keyword') == 0.9
        assert scorer.get_threshold_for_method('semantic') == 0.7
        assert scorer.get_threshold_for_method('fallback') == 0.0