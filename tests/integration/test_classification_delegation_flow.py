"""
Integration Tests for Classification → Delegation Flow
Tests the complete pipeline from user input to agent selection
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import pytest
import asyncio
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.atomic.classifier import RequestClassifier, TaskType
from src.delegation.engine import HybridDelegationEngine
from src.core.storage.delegation_repository import DelegationRepository


class TestClassificationDelegationFlow:
    """Integration tests for the complete classification and delegation flow"""
    
    @pytest.fixture
    async def system(self, tmp_path):
        """Initialize the complete system"""
        classifier = RequestClassifier()
        engine = HybridDelegationEngine()
        # Use a file-based database for tests to ensure persistence across connections
        db_path = tmp_path / "test_metrics.db"
        repository = DelegationRepository(str(db_path))
        
        await engine.initialize()
        await repository.initialize()
        
        return {
            'classifier': classifier,
            'engine': engine,
            'repository': repository
        }
        
    @pytest.mark.asyncio
    async def test_complete_flow_type_e_debugging(self, system):
        """Test complete flow for Type E debugging request"""
        user_input = "Debug the authentication error in the login module"
        
        # Step 1: Classification
        classification_result = await system['classifier'].classify(user_input)
        assert classification_result.task_type == TaskType.TYPE_E
        assert classification_result.confidence > 0.7
        
        # Step 2: Delegation
        delegation_result = await system['engine'].delegate(
            user_input,
            classification_result=classification_result
        )
        assert delegation_result.success
        assert delegation_result.selected_agent == "A1"  # Reasoner for debugging
        
        # Step 3: Save metrics
        keyword_result = delegation_result.stage_results.get('keyword')
        semantic_result = delegation_result.stage_results.get('semantic')
        fallback_result = delegation_result.stage_results.get('fallback')
        
        await system['repository'].save_delegation_metrics(
            request_id="test_001",
            classification_type=classification_result.task_type.value,
            confidence_score=delegation_result.confidence_score.overall_score,
            delegation_method=delegation_result.delegation_method,
            selected_agent=delegation_result.selected_agent,
            stage_latencies={
                'keyword': keyword_result.processing_time_ms if keyword_result else 0,
                'semantic': semantic_result.processing_time_ms if semantic_result else 0,
                'fallback': fallback_result.processing_time_ms if fallback_result else 0,
            },
            total_latency=delegation_result.total_processing_time_ms
        )
        
        # Verify metrics were saved
        metrics = await system['repository'].get_delegation_distribution(days_back=1)
        assert len(metrics) > 0
        
    @pytest.mark.asyncio
    async def test_complete_flow_type_c_research(self, system):
        """Test complete flow for Type C research request"""
        user_input = "Research the latest best practices for React performance optimization"
        
        # Step 1: Classification
        classification_result = await system['classifier'].classify(user_input)
        assert classification_result.task_type == TaskType.TYPE_C
        
        # Step 2: Delegation
        delegation_result = await system['engine'].delegate(
            user_input,
            classification_result=classification_result,
            context={'quality_score': 8.0}
        )
        assert delegation_result.success
        assert delegation_result.selected_agent == "R1"  # Researcher
        
        # Verify confidence factors
        assert delegation_result.confidence_score.factors['classification'] > 0
        
    @pytest.mark.asyncio
    async def test_complete_flow_ambiguous_to_pe(self, system):
        """Test that ambiguous requests route to PE"""
        user_input = "help me"
        
        # Step 1: Classification (should be low confidence)
        classification_result = await system['classifier'].classify(user_input)
        assert classification_result.confidence < 0.7
        
        # Step 2: Delegation (should go to PE fallback)
        delegation_result = await system['engine'].delegate(
            user_input,
            classification_result=classification_result
        )
        assert delegation_result.success
        assert delegation_result.selected_agent == "PE"
        assert delegation_result.delegation_method == "fallback"
        
    @pytest.mark.asyncio
    async def test_performance_across_types(self, system):
        """Test performance requirements across all task types"""
        test_cases = [
            ("What is an API?", TaskType.TYPE_A, 50),
            ("Build a REST API with authentication", TaskType.TYPE_B, 100),
            ("Find the latest Python documentation", TaskType.TYPE_C, 100),
            ("Test the login form", TaskType.TYPE_D, 100),
            ("Fix the database connection error", TaskType.TYPE_E, 100),
        ]
        
        for user_input, expected_type, max_time_ms in test_cases:
            start_time = time.perf_counter()
            
            # Classification
            classification_result = await system['classifier'].classify(user_input)
            
            # Delegation
            delegation_result = await system['engine'].delegate(
                user_input,
                classification_result=classification_result
            )
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            assert classification_result.task_type == expected_type
            assert delegation_result.success
            assert total_time < max_time_ms * 3, f"Total time {total_time}ms exceeds limit for '{user_input}'"
            
    @pytest.mark.asyncio
    async def test_accuracy_tracking(self, system):
        """Test classification accuracy tracking"""
        test_data = [
            ("Debug the error", TaskType.TYPE_E, TaskType.TYPE_E),
            ("Research Python", TaskType.TYPE_C, TaskType.TYPE_C),
            ("What is HTTP?", TaskType.TYPE_A, TaskType.TYPE_A),
            ("Build a feature", TaskType.TYPE_B, TaskType.TYPE_B),
            ("Test the UI", TaskType.TYPE_D, TaskType.TYPE_D),
        ]
        
        for prompt, predicted, actual in test_data:
            classification_result = await system['classifier'].classify(prompt)
            
            # Save classification history
            await system['repository'].save_classification_history(
                prompt_text=prompt,
                predicted_type=classification_result.task_type.value,
                confidence=classification_result.confidence,
                patterns_matched=classification_result.patterns_matched,
                processing_time_ms=classification_result.processing_time_ms,
                actual_type=actual.value
            )
            
        # Check accuracy
        accuracy = await system['repository'].get_classification_accuracy()
        assert accuracy['accuracy_percentage'] >= 95.0, f"Accuracy {accuracy['accuracy_percentage']:.1f}% below 95% target"
        
    @pytest.mark.asyncio
    async def test_delegation_method_distribution(self, system):
        """Test distribution of delegation methods"""
        # Run multiple delegations with different confidence levels
        test_inputs = [
            "Research machine learning trends",  # Should use semantic
            "enhance my prompt please",  # Should use keyword
            "something vague",  # Should use fallback
            "debug the Python script error",  # Should use keyword/semantic
            "test the web application",  # Should use semantic
        ]
        
        for input_text in test_inputs:
            classification = await system['classifier'].classify(input_text)
            delegation = await system['engine'].delegate(input_text, classification)
            
            await system['repository'].save_delegation_metrics(
                request_id=f"test_{input_text[:10]}",
                classification_type=classification.task_type.value,
                confidence_score=delegation.confidence_score.overall_score,
                delegation_method=delegation.delegation_method,
                selected_agent=delegation.selected_agent,
                stage_latencies={
                    'keyword': 5,
                    'semantic': 50,
                    'fallback': 100,
                },
                total_latency=delegation.total_processing_time_ms
            )
            
        # Check distribution
        distribution = await system['repository'].get_delegation_distribution(days_back=1)
        
        # Should have at least 2 different methods used
        methods_used = [d['method'] for d in distribution]
        assert len(set(methods_used)) >= 2, "Should use multiple delegation methods"
        
    @pytest.mark.asyncio
    async def test_confidence_factor_tracking(self, system):
        """Test that confidence factors are properly tracked"""
        user_input = "Implement OAuth authentication with JWT tokens"
        
        classification = await system['classifier'].classify(user_input)
        delegation = await system['engine'].delegate(
            user_input,
            classification_result=classification,
            context={'quality_score': 7.5, 'has_clear_intent': True}
        )
        
        # Save confidence factors
        await system['repository'].save_confidence_factors(
            request_id="test_factors",
            factors=delegation.confidence_score.factors,
            overall_confidence=delegation.confidence_score.overall_score
        )
        
        # Verify all factors are present
        assert 'classification' in delegation.confidence_score.factors
        assert 'keyword_match' in delegation.confidence_score.factors
        assert 'semantic_similarity' in delegation.confidence_score.factors
        assert 'context_quality' in delegation.confidence_score.factors
        assert 'input_clarity' in delegation.confidence_score.factors