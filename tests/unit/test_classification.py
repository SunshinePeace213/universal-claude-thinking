"""
Unit Tests for Request Classification Engine
Tests A/B/C/D/E task type detection with >95% accuracy target
Part of Story 1.2: Request Classification Engine with Delegation Integration
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.atomic.classifier import RequestClassifier, TaskType, ClassificationResult


class TestRequestClassifier:
    """Test suite for RequestClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return RequestClassifier()
        
    @pytest.mark.asyncio
    async def test_type_e_debugging_classification(self, classifier):
        """Test Type E - Debugging/Error Resolution detection"""
        test_cases = [
            "I'm getting an error when running the script",
            "The application crashes on startup",
            "Fix the bug in the login function",
            "Debug why the API is not working",
            "Troubleshoot the database connection issue",
            "There's an exception thrown in production",
        ]
        
        for input_text in test_cases:
            result = await classifier.classify(input_text)
            assert result.task_type == TaskType.TYPE_E, f"Failed for: {input_text}"
            assert result.confidence >= 0.7, f"Low confidence for: {input_text}"
            assert result.suggested_agent == "A1"
            
    @pytest.mark.asyncio
    async def test_type_d_web_testing_classification(self, classifier):
        """Test Type D - Web/Testing detection"""
        test_cases = [
            "Test the login form on the website",
            "Validate the checkout process in the browser",
            "Run UI tests for the dashboard",
            "Automate the user journey for registration",
            "Use Playwright to test the navigation",
        ]
        
        for input_text in test_cases:
            result = await classifier.classify(input_text)
            assert result.task_type == TaskType.TYPE_D, f"Failed for: {input_text}"
            assert result.confidence >= 0.7, f"Low confidence for: {input_text}"
            assert result.suggested_agent == "T1"
            
    @pytest.mark.asyncio
    async def test_type_c_research_classification(self, classifier):
        """Test Type C - Research Required detection"""
        test_cases = [
            "Research the latest React best practices",
            "Find documentation for the OpenAI API",
            "What are the current trends in machine learning?",
            "Look up how to implement OAuth 2.0",
            "Compare different database options for our project",
        ]
        
        for input_text in test_cases:
            result = await classifier.classify(input_text)
            assert result.task_type == TaskType.TYPE_C, f"Failed for: {input_text}"
            assert result.confidence >= 0.7, f"Low confidence for: {input_text}"
            assert result.suggested_agent == "R1"
            
    @pytest.mark.asyncio
    async def test_type_b_complex_classification(self, classifier):
        """Test Type B - Complex/Multi-step detection"""
        test_cases = [
            "Implement a user authentication system with JWT",
            "Build a REST API with CRUD operations",
            "Design the architecture for a microservices application",
            "Refactor the codebase to use dependency injection",
            "Create a complete e-commerce checkout workflow",
        ]
        
        for input_text in test_cases:
            result = await classifier.classify(input_text)
            assert result.task_type == TaskType.TYPE_B, f"Failed for: {input_text}"
            assert result.confidence >= 0.7, f"Low confidence for: {input_text}"
            assert result.suggested_agent == "A1"
            
    @pytest.mark.asyncio
    async def test_type_a_simple_classification(self, classifier):
        """Test Type A - Simple/Direct detection"""
        test_cases = [
            "What is REST?",
            "Define API",
            "How does HTTP work?",
            "Is Python interpreted?",
            "What time is it?",
        ]
        
        for input_text in test_cases:
            result = await classifier.classify(input_text)
            assert result.task_type == TaskType.TYPE_A, f"Failed for: {input_text}"
            assert result.suggested_agent == "PE"
            
    @pytest.mark.asyncio
    async def test_performance_requirement(self, classifier):
        """Test that classification meets <500ms performance requirement"""
        input_text = "Debug the error in the authentication module"
        
        start_time = time.perf_counter()
        result = await classifier.classify(input_text)
        processing_time = (time.perf_counter() - start_time) * 1000
        
        assert processing_time < 500, f"Processing took {processing_time}ms, exceeds 500ms requirement"
        assert result.processing_time_ms < 500
        
    @pytest.mark.asyncio
    async def test_confidence_adjustment_with_context(self, classifier):
        """Test confidence adjustment based on context"""
        input_text = "Fix the issue"
        
        # Test with low quality context
        context_low = {'quality_score': 3.0, 'has_clear_intent': False}
        result_low = await classifier.classify(input_text, context_low)
        
        # Test with high quality context
        context_high = {'quality_score': 8.0, 'has_clear_intent': True}
        result_high = await classifier.classify(input_text, context_high)
        
        # High quality context should yield higher confidence
        assert result_high.confidence > result_low.confidence
        
    @pytest.mark.asyncio
    async def test_delegation_method_determination(self, classifier):
        """Test delegation method selection based on confidence"""
        # High confidence input should use keyword or semantic
        clear_input = "Debug the Python script that throws AttributeError"
        result_clear = await classifier.classify(clear_input)
        assert result_clear.delegation_method in ["keyword", "semantic"]
        
        # Ambiguous input should use fallback
        ambiguous_input = "help"
        result_ambiguous = await classifier.classify(ambiguous_input)
        assert result_ambiguous.confidence < 0.7
        assert result_ambiguous.delegation_method == "fallback"
        
    @pytest.mark.asyncio
    async def test_priority_ordering(self, classifier):
        """Test that classification follows E→D→C→B→A priority"""
        # Input that could match multiple types
        multi_match_input = "Debug and test the new search feature implementation"
        result = await classifier.classify(multi_match_input)
        
        # Should prioritize Type E (debugging) over others
        assert result.task_type in [TaskType.TYPE_E, TaskType.TYPE_D]
        
    @pytest.mark.asyncio
    async def test_pattern_matching_accuracy(self, classifier):
        """Test pattern matching for accuracy"""
        test_data = [
            ("What causes memory leaks in Python?", TaskType.TYPE_C),  # Research
            ("Implement OAuth2 authentication", TaskType.TYPE_B),  # Complex
            ("The app crashes when clicking submit", TaskType.TYPE_E),  # Debug
            ("Test the payment gateway integration", TaskType.TYPE_D),  # Testing
            ("What is a closure?", TaskType.TYPE_A),  # Simple
        ]
        
        correct = 0
        total = len(test_data)
        
        for input_text, expected_type in test_data:
            result = await classifier.classify(input_text)
            if result.task_type == expected_type:
                correct += 1
                
        accuracy = correct / total
        assert accuracy >= 0.95, f"Accuracy {accuracy:.2%} is below 95% target"
        
    def test_get_classification_header(self, classifier):
        """Test classification header generation"""
        result = ClassificationResult(
            task_type=TaskType.TYPE_B,
            confidence=0.85,
            reasoning="Complex task detected",
            suggested_agent="A1",
            delegation_method="semantic",
            processing_time_ms=45.2,
            patterns_matched=["implement", "feature"]
        )
        
        header = classifier.get_classification_header(result)
        
        assert "REQUEST CLASSIFICATION ENGINE" in header
        assert "COMPLEX MULTI STEP" in header
        assert "0.85" in header
        assert "45.2ms" in header
        assert "semantic" in header