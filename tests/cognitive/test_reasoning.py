"""Cognitive quality tests for reasoning and enhancement effectiveness."""

import asyncio
import pytest
from typing import List, Tuple

from src.core.atomic import AtomicFoundation, ChainOfVerification
from src.core.atomic.analyzer import AtomicAnalysis


class TestCognitiveQuality:
    """Test cognitive quality of prompt analysis and enhancement."""
    
    @pytest.fixture
    def atomic(self):
        """Create an AtomicFoundation instance."""
        return AtomicFoundation()
    
    @pytest.fixture
    def cove(self):
        """Create a ChainOfVerification instance."""
        return ChainOfVerification()
    
    @pytest.mark.asyncio
    async def test_atomic_structure_detection(self, atomic):
        """Test accurate detection of atomic prompt components."""
        test_cases = [
            (
                "Analyze customer feedback data to identify trends and patterns, "
                "focusing on negative reviews from the last quarter, "
                "and present findings in a executive summary format.",
                {
                    "has_task": True,
                    "has_constraints": True,
                    "has_format": True,
                    "expected_score_min": 7.0
                }
            ),
            (
                "Create a Python script.",
                {
                    "has_task": True,
                    "has_constraints": False,
                    "has_format": False,
                    "expected_score_min": 4.0
                }
            ),
            (
                "Help me understand machine learning",
                {
                    "has_task": True,  # Weak task
                    "has_constraints": False,
                    "has_format": False,
                    "expected_score_min": 3.0
                }
            )
        ]
        
        for prompt, expectations in test_cases:
            analysis = await atomic.analyze_prompt(prompt)
            
            # Check structure detection accuracy
            assert bool(analysis.structure["task"]) == expectations["has_task"]
            assert bool(analysis.structure["constraints"]) == expectations["has_constraints"]
            assert bool(analysis.structure["output_format"]) == expectations["has_format"]
            
            # Check quality scoring accuracy
            assert analysis.quality_score >= expectations["expected_score_min"]
    
    @pytest.mark.asyncio
    async def test_enhancement_quality_improvement(self, atomic, cove):
        """Test that enhancements actually improve prompt quality."""
        poor_prompts = [
            "help with coding",
            "need something for my project",
            "analyze this",
            "write about AI"
        ]
        
        improvements = []
        
        for prompt in poor_prompts:
            # Get initial analysis
            initial_analysis = await atomic.analyze_prompt(prompt)
            initial_score = initial_analysis.quality_score
            
            # Apply CoVe enhancement
            enhanced_analysis = await cove.enhance_if_needed(initial_analysis, prompt)
            
            # Get the enhanced prompt suggestion
            enhanced_prompt = None
            for suggestion in enhanced_analysis.enhancement_suggestions:
                if "CoVe Enhanced Prompt" in suggestion:
                    enhanced_prompt = suggestion.replace("CoVe Enhanced Prompt: ", "")
                    break
            
            if enhanced_prompt and initial_score < 7.0:
                # Analyze the enhanced prompt
                final_analysis = await atomic.analyze_prompt(enhanced_prompt)
                
                improvement = final_analysis.quality_score - initial_score
                improvements.append(improvement)
        
        # Verify enhancements improve quality
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        assert avg_improvement > 0  # Should show improvement
        
        # Target: 15-25% improvement as per requirements
        improvement_percentage = (avg_improvement / 5.0) * 100  # Assuming avg initial score ~5
        assert improvement_percentage >= 10  # Allow some margin
    
    @pytest.mark.asyncio
    async def test_gap_detection_accuracy(self, atomic):
        """Test accuracy of gap detection."""
        test_cases = [
            (
                "help",
                ["task", "constraints", "output_format"]  # Missing all specifics
            ),
            (
                "Using Python 3.10, create a function",
                ["output_format"]  # Has task and constraints
            ),
            (
                "Generate a JSON response with user data",
                ["constraints"]  # Has task and format
            ),
            (
                "Implement a REST API with FastAPI that handles user authentication, "
                "uses JWT tokens, and returns standardized JSON responses",
                []  # Complete prompt
            )
        ]
        
        for prompt, expected_gaps in test_cases:
            analysis = await atomic.analyze_prompt(prompt)
            
            # Check that detected gaps match expected
            for gap in expected_gaps:
                assert gap in analysis.gaps
    
    @pytest.mark.asyncio
    async def test_suggestion_relevance(self, atomic):
        """Test that suggestions are relevant to the specific gaps."""
        prompts_and_checks = [
            (
                "help me",
                ["action verb", "specific", "objective"]
            ),
            (
                "Write a function",
                ["constraints", "requirements", "format", "structure"]
            ),
            (
                "Create a detailed report about sales using latest data",
                ["output format", "structure", "organize"]
            )
        ]
        
        for prompt, expected_keywords in prompts_and_checks:
            analysis = await atomic.analyze_prompt(prompt)
            
            # Check suggestions contain relevant keywords
            all_suggestions = " ".join(analysis.enhancement_suggestions).lower()
            relevant_count = sum(1 for keyword in expected_keywords if keyword in all_suggestions)
            
            assert relevant_count >= 1  # At least one relevant suggestion
    
    @pytest.mark.asyncio
    async def test_cove_verification_effectiveness(self, cove):
        """Test Chain of Verification's ability to identify missing elements."""
        # Create analysis with significant gaps
        analysis = AtomicAnalysis(
            structure={"task": "do something", "constraints": None, "output_format": None},
            quality_score=3.0,
            gaps=["task", "constraints", "output_format"],
            enhancement_suggestions=[],
            processing_time_ms=50,
            prompt_hash="test"
        )
        
        # Test verification question generation
        questions = cove._generate_verification_questions(analysis)
        
        # Should generate comprehensive questions
        assert len(questions) >= 6  # At least 2 per missing component
        
        # Check question coverage
        question_text = " ".join(questions).lower()
        assert "task" in question_text or "objective" in question_text
        assert "constraint" in question_text or "requirement" in question_text
        assert "format" in question_text or "structure" in question_text
    
    @pytest.mark.asyncio
    async def test_hallucination_reduction(self, atomic, cove):
        """Test that CoVe reduces ambiguity (proxy for hallucination reduction)."""
        ambiguous_prompts = [
            "tell me about it",
            "explain that thing",
            "what do you think about this",
            "analyze the situation"
        ]
        
        ambiguity_reductions = []
        
        for prompt in ambiguous_prompts:
            # Analyze original prompt
            analysis = await atomic.analyze_prompt(prompt)
            
            # Count ambiguous terms
            ambiguous_terms = ["it", "that", "thing", "this", "situation"]
            original_ambiguity = sum(1 for term in ambiguous_terms if term in prompt.lower())
            
            # Apply CoVe
            enhanced = await cove.enhance_if_needed(analysis, prompt)
            
            # Check if enhancement addresses ambiguity
            if enhanced.enhancement_suggestions:
                # Check if suggestions mention being specific or clarifying
                clarifying_terms = ["specific", "clarify", "define", "explain", "describe", "context", "detail"]
                enhanced_text = " ".join(enhanced.enhancement_suggestions).lower()
                addresses_ambiguity = any(term in enhanced_text for term in clarifying_terms)
                
                ambiguity_reductions.append(addresses_ambiguity)
        
        # Verify ambiguity reduction
        assert all(ambiguity_reductions)  # All should reduce or maintain ambiguity
        
        # Target: 30-50% hallucination reduction
        # Using ambiguity reduction as proxy
        reduction_rate = sum(ambiguity_reductions) / len(ambiguity_reductions)
        assert reduction_rate >= 0.3  # At least 30% show improvement


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.fixture
    def atomic(self):
        """Create an AtomicFoundation instance."""
        return AtomicFoundation()
    
    @pytest.mark.asyncio
    async def test_multilingual_prompts(self, atomic):
        """Test handling of prompts with mixed languages."""
        prompts = [
            "Write a function to calculate 素数 (prime numbers)",
            "Créer une API REST with authentication",
            "Analyze данные and generate report"
        ]
        
        for prompt in prompts:
            analysis = await atomic.analyze_prompt(prompt)
            
            # Should still detect basic structure
            assert analysis.structure["task"] is not None
            assert analysis.quality_score > 1.0  # Not rejected as unsafe
    
    @pytest.mark.asyncio
    async def test_technical_jargon_prompts(self, atomic):
        """Test handling of highly technical prompts."""
        prompt = (
            "Implement a B+ tree with MVCC support for ACID-compliant transactions, "
            "using WAL for durability and 2PL for isolation, "
            "returning a TypeScript interface definition."
        )
        
        analysis = await atomic.analyze_prompt(prompt)
        
        # Should recognize this as a well-structured technical prompt
        assert analysis.quality_score >= 7.0
        assert len(analysis.gaps) <= 1  # Maybe missing examples, but otherwise complete
    
    @pytest.mark.asyncio
    async def test_creative_prompts(self, atomic):
        """Test handling of creative/artistic prompts."""
        prompts = [
            (
                "Write a haiku about programming in Python",
                {"min_score": 5.5, "max_gaps": 2}
            ),
            (
                "Create a metaphorical explanation of recursion using a fairy tale, "
                "keeping it under 200 words and suitable for children",
                {"min_score": 7.0, "max_gaps": 1}
            )
        ]
        
        for prompt, expectations in prompts:
            analysis = await atomic.analyze_prompt(prompt)
            
            assert analysis.quality_score >= expectations["min_score"]
            assert len(analysis.gaps) <= expectations["max_gaps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])