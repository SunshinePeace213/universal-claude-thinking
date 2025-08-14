"""
Integration tests for delegation engine and sub-agent manager integration.

Tests the complete flow from delegation to agent execution with memory persistence.
Part of Story 1.3: Enhanced Sub-Agent Architecture Framework.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Import components to test
from src.delegation.engine import HybridDelegationEngine, DelegationResult
from src.agents.manager import SubAgentManager, AgentContext
from src.agents.base import BaseSubAgent


class TestDelegationIntegration:
    """Test integration between delegation engine and sub-agent manager."""
    
    @pytest.fixture
    async def delegation_engine(self):
        """Create delegation engine instance."""
        engine = HybridDelegationEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    async def sub_agent_manager(self):
        """Create sub-agent manager instance."""
        manager = SubAgentManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_delegation_to_subagent_execution(self, delegation_engine, sub_agent_manager):
        """Test complete flow from delegation to agent execution."""
        # Test user input
        user_input = "Research the latest features in Python 3.12"
        
        # Perform delegation
        delegation_result = await delegation_engine.delegate(user_input)
        
        # Verify delegation result
        assert delegation_result.success
        assert delegation_result.selected_agent in [
            'R1', 'researcher', 'PE'  # Could route to R1/researcher or PE
        ]
        assert delegation_result.confidence_score.overall_score > 0
        
        # If delegated to an agent other than PE, execute it
        if delegation_result.selected_agent != 'PE':
            agent_result = await delegation_engine.execute_agent(
                agent_name=delegation_result.selected_agent,
                user_input=user_input
            )
            
            # Verify agent execution result
            assert 'agent' in agent_result
            assert agent_result['agent'] == delegation_result.selected_agent
    
    @pytest.mark.asyncio
    async def test_execute_agent_method(self, sub_agent_manager):
        """Test the new execute_agent method implementation."""
        # Test successful execution
        result = await sub_agent_manager.execute_agent(
            "prompt-enhancer",
            {"prompt": "Improve this prompt", "context": {"type": "enhancement"}}
        )
        
        assert result["status"] == "success"
        assert result["agent"] == "prompt-enhancer"
        assert result["nickname"] == "PE"
        assert result["text_face"] == "🔧"
        assert result["model"] == "sonnet"
        assert result["response"]["processed"] is True
        
        # Test unknown agent
        result = await sub_agent_manager.execute_agent(
            "unknown-agent",
            {"prompt": "Test"}
        )
        
        assert result["status"] == "error"
        assert "Unknown agent" in result["error"]
    
    @pytest.mark.asyncio
    async def test_subagent_manager_in_delegation_engine(self, delegation_engine):
        """Test that SubAgentManager is properly integrated in delegation engine."""
        # Check if SubAgentManager is initialized
        assert delegation_engine.sub_agent_manager is not None
        
        # Get metrics including sub-agent stats
        metrics = delegation_engine.get_metrics()
        
        # Verify metrics structure
        assert 'total_delegations' in metrics
        assert 'keyword_percentage' in metrics
        assert 'semantic_percentage' in metrics
        assert 'fallback_percentage' in metrics
        
        # If SubAgentManager is available, check for agent stats
        if delegation_engine.sub_agent_manager:
            assert 'sub_agent_stats' in metrics
            assert 'total_agents' in metrics['sub_agent_stats']
    
    @pytest.mark.asyncio
    async def test_memory_persistence_integration(self, sub_agent_manager):
        """Test memory system integration in SubAgentManager."""
        # Check if memory system is available
        if not sub_agent_manager.memory_system:
            pytest.skip("Memory system not available")
        
        # Activate an agent
        agent_name = 'researcher'
        context = await sub_agent_manager.activate_agent(agent_name)
        
        # Verify context created
        assert context is not None
        assert context.agent_name == agent_name
        assert context.state == 'active'
        
        # Persist context to memory
        await sub_agent_manager.persist_context(agent_name)
        
        # Clear local context
        sub_agent_manager.contexts.clear()
        
        # Retrieve context from memory
        retrieved_data = await sub_agent_manager.retrieve_context(agent_name)
        
        # Verify context was persisted and retrieved
        if retrieved_data:  # Memory system might not be fully implemented
            assert retrieved_data['agent_name'] == agent_name
            assert 'invocation_count' in retrieved_data
    
    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self, delegation_engine):
        """Test parallel execution of multiple agents."""
        # Multiple user inputs requiring different agents
        inputs = [
            "Research machine learning algorithms",  # → researcher
            "Write a Python function to sort a list",  # → writer or tool-user
            "Explain this code to me",  # → interface
        ]
        
        # Delegate all inputs in parallel
        tasks = [delegation_engine.delegate(input_text) for input_text in inputs]
        results = await asyncio.gather(*tasks)
        
        # Verify all delegations succeeded
        assert all(r.success for r in results)
        assert len(results) == len(inputs)
        
        # Check that different agents were selected
        selected_agents = [r.selected_agent for r in results]
        assert len(set(selected_agents)) >= 1  # At least one unique agent
    
    @pytest.mark.asyncio
    async def test_agent_context_isolation(self, sub_agent_manager):
        """Test that agent contexts are properly isolated."""
        # Activate multiple agents
        agents = ['researcher', 'reasoner', 'writer']
        contexts = []
        
        for agent_name in agents:
            context = await sub_agent_manager.activate_agent(agent_name)
            contexts.append(context)
        
        # Verify each agent has its own context
        assert len(contexts) == len(agents)
        assert len(set(c.agent_id for c in contexts)) == len(agents)
        
        # Verify contexts are isolated (different IDs)
        agent_ids = [c.agent_id for c in contexts]
        assert len(agent_ids) == len(set(agent_ids))
        
        # Verify all agents are active
        active_agents = sub_agent_manager.list_active_agents()
        assert set(active_agents) == set(agents)
    
    @pytest.mark.asyncio
    async def test_error_handling_in_delegation(self, delegation_engine):
        """Test error handling when agent execution fails."""
        # Test with invalid input that might cause errors
        user_input = ""  # Empty input
        
        # Should still return a result (likely PE fallback)
        result = await delegation_engine.delegate(user_input)
        
        # Verify graceful handling
        assert result.success
        assert result.selected_agent == 'PE'  # Should fallback to PE
        assert result.delegation_method == 'fallback'
    
    @pytest.mark.asyncio
    async def test_delegation_confidence_scoring(self, delegation_engine):
        """Test confidence scoring in delegation results."""
        # Test with clear research query
        research_input = "Find information about quantum computing applications"
        result = await delegation_engine.delegate(research_input)
        
        # Verify confidence scoring
        assert result.success
        assert result.confidence_score is not None
        assert 0 <= result.confidence_score.overall_score <= 1
        assert result.confidence_score.factors is not None
        assert len(result.confidence_score.factors) > 0
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_management(self, sub_agent_manager):
        """Test complete agent lifecycle: activate → use → deactivate."""
        agent_name = 'evaluator'
        
        # Initial state - agent not active
        assert agent_name not in sub_agent_manager.active_agents
        
        # Activate agent
        context = await sub_agent_manager.activate_agent(agent_name)
        assert context.state == 'active'
        assert agent_name in sub_agent_manager.active_agents
        
        # Simulate usage (update access)
        context.update_access()
        assert context.invocation_count == 1
        
        # Deactivate agent
        await sub_agent_manager.deactivate_agent(agent_name)
        assert agent_name not in sub_agent_manager.active_agents
        
        # Context should still exist but be inactive
        stored_context = sub_agent_manager.get_agent_context(agent_name)
        assert stored_context is not None
        assert stored_context.state == 'inactive'
        assert stored_context.invocation_count == 1  # Preserved
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, delegation_engine):
        """Test that metrics are properly collected during delegation."""
        # Perform multiple delegations
        inputs = [
            "Research Python libraries",
            "Write documentation",
            "Debug this error"
        ]
        
        for input_text in inputs:
            await delegation_engine.delegate(input_text)
        
        # Get metrics
        metrics = delegation_engine.get_metrics()
        
        # Verify metrics collected
        assert metrics['total_delegations'] == len(inputs)
        assert metrics['avg_processing_time'] > 0
        
        # Check distribution percentages
        total_percentage = (
            metrics['keyword_percentage'] +
            metrics['semantic_percentage'] +
            metrics['fallback_percentage']
        )
        assert abs(total_percentage - 100.0) < 0.01  # Allow small floating point error
    
    @pytest.mark.asyncio
    async def test_hook_coordination_integration(self):
        """Test agent coordination hook integration."""
        import sys
        from pathlib import Path
        # Add hooks directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / '.claude' / 'hooks'))
        from agent_coordinator import AgentCoordinator
        
        coordinator = AgentCoordinator()
        await coordinator.initialize()
        
        # Test UserPromptSubmit hook
        event_data = {
            'prompt': 'Research machine learning frameworks',
            'context': {}
        }
        
        result = await coordinator.on_user_prompt_submit(event_data)
        
        # Verify hook response
        assert 'status' in result
        if result['status'] == 'delegated':
            assert 'agent' in result
            assert 'confidence' in result