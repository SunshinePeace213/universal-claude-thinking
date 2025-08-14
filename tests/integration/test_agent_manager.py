"""
Integration tests for SubAgentManager.

Tests the complete lifecycle, context isolation, and coordination
of sub-agents in realistic scenarios.
"""

import asyncio
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.agents.manager import SubAgentManager, AgentContext, AgentConfiguration
from src.agents.base import BaseSubAgent, AgentMessage, MessageType, MessagePriority


class MockSubAgent(BaseSubAgent):
    """Mock implementation of a sub-agent for testing."""
    
    async def process_request(self, message: AgentMessage) -> AgentMessage:
        """Process test requests."""
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            type=MessageType.RESPONSE,
            content={"result": f"Processed by {self.nickname}"},
            correlation_id=message.id
        )
    
    async def validate_input(self, content: dict) -> bool:
        """Validate test input."""
        return True


@pytest.mark.asyncio
class TestSubAgentManagerIntegration:
    """Integration tests for SubAgentManager."""
    
    async def test_manager_initialization(self):
        """Test manager initialization and configuration loading."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Check that all 7 agents are loaded
        assert len(manager.configurations) == 7
        assert "prompt-enhancer" in manager.configurations
        assert "researcher" in manager.configurations
        assert "reasoner" in manager.configurations
        assert "evaluator" in manager.configurations
        assert "tool-user" in manager.configurations
        assert "writer" in manager.configurations
        assert "interface" in manager.configurations
    
    async def test_context_isolation(self):
        """Test that contexts are properly isolated between agents."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Create contexts for two different agents
        context1 = await manager.activate_agent("prompt-enhancer")
        context2 = await manager.activate_agent("researcher")
        
        # Verify contexts are separate
        assert context1.agent_name == "prompt-enhancer"
        assert context2.agent_name == "researcher"
        assert context1.agent_id != context2.agent_id
        assert id(context1) != id(context2)
        
        # Verify metadata is correct
        assert context1.metadata["nickname"] == "PE"
        assert context2.metadata["nickname"] == "R1"
    
    async def test_agent_lifecycle(self):
        """Test agent activation, deactivation, and suspension."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Activate agent
        context = await manager.activate_agent("evaluator")
        assert context.state == "active"
        assert "evaluator" in manager.active_agents
        
        # Deactivate agent
        await manager.deactivate_agent("evaluator")
        context = manager.get_agent_context("evaluator")
        assert context.state == "inactive"
        assert "evaluator" not in manager.active_agents
        
        # Suspend agent
        await manager.suspend_agent("evaluator", "Test suspension")
        context = manager.get_agent_context("evaluator")
        assert context.state == "suspended"
        assert context.metadata["suspension_reason"] == "Test suspension"
    
    async def test_context_reuse(self):
        """Test that contexts are reused when agents are reactivated."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # First activation
        context1 = await manager.activate_agent("writer")
        initial_id = context1.agent_id
        initial_invocation_count = context1.invocation_count
        
        # Deactivate
        await manager.deactivate_agent("writer")
        
        # Reactivate - should reuse context
        context2 = await manager.activate_agent("writer")
        assert context2.agent_id == initial_id
        assert context2.invocation_count == initial_invocation_count + 1
    
    async def test_health_check(self):
        """Test health check functionality."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Initial health check - should be healthy
        health = await manager.health_check()
        assert health["status"] == "healthy"
        assert health["total_agents"] == 7
        assert health["active_agents"] == 0
        
        # Activate some agents
        await manager.activate_agent("interface")
        await manager.activate_agent("tool-user")
        
        health = await manager.health_check()
        assert health["active_agents"] == 2
        
        # Suspend an agent - should be degraded
        await manager.suspend_agent("tool-user", "Test issue")
        health = await manager.health_check()
        assert len(health["issues"]) > 0
        assert any(issue["type"] == "suspended_agents" for issue in health["issues"])
    
    async def test_agent_stats(self):
        """Test agent statistics collection."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Activate and get stats
        await manager.activate_agent("reasoner")
        stats = await manager.get_agent_stats("reasoner")
        
        assert stats["name"] == "reasoner"
        assert stats["nickname"] == "A1"
        assert stats["model"] == "opus"
        assert stats["state"] == "active"
        assert stats["invocations"] >= 0  # Starts at 0, increments on update_access
        assert isinstance(stats["created_at"], str)
        assert isinstance(stats["last_accessed"], str)
    
    async def test_context_cleanup(self):
        """Test cleanup of inactive contexts."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Create contexts
        await manager.activate_agent("prompt-enhancer")
        await manager.activate_agent("evaluator")
        
        # Deactivate one
        await manager.deactivate_agent("prompt-enhancer")
        
        # Manually set last_accessed to old time
        context = manager.get_agent_context("prompt-enhancer")
        context.last_accessed = datetime.now() - timedelta(hours=25)
        
        # Run cleanup
        await manager.cleanup_inactive_contexts(max_age_hours=24)
        
        # Check that old context was removed
        assert manager.get_agent_context("prompt-enhancer") is None
        assert manager.get_agent_context("evaluator") is not None
    
    async def test_concurrent_activations(self):
        """Test concurrent agent activations."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Activate multiple agents concurrently
        agents = ["prompt-enhancer", "researcher", "reasoner", "evaluator"]
        tasks = [manager.activate_agent(agent) for agent in agents]
        contexts = await asyncio.gather(*tasks)
        
        # Verify all contexts were created
        assert len(contexts) == 4
        assert all(ctx.state == "active" for ctx in contexts)
        assert len(manager.active_agents) == 4
    
    async def test_model_distribution(self):
        """Test that agents use appropriate models."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        sonnet_agents = ["prompt-enhancer", "evaluator", "interface"]
        opus_agents = ["researcher", "reasoner", "tool-user", "writer"]
        
        for agent_name in sonnet_agents:
            config = manager.get_agent_config(agent_name)
            assert config.model == "sonnet", f"{agent_name} should use sonnet"
        
        for agent_name in opus_agents:
            config = manager.get_agent_config(agent_name)
            assert config.model == "opus", f"{agent_name} should use opus"
    
    async def test_configuration_parsing(self):
        """Test that configurations are properly parsed from markdown."""
        manager = SubAgentManager(Path(".claude/agents"))
        await manager.initialize()
        
        # Check specific agent configuration
        config = manager.get_agent_config("tool-user")
        assert config.name == "tool-user"
        assert config.nickname == "T1"
        assert config.text_face == "🛠️"
        assert isinstance(config.description, str)
        assert len(config.description) > 0
        assert isinstance(config.tools, list)
        assert config.model == "opus"
        assert isinstance(config.content, str)
        assert len(config.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])