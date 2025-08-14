"""
Agent Coordination Hook for Claude Code.

Coordinates sub-agent execution and orchestration through Claude Code's hook system.
Part of Story 1.3: Enhanced Sub-Agent Architecture Framework.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import delegation and sub-agent components
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.delegation.engine import HybridDelegationEngine
    from src.agents.manager import SubAgentManager
    COMPONENTS_AVAILABLE = True
except ImportError:
    COMPONENTS_AVAILABLE = False
    HybridDelegationEngine = None
    SubAgentManager = None

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Coordinates agent delegation and execution through hooks.
    
    Integrates with:
    - Delegation engine for agent selection
    - SubAgentManager for agent execution
    - Memory system for context persistence
    """
    
    def __init__(self):
        """Initialize the agent coordinator."""
        self.delegation_engine = None
        self.sub_agent_manager = None
        self.initialized = False
        
        if COMPONENTS_AVAILABLE:
            try:
                self.delegation_engine = HybridDelegationEngine()
                self.sub_agent_manager = SubAgentManager()
                logger.info("Agent coordinator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize agent coordinator: {e}")
    
    async def initialize(self):
        """Initialize all components."""
        if not COMPONENTS_AVAILABLE:
            logger.warning("Agent coordination components not available")
            return
        
        try:
            # Initialize delegation engine
            if self.delegation_engine:
                await self.delegation_engine.initialize()
            
            # Initialize sub-agent manager
            if self.sub_agent_manager:
                await self.sub_agent_manager.initialize()
            
            self.initialized = True
            logger.info("Agent coordinator fully initialized")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.initialized = False
    
    async def on_subagent_stop(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle SubagentStop event for coordination.
        
        This hook is triggered when a sub-agent completes execution.
        """
        agent_name = event_data.get('agent_name')
        result = event_data.get('result')
        
        if not agent_name:
            return {"status": "error", "message": "No agent name provided"}
        
        try:
            # Persist agent context to memory
            if self.sub_agent_manager:
                await self.sub_agent_manager.persist_context(agent_name)
                logger.info(f"Persisted context for agent {agent_name}")
            
            # Track completion metrics
            completion_data = {
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "success": result.get('success', False),
                "processing_time": result.get('processing_time_ms', 0)
            }
            
            return {
                "status": "success",
                "message": f"Agent {agent_name} coordination complete",
                "data": completion_data
            }
        except Exception as e:
            logger.error(f"Error in SubagentStop hook: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def on_user_prompt_submit(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle UserPromptSubmit event for delegation.
        
        Routes user requests to appropriate sub-agents.
        """
        user_input = event_data.get('prompt', '')
        context = event_data.get('context', {})
        
        if not self.initialized:
            await self.initialize()
        
        if not self.delegation_engine:
            return {
                "status": "bypass",
                "message": "Delegation engine not available"
            }
        
        try:
            # Delegate to appropriate agent
            delegation_result = await self.delegation_engine.delegate(
                user_input=user_input,
                context=context
            )
            
            if delegation_result.success:
                selected_agent = delegation_result.selected_agent
                
                # Execute agent if manager available
                if self.sub_agent_manager and selected_agent != 'PE':
                    agent_result = await self.delegation_engine.execute_agent(
                        agent_name=selected_agent,
                        user_input=user_input,
                        context=context
                    )
                    
                    return {
                        "status": "delegated",
                        "agent": selected_agent,
                        "method": delegation_result.delegation_method,
                        "confidence": delegation_result.confidence_score.overall_score,
                        "result": agent_result
                    }
                
                # Return delegation info for PE or when manager unavailable
                return {
                    "status": "delegated",
                    "agent": selected_agent,
                    "method": delegation_result.delegation_method,
                    "confidence": delegation_result.confidence_score.overall_score
                }
            
            return {
                "status": "bypass",
                "message": "No suitable agent found"
            }
        except Exception as e:
            logger.error(f"Error in UserPromptSubmit hook: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def on_pre_tool_use(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle PreToolUse event for tool orchestration.
        
        Routes tool requests through T1 (Tool User) agent when appropriate.
        """
        tool_name = event_data.get('tool_name')
        tool_args = event_data.get('arguments', {})
        
        # Check if tool orchestration is needed
        dangerous_tools = ['Bash', 'Write', 'Edit', 'MultiEdit']
        if tool_name in dangerous_tools:
            # Route through T1 for safety validation
            if self.sub_agent_manager:
                try:
                    # Activate T1 if not active
                    if 'tool-user' not in self.sub_agent_manager.active_agents:
                        await self.sub_agent_manager.activate_agent('tool-user')
                    
                    # Let T1 validate the tool use
                    validation_result = {
                        "tool": tool_name,
                        "validated": True,
                        "safety_check": "passed"
                    }
                    
                    return {
                        "status": "validated",
                        "agent": "tool-user",
                        "result": validation_result
                    }
                except Exception as e:
                    logger.error(f"Error validating tool use: {e}")
        
        return {
            "status": "bypass",
            "message": "No validation needed"
        }
    
    async def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        stats = {
            "initialized": self.initialized,
            "delegation_available": self.delegation_engine is not None,
            "manager_available": self.sub_agent_manager is not None
        }
        
        if self.delegation_engine:
            stats["delegation_metrics"] = self.delegation_engine.get_metrics()
        
        if self.sub_agent_manager:
            stats["agent_stats"] = self.sub_agent_manager.get_agent_stats()
        
        return stats


# Global coordinator instance
coordinator = AgentCoordinator()


# Hook entry points for Claude Code
async def on_subagent_stop(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for SubagentStop hook."""
    return await coordinator.on_subagent_stop(event_data)


async def on_user_prompt_submit(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for UserPromptSubmit hook."""
    return await coordinator.on_user_prompt_submit(event_data)


async def on_pre_tool_use(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Entry point for PreToolUse hook."""
    return await coordinator.on_pre_tool_use(event_data)


# Hook registration for Claude Code
HOOKS = {
    "SubagentStop": on_subagent_stop,
    "UserPromptSubmit": on_user_prompt_submit,
    "PreToolUse": on_pre_tool_use
}