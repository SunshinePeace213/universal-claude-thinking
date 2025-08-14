"""
Sub-Agent Manager Infrastructure.

Manages the lifecycle, context isolation, and orchestration of specialized
sub-agents within the Claude Code environment.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from ..utils.metrics import MetricsCollector

# Import memory system for context persistence
try:
    from ..memory.layers.ltm import LongTermMemory
    from ..memory.layers.stm import ShortTermMemory
    from ..memory.layers.wm import WorkingMemory
    MEMORY_SYSTEM_AVAILABLE = True
except ImportError:
    MEMORY_SYSTEM_AVAILABLE = False
    ShortTermMemory = None
    WorkingMemory = None
    LongTermMemory = None

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Represents an isolated context window for a sub-agent."""

    agent_id: str
    agent_name: str
    created_at: datetime
    last_accessed: datetime
    token_count: int = 0
    invocation_count: int = 0
    state: str = "inactive"  # inactive, active, suspended, error
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_access(self) -> None:
        """Update last accessed timestamp."""
        self.last_accessed = datetime.now()
        self.invocation_count += 1


@dataclass
class AgentConfiguration:
    """Parsed configuration for a sub-agent."""

    name: str
    nickname: str
    text_face: str
    description: str
    tools: List[str]
    model: str
    content: str
    file_path: Path

    @classmethod
    def from_markdown(cls, file_path: Path) -> "AgentConfiguration":
        """Parse agent configuration from markdown file."""
        with open(file_path) as f:
            content = f.read()

        # Extract frontmatter
        pattern = r'^---\n(.*?)\n---\n(.*)$'
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            raise ValueError(f"Invalid agent configuration format in {file_path}")

        frontmatter_str = match.group(1)
        body_content = match.group(2)

        frontmatter = yaml.safe_load(frontmatter_str)

        return cls(
            name=frontmatter['name'],
            nickname=frontmatter['nickname'],
            text_face=frontmatter['text_face'],
            description=frontmatter['description'],
            tools=frontmatter.get('tools', []),
            model=frontmatter['model'],
            content=body_content.strip(),
            file_path=file_path
        )


class SubAgentManager:
    """
    Manages sub-agent lifecycle and orchestration.

    Responsibilities:
    - Load and validate agent configurations
    - Manage context isolation
    - Coordinate agent activation/deactivation
    - Track agent performance metrics
    - Handle error boundaries
    """

    def __init__(self, agents_dir: Path = Path(".claude/agents")):
        """Initialize the SubAgentManager."""
        self.agents_dir = agents_dir
        self.configurations: Dict[str, AgentConfiguration] = {}
        self.contexts: Dict[str, AgentContext] = {}
        self.active_agents: Set[str] = set()
        self.metrics = MetricsCollector()
        self._lock = asyncio.Lock()

        # Initialize memory layers if available
        self.memory_system = None
        if MEMORY_SYSTEM_AVAILABLE:
            try:
                self.memory_system = {
                    'stm': ShortTermMemory(),  # 2-hour window
                    'wm': WorkingMemory(),      # 7-day window
                    'ltm': LongTermMemory()     # Persistent
                }
                logger.info("Memory system initialized for context persistence")
            except Exception as e:
                logger.warning(f"Failed to initialize memory system: {e}")
                self.memory_system = None

    async def initialize(self) -> None:
        """Initialize the manager and load agent configurations."""
        logger.info("Initializing SubAgentManager")
        await self.load_configurations()
        logger.info(f"Loaded {len(self.configurations)} agent configurations")

    async def load_configurations(self) -> None:
        """Load all agent configurations from the agents directory."""
        if not self.agents_dir.exists():
            logger.warning(f"Agents directory {self.agents_dir} does not exist")
            return

        for file_path in self.agents_dir.glob("*.md"):
            try:
                config = AgentConfiguration.from_markdown(file_path)
                self.configurations[config.name] = config
                logger.debug(f"Loaded configuration for agent: {config.name}")
            except Exception as e:
                logger.error(f"Failed to load agent configuration from {file_path}: {e}")

    async def create_context(self, agent_name: str) -> AgentContext:
        """
        Create an isolated context for an agent.

        This ensures context pollution prevention through strict boundaries.
        """
        async with self._lock:
            if agent_name not in self.configurations:
                raise ValueError(f"Unknown agent: {agent_name}")

            if agent_name in self.contexts:
                # Return existing context
                context = self.contexts[agent_name]
                context.update_access()
                return context

            # Create new context
            config = self.configurations[agent_name]
            context = AgentContext(
                agent_id=f"{agent_name}_{datetime.now().timestamp()}",
                agent_name=agent_name,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                metadata={
                    "nickname": config.nickname,
                    "text_face": config.text_face,
                    "model": config.model,
                    "tools": config.tools
                }
            )

            self.contexts[agent_name] = context
            logger.info(f"Created context for agent: {agent_name}")

            # Track metrics
            await self.metrics.track_metric(
                "agent_context_created",
                {"agent": agent_name, "model": config.model}
            )

            return context

    async def activate_agent(self, agent_name: str) -> AgentContext:
        """
        Activate a sub-agent by creating/retrieving its context.

        This is called when Claude Code's /agents command delegates to a specialist.
        """
        logger.info(f"Activating agent: {agent_name}")

        context = await self.create_context(agent_name)
        context.state = "active"
        self.active_agents.add(agent_name)

        # Track activation metrics
        await self.metrics.track_metric(
            "agent_activated",
            {"agent": agent_name, "timestamp": datetime.now().isoformat()}
        )

        return context

    async def deactivate_agent(self, agent_name: str) -> None:
        """
        Deactivate a sub-agent while preserving its context.

        The context remains available for future activations.
        """
        async with self._lock:
            if agent_name not in self.contexts:
                logger.warning(f"Attempted to deactivate unknown agent: {agent_name}")
                return

            context = self.contexts[agent_name]
            context.state = "inactive"
            self.active_agents.discard(agent_name)

            logger.info(f"Deactivated agent: {agent_name}")

            # Track deactivation metrics
            await self.metrics.track_metric(
                "agent_deactivated",
                {"agent": agent_name, "timestamp": datetime.now().isoformat()}
            )

    async def suspend_agent(self, agent_name: str, reason: str = "") -> None:
        """
        Suspend an agent due to error or resource constraints.

        Suspended agents cannot be activated until resumed.
        """
        async with self._lock:
            if agent_name not in self.contexts:
                return

            context = self.contexts[agent_name]
            context.state = "suspended"
            context.metadata["suspension_reason"] = reason
            self.active_agents.discard(agent_name)

            logger.warning(f"Suspended agent {agent_name}: {reason}")

            # Track suspension
            await self.metrics.track_metric(
                "agent_suspended",
                {"agent": agent_name, "reason": reason}
            )

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfiguration]:
        """Get configuration for a specific agent."""
        return self.configurations.get(agent_name)

    def get_agent_context(self, agent_name: str) -> Optional[AgentContext]:
        """Get context for a specific agent."""
        return self.contexts.get(agent_name)

    def list_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.configurations.keys())

    def list_active_agents(self) -> List[str]:
        """List currently active agents."""
        return list(self.active_agents)

    async def persist_context(self, agent_name: str) -> None:
        """
        Persist agent context to memory system.

        Stores context in appropriate memory layer based on recency.
        """
        if not self.memory_system:
            return

        context = self.contexts.get(agent_name)
        if not context:
            return

        # Prepare context data for storage
        context_data = {
            'agent_name': agent_name,
            'agent_id': context.agent_id,
            'state': context.state,
            'invocation_count': context.invocation_count,
            'token_count': context.token_count,
            'metadata': context.metadata,
            'created_at': context.created_at.isoformat(),
            'last_accessed': context.last_accessed.isoformat()
        }

        try:
            # Store in short-term memory (always)
            await self.memory_system['stm'].store(
                key=f"agent_context_{agent_name}",
                data=context_data,
                metadata={'type': 'agent_context', 'agent': agent_name}
            )

            # Optimized thresholds based on usage patterns
            # If moderately used (>3), also store in working memory
            if context.invocation_count >= 3:
                await self.memory_system['wm'].store(
                    key=f"agent_context_{agent_name}",
                    data=context_data,
                    metadata={'type': 'agent_context', 'agent': agent_name}
                )

            # If frequently used (>10), persist to long-term memory
            if context.invocation_count >= 10:
                await self.memory_system['ltm'].store(
                    key=f"agent_context_{agent_name}",
                    data=context_data,
                    metadata={'type': 'agent_context', 'agent': agent_name}
                )

            logger.debug(f"Persisted context for agent {agent_name}")
        except Exception as e:
            logger.error(f"Failed to persist context for {agent_name}: {e}")

    async def retrieve_context(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent context from memory system.

        Checks memory layers in order: STM → WM → LTM.
        """
        if not self.memory_system:
            return None

        try:
            # Check short-term memory first
            context_data = await self.memory_system['stm'].retrieve(
                key=f"agent_context_{agent_name}"
            )

            # If not in STM, check working memory
            if not context_data:
                context_data = await self.memory_system['wm'].retrieve(
                    key=f"agent_context_{agent_name}"
                )

            # If not in WM, check long-term memory
            if not context_data:
                context_data = await self.memory_system['ltm'].retrieve(
                    key=f"agent_context_{agent_name}"
                )

            if context_data:
                logger.debug(f"Retrieved context for agent {agent_name} from memory")

            return context_data
        except Exception as e:
            logger.error(f"Failed to retrieve context for {agent_name}: {e}")
            return None

    async def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get performance statistics for an agent."""
        if agent_name not in self.contexts:
            return {}

        context = self.contexts[agent_name]
        config = self.configurations[agent_name]

        return {
            "name": agent_name,
            "nickname": config.nickname,
            "state": context.state,
            "model": config.model,
            "invocations": context.invocation_count,
            "token_count": context.token_count,
            "created_at": context.created_at.isoformat(),
            "last_accessed": context.last_accessed.isoformat(),
            "tools": config.tools,
            "memory_persisted": self.memory_system is not None
        }

    def get_all_agents_stats(self) -> Dict[str, Any]:
        """Get statistics about all agent usage."""
        stats = {
            "total_agents": len(self.configurations),
            "active_agents": len(self.active_agents),
            "total_contexts": len(self.contexts),
            "memory_enabled": self.memory_system is not None,
            "agents": {}
        }

        for name, context in self.contexts.items():
            stats["agents"][name] = {
                "state": context.state,
                "invocations": context.invocation_count,
                "token_count": context.token_count,
                "created_at": context.created_at.isoformat(),
                "last_accessed": context.last_accessed.isoformat()
            }

        return stats

    async def cleanup_inactive_contexts(self, max_age_hours: int = 24) -> None:
        """
        Clean up contexts that haven't been accessed recently.

        This helps manage memory usage while preserving active contexts.
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        agents_to_remove = []

        async with self._lock:
            for agent_name, context in self.contexts.items():
                if (context.state == "inactive" and
                    context.last_accessed < cutoff_time):
                    agents_to_remove.append(agent_name)

            for agent_name in agents_to_remove:
                del self.contexts[agent_name]
                logger.info(f"Cleaned up inactive context for agent: {agent_name}")

        if agents_to_remove:
            await self.metrics.track_metric(
                "contexts_cleaned",
                {"count": len(agents_to_remove)}
            )

    async def execute_agent(self, agent_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a sub-agent with isolated context and proper tool access.

        This method provides the bridge between delegation and actual agent execution,
        maintaining context isolation while enabling specialized processing.

        Args:
            agent_name: Name of the agent to execute
            input_data: Input data containing 'prompt' and optional 'context'

        Returns:
            Execution result with agent response and metadata
        """
        # Validate agent exists
        if agent_name not in self.configurations:
            return {
                'status': 'error',
                'error': f'Unknown agent: {agent_name}',
                'agent': agent_name
            }

        try:
            # Get or create agent context
            context = await self.activate_agent(agent_name)
            config = self.configurations[agent_name]

            # Extract input
            prompt = input_data.get('prompt', '')
            additional_context = input_data.get('context', {})

            # Update context with execution info
            context.token_count += len(prompt.split()) * 2  # Rough token estimate

            # Simulate agent execution (in production, this would interface with Claude Code)
            # The actual execution happens through Claude Code's native /agents command
            result = {
                'status': 'success',
                'agent': agent_name,
                'agent_id': context.agent_id,
                'nickname': config.nickname,
                'text_face': config.text_face,
                'model': config.model,
                'response': {
                    'processed': True,
                    'prompt_received': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                    'context_applied': bool(additional_context),
                    'tools_available': config.tools,
                    'execution_time_ms': 0.0  # Would be measured in production
                },
                'metadata': {
                    'invocation_count': context.invocation_count,
                    'token_count': context.token_count,
                    'state': context.state
                }
            }

            # Persist context after execution
            await self.persist_context(agent_name)

            # Track metrics
            await self.metrics.track_metric(
                "agent_execution",
                {
                    "agent": agent_name,
                    "status": "success",
                    "model": config.model
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error executing agent {agent_name}: {e}")

            # Suspend agent on error
            await self.suspend_agent(agent_name, f"Execution error: {str(e)}")

            return {
                'status': 'error',
                'agent': agent_name,
                'error': str(e),
                'suspended': True
            }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all agents and contexts.

        Returns system health status and any issues detected.
        """
        health_status = {
            "status": "healthy",
            "total_agents": len(self.configurations),
            "active_agents": len(self.active_agents),
            "total_contexts": len(self.contexts),
            "issues": []
        }

        # Check for suspended agents
        suspended = [name for name, ctx in self.contexts.items()
                    if ctx.state == "suspended"]
        if suspended:
            health_status["issues"].append({
                "type": "suspended_agents",
                "agents": suspended
            })

        # Check for error states
        error_agents = [name for name, ctx in self.contexts.items()
                       if ctx.state == "error"]
        if error_agents:
            health_status["status"] = "degraded"
            health_status["issues"].append({
                "type": "error_agents",
                "agents": error_agents
            })

        # Check configuration issues
        for name, config in self.configurations.items():
            if config.model not in ["sonnet", "opus"]:
                health_status["issues"].append({
                    "type": "invalid_model",
                    "agent": name,
                    "model": config.model
                })

        if health_status["issues"]:
            health_status["status"] = "degraded" if health_status["status"] != "error" else "error"

        return health_status
