"""
Base Sub-Agent class for coordination protocols.

Provides the foundation for all specialized sub-agents with
message passing, error handling, and coordination interfaces.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import uuid


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages that can be passed between agents."""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    STATUS = "status"
    COORDINATION = "coordination"
    RESULT = "result"


class MessagePriority(Enum):
    """Priority levels for message processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentMessage:
    """
    Message structure for inter-agent communication.
    
    Ensures consistent communication patterns between specialists.
    """
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    requires_response: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response
        }


class BaseSubAgent(ABC):
    """
    Abstract base class for all sub-agents.
    
    Provides coordination interfaces, message handling, and error boundaries.
    """
    
    def __init__(self, name: str, nickname: str, model: str = "sonnet"):
        """Initialize the base sub-agent."""
        self.name = name
        self.nickname = nickname
        self.model = model
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_handlers: Dict[str, Callable] = {}
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        
    @abstractmethod
    async def process_request(self, message: AgentMessage) -> AgentMessage:
        """
        Process an incoming request message.
        
        Must be implemented by each specialized sub-agent.
        """
        pass
    
    @abstractmethod
    async def validate_input(self, content: Dict[str, Any]) -> bool:
        """
        Validate input before processing.
        
        Helps prevent errors and ensures quality.
        """
        pass
    
    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent or the orchestrator."""
        logger.debug(f"{self.nickname} sending message to {message.to_agent}")
        await self.message_queue.put(message)
    
    async def receive_message(self) -> AgentMessage:
        """Receive a message from the queue."""
        return await self.message_queue.get()
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Handle an incoming message with error boundaries.
        
        Ensures isolated failures don't cascade.
        """
        try:
            # Validate input
            if not await self.validate_input(message.content):
                return self._create_error_response(
                    message,
                    "Invalid input format or content"
                )
            
            # Process based on message type
            if message.type == MessageType.REQUEST:
                response = await self.process_request(message)
                return response
            elif message.type == MessageType.COORDINATION:
                await self._handle_coordination(message)
                return None
            elif message.type == MessageType.STATUS:
                return self._create_status_response(message)
            else:
                logger.warning(f"Unknown message type: {message.type}")
                return None
                
        except Exception as e:
            logger.error(f"Error in {self.nickname} handling message: {e}")
            self.error_count += 1
            self.last_error = e
            return self._create_error_response(message, str(e))
    
    def _create_error_response(self, original: AgentMessage, error: str) -> AgentMessage:
        """Create an error response message."""
        return AgentMessage(
            from_agent=self.name,
            to_agent=original.from_agent,
            type=MessageType.ERROR,
            priority=original.priority,
            content={"error": error, "original_request": original.content},
            correlation_id=original.id
        )
    
    def _create_status_response(self, original: AgentMessage) -> AgentMessage:
        """Create a status response message."""
        return AgentMessage(
            from_agent=self.name,
            to_agent=original.from_agent,
            type=MessageType.STATUS,
            priority=MessagePriority.LOW,
            content={
                "status": "active" if self.is_running else "inactive",
                "error_count": self.error_count,
                "last_error": str(self.last_error) if self.last_error else None
            },
            correlation_id=original.id
        )
    
    async def _handle_coordination(self, message: AgentMessage) -> None:
        """Handle coordination messages from the orchestrator."""
        action = message.content.get("action")
        
        if action == "prepare":
            # Prepare for upcoming work
            await self.prepare()
        elif action == "reset":
            # Reset internal state
            await self.reset()
        elif action == "shutdown":
            # Graceful shutdown
            await self.shutdown()
        else:
            logger.warning(f"Unknown coordination action: {action}")
    
    async def prepare(self) -> None:
        """Prepare the agent for processing."""
        self.is_running = True
        logger.info(f"{self.nickname} prepared for processing")
    
    async def reset(self) -> None:
        """Reset the agent's internal state."""
        self.error_count = 0
        self.last_error = None
        logger.info(f"{self.nickname} state reset")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent."""
        self.is_running = False
        if self._task:
            self._task.cancel()
        logger.info(f"{self.nickname} shutdown complete")
    
    async def run(self) -> None:
        """
        Main processing loop for the agent.
        
        Continuously processes messages from the queue.
        """
        self.is_running = True
        logger.info(f"{self.nickname} started")
        
        try:
            while self.is_running:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(
                        self.receive_message(),
                        timeout=1.0
                    )
                    
                    # Handle the message
                    response = await self.handle_message(message)
                    
                    # Send response if required
                    if response and message.requires_response:
                        await self.send_message(response)
                        
                except asyncio.TimeoutError:
                    # No message received, continue
                    continue
                except Exception as e:
                    logger.error(f"Error in {self.nickname} run loop: {e}")
                    self.error_count += 1
                    
        except asyncio.CancelledError:
            logger.info(f"{self.nickname} cancelled")
        finally:
            self.is_running = False
            logger.info(f"{self.nickname} stopped")
    
    def start(self) -> None:
        """Start the agent's processing loop."""
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self.run())
    
    def stop(self) -> None:
        """Stop the agent's processing loop."""
        self.is_running = False
        if self._task and not self._task.done():
            self._task.cancel()


class CoordinationPattern:
    """
    Defines coordination patterns between agents.
    
    Common patterns for multi-agent workflows.
    """
    
    @staticmethod
    def sequential(agents: List[str]) -> List[tuple]:
        """
        Sequential processing pattern.
        
        Each agent processes in order, passing results to the next.
        """
        if len(agents) < 2:
            return []
        
        pattern = []
        for i in range(len(agents) - 1):
            pattern.append((agents[i], agents[i + 1]))
        return pattern
    
    @staticmethod
    def parallel(coordinator: str, workers: List[str]) -> List[tuple]:
        """
        Parallel processing pattern.
        
        Coordinator sends to all workers simultaneously.
        """
        pattern = []
        for worker in workers:
            pattern.append((coordinator, worker))
        return pattern
    
    @staticmethod
    def hierarchical(root: str, branches: Dict[str, List[str]]) -> List[tuple]:
        """
        Hierarchical processing pattern.
        
        Root delegates to branches, which delegate to leaves.
        """
        pattern = []
        for branch, leaves in branches.items():
            pattern.append((root, branch))
            for leaf in leaves:
                pattern.append((branch, leaf))
        return pattern
    
    @staticmethod
    def bidirectional(agent1: str, agent2: str) -> List[tuple]:
        """
        Bidirectional communication pattern.
        
        Two agents can communicate in both directions.
        """
        return [(agent1, agent2), (agent2, agent1)]