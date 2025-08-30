"""
StatusCollector - Central state aggregator for the Dynamic Header System.
Implements singleton pattern to maintain consistent system state.
"""

import asyncio
import copy
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StatusCollector:
    """
    Singleton collector that aggregates status from all system components.
    Maintains rolling window of events and current system state.
    """

    _instance: Optional['StatusCollector'] = None

    def __new__(cls) -> 'StatusCollector':
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the status collector (only once due to singleton)."""
        if self._initialized:
            return

        self._initialized = True
        self._event_history: deque = deque(maxlen=100)  # Rolling window of 100 events
        self._current_state: dict[str, Any] = {
            "layers": {},
            "agents": {},
            "memory": {},
            "classification": {},
            "performance": {},
            "timestamp": time.time()
        }
        self._subscribers: dict[str, list[Callable]] = {}
        self._subscription_counter = 0
        self._subscriptions: dict[str, tuple] = {}  # id -> (event_type, handler)
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> 'StatusCollector':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def subscribe(self, event_type: str, handler: Callable) -> str:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Callback function to handle events
            
        Returns:
            Subscription ID for later unsubscribe
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)

        # Generate subscription ID
        subscription_id = f"sub_{self._subscription_counter}"
        self._subscription_counter += 1
        self._subscriptions[subscription_id] = (event_type, handler)

        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
        """
        if subscription_id in self._subscriptions:
            event_type, handler = self._subscriptions[subscription_id]
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                except ValueError:
                    pass  # Handler already removed
            del self._subscriptions[subscription_id]

    async def publish_event(self, event_type: str, event_data: dict[str, Any]):
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: Type of event
            event_data: Event data dictionary
        """
        # Add to history
        self._add_to_history(event_data)

        # Update aggregated state based on event type
        async with self._lock:
            await self._update_state(event_type, event_data)

        # Notify subscribers
        if event_type in self._subscribers:
            for handler in self._subscribers[event_type]:
                try:
                    # Call handler without blocking
                    if asyncio.iscoroutinefunction(handler):
                        asyncio.create_task(handler(event_data))
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler: {e}")

    def _add_to_history(self, event: dict[str, Any]):
        """
        Add event to rolling history window.
        
        Args:
            event: Event to add to history
        """
        # Add timestamp if not present
        if "timestamp" not in event:
            event["timestamp"] = time.time()

        # Add to deque (automatically maintains max size)
        self._event_history.append(event)

    async def _update_state(self, event_type: str, event_data: dict[str, Any]):
        """
        Update aggregated state based on event type and data.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        # Update timestamp
        self._current_state["timestamp"] = time.time()

        # Route event to appropriate state updater
        if event_type == "layer_update":
            layer = event_data.get("layer")
            if layer:
                if "layers" not in self._current_state:
                    self._current_state["layers"] = {}
                self._current_state["layers"][layer] = {
                    "status": event_data.get("status"),
                    "confidence": event_data.get("confidence", 0.0),
                    "timestamp": time.time()
                }

        elif event_type == "agent_update":
            agent = event_data.get("agent")
            if agent:
                if "agents" not in self._current_state:
                    self._current_state["agents"] = {}
                self._current_state["agents"][agent] = {
                    "status": event_data.get("status"),
                    "timestamp": time.time()
                }

        elif event_type == "memory_update":
            if "memory" not in self._current_state:
                self._current_state["memory"] = {}

            # Update memory metrics
            for key in ["stm_usage", "wm_usage", "ltm_usage"]:
                if key in event_data:
                    self._current_state["memory"][key] = event_data[key]

        elif event_type == "classification_update":
            self._current_state["classification"] = {
                "type": event_data.get("classification"),
                "confidence": event_data.get("confidence", 0.0),
                "timestamp": time.time()
            }

        elif event_type == "performance_update":
            if "performance" not in self._current_state:
                self._current_state["performance"] = {}

            # Update performance metrics
            for key in ["latency_ms", "cpu_percent", "memory_mb"]:
                if key in event_data:
                    self._current_state["performance"][key] = event_data[key]

    def get_current_state(self) -> dict[str, Any]:
        """
        Get current aggregated system state.
        
        Returns:
            Deep copy of current state dictionary
        """
        # Return deep copy to prevent external modifications
        return copy.deepcopy(self._current_state)

    def get_event_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """
        Get event history.
        
        Args:
            limit: Maximum number of events to return (None for all)
            
        Returns:
            List of recent events
        """
        if limit is None:
            return list(self._event_history)
        else:
            return list(self._event_history)[-limit:]

    def clear_history(self):
        """Clear event history (useful for testing)."""
        self._event_history.clear()

    def reset_state(self):
        """Reset current state (useful for testing)."""
        self._current_state = {
            "layers": {},
            "agents": {},
            "memory": {},
            "classification": {},
            "performance": {},
            "timestamp": time.time()
        }
