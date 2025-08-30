"""
EventBus - Async message distribution system for the Dynamic Header System.
Implements publish-subscribe pattern with async event processing.
"""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Enumeration of all event types in the system."""
    LAYER_UPDATE = "layer_update"
    AGENT_UPDATE = "agent_update"
    MEMORY_UPDATE = "memory_update"
    CLASSIFICATION_UPDATE = "classification_update"
    PERFORMANCE_UPDATE = "performance_update"
    RESOURCE_UPDATE = "resource_update"
    STATUS_UPDATE = "status_update"
    PERFORMANCE_TEST = "performance_test"
    CONCURRENT_TEST = "concurrent_test"
    PRIORITY_TEST = "priority_test"
    SYSTEM_STATUS = "system_status"


@dataclass
class Event:
    """Event data structure for the event bus."""
    type: EventType
    source: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1-10, higher is more important
    correlation_id: str | None = None

    def __post_init__(self):
        """Validate event after creation."""
        if not isinstance(self.type, EventType):
            if isinstance(self.type, str):
                # Try to convert string to EventType
                try:
                    self.type = EventType(self.type)
                except ValueError:
                    raise ValueError(f"Invalid event type: {self.type}")
            else:
                raise TypeError(f"Event type must be EventType, got {type(self.type)}")

        if not 1 <= self.priority <= 10:
            raise ValueError(f"Priority must be 1-10, got {self.priority}")


class EventBus:
    """
    Asynchronous event bus for distributing events to subscribers.
    Ensures <10ms latency for event delivery.
    """

    def __init__(self):
        """Initialize the event bus."""
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)
        self._subscription_ids: dict[str, tuple] = {}  # id -> (event_type, handler)
        self._subscription_counter = 0
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = True
        self._processor_task: asyncio.Task | None = None
        self._start_processor()

    def _start_processor(self):
        """Start the event processor task."""
        if self._processor_task is None or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_events())

    async def _process_events(self):
        """Process events from the queue and deliver to subscribers."""
        while self._running:
            try:
                # Wait for event with timeout to allow checking _running flag
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )

                # Deliver to all subscribers of this event type
                if event.type in self._subscribers:
                    # Create delivery tasks for all subscribers
                    tasks = []
                    for handler in self._subscribers[event.type]:
                        tasks.append(self._deliver_event(handler, event))

                    # Run all deliveries concurrently
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

            except TimeoutError:
                continue  # Check _running flag
            except Exception as e:
                logger.error(f"Error processing event: {e}")

    async def _deliver_event(self, handler: Callable, event: Event):
        """
        Deliver event to a single handler with error isolation.
        
        Args:
            handler: Event handler function
            event: Event to deliver
        """
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Run sync handler in executor to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, event)
        except Exception as e:
            logger.error(f"Error in event handler {handler.__name__}: {e}")

    def subscribe(self, event_type: EventType, handler: Callable) -> str:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Async or sync function to handle events
            
        Returns:
            Subscription ID for later unsubscribe
        """
        if not isinstance(event_type, EventType):
            raise TypeError(f"event_type must be EventType, got {type(event_type)}")

        self._subscribers[event_type].append(handler)

        # Generate subscription ID
        subscription_id = f"sub_{self._subscription_counter}"
        self._subscription_counter += 1
        self._subscription_ids[subscription_id] = (event_type, handler)

        return subscription_id

    def unsubscribe(self, subscription_id: str):
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
        """
        if subscription_id in self._subscription_ids:
            event_type, handler = self._subscription_ids[subscription_id]
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(handler)
                except ValueError:
                    pass  # Handler already removed
            del self._subscription_ids[subscription_id]

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        if not isinstance(event, Event):
            raise TypeError(f"Must publish Event object, got {type(event)}")

        # Ensure processor is running
        if self._processor_task is None or self._processor_task.done():
            self._start_processor()

        # Add event to queue for processing
        await self._event_queue.put(event)

    async def stop(self):
        """Stop the event bus and clean up resources."""
        self._running = False

        # Cancel processor task
        if self._processor_task and not self._processor_task.done():
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Clear queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def clear_subscribers(self):
        """Clear all subscribers (useful for testing)."""
        self._subscribers.clear()
        self._subscription_ids.clear()

    def get_subscriber_count(self, event_type: EventType | None = None) -> int:
        """
        Get count of subscribers.
        
        Args:
            event_type: Specific event type or None for total
            
        Returns:
            Number of subscribers
        """
        if event_type is None:
            return sum(len(handlers) for handlers in self._subscribers.values())
        else:
            return len(self._subscribers.get(event_type, []))
