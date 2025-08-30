"""
HeaderFormatter with template support for dynamic status display.
Implements singleton pattern and integrates with EventBus for real-time updates.
"""

import asyncio
import time
from string import Template
from typing import Dict, Any, Optional, List
from collections import deque
import logging

from src.core.monitoring.event_bus import EventBus
from src.core.monitoring.display_adapters import (
    BaseAdapter,
    TerminalAdapter,
    JSONAdapter
)

logger = logging.getLogger(__name__)


class HeaderFormatter:
    """
    Singleton formatter for dynamic header display.
    Subscribes to EventBus and formats status data for various output modes.
    """
    
    _instance: Optional['HeaderFormatter'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls) -> 'HeaderFormatter':
        """Enforce singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize formatter with EventBus subscription and adapters."""
        # Prevent re-initialization
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        self._event_bus = EventBus()
        self._adapters: Dict[str, BaseAdapter] = {}
        self._event_queue = deque(maxlen=100)
        self._last_update_time = 0
        self._min_update_interval = 0.1  # 100ms rate limiting
        self._current_data: Dict[str, Any] = {}
        self._base_header: Optional[str] = None
        self._protocol_header: Optional[str] = None
        self._dynamic_header: Optional[str] = None
        self._templates: Dict[str, Template] = {}
        
        # Initialize default adapters
        self._initialize_default_adapters()
        
        # Try to start async tasks if event loop is running
        try:
            loop = asyncio.get_running_loop()
            # Subscribe to EventBus events
            asyncio.create_task(self._subscribe_to_events())
            # Start event processing loop
            asyncio.create_task(self._process_event_queue())
        except RuntimeError:
            # No event loop running (e.g., in tests or sync context)
            # Tasks will be started when async context is available
            logger.debug("No event loop available, async tasks deferred")
    
    def _initialize_default_adapters(self):
        """Initialize default display adapters."""
        self._adapters['terminal'] = TerminalAdapter()
        self._adapters['json'] = JSONAdapter()
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant EventBus events."""
        try:
            # Subscribe to various event types
            await self._event_bus.subscribe('StatusUpdate', self.handle_event)
            await self._event_bus.subscribe('LayerActivated', self.handle_event)
            await self._event_bus.subscribe('AgentStatusChanged', self.handle_event)
            await self._event_bus.subscribe('MemoryEvent', self.handle_event)
            logger.info("HeaderFormatter subscribed to EventBus")
        except Exception as e:
            logger.error(f"Failed to subscribe to EventBus: {e}")
    
    async def handle_event(self, event: Any):
        """Handle incoming events from EventBus."""
        # Add event to queue for processing
        self._event_queue.append(event)
    
    async def _process_event_queue(self):
        """Process queued events with rate limiting."""
        while True:
            try:
                if self._event_queue:
                    current_time = time.time()
                    time_since_last = current_time - self._last_update_time
                    
                    # Enforce rate limiting
                    if time_since_last < self._min_update_interval:
                        await asyncio.sleep(self._min_update_interval - time_since_last)
                    
                    # Process event
                    event = self._event_queue.popleft()
                    await self._process_update(self._extract_data(event))
                    self._last_update_time = time.time()
                else:
                    # No events, sleep briefly
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
                await asyncio.sleep(0.1)
    
    def _extract_data(self, event: Any) -> Dict[str, Any]:
        """Extract data from event for formatting."""
        if hasattr(event, 'data'):
            return event.data
        elif isinstance(event, dict):
            return event
        else:
            return {"raw_event": str(event)}
    
    async def _process_update(self, data: Dict[str, Any]):
        """Process status update with new data."""
        # Merge with current data
        self._current_data.update(data)
        
        # Format and store for retrieval
        for adapter_name in self._adapters:
            try:
                formatted = await self._adapters[adapter_name].render(self._current_data)
                # Store formatted output (in real implementation, might emit or cache)
            except Exception as e:
                logger.error(f"Error in adapter {adapter_name}: {e}")
    
    def _render_template(self, template_str: str, data: Dict[str, Any]) -> str:
        """Render template with provided data."""
        try:
            template = Template(template_str)
            # Create safe substitution dict with defaults for missing keys
            safe_data = {k: str(v) if v is not None else 'N/A' 
                        for k, v in data.items()}
            return template.safe_substitute(**safe_data)
        except Exception as e:
            logger.error(f"Template rendering error: {e}")
            return template_str  # Return original on error
    
    async def format(self, data: Dict[str, Any]) -> str:
        """Format data using default terminal adapter."""
        # Quick format for direct calls (not through event system)
        start_time = time.perf_counter()
        
        # Use terminal adapter by default
        result = await self._adapters['terminal'].render(data)
        
        # Verify performance
        elapsed = time.perf_counter() - start_time
        if elapsed > 0.010:  # 10ms
            logger.warning(f"Formatting took {elapsed*1000:.2f}ms, exceeds 10ms target")
        
        return result
    
    def register_adapter(self, name: str, adapter: BaseAdapter):
        """Register a custom display adapter."""
        if not isinstance(adapter, BaseAdapter):
            raise TypeError("Adapter must inherit from BaseAdapter")
        self._adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")
    
    async def get_output(self, adapter_name: str) -> str:
        """Get formatted output from specific adapter."""
        if adapter_name not in self._adapters:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        return await self._adapters[adapter_name].render(self._current_data)
    
    def set_base_header(self, header: str):
        """Set base header template (e.g., CLAUDE-v3.md format)."""
        self._base_header = header
    
    def set_header_structure(self, protocol: str, dynamic: str):
        """Set two-tier header structure."""
        self._protocol_header = protocol
        self._dynamic_header = dynamic
    
    def get_templates(self) -> Dict[str, Template]:
        """Get current template registry."""
        return self._templates
    
    def register_template(self, name: str, template_str: str):
        """Register a named template."""
        self._templates[name] = Template(template_str)
        logger.info(f"Registered template: {name}")
    
    def get_current_data(self) -> Dict[str, Any]:
        """Get current aggregated status data."""
        return self._current_data.copy()
    
    async def shutdown(self):
        """Clean shutdown of formatter."""
        logger.info("Shutting down HeaderFormatter")
        # Could clean up subscriptions and resources here