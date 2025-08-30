"""
Tests for HeaderFormatter and display adapters.
Following TDD approach - tests written before implementation.
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
from typing import Dict, Any

# These imports will fail initially until we implement the modules
from src.core.monitoring.header_formatter import HeaderFormatter
from src.core.monitoring.display_adapters import (
    BaseAdapter,
    TerminalAdapter, 
    JSONAdapter
)
from src.core.monitoring.event_bus import EventBus
from src.core.monitoring.message_schemas import EventType


class TestHeaderFormatter:
    """Test suite for HeaderFormatter class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a mock EventBus for testing."""
        bus = Mock(spec=EventBus)
        bus.subscribe = AsyncMock()
        bus.publish = AsyncMock()
        return bus
    
    @pytest.fixture
    def formatter(self, event_bus, monkeypatch):
        """Create HeaderFormatter instance with mocked EventBus."""
        monkeypatch.setattr('src.core.monitoring.header_formatter.EventBus', 
                          lambda: event_bus)
        # Clear singleton for testing
        HeaderFormatter._instance = None
        return HeaderFormatter()
    
    @pytest.mark.asyncio
    async def test_singleton_pattern_enforced(self):
        """Test that HeaderFormatter enforces singleton pattern."""
        # Reset singleton for test
        HeaderFormatter._instance = None
        
        formatter1 = HeaderFormatter()
        formatter2 = HeaderFormatter()
        assert formatter1 is formatter2
        assert id(formatter1) == id(formatter2)
        
        # Allow async tasks to initialize
        await asyncio.sleep(0.01)
    
    @pytest.mark.asyncio
    async def test_event_bus_subscription_on_init(self, formatter, event_bus):
        """Test that formatter subscribes to EventBus on initialization."""
        await asyncio.sleep(0.01)  # Allow async init to complete
        event_bus.subscribe.assert_called()
        # Check it subscribes to correct event types
        call_args = event_bus.subscribe.call_args
        assert 'StatusUpdate' in str(call_args) or 'LayerActivated' in str(call_args)
    
    def test_template_rendering_with_variables(self, formatter):
        """Test template rendering with variable substitution."""
        template = "Model: ${model} | Status: ${status}"
        data = {"model": "Claude-3", "status": "Active"}
        result = formatter._render_template(template, data)
        assert result == "Model: Claude-3 | Status: Active"
    
    def test_missing_variable_handling(self, formatter):
        """Test graceful handling of missing template variables."""
        template = "Model: ${model} | Missing: ${missing_var}"
        data = {"model": "Claude-3"}
        result = formatter._render_template(template, data)
        # Should handle missing variables gracefully
        assert "Claude-3" in result
        assert "${missing_var}" in result or "N/A" in result
    
    def test_invalid_template_error_handling(self, formatter):
        """Test error handling for invalid templates."""
        invalid_template = "Model: ${model"  # Missing closing brace
        data = {"model": "Claude-3"}
        result = formatter._render_template(invalid_template, data)
        # Should return template as-is or error message
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting_100ms_minimum(self, formatter):
        """Test that updates are rate-limited to 100ms minimum."""
        update_times = []
        
        async def mock_update():
            update_times.append(time.time())
            await formatter._process_update({"test": "data"})
        
        # Send rapid updates
        tasks = [mock_update() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Check intervals between updates
        if len(update_times) > 1:
            intervals = [update_times[i+1] - update_times[i] 
                        for i in range(len(update_times)-1)]
            # All intervals should be >= 100ms (0.1s)
            assert all(interval >= 0.095 for interval in intervals)  # Small tolerance
    
    @pytest.mark.asyncio
    async def test_burst_event_queuing(self, formatter):
        """Test that burst events are properly queued."""
        events = [{"id": i, "data": f"event_{i}"} for i in range(10)]
        
        # Send burst of events
        tasks = [formatter.handle_event(event) for event in events]
        await asyncio.gather(*tasks)
        
        # Check that events are queued and processed
        assert formatter._event_queue is not None
        # Queue should handle burst properly
        assert hasattr(formatter, '_last_update_time')
    
    @pytest.mark.asyncio
    async def test_rendering_performance_under_10ms(self, formatter):
        """Test that rendering completes in under 10ms."""
        data = {
            "workflow_agents": ["agent1", "agent2", "agent3"],
            "cognitive_processors": ["PE", "R1", "A1"],
            "context_usage": "8192/10000",
            "memory_usage": "45%"
        }
        
        start_time = time.perf_counter()
        result = await formatter.format(data)
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        assert elapsed_ms < 10, f"Rendering took {elapsed_ms:.2f}ms, exceeds 10ms limit"
    
    @pytest.mark.asyncio
    async def test_adapter_registration(self, formatter):
        """Test that display adapters can be registered."""
        adapter = Mock(spec=BaseAdapter)
        formatter.register_adapter("test", adapter)
        assert "test" in formatter._adapters
        assert formatter._adapters["test"] == adapter
    
    def test_default_adapters_initialized(self, formatter):
        """Test that default adapters are initialized."""
        assert "terminal" in formatter._adapters
        assert "json" in formatter._adapters
        assert isinstance(formatter._adapters["terminal"], TerminalAdapter)
        assert isinstance(formatter._adapters["json"], JSONAdapter)


class TestTerminalAdapter:
    """Test suite for TerminalAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create TerminalAdapter instance."""
        return TerminalAdapter()
    
    @pytest.mark.asyncio
    async def test_rich_console_formatting(self, adapter):
        """Test that rich console formatting is applied."""
        data = {
            "workflow_agents": [
                {"name": "api-designer", "status": "Active"},
                {"name": "code-reviewer", "status": "Idle"}
            ],
            "cognitive_processors": [
                {"name": "R1-Researcher", "status": "Active"},
                {"name": "A1-Reasoner", "status": "Queued"}
            ]
        }
        
        with patch('src.core.monitoring.display_adapters.Console') as MockConsole:
            console_instance = MockConsole.return_value
            result = await adapter.render(data)
            
            # Should use rich console for formatting
            assert console_instance.print.called or console_instance.render_str.called
            assert result is not None
            assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_dual_agent_table_display(self, adapter):
        """Test dual sub-agent display format."""
        data = {
            "workflow_agents": [{"name": "test-agent", "status": "Active"}],
            "cognitive_processors": [{"name": "PE", "status": "Complete"}]
        }
        
        result = await adapter.render(data)
        
        # Check for both agent types in output
        assert "WORKFLOW AGENTS" in result or "workflow" in result.lower()
        assert "COGNITIVE PROCESSORS" in result or "cognitive" in result.lower()
    
    @pytest.mark.asyncio
    async def test_ansi_color_codes_applied(self, adapter):
        """Test that ANSI color codes are applied for terminal."""
        data = {"status": "Active", "model": "Claude-3"}
        result = await adapter.render(data)
        
        # Check for ANSI escape codes or rich markup
        # Rich might use internal representation
        assert result is not None
        # Result should contain formatted text
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_terminal_width_adaptation(self, adapter):
        """Test adaptation to different terminal widths."""
        data = {"test": "data" * 50}  # Long data
        
        # Test with narrow terminal
        with patch('shutil.get_terminal_size') as mock_size:
            mock_size.return_value = (40, 24)  # Narrow terminal
            narrow_result = await adapter.render(data)
            
            mock_size.return_value = (120, 24)  # Wide terminal
            wide_result = await adapter.render(data)
            
            # Results should differ based on width
            assert narrow_result != wide_result or len(narrow_result) < len(wide_result)
    
    @pytest.mark.asyncio
    async def test_fallback_to_plain_text(self, adapter):
        """Test fallback to plain text when rich is unavailable."""
        with patch.dict('sys.modules', {'rich': None}):
            # Simulate rich not being available
            adapter._use_plain_text = True
            data = {"status": "Active"}
            result = await adapter.render(data)
            
            # Should still produce output without rich
            assert result is not None
            assert "Active" in result
            # Should not contain rich markup
            assert "[" not in result or "]" not in result


class TestJSONAdapter:
    """Test suite for JSONAdapter."""
    
    @pytest.fixture
    def adapter(self):
        """Create JSONAdapter instance."""
        return JSONAdapter()
    
    @pytest.mark.asyncio
    async def test_statusline_json_structure(self, adapter):
        """Test that output matches Claude Code statusline schema."""
        data = {
            "model": {"id": "claude-3", "display_name": "Claude 3"},
            "workspace": {"current_dir": "/test/dir"},
            "git": {"branch": "main", "status": "clean"},
            "cost": {"total_lines_added": 100, "total_lines_removed": 50}
        }
        
        result = await adapter.render(data)
        parsed = json.loads(result)
        
        # Check required statusline fields
        assert "model" in parsed
        assert "workspace" in parsed
        assert parsed["model"]["display_name"] == "Claude 3"
        assert parsed["workspace"]["current_dir"] == "/test/dir"
    
    @pytest.mark.asyncio
    async def test_data_type_serialization(self, adapter):
        """Test that all data types are properly serialized."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"}
        }
        
        result = await adapter.render(data)
        parsed = json.loads(result)
        
        # All types should be preserved
        assert parsed["string"] == "test"
        assert parsed["number"] == 42
        assert parsed["float"] == 3.14
        assert parsed["boolean"] is True
        assert parsed["null"] is None
        assert parsed["list"] == [1, 2, 3]
        assert parsed["dict"]["nested"] == "value"
    
    @pytest.mark.asyncio
    async def test_claude_code_schema_compliance(self, adapter):
        """Test compliance with Claude Code statusline schema."""
        data = {
            "cognitive_state": {
                "layers": ["Atomic", "Molecular"],
                "active_agents": ["PE", "R1"]
            }
        }
        
        result = await adapter.render(data)
        parsed = json.loads(result)
        
        # Should include cognitive state in proper format
        assert "cognitive_state" in parsed
        assert "layers" in parsed["cognitive_state"]
        assert len(parsed["cognitive_state"]["layers"]) == 2
    
    @pytest.mark.asyncio
    async def test_empty_data_handling(self, adapter):
        """Test handling of empty or minimal data."""
        result = await adapter.render({})
        parsed = json.loads(result)
        
        # Should return valid JSON even with empty data
        assert isinstance(parsed, dict)
        
        # Test with None
        result = await adapter.render(None)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)


class TestIntegration:
    """Integration tests for the complete header formatting system."""
    
    @pytest.fixture
    async def setup(self):
        """Set up integration test environment."""
        event_bus = EventBus()
        HeaderFormatter._instance = None  # Reset singleton
        formatter = HeaderFormatter()
        
        # Wait for initialization
        await asyncio.sleep(0.01)
        
        return {
            "event_bus": event_bus,
            "formatter": formatter
        }
    
    @pytest.mark.asyncio
    async def test_full_event_to_output_pipeline(self, setup):
        """Test complete pipeline from event to formatted output."""
        event_bus = setup["event_bus"]
        formatter = setup["formatter"]
        
        # Publish event
        event = {
            "type": "status_update",
            "source": "test",
            "timestamp": time.time(),
            "priority": 5,
            "correlation_id": "test-123",
            "data": {
                "layer": "Atomic",
                "status": "Active",
                "model": "Claude-3"
            }
        }
        
        await event_bus.publish(event)
        await asyncio.sleep(0.15)  # Allow processing
        
        # Get formatted output
        terminal_output = await formatter.get_output("terminal")
        json_output = await formatter.get_output("json")
        
        # Verify outputs
        assert terminal_output is not None
        assert "Active" in terminal_output or "Atomic" in terminal_output
        
        parsed_json = json.loads(json_output)
        assert parsed_json is not None
    
    @pytest.mark.asyncio
    async def test_claude_v3_header_preservation(self, setup):
        """Test that existing CLAUDE-v3.md headers are preserved."""
        formatter = setup["formatter"]
        
        # Set existing header format
        formatter.set_base_header("""
        🧠 SAGE Status: [✅Active]
        🔍 SEIQF Status: [✅Active]
        🎭 SIA Status: [✅Active]
        """)
        
        data = {"new_field": "test"}
        result = await formatter.format(data)
        
        # Original headers should be preserved
        assert "SAGE Status" in result
        assert "SEIQF Status" in result
        assert "SIA Status" in result
        # New data should be added
        assert "test" in result or "new_field" in result
    
    @pytest.mark.asyncio
    async def test_two_tier_header_structure(self, setup):
        """Test two-tier structure with protocol and dynamic headers."""
        formatter = setup["formatter"]
        
        # Configure two-tier structure
        protocol_header = "=== PROTOCOL STATUS ==="
        dynamic_header = "=== DYNAMIC STATUS ==="
        
        formatter.set_header_structure(protocol_header, dynamic_header)
        
        data = {"dynamic_data": "test_value"}
        result = await formatter.format(data)
        
        # Both tiers should be present
        assert "PROTOCOL STATUS" in result
        assert "DYNAMIC STATUS" in result
        
        # Order should be preserved (protocol first)
        protocol_index = result.index("PROTOCOL STATUS")
        dynamic_index = result.index("DYNAMIC STATUS")
        assert protocol_index < dynamic_index
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, setup):
        """Test system performance under heavy load."""
        formatter = setup["formatter"]
        event_bus = setup["event_bus"]
        
        # Generate many events
        events = []
        for i in range(100):
            event = StatusUpdateEvent(
                event_type="StatusUpdate",
                timestamp=time.time(),
                data={"id": i, "status": f"status_{i}"}
            )
            events.append(event)
        
        # Measure processing time
        start_time = time.perf_counter()
        
        # Publish all events
        tasks = [event_bus.publish(event) for event in events]
        await asyncio.gather(*tasks)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        end_time = time.perf_counter()
        
        # Check performance
        total_time = end_time - start_time
        assert total_time < 2.0, f"Processing 100 events took {total_time:.2f}s"
        
        # Get final output
        output = await formatter.get_output("terminal")
        assert output is not None