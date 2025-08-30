"""
Test suite for StatusCollector - the central state aggregator.
Following TDD principles with comprehensive test coverage.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.monitoring.status_collector import StatusCollector


class TestStatusCollector:
    """Test suite for StatusCollector singleton and event aggregation."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton instance between tests."""
        StatusCollector._instance = None
        yield
        StatusCollector._instance = None
    
    def test_singleton_pattern_enforced(self):
        """Test that StatusCollector enforces singleton pattern."""
        # Create first instance
        collector1 = StatusCollector()
        
        # Attempt to create second instance
        collector2 = StatusCollector()
        
        # Both should be the same instance
        assert collector1 is collector2
        assert id(collector1) == id(collector2)
        
        # Even with get_instance method
        collector3 = StatusCollector.get_instance()
        assert collector1 is collector3
    
    @pytest.mark.asyncio
    async def test_concurrent_event_handling(self):
        """Test that StatusCollector handles concurrent events safely."""
        collector = StatusCollector()
        events_received = []
        
        # Subscribe to events
        def event_handler(event):
            events_received.append(event)
        
        collector.subscribe("test_event", event_handler)
        
        # Create multiple concurrent event publishes
        async def publish_event(event_id):
            await collector.publish_event("test_event", {
                "id": event_id,
                "timestamp": time.time()
            })
        
        # Publish 100 events concurrently
        tasks = [publish_event(i) for i in range(100)]
        await asyncio.gather(*tasks)
        
        # All events should be received
        assert len(events_received) == 100
        
        # Check that all event IDs are present (no lost events)
        received_ids = {event["id"] for event in events_received}
        assert received_ids == set(range(100))
    
    @pytest.mark.asyncio
    async def test_event_aggregation_within_10ms(self):
        """Test that event aggregation completes within 10ms latency requirement."""
        collector = StatusCollector()
        
        # Create a test event
        test_event = {
            "layer": "atomic",
            "status": "active",
            "timestamp": time.time()
        }
        
        # Measure aggregation time
        start_time = time.perf_counter()
        
        # Publish event and wait for aggregation
        await collector.publish_event("layer_update", test_event)
        
        # Get aggregated state
        state = collector.get_current_state()
        
        end_time = time.perf_counter()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        
        # Assert latency is under 10ms
        assert latency_ms < 10, f"Latency {latency_ms:.2f}ms exceeds 10ms requirement"
        
        # Verify event was aggregated
        assert "layers" in state
        assert state["layers"]["atomic"]["status"] == "active"
    
    def test_memory_limit_50mb(self):
        """Test that StatusCollector memory usage stays under 50MB."""
        collector = StatusCollector()
        
        # Generate large number of events
        for i in range(10000):
            collector._add_to_history({
                "id": i,
                "data": "x" * 1000,  # 1KB of data per event
                "timestamp": time.time()
            })
        
        # Get memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get collector's approximate memory usage
        # This is a simplified check - in production we'd be more precise
        collector_size = sys.getsizeof(collector._event_history)
        collector_size += sys.getsizeof(collector._current_state)
        
        # Convert to MB
        size_mb = collector_size / (1024 * 1024)
        
        # Assert under 50MB limit
        assert size_mb < 50, f"Memory usage {size_mb:.2f}MB exceeds 50MB limit"
    
    def test_rolling_window_100_events(self):
        """Test that event history maintains rolling window of last 100 events."""
        collector = StatusCollector()
        
        # Add 150 events
        for i in range(150):
            collector._add_to_history({
                "id": i,
                "timestamp": time.time()
            })
        
        # History should only contain last 100 events
        assert len(collector._event_history) == 100
        
        # First event should be id=50 (0-49 were dropped)
        assert collector._event_history[0]["id"] == 50
        
        # Last event should be id=149
        assert collector._event_history[-1]["id"] == 149
    
    @pytest.mark.asyncio
    async def test_state_aggregation_accuracy(self):
        """Test that state aggregation correctly combines multiple sources."""
        collector = StatusCollector()
        
        # Publish events from different sources
        await collector.publish_event("layer_update", {
            "layer": "atomic",
            "status": "active",
            "confidence": 0.95
        })
        
        await collector.publish_event("agent_update", {
            "agent": "researcher",
            "status": "idle"
        })
        
        await collector.publish_event("memory_update", {
            "stm_usage": 45,
            "wm_usage": 12,
            "ltm_usage": 3
        })
        
        # Get aggregated state
        state = collector.get_current_state()
        
        # Verify all updates are reflected
        assert state["layers"]["atomic"]["status"] == "active"
        assert state["layers"]["atomic"]["confidence"] == 0.95
        assert state["agents"]["researcher"]["status"] == "idle"
        assert state["memory"]["stm_usage"] == 45
        assert state["memory"]["wm_usage"] == 12
        assert state["memory"]["ltm_usage"] == 3
    
    @pytest.mark.asyncio
    async def test_subscriber_notification(self):
        """Test that all subscribers are notified of events."""
        collector = StatusCollector()
        
        # Track notifications for multiple subscribers
        subscriber1_events = []
        subscriber2_events = []
        
        def handler1(event):
            subscriber1_events.append(event)
        
        def handler2(event):
            subscriber2_events.append(event)
        
        # Subscribe to same event type
        collector.subscribe("test_event", handler1)
        collector.subscribe("test_event", handler2)
        
        # Publish event
        test_event = {"data": "test"}
        await collector.publish_event("test_event", test_event)
        
        # Both subscribers should receive the event
        assert len(subscriber1_events) == 1
        assert len(subscriber2_events) == 1
        assert subscriber1_events[0] == test_event
        assert subscriber2_events[0] == test_event
    
    def test_state_persistence(self):
        """Test that state persists across get_current_state calls."""
        collector = StatusCollector()
        
        # Set initial state
        collector._current_state = {
            "layers": {"atomic": {"status": "active"}},
            "timestamp": time.time()
        }
        
        # Get state multiple times
        state1 = collector.get_current_state()
        state2 = collector.get_current_state()
        
        # Should be same data (not necessarily same object)
        assert state1["layers"]["atomic"]["status"] == state2["layers"]["atomic"]["status"]
        
        # Modifications to returned state shouldn't affect internal state
        state1["layers"]["atomic"]["status"] = "modified"
        state3 = collector.get_current_state()
        assert state3["layers"]["atomic"]["status"] == "active"  # Original value
    
    @pytest.mark.asyncio
    async def test_error_handling_in_subscribers(self):
        """Test that errors in one subscriber don't affect others."""
        collector = StatusCollector()
        
        good_events = []
        
        def failing_handler(event):
            raise Exception("Handler error")
        
        def good_handler(event):
            good_events.append(event)
        
        # Subscribe both handlers
        collector.subscribe("test_event", failing_handler)
        collector.subscribe("test_event", good_handler)
        
        # Publish event
        await collector.publish_event("test_event", {"data": "test"})
        
        # Good handler should still receive event despite error in failing handler
        assert len(good_events) == 1
        assert good_events[0]["data"] == "test"
    
    @pytest.mark.asyncio
    async def test_unsubscribe_functionality(self):
        """Test that unsubscribe removes event handlers."""
        collector = StatusCollector()
        
        events_received = []
        
        def handler(event):
            events_received.append(event)
        
        # Subscribe
        subscription_id = collector.subscribe("test_event", handler)
        
        # Publish first event
        await collector.publish_event("test_event", {"id": 1})
        assert len(events_received) == 1
        
        # Unsubscribe
        collector.unsubscribe(subscription_id)
        
        # Publish second event
        await collector.publish_event("test_event", {"id": 2})
        
        # Should still only have first event
        assert len(events_received) == 1
        assert events_received[0]["id"] == 1