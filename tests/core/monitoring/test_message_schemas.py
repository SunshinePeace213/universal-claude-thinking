"""
Test suite for message schemas validation.
Ensures all events conform to expected schemas for reliability.
"""

import pytest
import json
import time
from jsonschema import validate, ValidationError, Draft7Validator
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.monitoring.message_schemas import (
    EVENT_SCHEMAS,
    validate_event,
    create_event,
    EventType
)


class TestMessageSchemas:
    """Test suite for event message schema validation."""
    
    def test_event_schema_validation(self):
        """Test that valid events pass schema validation."""
        # Test layer update event
        layer_event = {
            "type": "layer_update",
            "source": "atomic_layer",
            "timestamp": time.time(),
            "data": {
                "layer": "atomic",
                "status": "active",
                "confidence": 0.95,
                "metrics": {
                    "processing_time_ms": 5.2,
                    "tokens_processed": 150
                }
            }
        }
        
        # Should not raise
        assert validate_event(layer_event) is True
        
        # Test agent update event
        agent_event = {
            "type": "agent_update",
            "source": "agent_manager",
            "timestamp": time.time(),
            "data": {
                "agent_id": "researcher_001",
                "agent_type": "cognitive",
                "name": "R1-Researcher",
                "status": "active",
                "current_task": "Gathering context",
                "progress": 45
            }
        }
        
        assert validate_event(agent_event) is True
        
        # Test memory update event
        memory_event = {
            "type": "memory_update",
            "source": "memory_monitor",
            "timestamp": time.time(),
            "data": {
                "stm_usage_percent": 45.5,
                "wm_usage_percent": 12.3,
                "ltm_usage_percent": 3.1,
                "total_items": {
                    "stm": 25,
                    "wm": 150,
                    "ltm": 1024
                }
            }
        }
        
        assert validate_event(memory_event) is True
        
        # Test classification update event
        classification_event = {
            "type": "classification_update",
            "source": "classifier",
            "timestamp": time.time(),
            "data": {
                "request_id": "req_123",
                "classification": "B",
                "confidence": 0.87,
                "required_tools": ["sequential_thinking", "systems_thinking"],
                "reasoning": "Complex multi-step task requiring analysis"
            }
        }
        
        assert validate_event(classification_event) is True
    
    def test_invalid_event_rejection(self):
        """Test that invalid events are rejected with appropriate errors."""
        # Missing required field (timestamp)
        invalid_event1 = {
            "type": "layer_update",
            "source": "atomic_layer",
            # "timestamp": missing
            "data": {"layer": "atomic", "status": "active"}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_event(invalid_event1)
        assert "timestamp" in str(exc_info.value)
        
        # Invalid type field
        invalid_event2 = {
            "type": "unknown_type",
            "source": "test",
            "timestamp": time.time(),
            "data": {}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_event(invalid_event2)
        assert "unknown_type" in str(exc_info.value)
        
        # Invalid data type (string instead of number)
        invalid_event3 = {
            "type": "memory_update",
            "source": "memory_monitor",
            "timestamp": time.time(),
            "data": {
                "stm_usage_percent": "forty-five",  # Should be number
                "wm_usage_percent": 12.3,
                "ltm_usage_percent": 3.1
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_event(invalid_event3)
        assert "stm_usage_percent" in str(exc_info.value)
        
        # Missing required data field
        invalid_event4 = {
            "type": "agent_update",
            "source": "agent_manager",
            "timestamp": time.time(),
            "data": {
                # "agent_id": missing required field
                "status": "active"
            }
        }
        
        with pytest.raises(ValidationError) as exc_info:
            validate_event(invalid_event4)
        assert "agent_id" in str(exc_info.value)
    
    def test_backwards_compatibility(self):
        """Test that schemas maintain backwards compatibility."""
        # Old format event (v1)
        old_event_v1 = {
            "type": "layer_update",
            "source": "atomic_layer",
            "timestamp": time.time(),
            "data": {
                "layer": "atomic",
                "status": "active"
                # Missing newer fields like confidence, metrics
            }
        }
        
        # Should still validate (new fields are optional)
        assert validate_event(old_event_v1) is True
        
        # Event with additional unknown fields (forward compatibility)
        future_event = {
            "type": "layer_update",
            "source": "atomic_layer",
            "timestamp": time.time(),
            "data": {
                "layer": "atomic",
                "status": "active",
                "confidence": 0.95,
                "new_field_v2": "some_value",  # Unknown field
                "experimental_feature": True    # Unknown field
            },
            "metadata": {  # Additional top-level field
                "version": "2.0"
            }
        }
        
        # Should validate with additionalProperties allowed
        assert validate_event(future_event, allow_additional=True) is True
    
    def test_event_creation_helpers(self):
        """Test helper functions for creating valid events."""
        # Create layer update event
        layer_event = create_event(
            EventType.LAYER_UPDATE,
            source="atomic_layer",
            data={
                "layer": "atomic",
                "status": "active",
                "confidence": 0.95
            }
        )
        
        # Should have all required fields
        assert layer_event["type"] == "layer_update"
        assert layer_event["source"] == "atomic_layer"
        assert "timestamp" in layer_event
        assert layer_event["data"]["layer"] == "atomic"
        
        # Should be valid
        assert validate_event(layer_event) is True
        
        # Create agent update event
        agent_event = create_event(
            EventType.AGENT_UPDATE,
            source="agent_manager",
            data={
                "agent_id": "researcher_001",
                "agent_type": "cognitive",
                "name": "R1-Researcher",
                "status": "idle"
            }
        )
        
        assert agent_event["type"] == "agent_update"
        assert validate_event(agent_event) is True
    
    def test_schema_documentation(self):
        """Test that all schemas have proper documentation."""
        for event_type, schema in EVENT_SCHEMAS.items():
            # Check that schema has description
            assert "description" in schema, f"Schema for {event_type} missing description"
            
            # Check that required fields are documented
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    if prop_name in schema.get("required", []):
                        assert "description" in prop_schema, \
                            f"Required field {prop_name} in {event_type} missing description"
    
    def test_event_type_enum_coverage(self):
        """Test that EventType enum covers all schema types."""
        # Get all event types from enum
        enum_types = {e.value for e in EventType}
        
        # Get all event types from schemas
        schema_types = set(EVENT_SCHEMAS.keys())
        
        # They should match
        assert enum_types == schema_types, \
            f"Mismatch between enum and schemas. Enum: {enum_types}, Schemas: {schema_types}"
    
    def test_complex_nested_validation(self):
        """Test validation of complex nested data structures."""
        complex_event = {
            "type": "system_status",
            "source": "status_collector",
            "timestamp": time.time(),
            "data": {
                "layers": {
                    "atomic": {"status": "active", "confidence": 0.95},
                    "molecular": {"status": "active", "confidence": 0.88},
                    "cellular": {"status": "idle", "confidence": 0.0},
                    "organ": {"status": "active", "confidence": 0.92},
                    "cognitive": {"status": "active", "confidence": 0.90},
                    "programming": {"status": "idle", "confidence": 0.0},
                    "subagent": {"status": "active", "confidence": 0.85}
                },
                "agents": {
                    "native": [
                        {"id": "api-designer", "status": "idle"},
                        {"id": "code-reviewer", "status": "active"}
                    ],
                    "cognitive": [
                        {"id": "PE", "name": "Prompt-Enhancer", "status": "active"},
                        {"id": "R1", "name": "Researcher", "status": "idle"},
                        {"id": "A1", "name": "Reasoner", "status": "queued"}
                    ]
                },
                "memory": {
                    "stm": {"usage": 45, "items": 25},
                    "wm": {"usage": 12, "items": 150},
                    "ltm": {"usage": 3, "items": 1024}
                },
                "performance": {
                    "latency_ms": 8.5,
                    "cpu_percent": 1.2,
                    "memory_mb": 35.5
                }
            }
        }
        
        # Should validate successfully
        assert validate_event(complex_event) is True
    
    def test_timestamp_validation(self):
        """Test that timestamp validation works correctly."""
        # Valid timestamp (current time)
        valid_event = {
            "type": "layer_update",
            "source": "test",
            "timestamp": time.time(),
            "data": {"layer": "atomic", "status": "active"}
        }
        assert validate_event(valid_event) is True
        
        # Invalid timestamp (negative)
        invalid_event1 = {
            "type": "layer_update",
            "source": "test",
            "timestamp": -1,
            "data": {"layer": "atomic", "status": "active"}
        }
        with pytest.raises(ValidationError):
            validate_event(invalid_event1)
        
        # Invalid timestamp (string)
        invalid_event2 = {
            "type": "layer_update",
            "source": "test",
            "timestamp": "2024-01-01",
            "data": {"layer": "atomic", "status": "active"}
        }
        with pytest.raises(ValidationError):
            validate_event(invalid_event2)
        
        # Future timestamp (should be allowed)
        future_event = {
            "type": "layer_update",
            "source": "test",
            "timestamp": time.time() + 3600,  # 1 hour in future
            "data": {"layer": "atomic", "status": "active"}
        }
        assert validate_event(future_event) is True
    
    def test_priority_field_validation(self):
        """Test validation of optional priority field."""
        # Event with valid priority
        high_priority_event = {
            "type": "classification_update",
            "source": "classifier",
            "timestamp": time.time(),
            "priority": 5,
            "data": {
                "request_id": "req_123",
                "classification": "A",
                "confidence": 0.95
            }
        }
        assert validate_event(high_priority_event) is True
        
        # Event without priority (optional)
        no_priority_event = {
            "type": "classification_update",
            "source": "classifier",
            "timestamp": time.time(),
            "data": {
                "request_id": "req_123",
                "classification": "A",
                "confidence": 0.95
            }
        }
        assert validate_event(no_priority_event) is True
        
        # Invalid priority (out of range)
        invalid_priority = {
            "type": "classification_update",
            "source": "classifier",
            "timestamp": time.time(),
            "priority": 11,  # Should be 1-10
            "data": {
                "request_id": "req_123",
                "classification": "A",
                "confidence": 0.95
            }
        }
        with pytest.raises(ValidationError):
            validate_event(invalid_priority)
    
    def test_correlation_id_field(self):
        """Test validation of optional correlation_id for tracing."""
        # Event with correlation ID
        traced_event = {
            "type": "layer_update",
            "source": "atomic_layer",
            "timestamp": time.time(),
            "correlation_id": "trace_123456",
            "data": {
                "layer": "atomic",
                "status": "active"
            }
        }
        assert validate_event(traced_event) is True
        
        # Event without correlation ID (optional)
        untraced_event = {
            "type": "layer_update",
            "source": "atomic_layer",
            "timestamp": time.time(),
            "data": {
                "layer": "atomic",
                "status": "active"
            }
        }
        assert validate_event(untraced_event) is True