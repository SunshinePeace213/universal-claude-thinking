"""
Message schemas for event validation in the Dynamic Header System.
Ensures all events conform to expected structure for reliability.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from jsonschema import ValidationError, validate


class EventType(Enum):
    """Event types matching the schemas."""
    LAYER_UPDATE = "layer_update"
    AGENT_UPDATE = "agent_update"
    MEMORY_UPDATE = "memory_update"
    CLASSIFICATION_UPDATE = "classification_update"
    PERFORMANCE_UPDATE = "performance_update"
    RESOURCE_UPDATE = "resource_update"
    STATUS_UPDATE = "status_update"
    SYSTEM_STATUS = "system_status"


@dataclass
class StatusUpdateEvent:
    """
    Event class for status updates in the monitoring system.
    Used by tests and EventBus for type-safe event handling.
    """
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    source: str = "unknown"
    priority: int = 5
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for EventBus."""
        result = {
            "type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "data": self.data,
            "priority": self.priority
        }
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        return result
    
    @classmethod
    def from_dict(cls, event_dict: Dict[str, Any]) -> 'StatusUpdateEvent':
        """Create from dictionary format."""
        return cls(
            event_type=event_dict.get("type", "status_update"),
            source=event_dict.get("source", "unknown"),
            timestamp=event_dict.get("timestamp", time.time()),
            data=event_dict.get("data", {}),
            priority=event_dict.get("priority", 5),
            correlation_id=event_dict.get("correlation_id")
        )


# JSON Schema definitions for each event type
EVENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "layer_update": {
        "type": "object",
        "description": "Event for cognitive layer status updates",
        "properties": {
            "type": {"type": "string", "enum": ["layer_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "layer": {
                        "type": "string",
                        "enum": ["atomic", "molecular", "cellular", "organ", "cognitive", "programming", "subagent"]
                    },
                    "status": {
                        "type": "string",
                        "enum": ["active", "idle", "processing", "error"]
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "processing_time_ms": {"type": "number", "minimum": 0},
                            "tokens_processed": {"type": "integer", "minimum": 0}
                        }
                    }
                },
                "required": ["layer", "status"]
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "agent_update": {
        "type": "object",
        "description": "Event for sub-agent status updates",
        "properties": {
            "type": {"type": "string", "enum": ["agent_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "agent_id": {"type": "string", "description": "Unique agent identifier"},
                    "agent_type": {
                        "type": "string",
                        "enum": ["native", "cognitive"],
                        "description": "Native Claude Code or cognitive processor"
                    },
                    "name": {"type": "string", "description": "Human-readable agent name"},
                    "status": {
                        "type": "string",
                        "enum": ["active", "idle", "queued", "error"]
                    },
                    "current_task": {"type": "string"},
                    "progress": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["agent_id", "status"]
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "memory_update": {
        "type": "object",
        "description": "Event for memory system updates",
        "properties": {
            "type": {"type": "string", "enum": ["memory_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "stm_usage_percent": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 100.0
                    },
                    "wm_usage_percent": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 100.0
                    },
                    "ltm_usage_percent": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 100.0
                    },
                    "total_items": {
                        "type": "object",
                        "properties": {
                            "stm": {"type": "integer", "minimum": 0},
                            "wm": {"type": "integer", "minimum": 0},
                            "ltm": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "classification_update": {
        "type": "object",
        "description": "Event for request classification updates",
        "properties": {
            "type": {"type": "string", "enum": ["classification_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "request_id": {"type": "string"},
                    "classification": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D", "E"],
                        "description": "Request type classification"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "required_tools": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "reasoning": {"type": "string"}
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "performance_update": {
        "type": "object",
        "description": "Event for performance metrics updates",
        "properties": {
            "type": {"type": "string", "enum": ["performance_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "latency_ms": {"type": "number", "minimum": 0},
                    "cpu_percent": {"type": "number", "minimum": 0, "maximum": 100},
                    "memory_mb": {"type": "number", "minimum": 0}
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "resource_update": {
        "type": "object",
        "description": "Event for resource utilization updates",
        "properties": {
            "type": {"type": "string", "enum": ["resource_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "cpu_cores": {"type": "integer", "minimum": 0},
                    "memory_available_mb": {"type": "number", "minimum": 0},
                    "disk_usage_percent": {"type": "number", "minimum": 0, "maximum": 100}
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "status_update": {
        "type": "object",
        "description": "Event for general status updates",
        "properties": {
            "type": {"type": "string", "enum": ["status_update"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "component": {"type": "string"},
                    "status": {"type": "string"},
                    "message": {"type": "string"}
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    },

    "system_status": {
        "type": "object",
        "description": "Comprehensive system status event",
        "properties": {
            "type": {"type": "string", "enum": ["system_status"], "description": "Event type identifier"},
            "source": {"type": "string", "description": "Source of the event"},
            "timestamp": {"type": "number", "minimum": 0, "description": "Unix timestamp"},
            "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            "correlation_id": {"type": "string"},
            "data": {
                "type": "object",
                "description": "Event-specific data payload",
                "properties": {
                    "layers": {
                        "type": "object",
                        "patternProperties": {
                            "^(atomic|molecular|cellular|organ|cognitive|programming|subagent)$": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string"},
                                    "confidence": {"type": "number"}
                                }
                            }
                        }
                    },
                    "agents": {
                        "type": "object",
                        "properties": {
                            "native": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "status": {"type": "string"}
                                    }
                                }
                            },
                            "cognitive": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "name": {"type": "string"},
                                        "status": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "memory": {
                        "type": "object",
                        "properties": {
                            "stm": {
                                "type": "object",
                                "properties": {
                                    "usage": {"type": "number"},
                                    "items": {"type": "integer"}
                                }
                            },
                            "wm": {
                                "type": "object",
                                "properties": {
                                    "usage": {"type": "number"},
                                    "items": {"type": "integer"}
                                }
                            },
                            "ltm": {
                                "type": "object",
                                "properties": {
                                    "usage": {"type": "number"},
                                    "items": {"type": "integer"}
                                }
                            }
                        }
                    },
                    "performance": {
                        "type": "object",
                        "properties": {
                            "latency_ms": {"type": "number", "minimum": 0},
                            "cpu_percent": {"type": "number", "minimum": 0, "maximum": 100},
                            "memory_mb": {"type": "number", "minimum": 0}
                        }
                    }
                }
            }
        },
        "required": ["type", "source", "timestamp", "data"]
    }
}


def validate_event(event: dict[str, Any], allow_additional: bool = False) -> bool:
    """
    Validate an event against its schema.
    
    Args:
        event: Event dictionary to validate
        allow_additional: Whether to allow additional properties
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If event is invalid
    """
    event_type = event.get("type")

    if event_type not in EVENT_SCHEMAS:
        raise ValidationError(f"Unknown event type: {event_type}")

    schema = EVENT_SCHEMAS[event_type]

    # Create validator with or without additional properties
    if not allow_additional:
        schema = {**schema, "additionalProperties": False}

    # Validate timestamp if present
    if "timestamp" in event:
        if not isinstance(event["timestamp"], (int, float)):
            raise ValidationError("timestamp must be a number")
        if event["timestamp"] < 0:
            raise ValidationError("timestamp cannot be negative")

    # Validate priority if present
    if "priority" in event:
        if not isinstance(event["priority"], int):
            raise ValidationError("priority must be an integer")
        if not 1 <= event["priority"] <= 10:
            raise ValidationError("priority must be between 1 and 10")

    # Validate against schema
    validate(event, schema)

    return True


def create_event(
    event_type: EventType,
    source: str,
    data: dict[str, Any],
    priority: int | None = None,
    correlation_id: str | None = None
) -> dict[str, Any]:
    """
    Create a valid event dictionary.
    
    Args:
        event_type: Type of event
        source: Source of the event
        data: Event data
        priority: Optional priority (1-10)
        correlation_id: Optional correlation ID for tracing
        
    Returns:
        Valid event dictionary
    """
    event = {
        "type": event_type.value if isinstance(event_type, EventType) else event_type,
        "source": source,
        "timestamp": time.time(),
        "data": data
    }

    if priority is not None:
        event["priority"] = priority

    if correlation_id is not None:
        event["correlation_id"] = correlation_id

    return event
