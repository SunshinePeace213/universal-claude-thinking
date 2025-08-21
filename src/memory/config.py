"""
Memory system configuration management.

Provides centralized configuration loading and validation for the
5-layer memory system with support for YAML-based configuration.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import yaml

# Singleton instance storage
_config_instance: Optional['MemoryConfig'] = None
_config_lock = Lock()


@dataclass
class MemoryConfig:
    """
    Configuration for the memory system.
    
    Holds all configuration parameters for the 5-layer memory architecture
    including TTLs, thresholds, and performance requirements.
    """

    stm: dict[str, Any] = field(default_factory=dict)
    wm: dict[str, Any] = field(default_factory=dict)
    ltm: dict[str, Any] = field(default_factory=dict)
    swarm: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    scoring: dict[str, Any] = field(default_factory=dict)
    storage: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get nested configuration value using dot notation.
        
        Args:
            key: Dot-separated path to configuration value (e.g., 'stm.ttl_hours')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        parts = key.split('.')
        value = self

        try:
            for part in parts:
                if isinstance(value, MemoryConfig):
                    value = getattr(value, part)
                elif isinstance(value, dict):
                    value = value[part]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default

    def update(self, key: str, value: Any) -> None:
        """
        Update nested configuration value using dot notation.
        
        Args:
            key: Dot-separated path to configuration value
            value: New value to set
        """
        parts = key.split('.')
        if len(parts) < 2:
            raise ValueError(f"Invalid configuration key: {key}")

        # Navigate to parent
        parent = self
        for part in parts[:-2]:
            parent = getattr(parent, part)

        # Get the container (should be a dict)
        container_name = parts[-2]
        container = getattr(parent, container_name)

        # Set the value
        if isinstance(container, dict):
            container[parts[-1]] = value
        else:
            raise ValueError(f"Cannot update non-dict configuration: {container_name}")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)


def load_config(config_path: str | None = None) -> MemoryConfig:
    """
    Load configuration from YAML file.
    
    Uses singleton pattern to ensure only one configuration instance exists.
    
    Args:
        config_path: Path to configuration file (uses default if None)
        
    Returns:
        MemoryConfig instance
        
    Raises:
        FileNotFoundError: If configuration file not found
        yaml.YAMLError: If configuration file is invalid
    """
    global _config_instance

    # Return existing instance if available
    if _config_instance is not None and config_path is None:
        return _config_instance

    # Determine config path
    if config_path is None:
        config_path = "config/memory_config.yaml"

    config_file = Path(config_path)

    # Check if file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML configuration
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    # Extract memory configuration section
    if 'memory' not in config_data:
        raise ValueError("Configuration file missing 'memory' section")

    memory_config = config_data['memory']

    # Create MemoryConfig instance
    with _config_lock:
        _config_instance = MemoryConfig(**memory_config)
        return _config_instance


def validate_config(config: MemoryConfig) -> None:
    """
    Validate configuration values.
    
    Checks that all configuration parameters are within valid ranges
    and have appropriate relationships.
    
    Args:
        config: MemoryConfig instance to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate STM configuration
    if config.stm.get('ttl_hours', 0) <= 0:
        raise ValueError("STM TTL must be positive")

    if config.stm.get('cache_size', 0) <= 0:
        raise ValueError("STM cache size must be positive")

    # Validate WM configuration
    if config.wm.get('ttl_days', 0) <= 0:
        raise ValueError("WM TTL must be positive")

    promotion_threshold = config.wm.get('promotion_threshold', 5.0)
    if not 0.0 <= promotion_threshold <= 10.0:
        raise ValueError("Promotion threshold must be between 0.0 and 10.0")

    # Validate LTM configuration
    promotion_score = config.ltm.get('promotion_score', 8.0)
    if not 0.0 <= promotion_score <= 10.0:
        raise ValueError("LTM promotion score must be between 0.0 and 10.0")

    if config.ltm.get('promotion_uses', 0) <= 0:
        raise ValueError("LTM promotion uses must be positive")

    # Validate scoring configuration
    scoring = config.scoring
    min_score = scoring.get('min_score', 0.0)
    max_score = scoring.get('max_score', 10.0)
    default_score = scoring.get('default_score', 5.0)

    if min_score >= max_score:
        raise ValueError("min_score must be less than max_score")

    if not min_score <= default_score <= max_score:
        raise ValueError(f"default_score must be between {min_score} and {max_score}")

    if scoring.get('positive_adjustment', 0) <= 0:
        raise ValueError("positive_adjustment must be positive")

    if scoring.get('negative_adjustment', 0) >= 0:
        raise ValueError("negative_adjustment must be negative")

    # Validate storage configuration
    if config.storage.get('connection_pool_size', 0) <= 0:
        raise ValueError("Connection pool size must be positive")

    if config.storage.get('batch_insert_size', 0) <= 0:
        raise ValueError("Batch insert size must be positive")

    # Validate embedding configuration
    if config.embedding.get('dimension', 0) <= 0:
        raise ValueError("Embedding dimension must be positive")

    if config.embedding.get('batch_size', 0) <= 0:
        raise ValueError("Embedding batch size must be positive")

    # Validate performance requirements
    if config.performance.get('max_retrieval_latency_ms', 0) <= 0:
        raise ValueError("Max retrieval latency must be positive")

    if config.performance.get('max_promotion_latency_ms', 0) <= 0:
        raise ValueError("Max promotion latency must be positive")

    if config.performance.get('max_pii_detection_latency_ms', 0) <= 0:
        raise ValueError("Max PII detection latency must be positive")


def reset_config() -> None:
    """
    Reset the singleton configuration instance.
    
    Mainly used for testing purposes to ensure clean state.
    """
    global _config_instance
    with _config_lock:
        _config_instance = None
