"""
Unit tests for memory system configuration.

Tests the configuration loading, validation, and access patterns for
the memory system configuration management.
"""

import pytest
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, mock_open
import yaml

from src.memory.config import MemoryConfig, load_config, validate_config


class TestMemoryConfig:
    """Test suite for memory configuration management."""
    
    @pytest.fixture
    def sample_config(self) -> Dict[str, Any]:
        """Provide sample configuration for testing."""
        return {
            'memory': {
                'stm': {
                    'ttl_hours': 2,
                    'cache_size': 1000,
                    'max_memories': 500
                },
                'wm': {
                    'ttl_days': 7,
                    'promotion_threshold': 5.0,
                    'max_memories': 2000
                },
                'ltm': {
                    'promotion_score': 8.0,
                    'promotion_uses': 5,
                    'max_memories': 10000
                },
                'swarm': {
                    'enabled': False,
                    'anonymization_required': True
                },
                'privacy': {
                    'pii_detection': True,
                    'anonymization': True,
                    'pii_patterns': ['email', 'phone', 'ssn'],
                    'spacy_model': 'en_core_web_lg'
                },
                'promotion': {
                    'stm_check_interval': 3600,
                    'wm_check_interval': 86400,
                    'batch_size': 100
                },
                'scoring': {
                    'default_score': 5.0,
                    'positive_adjustment': 0.3,
                    'negative_adjustment': -0.3,
                    'min_score': 0.0,
                    'max_score': 10.0
                },
                'storage': {
                    'database_path': 'data/memories/thinking_v2.db',
                    'connection_pool_size': 10,
                    'cache_ttl_seconds': 300,
                    'batch_insert_size': 100
                },
                'embedding': {
                    'model_path': 'embedding/Qwen3-Embedding-8B',
                    'dimension': 4096,
                    'batch_size': 32,
                    'use_mps': True
                },
                'performance': {
                    'max_retrieval_latency_ms': 100,
                    'max_promotion_latency_ms': 500,
                    'max_pii_detection_latency_ms': 50
                }
            }
        }
    
    def test_memory_config_initialization(self, sample_config):
        """Test MemoryConfig dataclass initialization."""
        config_data = sample_config['memory']
        config = MemoryConfig(**config_data)
        
        # Test STM configuration
        assert config.stm['ttl_hours'] == 2
        assert config.stm['cache_size'] == 1000
        assert config.stm['max_memories'] == 500
        
        # Test WM configuration
        assert config.wm['ttl_days'] == 7
        assert config.wm['promotion_threshold'] == 5.0
        
        # Test LTM configuration
        assert config.ltm['promotion_score'] == 8.0
        assert config.ltm['promotion_uses'] == 5
        
        # Test privacy configuration
        assert config.privacy['pii_detection'] is True
        assert 'email' in config.privacy['pii_patterns']
        
        # Test scoring configuration
        assert config.scoring['default_score'] == 5.0
        assert config.scoring['positive_adjustment'] == 0.3
    
    def test_load_config_from_file(self, sample_config, tmp_path):
        """Test loading configuration from YAML file."""
        # Create temporary config file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml.dump(sample_config))
        
        # Load configuration
        config = load_config(str(config_file))
        
        assert isinstance(config, MemoryConfig)
        assert config.stm['ttl_hours'] == 2
        assert config.embedding['dimension'] == 4096
    
    def test_load_config_default_path(self, sample_config):
        """Test loading configuration from default path."""
        yaml_content = yaml.dump(sample_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config = load_config()
                
                assert isinstance(config, MemoryConfig)
                assert config.storage['database_path'] == 'data/memories/thinking_v2.db'
    
    def test_load_config_file_not_found(self):
        """Test handling of missing configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
    
    def test_validate_config_valid(self, sample_config):
        """Test validation of valid configuration."""
        config_data = sample_config['memory']
        config = MemoryConfig(**config_data)
        
        # Should not raise any exception
        validate_config(config)
    
    def test_validate_config_invalid_ttl(self, sample_config):
        """Test validation catches invalid TTL values."""
        config_data = sample_config['memory']
        config_data['stm']['ttl_hours'] = -1
        config = MemoryConfig(**config_data)
        
        with pytest.raises(ValueError, match="STM TTL must be positive"):
            validate_config(config)
    
    def test_validate_config_invalid_threshold(self, sample_config):
        """Test validation catches invalid threshold values."""
        config_data = sample_config['memory']
        config_data['wm']['promotion_threshold'] = 11.0
        config = MemoryConfig(**config_data)
        
        with pytest.raises(ValueError, match="Promotion threshold must be between"):
            validate_config(config)
    
    def test_validate_config_invalid_scoring(self, sample_config):
        """Test validation catches invalid scoring parameters."""
        config_data = sample_config['memory']
        config_data['scoring']['min_score'] = 5.0
        config_data['scoring']['max_score'] = 5.0
        config = MemoryConfig(**config_data)
        
        with pytest.raises(ValueError, match="min_score must be less than max_score"):
            validate_config(config)
    
    def test_config_get_method(self, sample_config):
        """Test convenient get method for nested configuration."""
        config_data = sample_config['memory']
        config = MemoryConfig(**config_data)
        
        # Test nested access
        assert config.get('stm.ttl_hours') == 2
        assert config.get('wm.promotion_threshold') == 5.0
        assert config.get('scoring.default_score') == 5.0
        
        # Test with default value
        assert config.get('nonexistent.key', default=100) == 100
    
    def test_config_update_method(self, sample_config):
        """Test updating configuration values."""
        config_data = sample_config['memory']
        config = MemoryConfig(**config_data)
        
        # Update nested value
        config.update('stm.ttl_hours', 3)
        assert config.stm['ttl_hours'] == 3
        
        # Update scoring parameter
        config.update('scoring.default_score', 6.0)
        assert config.scoring['default_score'] == 6.0
    
    def test_config_to_dict(self, sample_config):
        """Test converting configuration to dictionary."""
        config_data = sample_config['memory']
        config = MemoryConfig(**config_data)
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['stm']['ttl_hours'] == 2
        assert config_dict['embedding']['dimension'] == 4096
    
    def test_config_singleton_pattern(self, sample_config):
        """Test that configuration uses singleton pattern."""
        yaml_content = yaml.dump(sample_config)
        
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            with patch('pathlib.Path.exists', return_value=True):
                config1 = load_config()
                config2 = load_config()
                
                # Should return the same instance
                assert config1 is config2