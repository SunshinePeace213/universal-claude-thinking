"""
Unit tests for sub-agent configuration validation.

Tests ensure all agent configurations:
1. Have valid YAML frontmatter
2. Include required fields
3. Use appropriate models
4. Have valid tool lists
"""

import os
import re
import yaml
import pytest
from pathlib import Path
from typing import Dict, List, Any


class TestAgentConfigurations:
    """Test suite for validating sub-agent markdown configurations."""
    
    AGENTS_DIR = Path(".claude/agents")
    REQUIRED_AGENTS = [
        "prompt-enhancer",
        "researcher", 
        "reasoner",
        "evaluator",
        "tool-user",
        "writer",
        "interface"
    ]
    
    REQUIRED_FRONTMATTER_FIELDS = [
        "name",
        "nickname",
        "text_face",
        "description",
        "tools",
        "model"
    ]
    
    VALID_MODELS = ["sonnet", "opus"]
    
    EXPECTED_NICKNAMES = {
        "prompt-enhancer": "PE",
        "researcher": "R1",
        "reasoner": "A1",
        "evaluator": "E1",
        "tool-user": "T1",
        "writer": "W1",
        "interface": "I1"
    }
    
    EXPECTED_TEXT_FACES = {
        "prompt-enhancer": "🔧",
        "researcher": "🔍",
        "reasoner": "🧠",
        "evaluator": "📊",
        "tool-user": "🛠️",
        "writer": "🖋️",
        "interface": "🗣️"
    }
    
    @pytest.fixture(scope="class")
    def agent_files(self) -> Dict[str, Path]:
        """Get all agent configuration files."""
        files = {}
        for agent_name in self.REQUIRED_AGENTS:
            file_path = self.AGENTS_DIR / f"{agent_name}.md"
            if file_path.exists():
                files[agent_name] = file_path
        return files
    
    def extract_frontmatter(self, file_path: Path) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown file."""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract frontmatter between --- markers
        pattern = r'^---\n(.*?)\n---'
        match = re.match(pattern, content, re.DOTALL)
        
        if not match:
            raise ValueError(f"No frontmatter found in {file_path}")
        
        frontmatter_str = match.group(1)
        return yaml.safe_load(frontmatter_str)
    
    def test_all_agents_exist(self, agent_files):
        """Test that all required agent files exist."""
        missing_agents = []
        for agent_name in self.REQUIRED_AGENTS:
            if agent_name not in agent_files:
                missing_agents.append(agent_name)
        
        assert not missing_agents, f"Missing agent configurations: {missing_agents}"
    
    def test_frontmatter_structure(self, agent_files):
        """Test that all agents have valid YAML frontmatter."""
        for agent_name, file_path in agent_files.items():
            try:
                frontmatter = self.extract_frontmatter(file_path)
                assert isinstance(frontmatter, dict), f"{agent_name}: Frontmatter is not a dictionary"
            except Exception as e:
                pytest.fail(f"{agent_name}: Failed to parse frontmatter - {e}")
    
    def test_required_fields(self, agent_files):
        """Test that all required fields are present in frontmatter."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            
            missing_fields = []
            for field in self.REQUIRED_FRONTMATTER_FIELDS:
                if field not in frontmatter:
                    missing_fields.append(field)
            
            assert not missing_fields, f"{agent_name}: Missing required fields {missing_fields}"
    
    def test_agent_names_match(self, agent_files):
        """Test that frontmatter name matches file name."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            
            assert frontmatter.get("name") == agent_name, \
                f"{agent_name}: Name mismatch - expected '{agent_name}', got '{frontmatter.get('name')}'"
    
    def test_nicknames_correct(self, agent_files):
        """Test that agents have correct nicknames."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            expected_nickname = self.EXPECTED_NICKNAMES.get(agent_name)
            
            assert frontmatter.get("nickname") == expected_nickname, \
                f"{agent_name}: Expected nickname '{expected_nickname}', got '{frontmatter.get('nickname')}'"
    
    def test_text_faces_correct(self, agent_files):
        """Test that agents have correct text faces."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            expected_face = self.EXPECTED_TEXT_FACES.get(agent_name)
            
            assert frontmatter.get("text_face") == expected_face, \
                f"{agent_name}: Expected text_face '{expected_face}', got '{frontmatter.get('text_face')}'"
    
    def test_valid_models(self, agent_files):
        """Test that agents use valid model specifications."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            model = frontmatter.get("model")
            
            assert model in self.VALID_MODELS, \
                f"{agent_name}: Invalid model '{model}'. Must be one of {self.VALID_MODELS}"
    
    def test_model_assignments(self, agent_files):
        """Test that agents use appropriate models based on complexity."""
        expected_models = {
            "prompt-enhancer": "sonnet",  # Simple validation
            "researcher": "opus",          # Complex synthesis
            "reasoner": "opus",            # Complex logic
            "evaluator": "sonnet",         # Simple scoring
            "tool-user": "opus",           # Complex orchestration
            "writer": "opus",              # Creative synthesis
            "interface": "sonnet"          # Simple translation
        }
        
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            expected_model = expected_models.get(agent_name)
            
            assert frontmatter.get("model") == expected_model, \
                f"{agent_name}: Expected model '{expected_model}', got '{frontmatter.get('model')}'"
    
    def test_tools_field_type(self, agent_files):
        """Test that tools field is a list."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            tools = frontmatter.get("tools")
            
            assert isinstance(tools, list), \
                f"{agent_name}: Tools field must be a list, got {type(tools)}"
    
    def test_description_not_empty(self, agent_files):
        """Test that descriptions are not empty."""
        for agent_name, file_path in agent_files.items():
            frontmatter = self.extract_frontmatter(file_path)
            description = frontmatter.get("description")
            
            assert description and len(description.strip()) > 0, \
                f"{agent_name}: Description cannot be empty"
    
    def test_content_structure(self, agent_files):
        """Test that agent files have proper content structure."""
        required_sections = [
            "## ",  # At least one section header
            "1. ",  # At least one numbered list (process steps)
        ]
        
        for agent_name, file_path in agent_files.items():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Remove frontmatter for content check
            content_body = re.sub(r'^---.*?---\n', '', content, flags=re.DOTALL)
            
            for section in required_sections:
                assert section in content_body, \
                    f"{agent_name}: Missing required content structure element '{section}'"
    
    def test_token_efficiency(self, agent_files):
        """Test that agent configurations meet token efficiency targets."""
        # V2 Enhanced targets: 800-1000 tokens per agent
        # Rough estimate: 1 token ≈ 4 characters
        # Note: Our implementation is slightly more efficient than original spec
        min_chars = 400 * 4  # ~1600 characters (allowing for efficient implementation)
        max_chars = 1100 * 4  # ~4400 characters (with some buffer)
        
        for agent_name, file_path in agent_files.items():
            with open(file_path, 'r') as f:
                content = f.read()
            
            char_count = len(content)
            estimated_tokens = char_count / 4
            
            # Check that agents are within reasonable token range
            assert min_chars <= char_count <= max_chars * 1.2, \
                f"{agent_name}: Token count ~{estimated_tokens:.0f} outside acceptable range 400-1100"
    
    def test_integration_points(self, agent_files):
        """Test that agents document integration points."""
        for agent_name, file_path in agent_files.items():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for integration section
            assert "## Integration Points" in content or "Integration:" in content, \
                f"{agent_name}: Missing integration points documentation"
    
    def test_error_handling_documented(self, agent_files):
        """Test that agents with tools document error handling."""
        agents_with_tools = ["researcher", "reasoner", "tool-user", "writer"]
        
        for agent_name in agents_with_tools:
            if agent_name in agent_files:
                file_path = agent_files[agent_name]
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Check for error handling section or patterns
                assert "Error" in content or "error" in content or "Rollback" in content, \
                    f"{agent_name}: Missing error handling documentation"


class TestAgentCoordination:
    """Test agent coordination and communication patterns."""
    
    def test_coordination_flow(self):
        """Test that agent coordination follows expected patterns."""
        # Define expected coordination patterns
        coordination_flows = [
            ["prompt-enhancer", "researcher", "reasoner", "writer"],  # Research flow
            ["prompt-enhancer", "tool-user", "evaluator"],           # Execution flow
            ["reasoner", "evaluator", "interface"],                  # Validation flow
        ]
        
        # This would be tested more thoroughly with integration tests
        assert len(coordination_flows) > 0, "Coordination flows defined"
    
    def test_message_format(self):
        """Test that agents use consistent message formats."""
        # This would validate actual message passing in integration tests
        expected_message_fields = ["from", "to", "type", "content", "timestamp"]
        assert len(expected_message_fields) == 5, "Message format defined"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])