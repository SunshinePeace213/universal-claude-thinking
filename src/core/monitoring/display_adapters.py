"""
Display adapters for different output formats.
Supports terminal (with rich formatting) and JSON (for statusline) outputs.
"""

import json
import shutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging

# Try to import rich, fall back to plain text if not available
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Table = None
    Panel = None
    Text = None

logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Abstract base class for display adapters."""
    
    @abstractmethod
    async def render(self, data: Dict[str, Any]) -> str:
        """Render data to specific format."""
        pass


class TerminalAdapter(BaseAdapter):
    """
    Terminal display adapter with rich formatting support.
    Falls back to plain text if rich is not available.
    """
    
    def __init__(self):
        """Initialize terminal adapter."""
        self._use_plain_text = not RICH_AVAILABLE
        if RICH_AVAILABLE:
            self._console = Console()
        else:
            self._console = None
            logger.warning("Rich library not available, using plain text output")
    
    async def render(self, data: Dict[str, Any]) -> str:
        """Render data for terminal display with formatting."""
        if self._use_plain_text:
            return self._render_plain_text(data)
        else:
            return self._render_rich(data)
    
    def _render_plain_text(self, data: Dict[str, Any]) -> str:
        """Render as plain text without rich formatting."""
        lines = []
        lines.append("=" * 60)
        lines.append(" WORKFLOW AGENTS (Claude Code Native)")
        lines.append("-" * 60)
        
        # Workflow agents
        if 'workflow_agents' in data:
            for agent in data.get('workflow_agents', []):
                if isinstance(agent, dict):
                    name = agent.get('name', 'Unknown')
                    status = agent.get('status', 'Unknown')
                    lines.append(f"  • {name}: {status}")
                else:
                    lines.append(f"  • {agent}")
        else:
            lines.append("  No workflow agents active")
        
        lines.append("")
        lines.append(" COGNITIVE PROCESSORS (Enhanced Architecture)")
        lines.append("-" * 60)
        
        # Cognitive processors
        if 'cognitive_processors' in data:
            for processor in data.get('cognitive_processors', []):
                if isinstance(processor, dict):
                    name = processor.get('name', 'Unknown')
                    status = processor.get('status', 'Unknown')
                    lines.append(f"  • {name}: {status}")
                else:
                    lines.append(f"  • {processor}")
        else:
            lines.append("  No cognitive processors active")
        
        lines.append("")
        lines.append(" RESOURCE USAGE")
        lines.append("-" * 60)
        
        # Resource usage
        context = data.get('context_usage', 'N/A')
        memory = data.get('memory_usage', 'N/A')
        lines.append(f"  Context: {context} | Memory: {memory}")
        
        # Additional status info
        if 'status' in data:
            lines.append(f"  Status: {data['status']}")
        if 'model' in data:
            lines.append(f"  Model: {data['model']}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _render_rich(self, data: Dict[str, Any]) -> str:
        """Render with rich formatting for enhanced terminal display."""
        # Get terminal width for adaptation
        terminal_width = shutil.get_terminal_size().columns
        
        # Create main container table
        main_table = Table(show_header=False, show_edge=False, 
                          width=min(terminal_width, 80), 
                          pad_edge=False)
        
        # Workflow Agents Section
        workflow_table = self._create_agent_table(
            "🎯 WORKFLOW AGENTS (Claude Code Native)",
            data.get('workflow_agents', []),
            "bold cyan"
        )
        
        # Cognitive Processors Section
        cognitive_table = self._create_agent_table(
            "🧠 COGNITIVE PROCESSORS (Enhanced Architecture)",
            data.get('cognitive_processors', []),
            "bold magenta"
        )
        
        # Resource Usage Section
        resource_text = self._create_resource_panel(data)
        
        # Combine sections
        main_table.add_row(workflow_table)
        main_table.add_row(cognitive_table)
        main_table.add_row(resource_text)
        
        # Render to string
        with self._console.capture() as capture:
            self._console.print(main_table)
        
        return capture.get()
    
    def _create_agent_table(self, title: str, agents: List, style: str) -> Panel:
        """Create a formatted table for agents/processors."""
        table = Table(show_header=False, show_edge=False, expand=True)
        table.add_column("Name", style="bold")
        table.add_column("Status")
        
        if agents:
            for agent in agents:
                if isinstance(agent, dict):
                    name = agent.get('name', 'Unknown')
                    status = agent.get('status', 'Unknown')
                    status_style = self._get_status_style(status)
                    table.add_row(f"• {name}", Text(status, style=status_style))
                else:
                    table.add_row(f"• {agent}", "")
        else:
            table.add_row("No agents active", "")
        
        return Panel(table, title=title, title_align="left", 
                    border_style=style, expand=True)
    
    def _create_resource_panel(self, data: Dict[str, Any]) -> Panel:
        """Create resource usage panel."""
        lines = []
        
        # Context usage
        context = data.get('context_usage', 'N/A')
        memory = data.get('memory_usage', 'N/A')
        
        # Create progress bar style display if numeric
        if isinstance(context, str) and '/' in context:
            try:
                used, total = context.split('/')
                used_int = int(used)
                total_int = int(total)
                percentage = (used_int / total_int) * 100
                bar_length = 20
                filled = int(bar_length * used_int / total_int)
                bar = '█' * filled + '░' * (bar_length - filled)
                context_display = f"[{bar}] {context} ({percentage:.1f}%)"
            except:
                context_display = context
        else:
            context_display = context
        
        lines.append(f"Context: {context_display}")
        lines.append(f"Memory: {memory}")
        
        # Additional info
        if 'model' in data:
            model_info = data['model']
            if isinstance(model_info, dict):
                lines.append(f"Model: {model_info.get('display_name', 'Unknown')}")
            else:
                lines.append(f"Model: {model_info}")
        
        resource_text = Text("\n".join(lines))
        return Panel(resource_text, title="📊 RESOURCE USAGE", 
                    title_align="left", border_style="bold green", 
                    expand=True)
    
    def _get_status_style(self, status: str) -> str:
        """Get style based on status value."""
        status_lower = status.lower()
        if 'active' in status_lower or 'running' in status_lower:
            return "bold green"
        elif 'idle' in status_lower or 'waiting' in status_lower:
            return "yellow"
        elif 'complete' in status_lower or 'done' in status_lower:
            return "blue"
        elif 'error' in status_lower or 'failed' in status_lower:
            return "bold red"
        elif 'queued' in status_lower:
            return "cyan"
        else:
            return "white"


class JSONAdapter(BaseAdapter):
    """
    JSON adapter for Claude Code statusline output.
    Formats data according to statusline schema.
    """
    
    async def render(self, data: Optional[Dict[str, Any]]) -> str:
        """Render data as JSON for statusline consumption."""
        if data is None:
            data = {}
        
        # Build statusline-compatible JSON structure
        output = {
            "hook_event_name": "Status",
            "session_id": data.get('session_id', 'unknown'),
            "cwd": data.get('cwd', '/'),
            "version": data.get('version', '1.0.0')
        }
        
        # Model information
        if 'model' in data:
            model_info = data['model']
            if isinstance(model_info, dict):
                output['model'] = model_info
            else:
                output['model'] = {
                    'id': str(model_info),
                    'display_name': str(model_info)
                }
        else:
            output['model'] = {
                'id': 'unknown',
                'display_name': 'Unknown'
            }
        
        # Workspace information
        if 'workspace' in data:
            output['workspace'] = data['workspace']
        else:
            output['workspace'] = {
                'current_dir': data.get('current_dir', '/'),
                'project_dir': data.get('project_dir', '/')
            }
        
        # Git information (if available)
        if 'git' in data:
            output['git'] = data['git']
        
        # Cost/metrics information
        if 'cost' in data:
            output['cost'] = data['cost']
        else:
            output['cost'] = {
                'total_lines_added': data.get('lines_added', 0),
                'total_lines_removed': data.get('lines_removed', 0)
            }
        
        # Cognitive state (custom extension)
        if 'cognitive_state' in data:
            output['cognitive_state'] = data['cognitive_state']
        else:
            # Build from available data
            cognitive_state = {
                'layers': [],
                'active_agents': []
            }
            
            # Extract active layers
            for key in ['layer', 'active_layer', 'layers']:
                if key in data:
                    if isinstance(data[key], list):
                        cognitive_state['layers'].extend(data[key])
                    else:
                        cognitive_state['layers'].append(str(data[key]))
            
            # Extract active agents
            if 'workflow_agents' in data:
                agents = data['workflow_agents']
                if isinstance(agents, list):
                    for agent in agents:
                        if isinstance(agent, dict) and agent.get('status') == 'Active':
                            cognitive_state['active_agents'].append(agent.get('name', 'Unknown'))
            
            if 'cognitive_processors' in data:
                processors = data['cognitive_processors']
                if isinstance(processors, list):
                    for proc in processors:
                        if isinstance(proc, dict) and proc.get('status') in ['Active', 'Running']:
                            cognitive_state['active_agents'].append(proc.get('name', 'Unknown'))
            
            if cognitive_state['layers'] or cognitive_state['active_agents']:
                output['cognitive_state'] = cognitive_state
        
        # Resource usage
        if 'context_usage' in data or 'memory_usage' in data:
            output['resources'] = {
                'context': data.get('context_usage', 'N/A'),
                'memory': data.get('memory_usage', 'N/A')
            }
        
        # Ensure valid JSON output
        try:
            return json.dumps(output, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            # Return minimal valid JSON on error
            return json.dumps({
                "error": "Serialization failed",
                "hook_event_name": "Status"
            })