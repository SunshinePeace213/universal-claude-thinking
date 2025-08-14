"""
Sub-agent infrastructure for Universal Claude Thinking v2.

This module provides the foundation for Claude Code's native sub-agent
capabilities with isolated contexts, coordination protocols, and 
performance monitoring.
"""

from .base import BaseSubAgent
from .manager import SubAgentManager

__all__ = ['BaseSubAgent', 'SubAgentManager']