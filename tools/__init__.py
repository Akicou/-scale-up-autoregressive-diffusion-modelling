"""
Tool calling module for function calling in language models.

This module provides:
- ToolRegistry: Register and execute tools
- ToolCallingMixin: Parse and handle tool calls from model output
"""

from .registry import ToolRegistry, Tool, ToolParameter
from .parsing import ToolCallParser, detect_tool_calls, parse_tool_result

__all__ = [
    "ToolRegistry",
    "Tool",
    "ToolParameter",
    "ToolCallParser",
    "detect_tool_calls",
    "parse_tool_result",
]
