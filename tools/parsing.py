#!/usr/bin/env python3
"""
Tool call parsing from model output.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ParsedToolCall:
    """A parsed tool call from model output."""
    
    tool_name: str
    arguments: Dict[str, Any]
    raw_text: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool_name,
            "args": self.arguments,
            "raw": self.raw_text,
        }


class ToolCallParser:
    """
    Parser for extracting tool calls from model output.
    
    Supports multiple formats:
    - JSON: {"tool": "name", "args": {...}}
    - XML: <tool_call name="name"><argument name="key">value</argument></tool_call>
    - Python-like: tool::name(arg1="value1")
    """
    
    def __init__(
        self,
        supported_formats: Optional[List[str]] = None,
        strict: bool = False,
    ):
        """
        Initialize parser.
        
        Args:
            supported_formats: List of formats to try ("json", "xml", "python")
            strict: If True, only parse exact format matches
        """
        self.supported_formats = supported_formats or ["json", "xml", "python"]
        self.strict = strict
    
    def parse(self, output: str) -> List[ParsedToolCall]:
        """
        Parse tool calls from output text.
        
        Args:
            output: Model output text
        
        Returns:
            List of parsed tool calls
        """
        tool_calls = []
        
        for fmt in self.supported_formats:
            if fmt == "json":
                tool_calls.extend(self._parse_json(output))
            elif fmt == "xml":
                tool_calls.extend(self._parse_xml(output))
            elif fmt == "python":
                tool_calls.extend(self._parse_python(output))
        
        return tool_calls
    
    def _parse_json(self, output: str) -> List[ParsedToolCall]:
        """Parse JSON format tool calls."""
        tool_calls = []
        
        # Pattern 1: {"tool": "name", "args": {...}}
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"([^"]+)"[^{}]*"args"\s*:\s*(\{[^}]*\})[^{}]*\}'
        
        for match in re.finditer(json_pattern, output, re.DOTALL):
            tool_name = match.group(1)
            args_str = match.group(2)
            
            try:
                args = json.loads(args_str)
                tool_calls.append(ParsedToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    raw_text=match.group(0),
                ))
            except json.JSONDecodeError:
                continue
        
        # Pattern 2: {"name": "tool_name", "arguments": {...}}
        if not tool_calls:
            alt_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^}]*\})[^{}]*\}'
            
            for match in re.finditer(alt_pattern, output, re.DOTALL):
                tool_name = match.group(1)
                args_str = match.group(2)
                
                try:
                    args = json.loads(args_str)
                    tool_calls.append(ParsedToolCall(
                        tool_name=tool_name,
                        arguments=args,
                        raw_text=match.group(0),
                    ))
                except json.JSONDecodeError:
                    continue
        
        # Try to find any JSON object with tool/name and args/arguments
        if not tool_calls and not self.strict:
            # Look for potential JSON blocks
            json_candidates = re.findall(r'\{[^{}]*\}', output)
            for candidate in json_candidates:
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        tool_name = data.get("tool") or data.get("name")
                        args = data.get("args") or data.get("arguments") or data.get("params", {})
                        
                        if tool_name and args:
                            tool_calls.append(ParsedToolCall(
                                tool_name=tool_name,
                                arguments=args,
                                raw_text=candidate,
                            ))
                except json.JSONDecodeError:
                    continue
        
        return tool_calls
    
    def _parse_xml(self, output: str) -> List[ParsedToolCall]:
        """Parse XML format tool calls."""
        tool_calls = []
        
        # Pattern: <tool_call name="name">...</tool_call>
        # or: <tool name="name">...</tool>
        xml_pattern = r'<(?:tool_call|tool)\s+name="([^"]+)"[^>]*>(.*?)</(?:tool_call|tool)>'
        
        for match in re.finditer(xml_pattern, output, re.DOTALL):
            tool_name = match.group(1)
            content = match.group(2)
            
            # Parse arguments from content
            args = {}
            
            # Pattern: <argument name="key">value</argument>
            arg_pattern = r'<argument\s+name="([^"]+)">([^<]*)</argument>'
            
            for arg_match in re.finditer(arg_pattern, content):
                key = arg_match.group(1)
                value = arg_match.group(2).strip()
                
                # Try to parse as JSON
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
                
                args[key] = value
            
            # Also look for key="value" format
            kv_pattern = r'(?:name|key)="([^"]+)"(?:\s*:)?\s*"([^"]+)"'
            if not args:
                for kv_match in re.finditer(kv_pattern, content):
                    key = kv_match.group(1)
                    value = kv_match.group(2)
                    
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    
                    args[key] = value
            
            if args:
                tool_calls.append(ParsedToolCall(
                    tool_name=tool_name,
                    arguments=args,
                    raw_text=match.group(0),
                ))
        
        return tool_calls
    
    def _parse_python(self, output: str) -> List[ParsedToolCall]:
        """Parse Python-like tool calls."""
        tool_calls = []
        
        # Pattern: tool::name(arg1="value1", arg2="value2")
        py_pattern = r'tool::([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)'
        
        for match in re.finditer(py_pattern, output, re.DOTALL):
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments
            args = {}
            
            # Split by comma, respecting quotes
            current_key = None
            current_value = ""
            in_quotes = False
            quote_char = None
            
            for char in args_str:
                if char in ('"', "'") and not in_quotes:
                    in_quotes = True
                    quote_char = char
                    current_value = ""
                elif char == quote_char and in_quotes:
                    in_quotes = False
                    quote_char = None
                    
                    # Try to parse as JSON
                    try:
                        current_value = json.loads(f'"{current_value}"')
                    except json.JSONDecodeError:
                        pass
                    
                    if current_key:
                        args[current_key] = current_value
                        current_key = None
                    current_value = ""
                elif char == "=" and not in_quotes:
                    current_key = current_value.strip()
                    current_value = ""
                elif char == "," and not in_quotes:
                    if current_key:
                        try:
                            current_value = json.loads(f'"{current_value.strip()}"')
                        except json.JSONDecodeError:
                            pass
                        args[current_key] = current_value
                    current_key = None
                    current_value = ""
                else:
                    current_value += char
            
            # Handle last argument
            if current_key and current_value:
                try:
                    current_value = json.loads(f'"{current_value.strip()}"')
                except json.JSONDecodeError:
                    pass
                args[current_key] = current_value
            
            tool_calls.append(ParsedToolCall(
                tool_name=tool_name,
                arguments=args,
                raw_text=match.group(0),
            ))
        
        return tool_calls


def detect_tool_calls(output: str) -> bool:
    """
    Detect if output contains any tool calls.
    
    Args:
        output: Model output text
    
    Returns:
        True if tool calls are detected
    """
    parser = ToolCallParser()
    return len(parser.parse(output)) > 0


def parse_tool_result(output: str) -> Optional[List[ParsedToolCall]]:
    """
    Convenience function to parse tool calls.
    
    Args:
        output: Model output text
    
    Returns:
        List of parsed tool calls or None
    """
    parser = ToolCallParser()
    return parser.parse(output)


# ============================================================================
# Tool Calling Mixin for Models
# ============================================================================

class ToolCallingMixin:
    """
    Mixin class for handling tool calling in language models.
    
    Use this with your model wrapper to add tool calling capabilities.
    """
    
    def __init__(self, tool_registry: Optional[Any] = None):
        self.tool_registry = tool_registry
        self.parser = ToolCallParser()
        self._conversation_history: List[Dict[str, str]] = []
    
    def add_tool_schema_to_prompt(self) -> str:
        """Add tool schemas to system prompt."""
        if not self.tool_registry:
            return ""
        
        return self.tool_registry.get_schemas_text()
    
    def handle_tool_call(self, output: str) -> str:
        """
        Handle tool calls in model output.
        
        Args:
            output: Model output that may contain tool calls
        
        Returns:
            Output with tool results inserted
        """
        tool_calls = self.parser.parse(output)
        
        if not tool_calls or not self.tool_registry:
            return output
        
        # Execute tool calls
        results = []
        for tc in tool_calls:
            try:
                result = self.tool_registry.execute(tc.tool_name, tc.arguments)
                results.append({
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result": result,
                })
            except Exception as e:
                results.append({
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "error": str(e),
                })
        
        # Format results for insertion
        result_text = "\n".join([
            f"Result: {json.dumps(r['result'])}" if "result" in r else f"Error: {r.get('error')}"
            for r in results
        ])
        
        # Append to output
        return f"{output}\n\n{result_text}"
    
    def update_conversation_history(self, role: str, content: str):
        """Update conversation history."""
        self._conversation_history.append({"role": role, "content": content})
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self._conversation_history = []


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test the parser
    from registry import get_default_registry
    
    # Get default tools
    registry = get_default_registry()
    print("Available tools:", registry.list_tools())
    print("\nTool schemas:")
    print(registry.get_schemas_text())
    
    # Test parsing
    parser = ToolCallParser()
    
    # Test JSON format
    test_json = 'Let me calculate that: {"tool": "calculate", "args": {"expression": "2+2"}}'
    print("\n--- JSON Test ---")
    print(f"Input: {test_json}")
    calls = parser.parse(test_json)
    for call in calls:
        print(f"Parsed: {call.tool_name}, args: {call.arguments}")
    
    # Test XML format
    test_xml = 'Let me search: <tool_call name="search"><argument name="query">Python</argument></tool_call>'
    print("\n--- XML Test ---")
    print(f"Input: {test_xml}")
    calls = parser.parse(test_xml)
    for call in calls:
        print(f"Parsed: {call.tool_name}, args: {call.arguments}")
    
    # Test execution
    print("\n--- Execution Test ---")
    result = registry.execute_from_string('{"tool": "calculate", "args": {"expression": "2**10"}}')
    print(f"2**10 = {result}")
