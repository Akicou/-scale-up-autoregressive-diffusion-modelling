#!/usr/bin/env python3
"""
Tool registry for registering and executing tools.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None


@dataclass
class Tool:
    """Definition of a tool."""
    
    name: str
    description: str
    func: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate tool definition."""
        if not self.name:
            raise ValueError("Tool name cannot be empty")
        if not callable(self.func):
            raise ValueError(f"Tool function must be callable, got {type(self.func)}")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            
            if param.default is not None:
                prop["default"] = param.default
            
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required else None,
            },
        }
    
    def execute(self, args: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments."""
        # Validate arguments
        for param in self.parameters:
            if param.required and param.name not in args:
                raise ValueError(f"Missing required parameter: {param.name}")
        
        # Call function
        try:
            result = self.func(**args)
            return result
        except TypeError as e:
            raise ValueError(f"Error executing tool {self.name}: {e}")


class ToolRegistry:
    """
    Registry for managing and executing tools.
    
    Supports registering tools, executing them, and generating tool schemas
    for prompt injection.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str = "",
        parameters: Optional[List[Union[ToolParameter, Dict]]] = None,
    ) -> "ToolRegistry":
        """
        Register a tool.
        
        Args:
            name: Tool name
            func: Function to execute
            description: Tool description
            parameters: List of parameter definitions
        
        Returns:
            Self for chaining
        """
        # Convert dict params to ToolParameter
        params = []
        if parameters:
            for p in parameters:
                if isinstance(p, ToolParameter):
                    params.append(p)
                elif isinstance(p, dict):
                    params.append(ToolParameter(**p))
        
        tool = Tool(name=name, description=description, func=func, parameters=params)
        self._tools[name] = tool
        
        return self
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def execute(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool by name with given arguments.
        
        Args:
            name: Tool name
            args: Arguments to pass to the tool
        
        Returns:
            Tool execution result
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found. Available tools: {list(self._tools.keys())}")
        
        return tool.execute(args)
    
    def execute_from_string(self, tool_call_str: str) -> Any:
        """
        Parse and execute a tool call from a string.
        
        Supports:
        - JSON format: {"tool": "name", "args": {...}}
        - Python-like: tool::name(arg1=value1, arg2=value2)
        
        Args:
            tool_call_str: String containing tool call
        
        Returns:
            Execution result
        """
        # Try JSON format first
        try:
            data = json.loads(tool_call_str)
            if isinstance(data, dict):
                name = data.get("tool") or data.get("name")
                args = data.get("args") or data.get("arguments", {})
                return self.execute(name, args)
        except json.JSONDecodeError:
            pass
        
        # Try Python-like format
        match = re.search(r'tool::([^(]+)\((.*?)\)', tool_call_str)
        if match:
            name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments
            args = {}
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    
                    # Try to parse as JSON
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    
                    args[key] = value
            
            return self.execute(name, args)
        
        raise ValueError(f"Could not parse tool call: {tool_call_str}")
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_schemas_text(self) -> str:
        """Get formatted text describing all available tools."""
        schemas = self.get_schemas()
        if not schemas:
            return "No tools available."
        
        lines = ["Available tools:"]
        for schema in schemas:
            lines.append(f"\n- {schema['name']}: {schema['description']}")
            lines.append(f"  Parameters:")
            
            params = schema["parameters"]["properties"]
            for name, info in params.items():
                required = name in schema["parameters"].get("required", [])
                req_str = " (required)" if required else ""
                lines.append(f"    {name}: {info['type']}{req_str}")
                if info.get("description"):
                    lines.append(f"      Description: {info['description']}")
        
        return "\n".join(lines)
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())


# ============================================================================
# Built-in Tools
# ============================================================================

def calculate(expression: str) -> Union[int, float]:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression (e.g., "2+2", "sin(pi/2)")
    
    Returns:
        Result of the expression
    """
    import math
    
    # Safe evaluation with limited imports
    allowed_names = {
        "abs": abs,
        "max": max,
        "min": min,
        "pow": pow,
        "round": round,
        "sum": sum,
        "len": len,
        "sorted": sorted,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "math": math,
        "pi": math.pi,
        "e": math.e,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


def search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results
    
    Returns:
        List of search results with title, url, snippet
    """
    # Placeholder - in production, would use search API
    return [
        {
            "title": f"Search result for: {query}",
            "url": "https://example.com",
            "snippet": "This is a mock search result. In production, integrate with search API.",
        }
    ][:max_results]


def get_weather(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Get weather information for a location.
    
    Args:
        location: Location name
        unit: Temperature unit (celsius or fahrenheit)
    
    Returns:
        Weather information
    """
    # Placeholder - in production, would use weather API
    return {
        "location": location,
        "temperature": 20,
        "unit": unit,
        "condition": "sunny",
        "humidity": 50,
        "wind_speed": 10,
    }


def get_current_time(timezone: str = "UTC") -> Dict[str, str]:
    """
    Get current time for a timezone.
    
    Args:
        timezone: Timezone name (e.g., "UTC", "America/New_York")
    
    Returns:
        Current time information
    """
    from datetime import datetime
    
    try:
        import pytz
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
    except ImportError:
        # Fallback without pytz
        now = datetime.now()
    
    return {
        "timezone": timezone,
        "datetime": now.isoformat(),
        "timestamp": int(now.timestamp()),
    }


def python(code: str) -> str:
    """
    Execute Python code.
    
    WARNING: This is a security risk. Only use in sandboxed environments.
    
    Args:
        code: Python code to execute
    
    Returns:
        Execution result as string
    """
    import io
    import sys
    
    # Capture output
    stdout = io.StringIO()
    stderr = io.StringIO()
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout
    sys.stderr = stderr
    
    try:
        exec(code, {"__builtins__": {}})
        result = stdout.getvalue()
        if stderr.getvalue():
            result += "\nErrors: " + stderr.getvalue()
    except Exception as e:
        result = f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result or "Code executed successfully."


def get_default_registry() -> ToolRegistry:
    """Get a registry with default tools."""
    registry = ToolRegistry()
    
    registry.register(
        "calculate",
        calculate,
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParameter("expression", "string", "The expression to evaluate", required=True),
        ],
    )
    
    registry.register(
        "search",
        search,
        description="Search the web for information",
        parameters=[
            ToolParameter("query", "string", "Search query", required=True),
            ToolParameter("max_results", "integer", "Maximum results", required=False, default=5),
        ],
    )
    
    registry.register(
        "get_weather",
        get_weather,
        description="Get weather information for a location",
        parameters=[
            ToolParameter("location", "string", "Location name", required=True),
            ToolParameter("unit", "string", "Temperature unit", required=False, default="celsius", enum=["celsius", "fahrenheit"]),
        ],
    )
    
    registry.register(
        "get_current_time",
        get_current_time,
        description="Get current time for a timezone",
        parameters=[
            ToolParameter("timezone", "string", "Timezone name", required=False, default="UTC"),
        ],
    )
    
    # WARNING: Python execution is dangerous
    # Only enable in sandboxed environments
    # registry.register(
    #     "python",
    #     python,
    #     description="Execute Python code",
    #     parameters=[
    #         ToolParameter("code", "string", "Python code to execute", required=True),
    #     ],
    # )
    
    return registry
