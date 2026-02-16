# Copyright 2025 Apple Inc.

"""
Tool schema to grammar conversion with auto-format detection.

This module converts tool definitions (JSON schema format) into Lark grammars
that can be used for grammar-constrained generation. It automatically detects
the tool call format from the model's chat template.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .jinja_analysis import (
    ToolCallFormat,
    analyze_chat_template,
    get_tool_format_from_tokenizer,
)


def build_tool_grammar(
    tools: List[Dict[str, Any]],
    tokenizer=None,
    format_override: Optional[ToolCallFormat] = None,
) -> str:
    """
    Build a Lark grammar for tool calling from tool definitions.

    This function automatically detects the tool call format from the
    tokenizer's chat template and generates a grammar that matches.

    Args:
        tools: List of tool definitions with 'name' and 'parameters' fields.
        tokenizer: Optional tokenizer to detect format from chat template.
        format_override: Optional explicit format specification.

    Returns:
        Lark grammar string for llguidance.

    Example::

        tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "units": {"enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        ]

        grammar = build_tool_grammar(tools, tokenizer)
        # Returns Lark grammar matching the tokenizer's format
    """
    # Detect format from tokenizer or use override
    if format_override:
        fmt = format_override
    elif tokenizer:
        fmt = get_tool_format_from_tokenizer(tokenizer)
    else:
        fmt = None

    # Default to JSON format if not detected
    if not fmt:
        fmt = ToolCallFormat(format_type="json")

    # Generate grammar based on format type
    if fmt.format_type == "json":
        return _build_json_grammar(tools, fmt)
    elif fmt.format_type == "xml":
        return _build_xml_grammar(tools, fmt)
    elif fmt.format_type == "python":
        return _build_python_grammar(tools, fmt)
    else:
        # Default to JSON
        return _build_json_grammar(tools, fmt)


def _build_json_grammar(
    tools: List[Dict[str, Any]],
    fmt: ToolCallFormat,
) -> str:
    """Build JSON-style tool call grammar."""
    # Build JSON schema for tool calls
    tool_schemas = []
    for tool in tools:
        tool_schema = _build_single_tool_schema(tool, fmt)
        tool_schemas.append(tool_schema)

    # Combine into anyOf
    if len(tool_schemas) == 1:
        combined_schema = tool_schemas[0]
    else:
        combined_schema = {"anyOf": tool_schemas}

    # Build Lark grammar with markers
    grammar_parts = []

    # Start rule with optional marker
    if fmt.start_marker:
        # Escape special characters for Lark
        escaped_start = _escape_for_lark(fmt.start_marker)
        grammar_parts.append(f'start: "{escaped_start}" tool_call')
    else:
        grammar_parts.append("start: tool_call")

    # Add end marker if present
    if fmt.end_marker:
        escaped_end = _escape_for_lark(fmt.end_marker)
        grammar_parts[0] += f' "{escaped_end}"'

    # Use %json directive for the tool call body
    schema_json = json.dumps(combined_schema)
    grammar_parts.append(f"tool_call: %json{{{schema_json}}}")

    return "\n".join(grammar_parts)


def _build_single_tool_schema(
    tool: Dict[str, Any],
    fmt: ToolCallFormat,
) -> Dict[str, Any]:
    """Build JSON schema for a single tool."""
    name = tool.get("name", "unknown")
    parameters = tool.get("parameters", {"type": "object", "properties": {}})

    # Handle nested function field (OpenAI style)
    if "." in fmt.name_field:
        # e.g., function.name -> nested structure
        return {
            "type": "object",
            "properties": {
                "function": {
                    "type": "object",
                    "properties": {
                        "name": {"const": name},
                        "arguments": parameters,
                    },
                    "required": ["name", "arguments"],
                }
            },
            "required": ["function"],
        }

    # Standard flat structure
    return {
        "type": "object",
        "properties": {
            fmt.name_field: {"const": name},
            fmt.arguments_field: parameters,
        },
        "required": [fmt.name_field, fmt.arguments_field],
    }


def _build_xml_grammar(
    tools: List[Dict[str, Any]],
    fmt: ToolCallFormat,
) -> str:
    """Build XML-style tool call grammar (e.g., <function=name>args</function>)."""
    # Build choice of tool names
    tool_names = [f'"{tool["name"]}"' for tool in tools]
    name_choice = " | ".join(tool_names)

    # Build parameter grammars for each tool
    tool_rules = []
    for tool in tools:
        name = tool["name"]
        params = tool.get("parameters", {})
        param_schema = json.dumps(params)

        tool_rules.append(
            f'tool_{_sanitize_name(name)}: "<function=" "{name}" ">" '
            f"%json{{{param_schema}}} " '"</function>"'
        )

    # Combine into grammar
    tool_alts = " | ".join(f"tool_{_sanitize_name(t['name'])}" for t in tools)

    grammar = f"start: {tool_alts}\n"
    grammar += "\n".join(tool_rules)

    return grammar


def _build_python_grammar(
    tools: List[Dict[str, Any]],
    fmt: ToolCallFormat,
) -> str:
    """Build Python-style tool call grammar (e.g., function_name(args))."""
    tool_rules = []
    for tool in tools:
        name = tool["name"]
        params = tool.get("parameters", {})

        # Convert parameters to Python-style arguments
        props = params.get("properties", {})
        required = params.get("required", [])

        arg_rules = []
        for prop_name, prop_schema in props.items():
            prop_grammar = _json_schema_to_lark(prop_schema)
            if prop_name in required:
                arg_rules.append(f'"{prop_name}=" {prop_grammar}')
            else:
                arg_rules.append(f'("{prop_name}=" {prop_grammar})?')

        args_grammar = ', '.join(arg_rules) if arg_rules else ""

        tool_rules.append(
            f'tool_{_sanitize_name(name)}: "{name}" "(" {args_grammar} ")"'
        )

    # Combine into grammar
    tool_alts = " | ".join(f"tool_{_sanitize_name(t['name'])}" for t in tools)

    grammar_parts = []
    if fmt.start_marker:
        escaped = _escape_for_lark(fmt.start_marker)
        grammar_parts.append(f'start: "{escaped}" ({tool_alts})')
    else:
        grammar_parts.append(f"start: {tool_alts}")

    grammar_parts.extend(tool_rules)

    return "\n".join(grammar_parts)


def _json_schema_to_lark(schema: Dict[str, Any]) -> str:
    """Convert JSON schema to Lark grammar fragment."""
    schema_type = schema.get("type", "string")

    if schema_type == "string":
        if "enum" in schema:
            alts = " | ".join(f'"{v}"' for v in schema["enum"])
            return f"({alts})"
        if "pattern" in schema:
            return f"/{schema['pattern']}/"
        return "STRING"

    elif schema_type == "integer":
        return "INT"

    elif schema_type == "number":
        return "NUMBER"

    elif schema_type == "boolean":
        return '("true" | "false")'

    elif schema_type == "array":
        items = schema.get("items", {"type": "string"})
        item_grammar = _json_schema_to_lark(items)
        return f'"[" ({item_grammar} ("," {item_grammar})*)? "]"'

    elif schema_type == "object":
        # Inline JSON schema reference
        return f"%json{{{json.dumps(schema)}}}"

    return "/.+/"


def _escape_for_lark(s: str) -> str:
    """Escape special characters for Lark string literals."""
    # Escape backslashes first, then quotes
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    return s


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a Lark rule name."""
    # Replace non-alphanumeric with underscore
    return re.sub(r"[^a-zA-Z0-9]", "_", name)


def tools_from_functions(functions: List[callable]) -> List[Dict[str, Any]]:
    """
    Create tool definitions from Python functions.

    Uses function signatures and docstrings to build tool schemas.

    Args:
        functions: List of Python callables.

    Returns:
        List of tool definitions.
    """
    import inspect
    from typing import get_type_hints

    tools = []
    for func in functions:
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Get type from hints
            param_type = hints.get(param_name, str)
            json_type = _python_type_to_json(param_type)

            properties[param_name] = {"type": json_type}

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        tool = {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
        tools.append(tool)

    return tools


def _python_type_to_json(py_type) -> str:
    """Convert Python type to JSON schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return type_map.get(py_type, "string")


def build_multi_tool_grammar(
    tools: List[Dict[str, Any]],
    tokenizer=None,
    allow_multiple: bool = True,
    format_override: Optional[ToolCallFormat] = None,
) -> str:
    """
    Build grammar that allows multiple sequential tool calls.

    Args:
        tools: List of tool definitions.
        tokenizer: Optional tokenizer for format detection.
        allow_multiple: If True, allows multiple tool calls in sequence.
        format_override: Optional explicit format specification.

    Returns:
        Lark grammar string.
    """
    # Get single tool grammar
    single_grammar = build_tool_grammar(tools, tokenizer, format_override)

    if not allow_multiple:
        return single_grammar

    # Modify to allow repetition
    # Replace 'start: ...' with 'start: (...)+'
    if single_grammar.startswith("start:"):
        # Extract the start rule content
        lines = single_grammar.split("\n")
        start_rule = lines[0]
        rest = "\n".join(lines[1:])

        # Get content after 'start:'
        start_content = start_rule.split(":", 1)[1].strip()

        # Allow one or more
        new_start = f"start: ({start_content})+"

        return f"{new_start}\n{rest}"

    return single_grammar
