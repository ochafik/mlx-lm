# Copyright 2025 Apple Inc.

"""
Jinja template differential analysis for auto-detecting tool call format.

This module analyzes a model's Jinja chat template to automatically detect
how tool calls are formatted, including:
- Start/end markers (e.g., <|tool_call|>)
- JSON structure (name field, arguments field, ordering)
- Whitespace and formatting conventions

The approach is "differential": we render the template with known dummy
tool calls and analyze the output to extract the format specification.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from jinja2 import Environment, BaseLoader, TemplateSyntaxError


@dataclass
class ToolCallFormat:
    """Detected tool call format specification."""

    # Format type: "json", "xml", "python", "custom"
    format_type: str = "json"

    # Markers around tool calls
    start_marker: Optional[str] = None
    end_marker: Optional[str] = None

    # For JSON format: field names and ordering
    name_field: str = "name"
    arguments_field: str = "arguments"

    # Additional format details
    uses_newlines: bool = False
    indent: int = 0
    arguments_as_string: bool = False  # Some formats stringify arguments

    # Raw captured examples
    examples: List[str] = field(default_factory=list)


@dataclass
class TemplateAnalysis:
    """Complete analysis result from a chat template."""

    # Whether tool calling is supported
    has_tool_calls: bool = False

    # Detected format specification
    format: Optional[ToolCallFormat] = None

    # Role markers
    assistant_prefix: Optional[str] = None
    assistant_suffix: Optional[str] = None

    # Raw template for debugging
    raw_template: str = ""


class JinjaTemplateAnalyzer:
    """
    Analyze Jinja templates for tool call patterns using differential rendering.

    The analyzer works by:
    1. Rendering the template with dummy tool calls
    2. Comparing outputs to extract format patterns
    3. Building a format specification for grammar generation
    """

    def __init__(self):
        self.env = Environment(loader=BaseLoader())
        # Add json filter that some templates expect
        self.env.filters["tojson"] = lambda x: json.dumps(x)
        self.env.globals["json"] = json

    def analyze(self, template_str: str) -> TemplateAnalysis:
        """
        Analyze a chat template for tool call patterns.

        Args:
            template_str: The Jinja2 chat template string.

        Returns:
            TemplateAnalysis with detected patterns.
        """
        analysis = TemplateAnalysis(raw_template=template_str)

        # Quick check for tool call indicators
        if not self._has_tool_call_indicators(template_str):
            return analysis

        analysis.has_tool_calls = True

        # Try to detect format via differential rendering
        try:
            format_spec = self._detect_format_by_rendering(template_str)
            if format_spec:
                analysis.format = format_spec
        except Exception:
            # Fall back to pattern matching
            format_spec = self._detect_format_by_pattern(template_str)
            if format_spec:
                analysis.format = format_spec

        # Detect assistant markers
        analysis.assistant_prefix, analysis.assistant_suffix = (
            self._detect_assistant_markers(template_str)
        )

        return analysis

    def _has_tool_call_indicators(self, template: str) -> bool:
        """Check if template has any tool call indicators."""
        indicators = [
            "tool_call",
            "tool_calls",
            "function_call",
            "tools",
            "<function",
            "python_tag",
        ]
        template_lower = template.lower()
        return any(ind in template_lower for ind in indicators)

    def _detect_format_by_rendering(
        self, template_str: str
    ) -> Optional[ToolCallFormat]:
        """
        Detect format by rendering template with dummy tool calls.

        This is the "differential analysis" approach - we render with known
        inputs and extract the format from the outputs.
        """
        # Create dummy messages with tool calls
        dummy_tool_call = {
            "name": "__TOOL_NAME_MARKER__",
            "arguments": {"__ARG_KEY__": "__ARG_VALUE__"},
        }

        # Also test with function field (some templates use this)
        dummy_tool_call_alt = {
            "function": {
                "name": "__TOOL_NAME_MARKER__",
                "arguments": {"__ARG_KEY__": "__ARG_VALUE__"},
            }
        }

        messages_with_tool = [
            {"role": "user", "content": "Test message"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [dummy_tool_call],
            },
        ]

        messages_with_tool_alt = [
            {"role": "user", "content": "Test message"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [dummy_tool_call_alt],
            },
        ]

        messages_without_tool = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response"},
        ]

        try:
            template = self.env.from_string(template_str)

            # Render with and without tools
            with_tool = self._safe_render(template, messages_with_tool)
            with_tool_alt = self._safe_render(template, messages_with_tool_alt)
            without_tool = self._safe_render(template, messages_without_tool)

            # Use whichever rendering worked
            tool_output = with_tool or with_tool_alt
            if not tool_output:
                return None

            # Extract the tool call portion by diffing
            format_spec = self._extract_format_from_diff(
                tool_output, without_tool or "", dummy_tool_call
            )

            return format_spec

        except (TemplateSyntaxError, Exception):
            return None

    def _safe_render(
        self,
        template,
        messages: List[Dict],
        **kwargs,
    ) -> Optional[str]:
        """Safely render template with error handling."""
        try:
            # Common template variables
            render_kwargs = {
                "messages": messages,
                "add_generation_prompt": False,
                "bos_token": "<s>",
                "eos_token": "</s>",
                **kwargs,
            }
            return template.render(**render_kwargs)
        except Exception:
            return None

    def _extract_format_from_diff(
        self,
        with_tool: str,
        without_tool: str,
        dummy_tool: Dict,
    ) -> Optional[ToolCallFormat]:
        """Extract format specification by analyzing rendered output."""
        format_spec = ToolCallFormat()

        # Find the tool call portion in the output
        if "__TOOL_NAME_MARKER__" not in with_tool:
            return None

        format_spec.examples.append(with_tool)

        # Detect JSON format indicators
        if '{"name"' in with_tool or '"name":' in with_tool:
            format_spec.format_type = "json"

            # Extract name/arguments field names
            name_match = re.search(
                r'"([^"]+)":\s*"__TOOL_NAME_MARKER__"', with_tool
            )
            if name_match:
                format_spec.name_field = name_match.group(1)

            args_match = re.search(
                r'"([^"]+)":\s*\{[^}]*"__ARG_KEY__"', with_tool
            )
            if args_match:
                format_spec.arguments_field = args_match.group(1)

        # Detect XML/function format
        elif "<function" in with_tool or "<tool" in with_tool:
            format_spec.format_type = "xml"

        # Detect Python-style format
        elif "__TOOL_NAME_MARKER__(" in with_tool:
            format_spec.format_type = "python"

        # Detect start/end markers
        format_spec.start_marker, format_spec.end_marker = (
            self._detect_markers(with_tool)
        )

        # Detect formatting
        format_spec.uses_newlines = "\n" in with_tool.split(
            "__TOOL_NAME_MARKER__"
        )[0][-50:]

        return format_spec

    def _detect_markers(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """Detect start and end markers around tool calls."""
        # Common marker patterns
        marker_patterns = [
            # Llama-3 style
            (r"(<\|tool_call\|>)", r"(<\|/tool_call\|>|</tool_call>)"),
            (r"(<\|python_tag\|>)", r"(<\|eom_id\|>)"),
            # Generic XML style
            (r"(<tool_call>)", r"(</tool_call>)"),
            (r"(<function[^>]*>)", r"(</function>)"),
            # Qwen style
            (r"(<\|tool▁call\|>)", r"(<\|/tool▁call\|>)"),
        ]

        for start_pattern, end_pattern in marker_patterns:
            start_match = re.search(start_pattern, output)
            end_match = re.search(end_pattern, output)

            if start_match:
                start_marker = start_match.group(1)
                end_marker = end_match.group(1) if end_match else None
                return start_marker, end_marker

        return None, None

    def _detect_format_by_pattern(
        self, template_str: str
    ) -> Optional[ToolCallFormat]:
        """Fall back to pattern matching on template source."""
        format_spec = ToolCallFormat()

        # Check for common patterns in template source
        if "<|tool_call|>" in template_str:
            format_spec.format_type = "json"
            format_spec.start_marker = "<|tool_call|>"
            if "<|/tool_call|>" in template_str:
                format_spec.end_marker = "<|/tool_call|>"
            elif "</tool_call>" in template_str:
                format_spec.end_marker = "</tool_call>"

        elif "<tool_call>" in template_str:
            format_spec.format_type = "json"
            format_spec.start_marker = "<tool_call>"
            format_spec.end_marker = "</tool_call>"

        elif "<|python_tag|>" in template_str:
            format_spec.format_type = "python"
            format_spec.start_marker = "<|python_tag|>"

        elif "<function=" in template_str:
            format_spec.format_type = "xml"

        # Detect field names from template
        if "tool_call['name']" in template_str or 'tool_call["name"]' in template_str:
            format_spec.name_field = "name"
        if "tool_call.function.name" in template_str:
            format_spec.name_field = "function.name"

        if (
            "tool_call['arguments']" in template_str
            or 'tool_call["arguments"]' in template_str
        ):
            format_spec.arguments_field = "arguments"
        if "tool_call.function.arguments" in template_str:
            format_spec.arguments_field = "function.arguments"

        return format_spec

    def _detect_assistant_markers(
        self, template_str: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Detect assistant role prefix/suffix markers."""
        # Common patterns
        patterns = [
            (r"<\|start_header_id\|>assistant<\|end_header_id\|>", None),
            (r"<\|im_start\|>assistant", r"<\|im_end\|>"),
            (r"\[INST\].*?\[/INST\]", None),
            (r"### Assistant:", None),
        ]

        for prefix_pattern, suffix_pattern in patterns:
            if re.search(prefix_pattern, template_str):
                prefix_match = re.search(prefix_pattern, template_str)
                prefix = prefix_match.group(0) if prefix_match else None

                suffix = None
                if suffix_pattern:
                    suffix_match = re.search(suffix_pattern, template_str)
                    suffix = suffix_match.group(0) if suffix_match else None

                return prefix, suffix

        return None, None


def analyze_chat_template(template_str: str) -> TemplateAnalysis:
    """
    Convenience function to analyze a chat template.

    Args:
        template_str: The Jinja2 chat template string.

    Returns:
        TemplateAnalysis with detected patterns.
    """
    analyzer = JinjaTemplateAnalyzer()
    return analyzer.analyze(template_str)


def get_tool_format_from_tokenizer(tokenizer) -> Optional[ToolCallFormat]:
    """
    Get tool call format from a tokenizer's chat template.

    Args:
        tokenizer: HuggingFace tokenizer with chat_template attribute.

    Returns:
        ToolCallFormat if tool calling is supported, None otherwise.
    """
    # Try to get the chat_template from the tokenizer directly.
    # Do NOT unwrap to ._tokenizer (fast tokenizer) as it lacks chat_template.
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None

    analysis = analyze_chat_template(template)
    return analysis.format if analysis.has_tool_calls else None
