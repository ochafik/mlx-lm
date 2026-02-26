# Copyright 2025 Apple Inc.
#
# Grammar-constrained generation for MLX-LM.
#
# This package provides grammar-based token masking for structured output
# generation, including JSON schema validation and tool calling.
#
# Key components:
# - GrammarState: Abstract base class for grammar state tracking
# - LLGuidanceState: Implementation using the llguidance Rust engine
# - build_tool_grammar: Auto-generate grammar from tool definitions
# - analyze_chat_template: Extract tool call format from Jinja templates

from .base import GrammarState
from .mask_ops import (
    apply_grammar_mask,
    bitmask_to_bool,
    bool_to_bitmask,
    count_allowed_tokens,
    get_allowed_token_ids,
)
from .jinja_analysis import (
    ToolCallFormat,
    TemplateAnalysis,
    JinjaTemplateAnalyzer,
    analyze_chat_template,
    get_tool_format_from_tokenizer,
)
from .tool_schema import (
    build_tool_grammar,
    build_tool_schema,
    build_multi_tool_grammar,
    parse_tool_call_output,
    tools_from_functions,
)

# Lazy import for llguidance (optional dependency)
def __getattr__(name):
    if name == "LLGuidanceState":
        from .llguidance_adapter import LLGuidanceState
        return LLGuidanceState
    elif name == "MockGrammarState":
        from .llguidance_adapter import MockGrammarState
        return MockGrammarState
    elif name == "is_llguidance_available":
        from .llguidance_adapter import is_llguidance_available
        return is_llguidance_available
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "GrammarState",
    # LLGuidance integration (lazy-loaded)
    "LLGuidanceState",
    "MockGrammarState",
    "is_llguidance_available",
    # Mask operations
    "apply_grammar_mask",
    "bitmask_to_bool",
    "bool_to_bitmask",
    "count_allowed_tokens",
    "get_allowed_token_ids",
    # Jinja template analysis
    "ToolCallFormat",
    "TemplateAnalysis",
    "JinjaTemplateAnalyzer",
    "analyze_chat_template",
    "get_tool_format_from_tokenizer",
    # Tool schema
    "build_tool_grammar",
    "build_tool_schema",
    "build_multi_tool_grammar",
    "parse_tool_call_output",
    "tools_from_functions",
]
