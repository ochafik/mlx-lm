# MLX-LM Grammar-Constrained Tool Calls: Complete Implementation Guide

## Overview

This guide provides step-by-step instructions for implementing grammar-constrained tool calling in MLX-LM. It builds on all previous research and provides concrete code for each component.

## Prerequisites

### Dependencies

```bash
# Core dependencies
pip install llguidance==1.4.0
pip install jinja2
pip install pydantic

# For development
pip install pytest pytest-benchmark
```

### Project Structure

Create the following structure:

```bash
mkdir -p mlx_lm/grammar
mkdir -p mlx_lm/tool_parsers
mkdir -p tests/test_grammar
mkdir -p examples/grammar
```

## Step 1: Base Grammar System

### 1.1 Create Base Classes

**File: `mlx_lm/grammar/__init__.py`**

```python
"""
Grammar-constrained generation for MLX-LM.

This module provides grammar-based constraints for language model generation,
enabling structured output and reliable tool calling.
"""

from mlx_lm.grammar.base import GrammarState
from mlx_lm.llguidance import LLGuidanceState
from mlx_lm.schema_compiler import JSONSchemaCompiler
from mlx_lm.mask_ops import apply_token_mask

__all__ = [
    "GrammarState",
    "LLGuidanceState",
    "JSONSchemaCompiler",
    "apply_token_mask",
]
```

**File: `mlx_lm/grammar/base.py`**

```python
"""
Base classes for grammar-constrained generation.
"""

from abc import ABC, abstractmethod
from typing import Optional
import mlx.core as mx


class GrammarState(ABC):
    """
    Abstract base class for grammar state tracking.

    A GrammarState tracks the current position in a grammar during generation,
    computes which tokens are valid at each step, and updates as tokens are generated.
    """

    @abstractmethod
    def compute_token_mask(self) -> mx.array:
        """
        Compute the token mask for the current position.

        Returns:
            mx.array: Boolean array where True indicates allowed tokens
        """
        pass

    @abstractmethod
    def update(self, token: int) -> bool:
        """
        Update the grammar state with a new token.

        Args:
            token: The token ID that was generated

        Returns:
            True if grammar is still active, False if complete/terminated
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if the grammar constraints are satisfied.

        Returns:
            True if grammar is complete and constraints can be removed
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the current state can accept end-of-sequence.

        Returns:
            True if EOS is allowed at this position
        """
        pass


class GrammarViolationError(Exception):
    """Raised when a generated token violates grammar constraints."""

    pass


class GrammarCompletionError(Exception):
    """Raised when grammar cannot be completed from current state."""

    pass
```

### 1.2 Implement LLGuidance Integration

**File: `mlx_lm/grammar/llguidance.py`**

```python
"""
LLGuidance integration for grammar-constrained generation.
"""

from typing import Optional
import json

import mlx.core as mx
import numpy as np

# Try to import llguidance, provide helpful error if missing
try:
    from llguidance import LLMatcher, LLTokenizer, grammar_from
    LLGUIDANCE_AVAILABLE = True
except ImportError:
    LLGUIDANCE_AVAILABLE = False
    LLMatcher = None
    LLTokenizer = None
    grammar_from = None


from mlx_lm.grammar.base import GrammarState, GrammarViolationError


class LLGuidanceState(GrammarState):
    """
    Grammar state using LLGuidance backend.

    This class wraps LLGuidance's LLMatcher to provide token masking
    for grammar-constrained generation.
    """

    def __init__(
        self,
        tokenizer,
        grammar_str: str,
        grammar_type: str = "lark",
        allow_early_termination: bool = False
    ):
        """
        Initialize LLGuidance state.

        Args:
            tokenizer: HuggingFace tokenizer
            grammar_str: Grammar string in specified format
            grammar_type: Type of grammar ("lark", "json_schema", "regex")
            allow_early_termination: Allow EOS before grammar is complete
        """
        if not LLGUIDANCE_AVAILABLE:
            raise ImportError(
                "llguidance is required but not installed. "
                "Install it with: pip install llguidance"
            )

        # Create LLGuidance tokenizer
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)
        self.vocab_size = tokenizer.vocab_size
        self.allow_early_termination = allow_early_termination

        # Compile grammar
        if grammar_type == "json_schema":
            self.grammar = grammar_from("json_schema", grammar_str)
        elif grammar_type == "lark":
            self.grammar = grammar_from("lark", grammar_str)
        elif grammar_type == "regex":
            self.grammar = grammar_from("regex", grammar_str)
        else:
            raise ValueError(f"Unknown grammar type: {grammar_type}")

        # Create matcher
        self.matcher = LLMatcher(self.ll_tokenizer, self.grammar)
        self._complete = False
        self._terminal = False

    def compute_token_mask(self) -> mx.array:
        """
        Compute token mask for current position.

        Returns:
            mx.array: Boolean array of allowed tokens
        """
        # Get mask from LLGuidance
        mask_np = self.matcher.get_token_mask()

        # Convert to MLX array
        return mx.array(mask_np, dtype=mx.bool_)

    def update(self, token: int) -> bool:
        """
        Update grammar state with new token.

        Args:
            token: Token ID that was generated

        Returns:
            True if grammar is still active
        """
        if self._complete:
            return False

        # Commit token to matcher
        self.matcher.commit_token(token)

        # Check if grammar is complete
        self._terminal = self.matcher.is_terminated()
        self._complete = self._terminal

        return not self._complete

    def is_complete(self) -> bool:
        """Check if grammar is complete."""
        return self._complete

    def is_terminal(self) -> bool:
        """Check if EOS is allowed."""
        return self._terminal or self.allow_early_termination

    @classmethod
    def from_json_schema(
        cls,
        tokenizer,
        schema: dict,
        **kwargs
    ) -> "LLGuidanceState":
        """
        Create grammar state from JSON schema.

        Args:
            tokenizer: HuggingFace tokenizer
            schema: JSON schema dictionary
            **kwargs: Additional arguments for LLGuidanceState

        Returns:
            LLGuidanceState instance
        """
        schema_str = json.dumps(schema)
        return cls(tokenizer, schema_str, grammar_type="json_schema", **kwargs)

    @classmethod
    def from_gbnf(
        cls,
        tokenizer,
        gbnf_string: str,
        **kwargs
    ) -> "LLGuidanceState":
        """
        Create grammar state from GBNF string.

        Args:
            tokenizer: HuggingFace tokenizer
            gbnf_string: GBNF grammar string
            **kwargs: Additional arguments for LLGuidanceState

        Returns:
            LLGuidanceState instance
        """
        return cls(tokenizer, gbnf_string, grammar_type="lark", **kwargs)

    @classmethod
    def from_regex(
        cls,
        tokenizer,
        pattern: str,
        **kwargs
    ) -> "LLGuidanceState":
        """
        Create grammar state from regex pattern.

        Args:
            tokenizer: HuggingFace tokenizer
            pattern: Regular expression pattern
            **kwargs: Additional arguments for LLGuidanceState

        Returns:
            LLGuidanceState instance
        """
        return cls(tokenizer, pattern, grammar_type="regex", **kwargs)
```

### 1.3 Implement Mask Operations

**File: `mlx_lm/grammar/mask_ops.py`**

```python
"""
Token mask operations for grammar-constrained generation.
"""

import mlx.core as mx


def apply_token_mask(logits: mx.array, mask: mx.array) -> mx.array:
    """
    Apply token mask to logits.

    Args:
        logits: Model output logits, shape (vocab_size,) or (batch, vocab_size)
        mask: Boolean mask of allowed tokens, same shape as logits

    Returns:
        mx.array: Logits with disallowed tokens set to -inf
    """
    # Create negative infinity value
    neg_inf = mx.array(-float('inf'), dtype=logits.dtype)

    # Apply mask: keep logits where mask is True, set to -inf where False
    return mx.where(mask, logits, neg_inf)


def apply_token_mask_inplace(logits: mx.array, mask: mx.array) -> None:
    """
    Apply token mask to logits in-place (where supported).

    This is a hint for optimization - actual behavior depends on MLX.

    Args:
        logits: Model output logits (modified in-place if possible)
        mask: Boolean mask of allowed tokens
    """
    # MLX doesn't support true in-place operations, so we return new array
    # This function is for API compatibility and potential future optimization
    result = apply_token_mask(logits, mask)
    return result


@mx.custom_function
def metal_bitmask_kernel(
    logits: mx.array,
    mask: mx.array,
    neg_inf: mx.array
) -> mx.array:
    """
    Metal kernel for efficient bitmask application.

    This kernel applies a compressed bitmask representation where one
    32-bit integer represents 32 tokens.

    Args:
        logits: Model logits, shape (batch, vocab_size)
        mask: Compressed bitmask, shape (batch, vocab_size // 32)
        neg_inf: Scalar negative infinity value

    Returns:
        Constrained logits
    """
    # Note: This is a placeholder. Actual Metal kernel implementation
    # would use mx.fast.metal_kernel() with custom shader code.
    # For now, we use the simpler boolean mask approach.

    # Expand compressed mask to full boolean mask
    batch, vocab_size = logits.shape
    num_words = mask.shape[1]

    bool_mask = mx.zeros((batch, vocab_size), dtype=mx.bool_)

    for word_idx in range(num_words):
        start = word_idx * 32
        end = min(start + 32, vocab_size)

        word_mask = mask[:, word_idx]

        for bit_idx in range(32):
            if start + bit_idx >= vocab_size:
                break

            bit_mask = (word_mask >> bit_idx) & 1
            bool_mask[:, start + bit_idx] = bit_mask == 1

    return mx.where(bool_mask, logits, neg_inf)
```

### 1.4 Create JSON Schema Compiler

**File: `mlx_lm/grammar/schema_compiler.py`**

```python
"""
JSON schema to grammar compiler.
"""

from typing import Dict, Any, Optional
import json


class JSONSchemaCompiler:
    """
    Compile JSON schemas to GBNF grammars.

    This converter handles common JSON schema constructs and produces
    efficient GBNF grammars for constrained generation.
    """

    def __init__(self, whitespace_flexible: bool = True):
        """
        Initialize compiler.

        Args:
            whitespace_flexible: Allow flexible whitespace in output
        """
        self.whitespace_flexible = whitespace_flexible

    def compile(self, schema: Dict[str, Any], root_rule: str = "root") -> str:
        """
        Compile JSON schema to GBNF grammar.

        Args:
            schema: JSON schema dictionary
            root_rule: Name of root rule

        Returns:
            GBNF grammar string
        """
        self._rule_counter = 0
        self._rules = {}

        # Compile the schema
        rule_body = self._compile_schema(schema)

        # Build final grammar
        rules_text = "\n".join(
            f"{name} ::= {body}\n" for name, body in self._rules.items()
        )
        rules_text += f"\n{root_rule} ::= {rule_body}\n"

        return rules_text

    def _compile_schema(self, schema: Dict[str, Any]) -> str:
        """Compile a schema node."""
        schema_type = schema.get("type")

        if "const" in schema:
            return json.dumps(schema["const"])

        if "enum" in schema:
            alternatives = [json.dumps(v) for v in schema["enum"]]
            return " | ".join(alternatives)

        if schema_type == "object":
            return self._compile_object(schema)
        elif schema_type == "array":
            return self._compile_array(schema)
        elif schema_type == "string":
            return self._compile_string(schema)
        elif schema_type in ["number", "integer"]:
            return self._compile_number(schema)
        elif schema_type == "boolean":
            return '"true" | "false"'
        elif schema_type == "null":
            return '"null"'
        else:
            # Any value
            return "any"

    def _compile_object(self, schema: Dict[str, Any]) -> str:
        """Compile object schema."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        if not properties:
            return '"{}"'

        # Compile each property
        prop_parts = []
        for i, (prop_name, prop_schema) in enumerate(properties.items()):
            prop_rule = self._compile_schema(prop_schema)
            is_required = prop_name in required
            is_last = i == len(properties) - 1

            if is_required:
                prop_parts.append(f'"\\"{prop_name}\\"": {prop_rule}')
            else:
                prop_parts.append(f'("\\"{prop_name}\\"": {prop_rule})?')

            if not is_last:
                prop_parts.append('","')

        # Join with flexible whitespace
        ws = self._ws()
        body = " ".join(prop_parts)
        return f'"{{" {ws} {body} {ws} "}}"'

    def _compile_array(self, schema: Dict[str, Any]) -> str:
        """Compile array schema."""
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", None)

        # Compile item schema
        item_rule = self._compile_schema(items_schema)

        # Build repetition
        ws = self._ws()

        if min_items == 0:
            # Zero or more
            return f'"[" {ws} ({item_rule} ({ws} "," {ws} {item_rule})* )? {ws} "]"'
        elif max_items is not None and min_items == max_items:
            # Exact count
            items = " ".join([item_rule for _ in range(min_items)])
            separators = " ".join([f'"{ws} , {ws}"' for _ in range(min_items - 1)])
            return f'"[" {ws} {items} {ws} "]"'
        else:
            # Range
            return f'"[" {ws} {item_rule} ({ws} "," {ws} {item_rule})* {ws} "]"'

    def _compile_string(self, schema: Dict[str, Any]) -> str:
        """Compile string schema."""
        pattern = schema.get("pattern")

        if pattern:
            # For simplicity, just use a generic string pattern
            # A full implementation would convert regex to GBNF
            return 'string'

        return 'string'

    def _compile_number(self, schema: Dict[str, Any]) -> str:
        """Compile number schema."""
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        # Simple number pattern
        # A full implementation would generate optimized ranges
        return 'number'

    def _ws(self) -> str:
        """Get whitespace pattern."""
        if self.whitespace_flexible:
            return '""'
        return '""'

    def _generate_rule_name(self) -> str:
        """Generate unique rule name."""
        name = f"rule_{self._rule_counter}"
        self._rule_counter += 1
        return name
```

## Step 2: Integration with Generation Loop

### 2.1 Modify generate_step

**File: `mlx_lm/generate.py`** (additions only shown)

```python
# In generate_step function signature, add:
def generate_step(
    model: BaseModel,
    tokens: mx.array,
    kwargs: dict,
    cache: Optional[List[Any]] = None,
    progress: Optional[Progress] = None,
    grammar_state: Optional[GrammarState] = None,  # NEW
    ...
) -> Tuple[int, mx.array, ...]:
    """
    Generate a single token.

    Args:
        model: Language model
        tokens: Current token sequence
        kwargs: Generation parameters
        cache: KV cache
        progress: Optional progress tracker
        grammar_state: Optional grammar constraints (NEW)
        ...
    """

    # ... existing prefill and decoding code ...

    # Extract logits for last position
    logits = logits[:, -1, :]

    # NEW: Apply grammar constraints (highest priority)
    if grammar_state is not None:
        mask = grammar_state.compute_token_mask()
        logits = apply_token_mask(logits, mask)

    # Apply logits processors
    if logits_processors and len(input_tokens) > 0:
        tokens_for_processor = mx.concat([tokens, input_tokens]) if tokens is not None else input_tokens
        for processor in logits_processors:
            logits = processor(tokens_for_processor, logits)

    # Sample token
    token = sample(
        logits,
        temperature=kwargs.get("temperature", 0.0),
        top_p=kwargs.get("top_p", 1.0),
        top_k=kwargs.get("top_k", -1),
        min_p=kwargs.get("min_p", 0.0),
    )

    # NEW: Update grammar state
    grammar_terminated = False
    if grammar_state is not None:
        still_active = grammar_state.update(int(token.item()))
        grammar_terminated = not still_active

    # ... existing code ...

    # Return grammar state in results
    return token, logits, cache, grammar_state, grammar_terminated, ...
```

### 2.2 Add Grammar Utilities to sample_utils.py

**File: `mlx_lm/sample_utils.py`** (additions)

```python
from mlx_lm.grammar.base import GrammarState
from mlx_lm.grammar.mask_ops import apply_token_mask


def create_grammar_logits_processor(grammar_state: GrammarState):
    """
    Create a logits processor that applies grammar constraints.

    Args:
        grammar_state: Grammar state for constraints

    Returns:
        Logits processor function
    """
    def processor(tokens: mx.array, logits: mx.array) -> mx.array:
        mask = grammar_state.compute_token_mask()
        return apply_token_mask(logits, mask)

    return processor
```

## Step 3: Tool Calling Support

### 3.1 Create Jinja Template Analyzer

**File: `mlx_lm/grammar/jinja_analyzer.py`**

```python
"""
Analyze Jinja templates to detect tool calling patterns.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ToolCallPattern:
    """Detected tool calling pattern in Jinja template."""
    format_type: str  # "json", "xml", "custom"
    start_marker: Optional[str]
    end_marker: Optional[str]
    structure: Dict[str, Any]

    def __init__(
        self,
        format_type: str,
        start_marker: Optional[str] = None,
        end_marker: Optional[str] = None,
        structure: Optional[Dict[str, Any]] = None
    ):
        self.format_type = format_type
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.structure = structure or {}


class JinjaTemplateAnalyzer:
    """Analyze Jinja templates for tool calling patterns."""

    def __init__(self):
        pass

    def analyze(self, template_str: str) -> Dict[str, Any]:
        """
        Analyze Jinja template for tool calling patterns.

        Args:
            template_str: Jinja template string

        Returns:
            Analysis dictionary with detected patterns
        """
        patterns = self._detect_patterns(template_str)

        return {
            "has_tool_calls": len(patterns) > 0,
            "patterns": patterns,
            "format": self._determine_format(patterns),
        }

    def _detect_patterns(self, template_str: str) -> List[ToolCallPattern]:
        """Detect tool calling patterns in template."""
        patterns = []

        # Check for special token markers
        if "<|tool_call|>" in template_str:
            patterns.append(ToolCallPattern(
                format_type="json",
                start_marker="<|tool_call|>",
                end_marker="<|tool_call_end|>",
                structure={"format": "special_token"}
            ))

        # Check for tool_calls loop
        if "tool_calls" in template_str and "for tool_call in" in template_str:
            structure = self._analyze_tool_loop(template_str)
            patterns.append(ToolCallPattern(
                format_type=structure.get("format", "json"),
                start_marker=None,
                end_marker=None,
                structure=structure
            ))

        return patterns

    def _analyze_tool_loop(self, template_str: str) -> Dict[str, Any]:
        """Analyze the tool call loop structure."""
        # Look for the loop
        match = re.search(
            r'{%-?\s*for\s+tool_call\s+in\s+message\[.tool_calls.\](.*?)}',
            template_str,
            re.DOTALL
        )

        if not match:
            return {"format": "unknown"}

        loop_content = match.group(1)

        # Detect format from content
        if '"name":' in loop_content and '"arguments":' in loop_content:
            return {
                "format": "json",
                "has_name": True,
                "has_arguments": True,
            }
        elif "<function=" in loop_content:
            return {
                "format": "xml",
                "tag_style": "attribute",
            }

        return {"format": "custom"}

    def _determine_format(self, patterns: List[ToolCallPattern]) -> str:
        """Determine overall format from patterns."""
        if not patterns:
            return "none"

        # Use first pattern's format
        return patterns[0].format_type
```

### 3.2 Create Tool Call Grammar Generator

**File: `mlx_lm/grammar/tool_grammar.py`**

```python
"""
Generate grammars for tool calling.
"""

from typing import Dict, Any, List
import json


class ToolCallGrammarGenerator:
    """Generate grammars for tool calling from tool definitions."""

    def __init__(self):
        self.jinja_analyzer = JinjaTemplateAnalyzer()

    def generate_from_tools(
        self,
        tools: List[Dict[str, Any]],
        format_type: str = "json"
    ) -> str:
        """
        Generate grammar from tool definitions.

        Args:
            tools: List of tool definitions
            format_type: Format type ("json", "xml", "custom")

        Returns:
            GBNF grammar string
        """
        if format_type == "json":
            return self._generate_json_grammar(tools)
        elif format_type == "xml":
            return self._generate_xml_grammar(tools)
        else:
            return self._generate_custom_grammar(tools, format_type)

    def _generate_json_grammar(self, tools: List[Dict[str, Any]]) -> str:
        """Generate JSON-style tool call grammar."""
        tool_alternatives = []

        for tool in tools:
            name = tool["name"]
            parameters = tool.get("parameters", {})

            # Generate parameter grammar
            param_grammar = self._schema_to_grammar(parameters)

            tool_alternatives.append(
                f'{{"name": "{name}", "arguments": {param_grammar}}}'
            )

        tools_rule = " | \\\n    ".join(tool_alternatives)

        return f"""root ::= tool_call
tool_call ::= {tools_rule}
"""

    def _generate_xml_grammar(self, tools: List[Dict[str, Any]]) -> str:
        """Generate XML-style tool call grammar."""
        # Simplified XML grammar
        tool_alternatives = []

        for tool in tools:
            name = tool["name"]
            parameters = tool.get("parameters", {})

            if not parameters.get("properties"):
                tool_alternatives.append(f'<function={name} />')
            else:
                tool_alternatives.append(
                    f'<function={name}> <parameter=.* /> </function>'
                )

        tools_rule = " | ".join(tool_alternatives)

        return f"""root ::= {tools_rule}
"""

    def _generate_custom_grammar(
        self,
        tools: List[Dict[str, Any]],
        format_type: str
    ) -> str:
        """Generate custom format grammar."""
        # Placeholder for custom formats
        return self._generate_json_grammar(tools)

    def _schema_to_grammar(self, schema: Dict[str, Any]) -> str:
        """Convert JSON schema to GBNF grammar fragment."""
        schema_type = schema.get("type")

        if schema_type == "object":
            return self._object_to_grammar(schema)
        elif schema_type == "array":
            return self._array_to_grammar(schema)
        elif schema_type == "string":
            return '"string"'
        elif schema_type in ["number", "integer"]:
            return "number"
        elif schema_type == "boolean":
            return '"true" | "false"'
        elif "enum" in schema:
            return " | ".join(f'"{v}"' for v in schema["enum"])
        else:
            return "any"

    def _object_to_grammar(self, schema: Dict[str, Any]) -> str:
        """Convert object schema to grammar."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        if not properties:
            return '"{}"'

        parts = []
        for i, (prop_name, prop_schema) in enumerate(properties.items()):
            prop_grammar = self._schema_to_grammar(prop_schema)
            is_required = prop_name in required
            is_last = i == len(properties) - 1

            if is_required:
                parts.append(f'"\\"{prop_name}\\"": {prop_grammar}')
            else:
                parts.append(f'("\\"{prop_name}\\"": {prop_grammar})?')

            if not is_last:
                parts.append('","')

        return '"{" " ".join(parts) + " }"'

    def _array_to_grammar(self, schema: Dict[str, Any]) -> str:
        """Convert array schema to grammar."""
        items = schema.get("items", {})
        item_grammar = self._schema_to_grammar(items)

        return f'"[" {item_grammar} ("," {item_grammar})* "]"'
```

### 3.3 Create Grammar-Based Tool Parser

**File: `mlx_lm/tool_parsers/grammar_tools.py`**

```python
"""
Grammar-based tool parser for MLX-LM.
"""

from typing import Dict, Any, List, Optional
import json


def parse_tool_call(
    text: str,
    tools: List[Dict[str, Any]],
    tokenizer
) -> Optional[Dict[str, Any]]:
    """
    Parse tool call from generated text.

    This is the entry point that matches existing tool parsers.

    Args:
        text: Generated text
        tools: List of tool definitions
        tokenizer: Tokenizer

    Returns:
        Parsed tool call or None
    """
    try:
        # Try to parse as JSON
        tool_call = json.loads(text.strip())

        # Validate structure
        if "name" not in tool_call:
            return None

        # Validate tool name
        tool_names = {tool["name"] for tool in tools}
        if tool_call["name"] not in tool_names:
            return None

        return {
            "name": tool_call["name"],
            "arguments": tool_call.get("arguments", {})
        }

    except (json.JSONDecodeError, TypeError):
        return None


def create_tool_grammar(
    tools: List[Dict[str, Any]],
    format_type: str = "json"
) -> str:
    """
    Create grammar for tool calling.

    Args:
        tools: List of tool definitions
        format_type: Format type

    Returns:
        GBNF grammar string
    """
    from mlx_lm.grammar.tool_grammar import ToolCallGrammarGenerator

    generator = ToolCallGrammarGenerator()
    return generator.generate_from_tools(tools, format_type)
```

## Step 4: High-Level API

### 4.1 Create Constrained Generation Module

**File: `mlx_lm/constrained_generate.py`**

```python"""
High-level API for grammar-constrained generation.
"""

from typing import Dict, Any, List, Optional, Union
import json

import mlx.core as mx

from mlx_lm.utils import generate_step
from mlx_lm.grammar.base import GrammarState
from mlx_lm.llguidance import LLGuidanceState


def generate_constrained(
    model,
    prompt: str,
    tokenizer,
    schema: Optional[Dict[str, Any]] = None,
    grammar: Optional[str] = None,
    grammar_state: Optional[GrammarState] = None,
    max_tokens: int = 100,
    **kwargs
) -> str:
    """
    Generate text with grammar constraints.

    Args:
        model: Language model
        prompt: Input prompt
        tokenizer: Tokenizer
        schema: JSON schema for output (alternative to grammar)
        grammar: GBNF grammar string (alternative to schema)
        grammar_state: Pre-created grammar state
        max_tokens: Maximum tokens to generate
        **kwargs: Additional generation parameters

    Returns:
        Generated text
    """
    # Create grammar state if not provided
    if grammar_state is None:
        if schema is not None:
            grammar_state = LLGuidanceState.from_json_schema(tokenizer, schema)
        elif grammar is not None:
            grammar_state = LLGuidanceState.from_gbnf(tokenizer, grammar)
        else:
            raise ValueError("Must provide schema, grammar, or grammar_state")

    # Tokenize prompt
    if isinstance(prompt, str):
        input_tokens = mx.array(tokenizer.encode(prompt))
    else:
        input_tokens = prompt

    # Create cache
    cache = None

    # Generate tokens
    tokens = input_tokens
    generated_count = 0

    while generated_count < max_tokens:
        token, _, cache, new_grammar_state, terminated = generate_step(
            model,
            tokens,
            {"max_tokens": max_tokens, **kwargs},
            cache=cache,
            grammar_state=grammar_state,
            **kwargs
        )

        tokens = mx.concat([tokens, mx.array([token])])
        generated_count += 1

        # Check termination
        if grammar_state.is_complete():
            break

        if terminated:
            break

    # Decode output
    output_tokens = tokens[len(input_tokens):].tolist()
    return tokenizer.decode(output_tokens)


def generate_tool_call(
    model,
    prompt: str,
    tokenizer,
    tools: List[Dict[str, Any]],
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Generate a tool call with grammar constraints.

    Args:
        model: Language model
        prompt: Input prompt
        tokenizer: Tokenizer
        tools: List of tool definitions
        **kwargs: Additional generation parameters

    Returns:
        Tool call dictionary or None
    """
    from mlx_lm.grammar.tool_grammar import create_tool_grammar
    from mlx_lm.tool_parsers.grammar_tools import parse_tool_call

    # Create grammar
    grammar = create_tool_grammar(tools)

    # Generate with grammar constraints
    output = generate_constrained(
        model,
        prompt,
        tokenizer,
        grammar=grammar,
        max_tokens=kwargs.get("max_tokens", 256),
        **kwargs
    )

    # Parse tool call
    return parse_tool_call(output, tools, tokenizer)
```

## Step 5: Examples

### 5.1 Basic Grammar Example

**File: `examples/grammar/basic_grammar.py`**

```python"""
Basic grammar-constrained generation example.
"""

from mlx_lm import load, generate_constrained


def main():
    # Load model
    model, tokenizer = load("meta-llama/Llama-3.1-8B-Instruct")

    # Example 1: Generate JSON
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string"}
        },
        "required": ["name", "age"]
    }

    prompt = "Generate a user profile for John Doe, age 30"
    result = generate_constrained(
        model,
        prompt,
        tokenizer,
        schema=schema,
        max_tokens=100
    )

    print("Generated JSON:")
    print(result)

    # Example 2: Generate with regex
    import re
    phone_pattern = r"\d{3}-\d{3}-\d{4}"

    # Create regex grammar
    grammar = f'root ::= "{phone_pattern}"'

    prompt = "Generate a phone number"
    result = generate_constrained(
        model,
        prompt,
        tokenizer,
        grammar=grammar,
        max_tokens=20
    )

    print("\nGenerated phone number:")
    print(result)


if __name__ == "__main__":
    main()
```

### 5.2 Tool Calling Example

**File: `examples/grammar/tool_calling.py`**

```python"""
Tool calling with grammar constraints example.
"""

from mlx_lm import load, generate_tool_call


def main():
    # Load model
    model, tokenizer = load("meta-llama/Llama-3.1-8B-Instruct")

    # Define tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        },
        {
            "name": "calculate",
            "description": "Perform a calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, e.g. 2 + 2"
                    }
                },
                "required": ["expression"]
            }
        }
    ]

    # Example 1: Weather tool
    prompt = "What's the weather like in Paris?"
    tool_call = generate_tool_call(
        model,
        prompt,
        tokenizer,
        tools=tools,
        max_tokens=100
    )

    print("Weather tool call:")
    print(f"  Tool: {tool_call['name']}")
    print(f"  Arguments: {tool_call['arguments']}")

    # Example 2: Calculator tool
    prompt = "What is 25 times 17?"
    tool_call = generate_tool_call(
        model,
        prompt,
        tokenizer,
        tools=tools,
        max_tokens=100
    )

    print("\nCalculator tool call:")
    print(f"  Tool: {tool_call['name']}")
    print(f"  Arguments: {tool_call['arguments']}")


if __name__ == "__main__":
    main()
```

## Step 6: Testing

### 6.1 Unit Tests

**File: `tests/test_grammar/test_base.py`**

```python"""
Tests for grammar base classes.
"""

import pytest
from mlx_lm.grammar.base import GrammarState
from mlx_lm.llguidance import LLGuidanceState


def test_llguidance_state_creation(tokenizer):
    """Test creating LLGuidance state."""
    grammar = 'root ::= "hello" "world"'
    state = LLGuidanceState.from_gbnf(tokenizer, grammar)

    assert state is not None
    assert not state.is_complete()


def test_llguidance_state_mask(tokenizer):
    """Test token mask computation."""
    grammar = 'root ::= "hello"'
    state = LLGuidanceState.from_gbnf(tokenizer, grammar)

    mask = state.compute_token_mask()
    assert mask.shape == (tokenizer.vocab_size,)


def test_llguidance_state_update(tokenizer):
    """Test state updates."""
    grammar = 'root ::= "hello" "world"'
    state = LLGuidanceState.from_gbnf(tokenizer, grammar)

    # Encode "hello"
    hello_tokens = tokenizer.encode("hello")

    # Update with hello tokens
    for token in hello_tokens:
        state.update(token)

    # State should still be active (need "world")
    assert not state.is_complete()


def test_json_schema_state(tokenizer):
    """Test JSON schema grammar."""
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "number"}
        }
    }

    state = LLGuidanceState.from_json_schema(tokenizer, schema)
    mask = state.compute_token_mask()

    # Check that some tokens are allowed
    assert mx.any(mask).item()  # At least some tokens should be allowed
```

### 6.2 Integration Tests

**File: `tests/test_grammar/test_integration.py`**

```python"""
Integration tests for grammar-constrained generation.
"""

import json
import pytest
from mlx_lm import load, generate_constrained, generate_tool_call


@pytest.mark.slow
def test_constrained_json_generation(model, tokenizer):
    """Test generating valid JSON."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"}
        },
        "required": ["name", "count"]
    }

    prompt = "Generate an object with name and count"
    output = generate_constrained(
        model,
        prompt,
        tokenizer,
        schema=schema,
        max_tokens=50
    )

    # Verify valid JSON
    result = json.loads(output)
    assert "name" in result
    assert "count" in result
    assert isinstance(result["count"], int)


@pytest.mark.slow
def test_tool_call_generation(model, tokenizer):
    """Test tool call generation."""
    tools = [
        {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string"}
                }
            }
        }
    ]

    prompt = "Call test_func with arg1"
    tool_call = generate_tool_call(
        model,
        prompt,
        tokenizer,
        tools=tools,
        max_tokens=100
    )

    assert tool_call is not None
    assert tool_call["name"] == "test_func"
    assert "arg1" in tool_call["arguments"]


@pytest.mark.slow
def test_grammar_termination(model, tokenizer):
    """Test that grammar terminates properly."""
    schema = {
        "type": "string"
    }

    prompt = "Generate a string"
    output = generate_constrained(
        model,
        prompt,
        tokenizer,
        schema=schema,
        max_tokens=100
    )

    # Should get valid output without error
    assert isinstance(output, str)
    assert len(output) > 0
```

## Step 7: Documentation

### 7.1 API Documentation

Add to `mlx_lm/docs/grammar.md`:

```markdown
# Grammar-Constrained Generation

MLX-LM supports grammar-constrained generation for structured output and reliable tool calling.

## Basic Usage

### JSON Schema Constraints

```python
from mlx_lm import load, generate_constrained

model, tokenizer = load("model-name")

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"}
    }
}

output = generate_constrained(
    model,
    "Generate a user profile",
    tokenizer,
    schema=schema
)
```

### Tool Calling

```python
from mlx_lm import generate_tool_call

tools = [{
    "name": "my_tool",
    "parameters": {
        "type": "object",
        "properties": {
            "arg": {"type": "string"}
        }
    }
}]

tool_call = generate_tool_call(
    model,
    "Use my_tool",
    tokenizer,
    tools=tools
)
```
```

## Next Steps

1. Implement Phase 1 (foundation)
2. Add comprehensive tests
3. Benchmark performance
4. Implement Phase 2 (tool calling)
5. Optimize (Phase 3)
6. Add advanced features (Phase 4)

---

**Document Version:** 1.0
**Last Updated:** 2024
