# Implementation Architecture: Grammar-Constrained Tool Calling for MLX-LM

## Executive Summary

After reviewing the 11 previous research documents, I've synthesized a concrete implementation plan. The architecture leverages LLGuidance's Rust core for efficient grammar parsing while providing a clean Python API that integrates seamlessly with mlx-lm's existing generation infrastructure.

## Core Design Decisions

### Decision 1: Use LLGuidance as Primary Grammar Engine

**Rationale**:
- Rust core provides 1-10μs token mask computation (vs 50-100μs for pure Python)
- Already supports Lark grammar syntax and JSON schema
- Maintains GBNF compatibility with llama.cpp
- Well-tested, battle-hardened codebase

**Alternative Considered**: Custom PEG parser from scratch
- Rejected due to development time and error-prone implementation
- Would require extensive testing for edge cases

### Decision 2: Logits Processor Integration Pattern

**Approach**: Implement grammar constraints as a `logits_processor` compatible with mlx-lm's existing hook.

**Why**:
```python
# From mlx_lm/generate.py:493-500
if logits_processors and len(input_tokens) > 0:
    tokens = mx.concat([tokens, input_tokens])
    for processor in logits_processors:
        logits = processor(tokens, logits)
```

This is the **most minimal-change approach**:
1. No modifications to `generate_step` signature
2. Works with all generation modes (stream, batch, speculative)
3. Composable with existing processors (repetition penalty, etc.)

### Decision 3: Lazy Grammar State Management

The grammar state is managed inside the logits processor itself, rather than passed as a separate parameter. This simplifies the API:

```python
# Usage
grammar = LLGuidanceState.from_json_schema(tokenizer, schema)
processor = GrammarLogitsProcessor(grammar)
response = generate(model, prompt, tokenizer, logits_processors=[processor])
```

### Decision 4: Fallback Strategy for Missing LLGuidance

If LLGuidance is not installed, the system falls back to:
1. Post-generation JSON parsing (existing behavior)
2. Warning message suggesting grammar constraints

This ensures backward compatibility.

## Implementation Architecture

```
mlx_lm/
├── grammar/                     # NEW PACKAGE
│   ├── __init__.py             # Public API exports
│   ├── base.py                 # GrammarState ABC
│   ├── llguidance_adapter.py   # LLGuidance integration
│   ├── mask_ops.py             # MLX-optimized mask operations
│   └── tool_schema.py          # Tool schema → grammar conversion
├── sample_utils.py             # Add GrammarLogitsProcessor
└── tool_parsers/
    └── grammar_constrained.py  # Grammar-aware tool parser
```

## Core Classes

### 1. GrammarState (Abstract Base Class)

```python
# mlx_lm/grammar/base.py
from abc import ABC, abstractmethod
from typing import Optional
import mlx.core as mx

class GrammarState(ABC):
    """
    Abstract base class for grammar-constrained generation state.

    A GrammarState tracks the current position in a grammar during token
    generation and provides token masks to constrain the model's output.
    """

    @abstractmethod
    def get_token_mask(self) -> mx.array:
        """
        Get boolean mask of allowed tokens for current grammar position.

        Returns:
            mx.array of shape (vocab_size,) with True for allowed tokens.
        """
        pass

    @abstractmethod
    def update(self, token_id: int) -> None:
        """
        Update grammar state with the generated token.

        Args:
            token_id: The token that was generated.
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if the grammar has been fully satisfied.

        Returns:
            True if generation should stop (grammar complete).
        """
        pass

    @property
    @abstractmethod
    def partial_output(self) -> str:
        """
        Get the partial UTF-8 output generated so far.
        """
        pass
```

### 2. LLGuidanceState (Concrete Implementation)

```python
# mlx_lm/grammar/llguidance_adapter.py
from typing import Union, Dict, Any, Optional
import json
import mlx.core as mx
import numpy as np

from .base import GrammarState

# Lazy import to handle optional dependency
_llguidance = None

def _get_llguidance():
    global _llguidance
    if _llguidance is None:
        try:
            import llguidance
            _llguidance = llguidance
        except ImportError:
            raise ImportError(
                "llguidance is required for grammar-constrained generation. "
                "Install with: pip install llguidance"
            )
    return _llguidance

class LLGuidanceState(GrammarState):
    """
    Grammar state implementation using LLGuidance.

    LLGuidance provides a Rust-based grammar engine with Lark grammar
    support and efficient token mask computation.
    """

    def __init__(self, tokenizer, grammar: str):
        """
        Initialize from a Lark grammar string.

        Args:
            tokenizer: HuggingFace tokenizer (or TokenizerWrapper)
            grammar: Lark grammar string
        """
        llg = _get_llguidance()

        # Handle TokenizerWrapper
        hf_tokenizer = getattr(tokenizer, '_tokenizer', tokenizer)

        self._ll_tokenizer = llg.LLTokenizer.from_hf_tokenizer(hf_tokenizer)
        self._matcher = llg.LLMatcher(self._ll_tokenizer, grammar)
        self._vocab_size = hf_tokenizer.vocab_size
        self._completed = False

    @classmethod
    def from_json_schema(cls, tokenizer, schema: Union[str, Dict[str, Any]]) -> 'LLGuidanceState':
        """
        Create grammar state from JSON schema.

        Args:
            tokenizer: HuggingFace tokenizer
            schema: JSON schema as dict or JSON string
        """
        llg = _get_llguidance()

        if isinstance(schema, dict):
            schema = json.dumps(schema)

        grammar = llg.grammar_from("json_schema", schema)
        return cls(tokenizer, grammar)

    @classmethod
    def from_regex(cls, tokenizer, pattern: str) -> 'LLGuidanceState':
        """
        Create grammar state from regex pattern.

        Args:
            tokenizer: HuggingFace tokenizer
            pattern: Regular expression pattern
        """
        llg = _get_llguidance()
        grammar = llg.grammar_from("regex", pattern)
        return cls(tokenizer, grammar)

    @classmethod
    def from_choices(cls, tokenizer, choices: list) -> 'LLGuidanceState':
        """
        Create grammar state for selecting from choices.

        Args:
            tokenizer: HuggingFace tokenizer
            choices: List of allowed string values
        """
        llg = _get_llguidance()
        grammar = llg.grammar_from("choice", json.dumps(choices))
        return cls(tokenizer, grammar)

    def get_token_mask(self) -> mx.array:
        """Get boolean mask of allowed tokens."""
        if self._completed:
            # If complete, only allow EOS-like tokens
            return mx.zeros((self._vocab_size,), dtype=mx.bool_)

        mask_np = self._matcher.get_token_mask()
        return mx.array(mask_np.astype(np.bool_))

    def update(self, token_id: int) -> None:
        """Update grammar state with generated token."""
        if not self._completed:
            self._matcher.commit_token(token_id)
            self._completed = self._matcher.is_terminated()

    def is_complete(self) -> bool:
        """Check if grammar is complete."""
        return self._completed

    @property
    def partial_output(self) -> str:
        """Get partial UTF-8 output."""
        return self._matcher.get_partial_utf8()
```

### 3. GrammarLogitsProcessor

```python
# In mlx_lm/sample_utils.py (add to existing file)

from typing import Optional
import mlx.core as mx

class GrammarLogitsProcessor:
    """
    Logits processor that applies grammar constraints during generation.

    Integrates with mlx-lm's existing logits_processors mechanism.
    """

    def __init__(self, grammar_state: 'GrammarState'):
        """
        Args:
            grammar_state: Grammar state object implementing get_token_mask()
        """
        self.grammar = grammar_state
        self._last_token = None

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply grammar constraint to logits.

        Args:
            tokens: Previously generated tokens (unused, for API compat)
            logits: Logits from the model for next token prediction

        Returns:
            Constrained logits with -inf for disallowed tokens
        """
        # Update grammar state with previous token (if any)
        if self._last_token is not None and not self.grammar.is_complete():
            self.grammar.update(self._last_token)

        if self.grammar.is_complete():
            return logits  # No more constraints

        # Get token mask from grammar
        mask = self.grammar.get_token_mask()

        # Apply constraint: set disallowed tokens to -inf
        constrained = mx.where(
            mask,
            logits,
            mx.full_like(logits, float('-inf'))
        )

        return constrained

    def on_token_generated(self, token_id: int):
        """
        Callback to track generated tokens.

        Should be called after each token is generated.
        """
        self._last_token = token_id

    @property
    def is_complete(self) -> bool:
        """Check if grammar generation is complete."""
        return self.grammar.is_complete()
```

### 4. Tool Schema Converter

```python
# mlx_lm/grammar/tool_schema.py
from typing import List, Dict, Any, Optional
import json

def build_tool_grammar(
    tools: List[Dict[str, Any]],
    format_type: str = "json",
    markers: Optional[Dict[str, str]] = None
) -> str:
    """
    Build a Lark grammar for tool calling from tool definitions.

    Args:
        tools: List of tool definitions with name and parameters
        format_type: Output format ("json" or "xml")
        markers: Optional start/end markers (e.g., {"start": "<|tool_call|>", "end": "<|/tool_call|>"})

    Returns:
        Lark grammar string
    """
    if format_type == "json":
        return _build_json_tool_grammar(tools, markers)
    elif format_type == "xml":
        return _build_xml_tool_grammar(tools, markers)
    else:
        raise ValueError(f"Unknown format_type: {format_type}")

def _build_json_tool_grammar(
    tools: List[Dict[str, Any]],
    markers: Optional[Dict[str, str]] = None
) -> str:
    """Build JSON-style tool call grammar."""

    # Build anyOf schema for all tools
    tool_schemas = []
    for tool in tools:
        tool_schema = {
            "type": "object",
            "properties": {
                "name": {"const": tool["name"]},
                "arguments": tool.get("parameters", {"type": "object"})
            },
            "required": ["name", "arguments"]
        }
        tool_schemas.append(tool_schema)

    combined_schema = {
        "anyOf": tool_schemas
    }

    # Build Lark grammar with optional markers
    grammar_parts = ["start:"]

    if markers and markers.get("start"):
        grammar_parts.append(f'"{markers["start"]}"')

    # Use %json directive for schema
    grammar_parts.append(f"%json{{{json.dumps(combined_schema)}}}")

    if markers and markers.get("end"):
        grammar_parts.append(f'"{markers["end"]}"')

    return " ".join(grammar_parts)

def _build_xml_tool_grammar(
    tools: List[Dict[str, Any]],
    markers: Optional[Dict[str, str]] = None
) -> str:
    """Build XML-style tool call grammar."""

    # Build choice of tool names
    tool_names = [f'"{tool["name"]}"' for tool in tools]
    name_choice = " | ".join(tool_names)

    grammar = f'''
start: tool_call
tool_call: "<function=" name ">" args "</function>"
name: {name_choice}
args: /[^<]*/
'''
    return grammar.strip()

def tools_from_chat_template(tokenizer) -> Optional[Dict[str, Any]]:
    """
    Extract tool calling format from chat template.

    Analyzes the Jinja2 chat template to detect:
    - Tool call markers (start/end tokens)
    - Output format (JSON, XML, etc.)
    - Field names and structure

    Args:
        tokenizer: Tokenizer with chat_template attribute

    Returns:
        Dict with format specification or None if no tool calling detected
    """
    template = getattr(tokenizer, 'chat_template', None)
    if not template:
        return None

    # Detect common patterns
    format_spec = {}

    # Check for JSON tool call pattern
    if 'tool_call' in template and ('json' in template.lower() or '"name"' in template):
        format_spec['type'] = 'json'

        # Detect markers
        if '<|tool_call|>' in template:
            format_spec['markers'] = {
                'start': '<|tool_call|>',
                'end': '</tool_call>' if '</tool_call>' in template else '<|/tool_call|>'
            }
        elif '<tool_call>' in template:
            format_spec['markers'] = {
                'start': '<tool_call>',
                'end': '</tool_call>'
            }

    # Check for XML/function style
    elif '<function=' in template:
        format_spec['type'] = 'xml'

    return format_spec if format_spec else None
```

## Integration with Existing Code

### Changes to sample_utils.py

Add the `GrammarLogitsProcessor` class and import it in `__init__`:

```python
# At the end of mlx_lm/sample_utils.py

# Import for public API
__all__ = [
    'make_sampler',
    'make_logits_processors',
    'GrammarLogitsProcessor',  # NEW
    # ... existing exports
]
```

### High-Level Usage API

```python
# Example usage: mlx_lm/examples/constrained_tool_use.py

from mlx_lm import load, generate
from mlx_lm.grammar import LLGuidanceState
from mlx_lm.sample_utils import GrammarLogitsProcessor

# Load model
model, tokenizer = load("meta-llama/Llama-3.1-8B-Instruct")

# Define tools
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

# Build grammar from tools
from mlx_lm.grammar.tool_schema import build_tool_grammar
grammar_str = build_tool_grammar(tools, format_type="json")

# Create grammar state
grammar = LLGuidanceState(tokenizer, grammar_str)

# Create logits processor
processor = GrammarLogitsProcessor(grammar)

# Generate with constraints
prompt = tokenizer.apply_chat_template([
    {"role": "user", "content": "What's the weather in Paris?"}
], tokenize=False, add_generation_prompt=True)

response = generate(
    model,
    tokenizer,
    prompt,
    logits_processors=[processor]
)

# Response is guaranteed to be valid JSON matching the tool schema
print(response)
# Output: {"name": "get_weather", "arguments": {"city": "Paris"}}
```

## Token Mask Application Strategy

### Efficient MLX Operations

```python
# mlx_lm/grammar/mask_ops.py

import mlx.core as mx
import numpy as np

def apply_grammar_mask(logits: mx.array, mask: mx.array) -> mx.array:
    """
    Apply boolean token mask to logits.

    Args:
        logits: Shape (batch, vocab_size) or (vocab_size,)
        mask: Boolean shape (vocab_size,) with True for allowed tokens

    Returns:
        Constrained logits with -inf for disallowed tokens
    """
    # Ensure mask is boolean
    if mask.dtype != mx.bool_:
        mask = mask.astype(mx.bool_)

    # Use where for efficient masking
    return mx.where(mask, logits, float('-inf'))

def bitmask_to_bool(bitmask: np.ndarray, vocab_size: int) -> mx.array:
    """
    Convert compressed bitmask to boolean array.

    LLGuidance returns bitmasks as uint32 arrays for efficiency.
    This expands them to full boolean arrays for MLX.

    Args:
        bitmask: Shape (vocab_size // 32 + 1,) uint32 array
        vocab_size: Total vocabulary size

    Returns:
        Boolean array of shape (vocab_size,)
    """
    # Expand bitmask to boolean
    result = np.zeros(vocab_size, dtype=np.bool_)

    for i, word in enumerate(bitmask):
        for bit in range(32):
            token_id = i * 32 + bit
            if token_id < vocab_size:
                result[token_id] = bool(word & (1 << bit))

    return mx.array(result)
```

## Error Handling

### No Valid Tokens Scenario

When the grammar state has no valid next tokens (edge case), the processor should:
1. Log a warning
2. Return original logits to allow any token
3. Mark the grammar as failed

```python
def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
    if self.grammar.is_complete():
        return logits

    mask = self.grammar.get_token_mask()

    # Check if any tokens are allowed
    if not mx.any(mask):
        import warnings
        warnings.warn(
            "Grammar constraint has no valid tokens. "
            "Falling back to unconstrained generation."
        )
        return logits

    return mx.where(mask, logits, float('-inf'))
```

## Performance Optimization Notes

### Grammar Caching

For repeated generations with the same tools:

```python
# Cache compiled grammars
_grammar_cache = {}

def get_cached_grammar(tools_hash: str, tokenizer) -> str:
    if tools_hash not in _grammar_cache:
        _grammar_cache[tools_hash] = build_tool_grammar(tools)
    return _grammar_cache[tools_hash]
```

### Batch Generation Considerations

For batch generation, each batch item may need different grammar states. The current design supports this by creating separate `GrammarLogitsProcessor` instances per batch item.

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
   - `test_grammar_base.py`: GrammarState interface
   - `test_llguidance_adapter.py`: LLGuidance integration
   - `test_tool_schema.py`: Schema to grammar conversion
   - `test_mask_ops.py`: MLX mask operations

2. **Integration Tests**: Test the full pipeline
   - `test_constrained_generation.py`: End-to-end generation

3. **Benchmark Tests**: Performance regression tests
   - Token mask computation time
   - Full generation overhead

## Dependencies

### Required
- `mlx>=0.6.0` (existing)
- `transformers>=4.30.0` (existing)

### Optional (for grammar constraints)
- `llguidance>=1.4.0` (new, optional)

### Installation
```bash
pip install llguidance
```

## Summary

This architecture provides:
1. **Minimal changes** to existing mlx-lm code
2. **Clean separation** via the grammar package
3. **Efficient implementation** using LLGuidance's Rust core
4. **Backward compatibility** with graceful fallback
5. **Flexible API** supporting JSON schema, regex, and custom grammars

The implementation follows mlx-lm's existing patterns and integrates naturally with the logits processor system.

---

**Document Version**: 1.0
**Author**: Claude (Implementation Agent)
**Date**: 2025-01-28
**Status**: Ready for Implementation
