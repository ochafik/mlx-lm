# LLGuidance Library: Comprehensive Analysis

## Overview

LLGuidance is a low-level grammar-constrained generation library built in Rust with Python bindings. It provides the foundation for structured generation by compiling grammars to token-level constraints that can be applied during LLM inference.

## Core Architecture

### Component Structure

```
llguidance/
├── parser/              # Rust core implementation
│   ├── earley_parser.rs     # Earley parser implementation
│   ├── grammar.rs            # Grammar compilation
│   ├── json_schema.rs        # JSON schema to grammar
│   └── tokenizer.rs          # Tokenizer abstraction
├── python/             # Python bindings
│   ├── llguidance/
│   │   ├── __init__.py
│   │   ├── matcher.py        # LLMatcher class
│   │   ├── tokenizer.py      # LLTokenizer class
│   │   └── executor.py       # LLExecutor class
└── integrations/       # Framework integrations
    ├── torch.py             # PyTorch JIT kernels
    ├── mlx.py               # MLX Metal shaders
    ├── hf.py                # HuggingFace tokenizers
    └── llamacpp.py          # llama.cpp vocab
```

### Key Design Decisions

1. **Rust Core**: Performance-critical parsing in Rust
2. **PyO3 Bindings**: Zero-copy Python-Rust interop
3. **Earley Parser**: Handles arbitrary context-free grammars
4. **Bitmask Compression**: 32-bit integers represent token constraints
5. **Tokenizer Abstraction**: Works with any tokenizer backend

## Core Classes and APIs

### 1. LLMatcher - Grammar Constraint Engine

```python
from llguidance import LLMatcher

# Create matcher from grammar
matcher = LLMatcher(
    tokenizer=ll_tokenizer,
    grammar='start: "Hello" NAME',
    allow_indent=False
)

# Get token mask for current position
mask = matcher.get_token_mask()  # Returns numpy array of bools

# Update state with new token
matcher.commit_token(token_id)

# Check if grammar is complete
is_complete = matcher.is_terminated()

# Get partial UTF-8 representation
utf8_bytes = matcher.get_partial_utf8()
```

### 2. LLTokenizer - Tokenizer Abstraction

```python
from llguidance import LLTokenizer

# Create from HuggingFace tokenizer
from transformers import AutoTokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
ll_tokenizer = LLTokenizer.from_hf_tokenizer(hf_tokenizer)

# Or from configuration
ll_tokenizer = LLTokenizer(config='{
    "type": "huggingface",
    "tokenizer": "meta-llama/Llama-3.1-8B"
}')

# Key methods:
tokens = ll_tokenizer.tokenize("Hello world")
token_str = ll_tokenizer.token_id_to_str(1234)
special_tokens = ll_tokenizer.get_special_token_ids()
```

### 3. LLExecutor - Parallel Processing

```python
from llguidance import LLExecutor

# Execute multiple grammars in parallel
executor = LLExecutor()
matchers = [
    LLMatcher(tokenizer, grammar1),
    LLMatcher(tokenizer, grammar2),
    LLMatcher(tokenizer, grammar3)
]

# Batch process - critical for performance
masks = executor.get_token_mask_parallel(matchers)
```

### 4. Grammar Compilation

```python
from llguidance import grammar_from

# JSON Schema to Grammar
json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    }
}
grammar = grammar_from("json_schema", json.dumps(json_schema))

# Regex to Grammar
grammar = grammar_from("regex", r"\d{3}-\d{3}-\d{4}")

# Choice List to Grammar
grammar = grammar_from("choice", '["red", "green", "blue"]')

# Lark Grammar with Extensions
grammar = grammar_from("lark", """
start: greeting | farewell
greeting: "Hello" NAME "!"
farewell: "Goodbye" NAME "!"
NAME: /[a-zA-Z]+/
""")
```

## Grammar Syntax Extensions

LLGuidance extends Lark grammar syntax with special features:

### 1. Special Tokens

```lark
start: TEXT | tool_call
TEXT: /[^{](.|\n)*/                           # Regular text
tool_call: <|python_tag|> json_body <|eom_id|>  # Special token markers
```

Special tokens are prefixed with `<|` and suffixed with `|>`.

### 2. Inline JSON Schemas

```lark
start: tool_call
tool_call: %json{
    "type": "object",
    "properties": {
        "name": {"const": "get_weather"},
        "arguments": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
}
```

### 3. Parametric Grammars

```lark
# Unique list generation
start: "[" %unique(WORD) * "]"
WORD: /[a-z]+/

# Permutation (all items exactly once)
start: digits=6 %permutation([0-9])
```

### 4. Lazy Lexemes

```lark
# Efficient boundary handling
start: WORD lazy_boundary
lazy_boundary: |1|  # Lookahead 1 character
WORD: /[a-zA-Z]+/
```

## JSON Schema Support

### Supported Schema Features

```python
# Core types
"type": "string" | "number" | "integer" | "boolean" | "object" | "array" | "null"

# Object constraints
"properties": {...}           # Property definitions
"required": ["prop1", ...]    # Required properties
"additionalProperties": {...} # Schema for additional props
"patternProperties": {...}    # Pattern-based properties

# Array constraints
"items": {...}                # Schema for all items
"prefixItems": [{...}, ...]   # Schema for tuple validation
"minItems": 0                 # Minimum items
"maxItems": 10                # Maximum items
"uniqueItems": true           # All items must be unique

# String constraints
"minLength": 1                # Minimum length
"maxLength": 100              # Maximum length
"pattern": "^[a-z]+$"         # Regex pattern
"format": "email"             # Format validation (74% support)

# Number constraints
"minimum": 0                  # Minimum value
"maximum": 100                # Maximum value
"exclusiveMinimum": 0         # Exclusive minimum
"exclusiveMaximum": 100       # Exclusive maximum
"multipleOf": 0.5             # Must be multiple of

# Composition
"anyOf": [{...}, ...]         # Must match one
"allOf": [{...}, ...]         # Must match all
"oneOf": [{...}, ...]         # Must match exactly one

# Other
"enum": ["a", "b", "c"]       # Enumerated values
"const": "fixed_value"        # Constant value
"$ref": "#/definitions/..."   # Reference
```

### Configuration via x-guidance

```python
schema = {
    "type": "object",
    "x-guidance": {
        "whitespace_flexible": False,
        "item_separator": "\\s{0,2},\\s{0,2}",
        "key_separator": "\\s{0,2}:\\s{0,2}",
        "pattern_to_alternatives": True
    },
    "properties": {
        "a": {"type": "number"}
    }
}
```

## Framework Integration

### MLX Integration

```python
# From llguidance.mlx import MLXDevice
import llguidance.mlx as llmlx

# Create MLX-optimized matcher
device = llmlx.MLXDevice()
matcher = LLMatcher(
    tokenizer=ll_tokenizer,
    grammar=grammar,
    device=device
)

# Get MLX array mask
mask = matcher.get_token_mask_mlx()  # Returns mx.array
```

The MLX integration uses Metal shader kernels for:
- Bitmask application
- Parallel token constraint evaluation
- GPU-accelerated mask computation

### PyTorch Integration

```python
import torch
from llguidance.torch import TorchDevice

device = TorchDevice()
matcher = LLMatcher(tokenizer, grammar, device=device)

# JIT-compiled mask application
mask_tensor = matcher.get_token_mask_torch()  # Returns torch.Tensor

# Apply to logits
logits[~mask] = -float('inf')
```

## Performance Characteristics

### Benchmarks

From llguidance test suite:

| Operation | Time | Notes |
|-----------|------|-------|
| Single grammar mask | ~1-5 μs | Depends on grammar complexity |
| 50 parallel grammars | ~20-40 μs | Using LLExecutor |
| JSON schema compilation | ~100-500 μs | One-time cost per schema |
| Token state update | ~1-2 μs | Per token generated |

### Memory Efficiency

- **Bitmask Compression**: 32x reduction vs boolean array
- **Lazy Evaluation**: Grammar states only when needed
- **Zero-Copy**: PyO3 avoids data copying between Rust/Python

### Optimization Techniques

1. **Left-Recursion Handling**: Efficient repetition without stack overflow
2. **Regex Vectorization**: Compile multiple regex patterns together
3. **Token Trie**: Efficient vocabulary lookup for regex patterns
4. **Slicing**: Configurable token slice for optimization

## Tool Calling Examples

### Model-Specific Grammars

```python
# Phi-4 Mini format
phi4_grammar = """
start: <|tool_call|> tools <|/tool_call|>
tools: %json{
    "type": "array",
    "items": {
        "anyOf": [
            {"$ref": "#/$defs/tool_gmail_search"},
            {"$ref": "#/$defs/tool_gmail_create"}
        ]
    },
    "minItems": 1
}
"""

# Qwen-3 format
qwen_grammar = """
start: tool*
tool: "<tool_call>" name "</tool_call>" parameters
name: %json{"type": "string", "const": "search"}
parameters: "<parameter>" "<item>" key "=" value "</item>" "</parameter>"
"""

# Llama-3.1 format
llama_grammar = """
start: ("Move " digit " " ("right" | "left"))*
digit: /[0-9]+/
"""
```

### Multi-Tool Schema

```python
# Define multiple tools
tools_schema = {
    "type": "array",
    "items": {
        "anyOf": [
            {
                "type": "object",
                "properties": {
                    "name": {"const": "get_weather"},
                    "arguments": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "object",
                "properties": {
                    "name": {"const": "calculator"},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                            "precision": {"type": "integer"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ]
    }
}

# Compile to grammar
grammar = grammar_from("json_schema", json.dumps(tools_schema))
```

## MLX-LM Integration Strategy

### Phase 1: Direct Integration

```python
# In mlx_lm/sample_utils.py
from llguidance import LLMatcher, LLTokenizer

class GrammarConstrainedSampler:
    def __init__(self, tokenizer, grammar):
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)
        self.matcher = LLMatcher(self.ll_tokenizer, grammar)

    def apply_constraint(self, logits: mx.array) -> mx.array:
        """Apply grammar mask to logits"""
        mask = self.matcher.get_token_mask()
        mask_mx = mx.array(mask)
        return mx.where(mask_mx, logits, -mx.inf)

    def update(self, token: int):
        """Update grammar state"""
        self.matcher.commit_token(token)
        return not self.matcher.is_terminated()
```

### Phase 2: MLX Kernels

For better performance, use MLX-specific operations:

```python
# Use MLX executor from llguidance.mlx
import llguidance.mlx as llmlx

device = llmlx.MLXDevice()
matcher = LLMatcher(tokenizer, grammar, device=device)

# Get MLX-native mask
mask = matcher.get_token_mask_mlx()  # mx.array
constrained_logits = logits * mask - mx.inf * (1 - mask)
```

### Phase 3: Native MLX Implementation

For maximum performance, reimplement core algorithms in MLX:

```python
# Native MLX bitmask operations
class MLXGrammarConstraint:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.mask_size = (vocab_size + 31) // 32
        self.masks = mx.zeros((self.mask_size,), dtype=mx.uint32)

    def set_allowed(self, token_id: int):
        idx = token_id // 32
        bit = token_id % 32
        self.masks = mx.zeros((self.mask_size,), dtype=mx.uint32)
        self.masks[idx] |= (1 << bit)

    def apply(self, logits: mx.array) -> mx.array:
        # Expand masks to full vocabulary
        full_mask = mx.zeros((self.vocab_size,), dtype=mx.bool_)
        for i in range(self.mask_size):
            start = i * 32
            end = min(start + 32, self.vocab_size)
            full_mask[start:end] = ((self.masks[i].astype(mx.uint32) >> np.arange(32)) & 1).astype(mx.bool_)[:end-start]
        return mx.where(full_mask, logits, -mx.inf)
```

## Key Dependencies

```toml
# llguidance Rust dependencies
[dependencies]
pyo3 = "0.20"
numpy = "0.20"
serde_json = "1.0"
regex = "1.10"
lark = "1.1"

# Python dependencies
llguidance==1.4.0
numpy
```

## Open Questions

1. **Custom Tokenizers**: How to integrate with custom tokenizer implementations?
2. **Streaming Generation**: Best practices for updating grammar state during streaming?
3. **Error Handling**: What happens when no tokens match the grammar?
4. **Recursive Grammars**: Performance implications for highly recursive schemas?

## References

- LLGuidance GitHub: https://github.com/guidance-ai/llguidance
- LLGuidance Documentation: https://guidance-ai.github.io/llguidance/
- Earley Parsing: https://en.wikipedia.org/wiki/Earley_parser
- JSON Schema: https://json-schema.org/
