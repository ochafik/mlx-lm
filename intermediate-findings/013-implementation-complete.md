# Implementation Complete: Grammar-Constrained Tool Calling for MLX-LM

## Summary

This document summarizes the complete implementation of a grammar-constrained tool calling system for mlx-lm, inspired by llama.cpp's "autoparser" approach but using LLGuidance and Lark grammar syntax.

## Implemented Components

### 1. Grammar Package (`mlx_lm/grammar/`)

#### `base.py` - Abstract Base Class
- `GrammarState`: Abstract interface for grammar state tracking
- Methods: `get_token_mask()`, `update()`, `is_complete()`, `reset()`, `clone()`

#### `jinja_analysis.py` - Template Differential Analysis
- `JinjaTemplateAnalyzer`: Analyzes Jinja chat templates
- `ToolCallFormat`: Data class for detected format specification
- `analyze_chat_template()`: Main entry point for template analysis
- **Key Innovation**: Uses differential rendering to detect tool call format

#### `tool_schema.py` - Tool Schema to Grammar Conversion
- `build_tool_grammar()`: Converts tool definitions to Lark grammar
- `tools_from_functions()`: Creates tool schemas from Python functions
- `build_multi_tool_grammar()`: Supports multiple sequential tool calls
- Automatically uses detected format from Jinja template analysis

#### `mask_ops.py` - MLX Token Mask Operations
- `apply_grammar_mask()`: Efficiently applies boolean masks to logits
- `bitmask_to_bool()` / `bool_to_bitmask()`: Compression utilities
- `count_allowed_tokens()` / `get_allowed_token_ids()`: Debug helpers

#### `llguidance_adapter.py` - LLGuidance Integration
- `LLGuidanceState`: Full grammar state implementation using LLGuidance
- Factory methods: `from_json_schema()`, `from_regex()`, `from_choices()`, `from_tools()`
- `MockGrammarState`: Testing without LLGuidance dependency
- `is_llguidance_available()`: Runtime dependency check

### 2. Sample Utils Integration (`mlx_lm/sample_utils.py`)

#### `GrammarLogitsProcessor`
- Compatible with existing `logits_processors` mechanism
- Handles batch dimensions correctly
- Tracks completion state
- Warns on empty masks

#### `make_grammar_logits_processor()`
- Convenience factory function
- Supports: `json_schema`, `regex`, `choices`, `tools`, `grammar`

### 3. Test Suite (`tests/test_grammar.py`)

22 comprehensive tests covering:
- Jinja template analysis
- Tool schema conversion
- Mask operations
- Logits processor
- Mock grammar state
- Tokenizer integration

## How It Works

### The "Autoparser" Approach

1. **Template Analysis**: When tools are provided, the system analyzes the model's Jinja chat template using differential rendering

2. **Format Detection**: By rendering with dummy tool calls, we extract:
   - Start/end markers (e.g., `<tool_call>`, `</tool_call>`)
   - JSON structure (field names, ordering)
   - Format type (JSON, XML, Python-style)

3. **Grammar Generation**: The detected format is used to generate a Lark grammar that matches the model's expected output format

4. **Constrained Generation**: During generation, the grammar state provides token masks that constrain the model to produce valid output

### Example Flow

```python
# 1. Load model
model, tokenizer = load("model-path")

# 2. Define tools
tools = [{"name": "get_weather", "parameters": {...}}]

# 3. Analyze template (automatic)
# The system detects: JSON format with <tool_call> markers

# 4. Build grammar (automatic)
grammar = build_tool_grammar(tools, tokenizer)

# 5. Create processor
processor = make_grammar_logits_processor(tokenizer, tools=tools)

# 6. Generate with constraints
response = generate(model, tokenizer, prompt, logits_processors=[processor])
# Output is guaranteed valid JSON matching tool schema
```

## Key Design Decisions

1. **LLGuidance over Custom Parser**: Uses battle-tested Rust implementation for reliability

2. **Logits Processor Pattern**: Integrates with existing mlx-lm infrastructure without modifying core generation loop

3. **Lazy Dependencies**: LLGuidance is optional - system degrades gracefully

4. **Differential Analysis**: Auto-detects format instead of requiring manual configuration per model

## Dependencies

- **Required**: `mlx`, `transformers`, `jinja2`
- **Optional**: `llguidance` (for grammar constraints)

## Installation

```bash
# For grammar constraints
pip install llguidance
```

## Usage Examples

### Basic JSON Schema Constraint

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_grammar_logits_processor

model, tokenizer = load("model-path")

processor = make_grammar_logits_processor(
    tokenizer,
    json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
)

response = generate(model, tokenizer, prompt, logits_processors=[processor])
```

### Tool Calling

```python
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }
]

processor = make_grammar_logits_processor(tokenizer, tools=tools)
response = generate(model, tokenizer, prompt, logits_processors=[processor])
```

### Regex Constraint

```python
processor = make_grammar_logits_processor(
    tokenizer,
    regex=r"[a-z]+@[a-z]+\.[a-z]+"  # Email pattern
)
```

## File Summary

### New Files Created

```
mlx_lm/grammar/
├── __init__.py           # Package exports
├── base.py               # GrammarState ABC
├── jinja_analysis.py     # Template differential analysis
├── tool_schema.py        # Tool schema to grammar
├── mask_ops.py           # MLX mask operations
└── llguidance_adapter.py # LLGuidance integration

tests/
└── test_grammar.py       # Comprehensive test suite

examples/
└── constrained_tool_use.py  # End-to-end example

intermediate-findings/
├── 012-implementation-architecture.md  # Architecture decisions
└── 013-implementation-complete.md      # This summary
```

### Modified Files

```
mlx_lm/sample_utils.py    # Added GrammarLogitsProcessor
```

## Test Results

```
======================== 22 passed, 0 failed ========================
```

## Future Enhancements

1. **Batch Grammar States**: Different grammars per batch item for batch generation

2. **Grammar Caching**: Cache compiled grammars for repeated use

3. **Streaming Integration**: Better integration with streaming generation

4. **More Format Types**: Support for additional tool call formats

5. **Native MLX Kernels**: Metal shaders for mask operations (currently uses numpy)

---

**Status**: Implementation Complete
**Author**: Claude (Implementation Agent)
**Date**: 2025-01-28
