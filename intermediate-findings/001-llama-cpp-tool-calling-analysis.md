# LLama.cpp Automatic Grammar Generation for Tool Calling

## Overview

This document summarizes findings from analyzing llama.cpp's approach to automatic grammar and parser generation for tool calling. The system uses differential analysis of Jinja templates to dynamically generate grammars that constrain model output.

## Core Architecture

### Template Format Detection

The system in `common/chat.cpp` uses pattern matching to detect different tool call formats:

```cpp
// Format detection patterns (from llama.cpp source)
- "tool_call" format: Special token-based tool calls (e.g., Phi-4)
- "python_tool" format: Python-style function calls
- "lambda" format: Lambda-based expressions
- "raw" format: Raw text without special formatting
```

### Grammar Generation System

The grammar compiler in `llama.h` and `llama-grammar.cpp` provides:

1. **GBNF (Guidance BNF) Format**: Extended BNF grammar syntax
2. **JSON Schema Compilation**: Converts JSON schemas to grammars
3. **Token Bitmask Application**: Constrains logits using bitmasks
4. **Pushdown Automaton**: Stack-based parser state tracking

## Key Implementation Details

### 1. Grammar Compilation Pipeline

```
Jinja Template → Differential Analysis → Format Detection → Grammar Generation → Token Constraints
```

### 2. Format-Specific Grammars

#### Python Tool Format
```lark
tool_call: functionName "{" parameters "}"
parameters: key "=" value ("," key "=" value)*
```

#### Special Token Format
```lark
tool_call: "<|tool_call|>" json_body "<|/tool_call|>"
```

#### Lambda Format
```lark
tool_call: "auto_tool_call(" function_name "," parameters ")"
```

### 3. Token Constraint Application

The system applies grammar constraints through:

```cpp
// Apply token mask at each generation step
void llama_grammar_sample(
    struct llama_grammar * grammar,
    struct llama_vocab * vocab,
    const float * logits,
    float * logits_out,
    size_t * n_tokens
);
```

## Integration with Chat Templates

### Template Analysis

The system analyzes chat templates to identify:

1. **Tool Call Start/End Markers**: Special tokens or text patterns
2. **Parameter Serialization Format**: JSON, XML, or custom formats
3. **Function Name Encoding**: How function names are represented
4. **Argument Ordering**: Positional vs. keyword arguments

### Differential Analysis

The "differential" approach means:

1. Parse the chat template to find tool call placeholders
2. Extract the format specification from template variables
3. Generate grammar matching that specific format
4. Apply constraints only during tool call generation

## JSON Schema to Grammar Compilation

### Supported JSON Schema Features

From `llama-grammar.cpp`:

```cpp
// Supported schema keywords
- type: string, number, integer, boolean, object, array
- properties: Object property definitions
- items / prefixItems: Array item definitions
- required: Required properties
- enum: Enumerated values
- const: Constant values
- anyOf, allOf, oneOf: Composition operators
- pattern: Regular expression patterns for strings
```

### Grammar Compilation Process

```cpp
// Pseudo-code from llama.cpp
json_schema_to_grammar(schema):
    grammar = base_grammar()

    match schema.type:
        case "object":
            for prop in schema.properties:
                grammar.add(prop.name, json_schema_to_grammar(prop.schema))
        case "array":
            item_grammar = json_schema_to_grammar(schema.items)
            grammar.add_sequence(item_grammar, schema.min_items, schema.max_items)
        case "string":
            if schema.pattern:
                grammar.add_regex(schema.pattern)
            else:
                grammar.add_terminal("STRING")
        # ... other types

    return grammar
```

## Performance Optimizations

### 1. Token Bitmask Compression

```cpp
// Compress token mask using 32-bit integers
struct token_bitmask {
    uint32_t masks[MAX_TOKENS / 32];
};
```

### 2. Lazy Grammar Evaluation

- Grammar compilation happens once per tool schema
- State is tracked incrementally during generation
- Only active paths are maintained

### 3. Early Termination

```cpp
// Stop constraint application when tool call is complete
if (grammar_is_complete(grammar_state)) {
    llama_grammar_free(grammar_state);
    grammar_state = nullptr;
}
```

## MLX-LM Integration Considerations

### 1. Required Components

```python
# Grammar compiler (similar to llama.cpp)
class GrammarCompiler:
    def compile_json_schema(self, schema: dict) -> str:
        """Convert JSON schema to GBNF grammar"""

    def compile_tool_schema(self, tools: List[Tool]) -> str:
        """Compile tool schemas to grammar"""

# Token constraint applier
class TokenConstraint:
    def apply_mask(self, logits: mx.array, mask: np.ndarray) -> mx.array:
        """Apply token mask to logits"""

# Grammar state tracker
class GrammarState:
    def update(self, token: int) -> bool:
        """Update state with new token, return True if complete"""
```

### 2. Integration Points in mlx-lm

Based on the analysis, the key integration points are:

1. **`mlx_lm/tool_parsers/grammar_tools.py`**: New grammar-aware parser
2. **`mlx_lm/sample_utils.py`**: Add grammar mask generation
3. **`mlx_lm/generate.py`**: Apply grammar constraints in generation loop
4. **`mlx_lm/models/cache.py`**: Track grammar state across generations

### 3. Recommended Implementation Approach

```python
# In generate_step function
def generate_step(model, tokens, grammar_state=None, ...):
    # ... existing code ...

    if grammar_state is not None:
        # Get allowed tokens from grammar
        allowed_mask = grammar_state.compute_token_mask()
        # Apply mask to logits
        logits = apply_grammar_constraint(logits, allowed_mask)

    # Sample token
    token = sample(logits, temperature, top_p, ...)

    # Update grammar state
    if grammar_state is not None:
        grammar_state.update(token)
        if grammar_state.is_complete():
            # Remove grammar constraints
            grammar_state = None

    return token, ...
```

## Key Files and References

### llama.cpp Files

- `common/chat.cpp`: Template format detection
- `llama.h`: Grammar API definitions
- `llama-grammar.cpp`: Grammar compilation and application
- `examples/llama-bench`: Grammar performance benchmarks

### Relevant Patterns

- GBNF Grammar Syntax: Extended BNF with regex support
- Token Masking: Bitmask-based token filtering
- Pushdown Automaton: Stack-based parser state
- JSON Schema Compilation: Schema to grammar conversion

## Open Questions

1. **Complex Tool Schemas**: How does llama.cpp handle nested or recursive tool schemas?
2. **Error Recovery**: What happens when grammar constraints cannot be satisfied?
3. **Multi-Tool Calls**: How are sequential tool calls handled?
4. **Streaming**: How does grammar constraint work with streaming generation?

## Next Steps for MLX-LM

1. Implement grammar compiler based on llama.cpp approach
2. Add token masking utilities using MLX operations
3. Create tool schema to grammar converter
4. Integrate with existing tool parser infrastructure
5. Add performance optimizations (bitmask compression, lazy evaluation)

## References

- llama.cpp repository: https://github.com/ggml-org/llama.cpp
- PR #18675 (if merged): Tool calling grammar generation
- GBNF Grammar Format: Documented in llama.cpp examples
- JSON Schema Specification: https://json-schema.org/
