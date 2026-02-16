# MLX-LM Integration Analysis

## Overview

This document summarizes the current state of mlx-lm and identifies specific integration points for grammar-constrained tool calling functionality.

## Current MLX-LM Architecture

### Key Modules

```
mlx_lm/
├── generate.py           # Core generation engine (1,551 lines)
├── models/
│   ├── base.py          # Base model utilities
│   ├── cache.py         # KV cache implementations
│   └── model_wrapper.py # Model wrapping for KV press
├── tokenizer_utils.py   # Tokenizer wrapper and utilities
├── sample_utils.py      # Sampling strategies and logits processors
├── tool_parsers/        # Tool calling parsers
│   ├── __init__.py
│   ├── json_tools.py
│   ├── function_gemma.py
│   ├── glm47.py
│   ├── minimax_m2.py
│   └── qwen3_coder.py
└── utils.py             # Model loading and conversion
```

## Generation Flow Analysis

### Main Generation Entry Points

```python
# From mlx_lm/generate.py

def generate(
    model: Type[BaseModel],
    prompt: Union[str, List[int]],
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> str:
    """Simple synchronous generation"""

def stream_generate(
    model: Type[BaseModel],
    prompt: Union[str, List[int]],
    **kwargs
) -> Iterator[GenerationResponse]:
    """Streaming generation with metadata"""

def batch_generate(
    model: Type[BaseModel],
    prompts: List[Union[str, List[int]]],
    **kwargs
) -> List[GenerationResponse]:
    """Batch processing"""
```

### Core Generation Loop (generate_step)

**Location**: `mlx_lm/generate.py:339-555`

```python
def generate_step(
    model: BaseModel,
    tokens: mx.array,
    kwargs: dict,
    cache: Optional[List[Any]] = None,
    progress: Optional[Progress] = None
) -> Tuple[int, mx.array, ...]:
    """
    Core token generation loop

    Returns:
        (token, logits, cache, metadata)
    """

    # Prefill phase
    if cache is None or tokens.shape[-1] > 1:
        # Process prompt in chunks
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            logits = model(chunk[None], cache=cache)
            mx.eval(logits)

    # Decoding phase
    else:
        # Single token generation
        logits = model(tokens[None], cache=cache)
        mx.eval(logits)

    # Logits processing
    logits = logits[:, -1, :]  # Get last token logits

    # Apply logits processors (EXISTING HOOK!)
    if logits_processors and len(input_tokens) > 0:
        tokens = mx.concat([tokens, input_tokens])
        for processor in logits_processors:
            logits = processor(tokens, logits)

    # Sample token
    token = sample(logits, temperature, top_p, top_k, min_p)

    return token, logits, cache, ...
```

## Existing Tool Calling Infrastructure

### Tool Parser System

**Location**: `mlx_lm/tokenizer_utils.py:520-542`

```python
# Auto-detect tool parser from tokenizer config
tool_parser_type = tokenizer_config.get(
    "tool_parser_type",
    _infer_tool_parser(tokenizer.chat_template)
)

if tool_parser_type is not None:
    tool_module = importlib.import_module(f"mlx_lm.tool_parsers.{tool_parser_type}")
    tool_parser = tool_module.parse_tool_call
```

### Supported Tool Formats

| Parser | Format | Example Models |
|--------|--------|----------------|
| `json_tools` | JSON: `{"name": "func", "arguments": {...}}` | Qwen, Llama-3 |
| `function_gemma` | XML: `<function=NAME>...</function>` | Gemma |
| `glm47` | Custom: `func_name {args}` | GLM-4 |
| `minimax_m2` | Mixed format | MiniMax M2 |
| `qwen3_coder` | Line-delimited | Qwen3-Coder |

### Tool Parser Interface

**Location**: `mlx_lm/tool_parsers/json_tools.py`

```python
def parse_tool_call(
    text: str,
    tools: List[Dict],
    tokenizer: PreTrainedTokenizer
) -> Optional[ToolCall]:
    """
    Parse tool call from generated text

    Args:
        text: Generated text
        tools: Available tools (from chat template)
        tokenizer: Tokenizer for validation

    Returns:
        ToolCall object or None
    """
    # Parse JSON tool call
    try:
        tool_data = json.loads(text.strip())
        return ToolCall(
            name=tool_data.get("name"),
            arguments=tool_data.get("arguments", {})
        )
    except json.JSONDecodeError:
        return None
```

### Chat Template Integration

**Location**: `mlx_lm/examples/tool_use.py`

```python
from mlx_lm import load, generate

# Define tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b

tools = {
    "multiply": multiply
}

# Load model
model, tokenizer = load("meta-llama/Llama-3.1-8B-Instruct")

# Apply chat template with tools
messages = [
    {"role": "user", "content": "What is 5 times 7?"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=list(tools.values()),
    return_tensors="pt"
)

# Generate
response = generate(model, prompt, tokenizer)
```

## KV Cache System

### Cache Types

**Location**: `mlx_lm/models/cache.py`

| Cache Type | Description | Use Case |
|------------|-------------|----------|
| `KVCache` | Standard cache with dynamic expansion | Default |
| `RotatingKVCache` | Sliding window cache | Long context |
| `QuantizedKVCache` | 4-8 bit quantized cache | Memory efficiency |
| `BatchKVCache` | Batch-aware cache | Batch generation |
| `PrunedKVCache` | Token compaction | KV pressing |

### Cache Creation

```python
# From mlx_lm/models/cache.py
def make_prompt_cache(model, max_kv_size=None):
    """Create appropriate cache for model"""
    if hasattr(model, "make_cache"):
        return model.make_cache()
    elif max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size)
                for _ in range(num_layers)]
    else:
        return [KVCache() for _ in range(num_layers)]
```

## Logits Processing System

### Logits Processors

**Location**: `mlx_lm/sample_utils.py`

```python
# Existing processors
def repetition_penalty(
    tokens: mx.array,
    logits: mx.array,
    penalty: float = 1.0
) -> mx.array:
    """Apply repetition penalty"""
    ...

def dry_processor(
    tokens: mx.array,
    logits: mx.array,
    ...
) -> mx.array:
    """Apply DRY (Don't Repeat Yourself) penalty"""
    ...
```

### Processor Usage

**Location**: `mlx_lm/generate.py:493-500`

```python
# Apply logits processors
if logits_processors and len(input_tokens) > 0:
    tokens = mx.concat([tokens, input_tokens])
    for processor in logits_processors:
        logits = processor(tokens, logits)
```

## Integration Points for Grammar Constraints

### 1. Logits Processor Integration (Recommended)

**Advantages**:
- Uses existing hook
- Minimal code changes
- Works with all generation modes

**Implementation**:

```python
# In mlx_lm/sample_utils.py

class GrammarLogitsProcessor:
    """Apply grammar constraints to logits"""

    def __init__(self, grammar_constraint: 'GrammarConstraint'):
        self.grammar = grammar_constraint
        self.active = True

    def __call__(
        self,
        tokens: mx.array,
        logits: mx.array
    ) -> mx.array:
        if not self.active:
            return logits

        # Get token mask from grammar
        mask = self.grammar.compute_token_mask(tokens)

        # Apply constraint
        vocab_size = logits.shape[-1]
        constrained = mx.where(
            mask,
            logits,
            mx.full_like(logits, -float('inf'))
        )

        # Update grammar state
        last_token = int(tokens[-1].item())
        self.grammar.update(last_token)

        # Check if grammar is complete
        if self.grammar.is_complete():
            self.active = False

        return constrained

# Usage
from mlx_lm import generate

grammar = JSONGrammar(schema=my_schema)
processor = GrammarLogitsProcessor(grammar)

response = generate(
    model,
    prompt,
    tokenizer,
    logits_processors=[processor]
)
```

### 2. Tool Parser Extension

**Location**: `mlx_lm/tool_parsers/grammar_tools.py`

```python
"""
Grammar-based tool parser for mlx-lm

Uses llguidance for grammar compilation and token masking.
"""

from typing import List, Dict, Optional, Any
import json
from llguidance import LLMatcher, LLTokenizer, grammar_from

class GrammarToolParser:
    """Parse tool calls with grammar constraints"""

    def __init__(
        self,
        tools: List[Dict[str, Any]],
        tokenizer: Any
    ):
        self.tools = tools
        self.tokenizer = tokenizer

        # Create llguidance tokenizer
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)

        # Build tool schema
        self.tool_schema = self._build_tool_schema(tools)

        # Compile grammar
        grammar_str = grammar_from(
            "json_schema",
            json.dumps(self.tool_schema)
        )
        self.matcher = LLMatcher(self.ll_tokenizer, grammar_str)

    def _build_tool_schema(self, tools: List[Dict]) -> Dict:
        """Build JSON schema for tools"""
        tool_defs = []
        for tool in tools:
            tool_def = {
                "type": "object",
                "properties": {
                    "name": {"const": tool["name"]},
                    "arguments": tool.get("parameters", {})
                },
                "required": ["name", "arguments"]
            }
            tool_defs.append(tool_def)

        return {
            "type": "object",
            "anyOf": tool_defs
        }

    def get_token_mask(self) -> List[bool]:
        """Get allowed token mask for current position"""
        return self.matcher.get_token_mask()

    def update(self, token: int) -> bool:
        """Update grammar state with new token"""
        self.matcher.commit_token(token)
        return not self.matcher.is_terminated()

    def parse(self, text: str) -> Optional[Dict]:
        """Parse completed tool call"""
        try:
            tool_call = json.loads(text)
            return {
                "name": tool_call.get("name"),
                "arguments": tool_call.get("arguments", {})
            }
        except json.JSONDecodeError:
            return None

def parse_tool_call(
    text: str,
    tools: List[Dict],
    tokenizer: Any
) -> Optional[Dict]:
    """
    Parse tool call using grammar constraints

    Entry point matching existing tool parsers
    """
    parser = GrammarToolParser(tools, tokenizer)
    return parser.parse(text)
```

### 3. Generation Loop Modification

**Location**: `mlx_lm/generate.py:339-555`

```python
# In generate_step function, add grammar parameter

def generate_step(
    model: BaseModel,
    tokens: mx.array,
    kwargs: dict,
    cache: Optional[List[Any]] = None,
    grammar_state: Optional['GrammarState'] = None,  # NEW
    ...
) -> Tuple[int, mx.array, ...]:
    """
    Core token generation loop with grammar constraints
    """

    # ... prefill and decoding phases ...

    # Apply grammar constraints (NEW)
    if grammar_state is not None:
        # Get token mask from grammar
        mask = grammar_state.compute_token_mask()

        # Convert to MLX array
        mask_mx = mx.array(mask)

        # Apply constraint
        logits = mx.where(mask_mx, logits, -mx.inf)

    # Apply logits processors
    if logits_processors and len(input_tokens) > 0:
        tokens = mx.concat([tokens, input_tokens])
        for processor in logits_processors:
            logits = processor(tokens, logits)

    # Sample token
    token = sample(logits, temperature, top_p, top_k, min_p)

    # Update grammar state (NEW)
    if grammar_state is not None:
        grammar_complete = grammar_state.update(int(token.item()))
        if grammar_complete:
            grammar_state = None

    return token, logits, cache, grammar_state, ...
```

## Recommended Implementation Plan

### Phase 1: Grammar Constraint Foundation

**Files to create**:
1. `mlx_lm/grammar/__init__.py` - Package init
2. `mlx_lm/grammar/base.py` - Base grammar classes
3. `mlx_lm/grammar/json_schema.py` - JSON schema compiler
4. `mlx_lm/grammar/llguidance_adapter.py` - LLGuidance integration

**Key classes**:
```python
# mlx_lm/grammar/base.py
class GrammarState(ABC):
    """Base class for grammar state tracking"""

    @abstractmethod
    def compute_token_mask(self) -> mx.array:
        """Get allowed token mask"""
        pass

    @abstractmethod
    def update(self, token: int) -> bool:
        """Update state with new token, return True if complete"""
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if grammar is satisfied"""
        pass
```

### Phase 2: Tool Calling Integration

**Files to modify**:
1. `mlx_lm/tool_parsers/grammar_tools.py` - Create new parser
2. `mlx_lm/tokenizer_utils.py` - Auto-detect grammar parser
3. `mlx_lm/sample_utils.py` - Add grammar logits processor

**Key additions**:
```python
# In tokenizer_utils.py
def _infer_tool_parser(chat_template: str) -> Optional[str]:
    """Auto-detect tool parser type"""
    # Check for tool call markers in template
    if "<|tool_call|>" in chat_template:
        return "grammar_tools"  # Use grammar-based parser
    # ... existing logic ...
```

### Phase 3: API Layer

**Files to create**:
1. `mlx_lm/constrained_generate.py` - High-level API
2. `mlx_lm/examples/constrained_tool_use.py` - Example usage

**API design**:
```python
# mlx_lm/constrained_generate.py

from typing import List, Dict, Any, Optional
import mxnet as mx

class ConstrainedGenerator:
    """Generator with grammar constraints"""

    def __init__(
        self,
        model: BaseModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = None

    def generate_tool_call(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a tool call with grammar constraints

        Args:
            prompt: Input prompt
            tools: List of tool definitions
            **kwargs: Generation arguments

        Returns:
            Tool call dictionary
        """
        from mlx_lm.grammar import ToolCallGrammar

        # Create grammar from tools
        grammar = ToolCallGrammar(tools, self.tokenizer)

        # Generate with constraints
        response = generate(
            self.model,
            prompt,
            self.tokenizer,
            grammar_state=grammar,
            **kwargs
        )

        # Parse tool call
        return grammar.parse(response.text)
```

### Phase 4: Optimization

**Optimizations**:
1. MLX-native bitmask operations
2. Grammar state caching
3. Parallel mask computation
4. Lazy grammar evaluation

**Files to modify**:
1. `mlx_lm/grammar/mask_ops.py` - MLX bitmask operations
2. `mlx_lm/grammar/cache.py` - Grammar state cache

## Performance Considerations

### Memory Overhead

| Component | Memory | Notes |
|-----------|--------|-------|
| Grammar compilation | 1-10 MB | One-time per schema |
| Token mask | vocab_size / 32 bytes | Compressed bitmask |
| Grammar state | 1-100 KB | Per active generation |

### Latency Impact

| Operation | Overhead | Optimization |
|-----------|----------|--------------|
| Grammar compilation | 100-500 μs | Cache compiled grammars |
| Token mask computation | 1-10 μs | MLX kernel |
| Mask application | <1 μs | Native MX operations |
| State update | 1-2 μs | Optimized data structures |

### Optimization Strategies

1. **Grammar Caching**: Compile once, reuse many times
2. **Mask Compression**: Use bitmask (32x reduction)
3. **Lazy Evaluation**: Only apply when needed
4. **MLX Kernels**: Native operations for speed
5. **Parallel Processing**: Batch multiple grammars

## Key Files Summary

### Files to Create

```
mlx_lm/
├── grammar/
│   ├── __init__.py
│   ├── base.py              # Base grammar classes
│   ├── json_schema.py       # JSON schema compiler
│   ├── llguidance_adapter.py # LLGuidance integration
│   ├── mask_ops.py          # MLX bitmask operations
│   └── cache.py             # Grammar state cache
├── tool_parsers/
│   └── grammar_tools.py     # Grammar-based tool parser
├── constrained_generate.py  # High-level API
└── examples/
    └── constrained_tool_use.py
```

### Files to Modify

```
mlx_lm/
├── generate.py              # Add grammar_state parameter
├── sample_utils.py          # Add grammar logits processor
└── tokenizer_utils.py       # Auto-detect grammar parser
```

## Open Questions

1. **Streaming with Grammar**: How to handle streaming generation with grammar constraints?
2. **Error Recovery**: What to do when no tokens match the grammar?
3. **Batch Generation**: How to apply different grammars per batch item?
4. **Multi-Tool Calls**: How to handle sequences of tool calls?

## Next Steps

1. Implement base grammar classes
2. Integrate LLGuidance for grammar compilation
3. Add logits processor for grammar constraints
4. Create tool parser with grammar validation
5. Add performance optimizations
6. Write tests and examples
