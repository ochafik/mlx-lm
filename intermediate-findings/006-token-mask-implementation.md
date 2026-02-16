# Token Masking Implementation for Grammar-Constrained Generation

## Overview

Token masking is the core technique for enforcing grammar constraints during LLM generation. This document provides a complete implementation guide for integrating token masking into MLX-LM.

## Core Concept

At each generation step, we:
1. Compute a bitmask of valid tokens from the grammar state
2. Apply the mask to the model's output logits
3. Sample from the constrained distribution
4. Update the grammar state with the selected token

## Bitmask Representation

### Data Structure

```python
import numpy as np
import mlx.core as mx
from typing import Tuple

def get_bitmask_shape(vocab_size: int) -> int:
    """Calculate number of 32-bit integers needed for vocabulary"""
    return (vocab_size + 31) // 32

class TokenBitmask:
    """Compressed token bitmask using 32-bit integers"""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.num_words = get_bitmask_shape(vocab_size)
        # -1 means all bits set (all tokens allowed)
        self.mask = np.full(self.num_words, -1, dtype=np.int32)

    def set_allowed(self, token_id: int):
        """Set a specific token as allowed"""
        word_idx = token_id // 32
        bit_idx = token_id % 32
        self.mask[word_idx] |= (1 << bit_idx)

    def set_forbidden(self, token_id: int):
        """Set a specific token as forbidden"""
        word_idx = token_id // 32
        bit_idx = token_id % 32
        self.mask[word_idx] &= ~(1 << bit_idx)

    def is_allowed(self, token_id: int) -> bool:
        """Check if a token is allowed"""
        word_idx = token_id // 32
        bit_idx = token_id % 32
        return (self.mask[word_idx] >> bit_idx) & 1

    def to_bool_array(self) -> np.ndarray:
        """Convert to boolean array (for debugging)"""
        result = np.zeros(self.vocab_size, dtype=bool)
        for i in range(self.vocab_size):
            result[i] = self.is_allowed(i)
        return result
```

## Grammar State Management

### Base Grammar State

```python
from abc import ABC, abstractmethod
from typing import Optional

class GrammarState(ABC):
    """Base class for grammar state tracking"""

    @abstractmethod
    def compute_token_mask(self) -> TokenBitmask:
        """Get allowed token mask for current position"""
        pass

    @abstractmethod
    def update(self, token: int) -> bool:
        """
        Update state with new token

        Returns:
            True if grammar is still active
            False if grammar is complete/terminated
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if grammar constraints are satisfied"""
        pass

    @abstractmethod
    def clone(self) -> 'GrammarState':
        """Create a copy of this state"""
        pass
```

### LLGuidance Integration

```python
from llguidance import LLMatcher, LLTokenizer, grammar_from
import json

class LLGuidanceState(GrammarState):
    """Grammar state using llguidance backend"""

    def __init__(self, tokenizer, grammar_str: str, grammar_type: str = "lark"):
        # Create llguidance tokenizer
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)

        # Compile grammar
        grammar = grammar_from(grammar_type, grammar_str)
        self.matcher = LLMatcher(self.ll_tokenizer, grammar)
        self.vocab_size = tokenizer.vocab_size
        self._complete = False

    def compute_token_mask(self) -> TokenBitmask:
        """Get allowed token mask"""
        mask_np = self.matcher.get_token_mask()
        bitmask = TokenBitmask(self.vocab_size)

        for token_id in range(self.vocab_size):
            if mask_np[token_id]:
                bitmask.set_allowed(token_id)
            else:
                bitmask.set_forbidden(token_id)

        return bitmask

    def update(self, token: int) -> bool:
        """Update grammar state"""
        self.matcher.commit_token(token)
        self._complete = self.matcher.is_terminated()
        return not self._complete

    def is_complete(self) -> bool:
        return self._complete

    def clone(self) -> 'LLGuidanceState':
        """Clone the state"""
        # Note: LLGuidance doesn't directly support cloning
        # This would require implementing custom state serialization
        raise NotImplementedError("Cloning not supported for LLGuidanceState")

    @classmethod
    def from_json_schema(cls, tokenizer, schema: dict) -> 'LLGuidanceState':
        """Create from JSON schema"""
        grammar_str = grammar_from("json_schema", json.dumps(schema))
        return cls(tokenizer, grammar_str, "json_schema")
```

## Mask Application

### NumPy Implementation (CPU)

```python
def apply_bitmask_numpy(
    logits: np.ndarray,
    mask: TokenBitmask
) -> np.ndarray:
    """Apply token mask using NumPy operations"""
    result = logits.copy()

    # Expand mask bits
    for word_idx in range(mask.num_words):
        start = word_idx * 32
        end = min(start + 32, mask.vocab_size)

        word_mask = mask.mask[word_idx]
        if word_mask == -1:  # All bits set
            continue

        for bit_idx in range(end - start):
            if not (word_mask >> bit_idx) & 1:
                result[start + bit_idx] = -float('inf')

    return result
```

### MLX Implementation (GPU)

```python
import mlx.core as mx

def apply_bitmask_mlx(
    logits: mx.array,
    mask: TokenBitmask
) -> mx.array:
    """Apply token mask using MLX operations"""

    # Create boolean mask from bitmask
    bool_mask = mx.zeros(mask.vocab_size, dtype=mx.bool_)

    for word_idx in range(mask.num_words):
        start = word_idx * 32
        end = min(start + 32, mask.vocab_size)

        word_mask = mask.mask[word_idx]
        if word_mask == -1:  # All bits set
            bool_mask[start:end] = True
            continue

        # Extract bits
        for bit_idx in range(end - start):
            bool_mask[start + bit_idx] = ((word_mask >> bit_idx) & 1) == 1

    # Apply mask
    return mx.where(bool_mask, logits, -mx.inf)
```

### Optimized MLX Metal Kernel

```python
@mx.custom_function
def bitmask_kernel_metal(
    logits: mx.array,
    mask_words: mx.array,
    neg_inf: mx.array
) -> mx.array:
    """Metal kernel for efficient bitmask application"""

    source = """
    uint batch = thread_position_in_grid.y;
    uint token = thread_position_in_grid.x;

    uint word_idx = token / 32;
    uint bit_idx = token % 32;

    // Check if bit is set
    bool allowed = false;
    if (word_idx < mask_words_shape[1]) {
        uint word = mask_words[batch * mask_words_shape[1] + word_idx];
        allowed = (word >> bit_idx) & 1;
    }

    // Apply constraint
    float logit = logits[batch * logits_shape[1] + token];
    out[batch * out_shape[1] + token] = allowed ? logit : neg_inf[0];
    """

    kernel = mx.fast.metal_kernel(
        name="bitmask_apply",
        input_names=["logits", "mask_words", "neg_inf"],
        output_names=["out"],
        source=source,
    )

    outputs = kernel(
        inputs=[logits, mask_words, neg_inf],
        template=[("T", logits.dtype)],
        grid=(logits.shape[1], logits.shape[0], 1),
        threadgroup=(256, 1, 1),
        output_shapes=[logits.shape],
        output_dtypes=[logits.dtype],
    )

    return outputs[0]

def apply_bitmask_mxl_optimized(
    logits: mx.array,
    mask_batch: np.ndarray  # Shape: (batch, num_words)
) -> mx.array:
    """Apply bitmask using optimized Metal kernel"""
    neg_inf = mx.array([-float('inf')], dtype=logits.dtype)
    mask_mx = mx.array(mask_batch)
    return bitmask_kernel_metal(logits, mask_mx, neg_inf)
```

## Logits Processor Integration

### Grammar-Aware Logits Processor

```python
from typing import Callable, List, Optional

class GrammarLogitsProcessor:
    """Logits processor that applies grammar constraints"""

    def __init__(self, grammar_state: GrammarState):
        self.grammar_state = grammar_state
        self.active = True

    def __call__(
        self,
        tokens: mx.array,
        logits: mx.array
    ) -> mx.array:
        """Apply grammar constraints to logits"""
        if not self.active:
            return logits

        # Get token mask
        mask = self.grammar_state.compute_token_mask()

        # Apply mask
        constrained = apply_bitmask_mlx(logits, mask)

        return constrained

    def update_state(self, token: int):
        """Update grammar state after sampling"""
        if self.active:
            still_active = self.grammar_state.update(token)
            self.active = still_active

    @property
    def is_complete(self) -> bool:
        return not self.active or self.grammar_state.is_complete()
```

### Integration with Existing Processors

```python
# In mlx_lm/sample_utils.py

def create_grammar_processor(
    grammar_state: GrammarState
) -> GrammarLogitsProcessor:
    """Factory function for grammar logits processor"""
    return GrammarLogitsProcessor(grammar_state)

# Combine with other processors
def create_logits_processors(
    temperature: float = 1.0,
    repetition_penalty: Optional[float] = None,
    grammar_state: Optional[GrammarState] = None,
    **kwargs
) -> List[Callable]:
    """Create list of logits processors"""

    processors = []

    # Add repetition penalty processor
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(
            lambda tokens, logits: apply_repetition_penalty(
                logits, tokens, repetition_penalty
            )
        )

    # Add grammar processor (highest priority)
    if grammar_state is not None:
        processors.append(create_grammar_processor(grammar_state))

    return processors
```

## Generation Loop Integration

### Modified generate_step

```python
# In mlx_lm/generate.py

def generate_step(
    model: BaseModel,
    tokens: mx.array,
    cache: Optional[List[Any]] = None,
    grammar_processor: Optional[GrammarLogitsProcessor] = None,
    **kwargs
) -> Tuple[int, mx.array, ...]:
    """
    Generate one token with optional grammar constraints

    Args:
        model: The language model
        tokens: Current token sequence
        cache: KV cache
        grammar_processor: Optional grammar constraints
        **kwargs: Other generation parameters

    Returns:
        (token, logits, cache, grammar_processor)
    """

    # Forward pass
    logits = model(tokens[None, -1:], cache=cache)
    mx.eval(logits)

    # Extract logits for last position
    logits = logits[0, -1, :]

    # Apply grammar constraints
    if grammar_processor is not None:
        logits = grammar_processor(tokens, logits)

    # Apply other logits processors
    logits_processors = kwargs.get('logits_processors', [])
    if logits_processors and len(tokens) > 0:
        for processor in logits_processors:
            logits = processor(tokens, logits)

    # Sample token
    token = sample(
        logits,
        temperature=kwargs.get('temperature', 0.0),
        top_p=kwargs.get('top_p', 1.0),
        top_k=kwargs.get('top_k', -1)
    )

    # Update grammar state
    if grammar_processor is not None:
        grammar_processor.update_state(int(token.item()))

    return token, logits, cache, grammar_processor
```

### High-Level API

```python
# In mlx_lm/constrained_generate.py

def generate_with_grammar(
    model: BaseModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    grammar: GrammarState,
    max_tokens: int = 100,
    **kwargs
) -> str:
    """
    Generate text with grammar constraints

    Args:
        model: The language model
        tokenizer: Tokenizer
        prompt: Input prompt
        grammar: Grammar state for constraints
        max_tokens: Maximum tokens to generate
        **kwargs: Generation parameters

    Returns:
        Generated text
    """
    from mlx_lm.utils import generate_step

    # Tokenize prompt
    input_tokens = mx.array(tokenizer.encode(prompt))

    # Create grammar processor
    grammar_processor = GrammarLogitsProcessor(grammar)

    # Create cache
    cache = [None] * len(model.layers) if hasattr(model, 'layers') else None

    # Generate tokens
    tokens = input_tokens
    for i in range(max_tokens):
        token, _, cache, grammar_processor = generate_step(
            model,
            tokens,
            cache,
            grammar_processor=grammar_processor,
            **kwargs
        )

        tokens = mx.concat([tokens, mx.array([token])])

        # Check if grammar is complete
        if grammar_processor.is_complete:
            break

    # Decode output
    output_tokens = tokens[len(input_tokens):].tolist()
    return tokenizer.decode(output_tokens)
```

## Tool Calling Integration

### Tool Call Grammar

```python
class ToolCallGrammar(LLGuidanceState):
    """Grammar for tool calling"""

    @classmethod
    def from_tools(
        cls,
        tokenizer,
        tools: List[Dict[str, Any]]
    ) -> 'ToolCallGrammar':
        """Create grammar from tool definitions"""

        # Build tool schema
        tool_schemas = []
        for tool in tools:
            schema = {
                "type": "object",
                "properties": {
                    "name": {"const": tool["name"]},
                    "arguments": tool.get("parameters", {})
                },
                "required": ["name", "arguments"]
            }
            tool_schemas.append(schema)

        # Create wrapper schema
        wrapper_schema = {
            "type": "object",
            "anyOf": tool_schemas
        }

        return cls.from_json_schema(tokenizer, wrapper_schema)

def generate_tool_call(
    model: BaseModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    tools: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """Generate a tool call with grammar constraints"""

    # Create grammar
    grammar = ToolCallGrammar.from_tools(tokenizer, tools)

    # Generate
    output = generate_with_grammar(
        model,
        tokenizer,
        prompt,
        grammar,
        **kwargs
    )

    # Parse tool call
    import json
    try:
        tool_call = json.loads(output.strip())
        return {
            "name": tool_call.get("name"),
            "arguments": tool_call.get("arguments", {})
        }
    except json.JSONDecodeError:
        return None
```

## Performance Optimization

### Grammar Caching

```python
from functools import lru_cache
import hashlib

class GrammarCache:
    """Cache compiled grammars"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache = {}

    def _get_key(self, schema: dict) -> str:
        """Generate cache key from schema"""
        return hashlib.md5(
            json.dumps(schema, sort_keys=True).encode()
        ).hexdigest()

    def get(self, schema: dict) -> Optional[GrammarState]:
        """Get cached grammar"""
        key = self._get_key(schema)
        return self._cache.get(key)

    def set(self, schema: dict, grammar: GrammarState):
        """Cache grammar"""
        key = self._get_key(schema)
        if len(self._cache) >= self.max_size:
            # Remove oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = grammar
```

### Batch Processing

```python
class BatchGrammarProcessor:
    """Process multiple grammars in parallel"""

    def __init__(self, grammars: List[GrammarState]):
        self.grammars = grammars
        self.processors = [
            GrammarLogitsProcessor(g) for g in grammars
        ]

    def apply_batch(
        self,
        logits: mx.array  # Shape: (batch_size, vocab_size)
    ) -> mx.array:
        """Apply grammar masks to batch"""
        result = logits.copy()

        for i, processor in enumerate(self.processors):
            if processor.active:
                result[i] = processor(None, logits[i])

        return result

    def update_batch(self, tokens: List[int]):
        """Update all grammar states"""
        for i, (token, processor) in enumerate(zip(tokens, self.processors)):
            processor.update_state(token)
```

## Testing

### Unit Tests

```python
import unittest

class TestTokenBitmask(unittest.TestCase):
    def test_bitmask_creation(self):
        mask = TokenBitmask(100)
        self.assertEqual(mask.num_words, 4)

    def test_set_allowed(self):
        mask = TokenBitmask(100)
        mask.set_allowed(50)
        self.assertTrue(mask.is_allowed(50))
        self.assertFalse(mask.is_allowed(51))

    def test_all_forbidden(self):
        mask = TokenBitmask(100)
        for i in range(100):
            mask.set_forbidden(i)
        for i in range(100):
            self.assertFalse(mask.is_allowed(i))

class TestGrammarProcessor(unittest.TestCase):
    def test_simple_grammar(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        grammar = LLGuidanceState(
            tokenizer,
            'start: "hello" | "world"'
        )

        mask = grammar.compute_token_mask()
        self.assertIsNotNone(mask)
```

## References

- LLGuidance: https://github.com/guidance-ai/llguidance
- llama.cpp Grammar Implementation: https://github.com/ggml-org/llama.cpp/blob/master/src/llama-grammar.cpp
- MLX Documentation: https://ml-explore.github.io/mlx/
