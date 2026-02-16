# Grammar-Constrained Tool Calls in MLX-LM: Synthesis and Recommendations

## Executive Summary

This document synthesizes all research findings and provides concrete recommendations for implementing grammar-constrained tool calling in MLX-LM. Based on analysis of llama.cpp's approach, llguidance/guidance libraries, and the current MLX-LM architecture, I recommend a phased implementation that leverages LLGuidance for grammar compilation while building MLX-native optimizations.

## Key Findings Summary

### 1. Llama.cpp Approach

**Strengths:**
- Mature, production-tested implementation
- GBNF grammar format is well-specified and efficient
- Automatic grammar generation from Jinja templates
- Comprehensive JSON schema to grammar conversion
- Excellent performance through bitmask compression

**Key Insights:**
- Uses pushdown automaton for grammar state tracking
- 32-bit integer bitmask compression (32x memory savings)
- Lazy grammar evaluation with trigger patterns
- Token-level and character-level constraint support

### 2. LLGuidance Library

**Strengths:**
- Rust core for performance-critical parsing
- Python bindings via PyO3 (zero-copy)
- MLX integration with Metal shaders
- Comprehensive JSON schema support
- Earley parser for arbitrary CFGs

**Key Insights:**
- Slicer optimization for common token patterns
- TokTrie for efficient vocabulary matching
- LLExecutor for parallel grammar processing
- Low latency: 1-10μs per token mask computation

### 3. Guidance Library

**Strengths:**
- High-level Pythonic API
- Multi-backend support (Transformers, Llama.cpp, OpenAI)
- Built-in tool calling support
- Token healing for better tokenization
- Composable stateless functions

**Key Insights:**
- Depends on llguidance 1.4.0 for grammar compilation
- Abstracts grammar complexity behind intuitive functions
- Model-agnostic design
- Good for prototyping and high-level applications

### 4. MLX-LM Architecture

**Strengths:**
- Already has tool calling infrastructure
- Logits processor hook in generation loop
- Modular design with clear extension points
- KV cache system supports advanced features

**Integration Points:**
- `generate_step()`: Add grammar_state parameter
- `sample_utils.py`: Add grammar logits processor
- `tool_parsers/`: Create grammar-based parser
- `tokenizer_utils.py`: Auto-detect grammar parser

## Recommended Implementation Strategy

### Phase 1: Foundation (Week 1-2)

**Goal:** Basic grammar-constrained generation using LLGuidance

**Deliverables:**
1. Grammar state management system
2. LLGuidance integration
3. Token mask application
4. Basic tool calling support

**Implementation:**

```python
# New files to create:
mlx_lm/grammar/
├── __init__.py
├── base.py              # GrammarState ABC
├── llguidance.py        # LLGuidance integration
└── mask_ops.py          # Mask application utilities

# Files to modify:
mlx_lm/generate.py       # Add grammar_state parameter
mlx_lm/sample_utils.py   # Add grammar logits processor
```

**Key Classes:**

```python
# mlx_lm/grammar/base.py
class GrammarState(ABC):
    @abstractmethod
    def compute_token_mask(self) -> mx.array:
        """Get allowed token mask"""
        pass

    @abstractmethod
    def update(self, token: int) -> bool:
        """Update state, return True if active"""
        pass

# mlx_lm/grammar/llguidance.py
class LLGuidanceState(GrammarState):
    def __init__(self, tokenizer, grammar_str: str):
        from llguidance import LLMatcher, LLTokenizer
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)
        self.matcher = LLMatcher(self.ll_tokenizer, grammar_str)

    def compute_token_mask(self) -> mx.array:
        mask_np = self.matcher.get_token_mask()
        return mx.array(mask_np)

    def update(self, token: int) -> bool:
        self.matcher.commit_token(token)
        return not self.matcher.is_terminated()
```

### Phase 2: Tool Calling (Week 3-4)

**Goal:** Tool calling with automatic grammar generation

**Deliverables:**
1. Jinja template analyzer
2. Tool schema to grammar converter
3. Grammar-based tool parser
4. Auto-detection system

**Implementation:**

```python
# New files:
mlx_lm/grammar/
├── jinja_analyzer.py    # Template analysis
├── schema_compiler.py   # JSON schema to grammar
└── tool_grammar.py      # Tool-specific grammar

mlx_lm/tool_parsers/
└── grammar_tools.py     # Grammar-based parser

# Files to modify:
mlx_lm/tokenizer_utils.py # Auto-detect grammar parser
```

**Key Features:**

```python
# Auto-detect and generate grammar
def auto_generate_tool_grammar(tokenizer, tools: List[Dict]) -> str:
    """Auto-generate tool call grammar from template"""
    analyzer = JinjaTemplateAnalyzer()
    generator = ToolCallGrammarGenerator()

    analysis = analyzer.analyze(tokenizer.chat_template)
    grammar = generator.generate(analysis['format_spec'], tools)

    return grammar

# Unified tool parser
class GrammarToolParser:
    def __init__(self, tokenizer, tools, grammar=None):
        if grammar is None:
            grammar = auto_generate_tool_grammar(tokenizer, tools)

        self.grammar_state = LLGuidanceState(tokenizer, grammar)

    def get_token_mask(self):
        return self.grammar_state.compute_token_mask()

    def parse(self, text: str) -> Dict:
        return json.loads(text)
```

### Phase 3: Optimization (Week 5-6)

**Goal:** MLX-native performance optimizations

**Deliverables:**
1. MLX Metal kernels for mask application
2. Grammar state caching
3. Batch processing support
4. Performance benchmarking

**Implementation:**

```python
# New files:
mlx_lm/grammar/
├── mlx_kernels.py       # MLX Metal kernels
└── cache.py             # Grammar state cache

# Optimized mask application
@mx.custom_function
def apply_bitmask_kernel(logits, mask_words, neg_inf):
    """Metal kernel for bitmask application"""
    source = """
    uint batch = thread_position_in_grid.y;
    uint token = thread_position_in_grid.x;
    uint word_idx = token / 32;
    uint bit_idx = token % 32;

    uint word = mask_words[batch * mask_shape[1] + word_idx];
    bool allowed = (word >> bit_idx) & 1;

    float logit = logits[batch * logits_shape[1] + token];
    out[batch * out_shape[1] + token] = allowed ? logit : neg_inf[0];
    """

    kernel = mx.fast.metal_kernel(
        name="bitmask_apply",
        input_names=["logits", "mask_words", "neg_inf"],
        output_names=["out"],
        source=source,
    )

    return kernel(...)

# Grammar cache
class GrammarCache:
    def __init__(self, max_size=100):
        self._cache = {}

    def get_or_create(self, schema: dict, tokenizer) -> GrammarState:
        key = self._hash_schema(schema)
        if key not in self._cache:
            self._cache[key] = LLGuidanceState.from_json_schema(tokenizer, schema)
        return self._cache[key]
```

### Phase 4: Advanced Features (Week 7-8)

**Goal:** Production-ready features

**Deliverables:**
1. Streaming generation with grammar
2. Multi-tool call sequences
3. Error handling and fallback
4. Comprehensive testing

**Implementation:**

```python
# Streaming support
class StreamingGrammarGenerator:
    def __init__(self, model, tokenizer, grammar):
        self.model = model
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.state = None

    def generate_stream(self, prompt: str):
        """Generate with grammar constraints, streaming"""
        self.state = LLGuidanceState(self.tokenizer, self.grammar)
        tokens = self.tokenizer.encode(prompt)

        for token in self._generate_step(tokens):
            text = self.tokenizer.decode([token])
            yield text

            if not self.state.update(token):
                break

# Error handling
class RobustGrammarGenerator:
    def generate_with_fallback(
        self,
        model,
        prompt,
        grammar,
        max_retries=3
    ):
        """Generate with fallback to unconstrained on error"""
        for attempt in range(max_retries):
            try:
                return self._generate_constrained(model, prompt, grammar)
            except GrammarViolationError:
                if attempt == max_retries - 1:
                    return self._generate_unconstrained(model, prompt)
```

## Architecture Decision Records

### Decision 1: Use LLGuidance vs. Custom Implementation

**Choice:** Use LLGuidance for grammar compilation

**Rationale:**
- LLGuidance is well-tested and performant
- Rust core provides speed where it matters
- Python bindings integrate well with MLX-LM
- MLX integration already exists
- Reduces development time significantly

**Trade-offs:**
- Adds external dependency
- Less control over parsing algorithm
- Potential version compatibility issues

### Decision 2: GBNF vs. JSON Schema as Primary Format

**Choice:** Support both, with JSON schema as primary API

**Rationale:**
- JSON schema is more developer-friendly
- GBNF is better for complex custom formats
- Can auto-convert JSON schema to GBNF
- Matches existing tool calling conventions

**Implementation:**
```python
# Primary API: JSON schema
grammar = GrammarState.from_json_schema(tokenizer, schema)

# Advanced API: GBNF
grammar = GrammarState.from_gbnf(tokenizer, gbnf_string)
```

### Decision 3: Grammar State in Generation Loop vs. Logits Processor

**Choice:** Both - grammar state as parameter, applied via processor

**Rationale:**
- Separation of concerns
- Grammar state needs special handling (completion detection)
- Logits processor is the right place for mask application
- Maintains flexibility for other uses

**Implementation:**
```python
def generate_step(
    model,
    tokens,
    cache,
    grammar_state=None,  # Special parameter
    logits_processors=None,
    ...
):
    # Forward pass
    logits = model(tokens, cache=cache)

    # Apply grammar (highest priority)
    if grammar_state:
        mask = grammar_state.compute_token_mask()
        logits = apply_mask(logits, mask)

    # Apply other processors
    for processor in logits_processors:
        logits = processor(tokens, logits)

    # Sample
    token = sample(logits, ...)

    # Update grammar state
    if grammar_state:
        grammar_state.update(token)

    return token
```

## File Structure Summary

### New Files

```
mlx_lm/
├── grammar/
│   ├── __init__.py              # Public API exports
│   ├── base.py                  # GrammarState ABC
│   ├── llguidance.py            # LLGuidance integration
│   ├── mask_ops.py              # Mask application utilities
│   ├── jinja_analyzer.py        # Jinja template analysis
│   ├── schema_compiler.py       # JSON schema to grammar
│   ├── tool_grammar.py          # Tool-specific grammars
│   ├── mlx_kernels.py           # MLX Metal kernels
│   └── cache.py                 # Grammar state cache
│
├── tool_parsers/
│   └── grammar_tools.py         # Grammar-based tool parser
│
├── constrained_generate.py      # High-level API
│
└── examples/
    ├── basic_grammar.py         # Basic usage
    ├── tool_calling.py          # Tool calling examples
    ├── json_generation.py       # JSON generation
    └── custom_grammar.py        # Custom GBNF examples
```

### Modified Files

```
mlx_lm/
├── generate.py                  # Add grammar_state parameter
├── sample_utils.py              # Add grammar processor utilities
└── tokenizer_utils.py           # Auto-detect grammar parser
```

## API Design

### High-Level API

```python
from mlx_lm import load, generate_constrained

# Load model
model, tokenizer = load("meta-llama/Llama-3.1-8B-Instruct")

# Constrain generation to JSON schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    }
}

response = generate_constrained(
    model,
    "Generate a user profile",
    tokenizer=tokenizer,
    schema=schema
)

# Tool calling
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            }
        }
    }
]

tool_call = generate_tool_call(
    model,
    "What's the weather in Paris?",
    tokenizer=tokenizer,
    tools=tools
)
```

### Low-Level API

```python
from mlx_lm.grammar import LLGuidanceState, GrammarLogitsProcessor
from mlx_lm import generate_step

# Create grammar state
grammar = LLGuidanceState.from_json_schema(tokenizer, schema)

# Create processor
processor = GrammarLogitsProcessor(grammar)

# Generate step by step
tokens = tokenizer.encode(prompt)
cache = model.make_cache()

for i in range(max_tokens):
    token, logits, cache, grammar = generate_step(
        model,
        tokens,
        cache,
        grammar_state=grammar,
        logits_processors=[processor]
    )

    tokens.append(token)

    if grammar.is_complete():
        break
```

## Testing Strategy

### Unit Tests

```python
# tests/test_grammar_base.py
def test_grammar_state_compute_mask():
    """Test token mask computation"""
    grammar = LLGuidanceState(tokenizer, 'root ::= "hello"')
    mask = grammar.compute_token_mask()
    assert mask.shape == (vocab_size,)

def test_grammar_state_update():
    """Test state update"""
    grammar = LLGuidanceState(tokenizer, 'root ::= "hello" "world"')

    # Update with "hello"
    hello_token = tokenizer.encode("hello")[0]
    assert grammar.update(hello_token)  # Still active

    # Update with "world"
    world_token = tokenizer.encode("world")[0]
    assert not grammar.update(world_token)  # Complete

# tests/test_tool_calling.py
def test_tool_call_generation():
    """Test tool call generation"""
    tools = [{"name": "test", "parameters": {"type": "object"}}]

    tool_call = generate_tool_call(
        model,
        "Use the test tool",
        tokenizer,
        tools
    )

    assert tool_call["name"] == "test"
    assert isinstance(tool_call["arguments"], dict)
```

### Integration Tests

```python
# tests/test_grammar_integration.py
def test_end_to_end_constrained_generation():
    """Test complete constrained generation pipeline"""
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "number"}
        }
    }

    output = generate_constrained(
        model,
        "Generate a number",
        tokenizer,
        schema=schema
    )

    result = json.loads(output)
    assert isinstance(result["value"], (int, float))

def test_tool_call_with_complex_schema():
    """Test tool calling with nested schemas"""
    tools = [{
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"}
                    }
                }
            }
        }
    }]

    tool_call = generate_tool_call(model, "Search", tokenizer, tools)
    assert "query" in tool_call["arguments"]
```

### Performance Tests

```python
# tests/test_grammar_performance.py
def test_mask_application_latency():
    """Test mask application performance"""
    import time

    grammar = LLGuidanceState(tokenizer, complex_grammar)

    start = time.perf_counter()
    for _ in range(1000):
        mask = grammar.compute_token_mask()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.1  # Should be < 100μs per mask

def test_generation_overhead():
    """Test generation overhead with grammar"""
    start = time.perf_counter()

    # Constrained generation
    output = generate_constrained(model, prompt, tokenizer, schema)

    constrained_time = time.perf_counter() - start

    start = time.perf_counter()

    # Unconstrained generation
    output = generate(model, prompt, tokenizer)

    unconstrained_time = time.perf_counter() - start

    # Grammar overhead should be < 20%
    overhead = (constrained_time - unconstrained_time) / unconstrained_time
    assert overhead < 0.2
```

## Performance Expectations

Based on llama.cpp and llguidance benchmarks:

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Grammar compilation | 100-500 μs | One-time per schema |
| Token mask computation | 1-10 μs | Per token |
| Mask application (MLX) | <1 μs | Metal kernel |
| State update | 1-2 μs | Per token |
| Total overhead per token | <10 μs | < 5% for typical generation |

## Success Criteria

### Phase 1 Success Criteria
- [ ] Generate text constrained to simple grammars
- [ ] JSON schema to grammar conversion
- [ ] Integration with existing generation loop
- [ ] Unit tests pass

### Phase 2 Success Criteria
- [ ] Tool calling with grammar constraints
- [ ] Auto-detection from Jinja templates
- [ ] Support for at least 3 model formats
- [ ] Integration tests pass

### Phase 3 Success Criteria
- [ ] MLX Metal kernel for mask application
- [ ] Grammar caching working
- [ ] Performance overhead < 10%
- [ ] Performance benchmarks pass

### Phase 4 Success Criteria
- [ ] Streaming generation with grammar
- [ ] Multi-tool call sequences
- [ ] Error handling and fallback
- [ ] All tests pass

## Risk Mitigation

### Risk 1: LLGuidance Dependency Issues

**Mitigation:**
- Pin specific version in requirements
- Implement fallback to pure Python parser
- Consider vendoring critical components

### Risk 2: Performance Overhead Too High

**Mitigation:**
- Profile early and often
- Optimize hot paths first
- Consider native MLX reimplementation if needed
- Use caching aggressively

### Risk 3: Complex Grammars Fail

**Mitigation:**
- Start with simple grammars
- Add complexity incrementally
- Provide good error messages
- Implement validation tools

### Risk 4: Template Auto-Detection Unreliable

**Mitigation:**
- Allow manual grammar override
- Validate auto-detected grammars
- Provide detection debugging tools
- Document supported template patterns

## Next Steps

1. **Week 1**: Set up project structure and dependencies
2. **Week 2**: Implement Phase 1 (foundation)
3. **Week 3-4**: Implement Phase 2 (tool calling)
4. **Week 5-6**: Implement Phase 3 (optimization)
5. **Week 7-8**: Implement Phase 4 (advanced features)
6. **Week 9**: Documentation and examples
7. **Week 10**: Testing and refinement

## References

### Knowledge Base Documents
1. `001-llama-cpp-tool-calling-analysis.md` - Llama.cpp analysis
2. `002-llguidance-library-analysis.md` - LLGuidance deep dive
3. `003-guidance-library-analysis.md` - Guidance library analysis
4. `004-mlx-lm-integration-plan.md` - MLX-LM integration
5. `005-gbnf-grammar-reference.md` - GBNF format reference
6. `006-token-mask-implementation.md` - Token masking guide
7. `007-jinja-template-analysis.md` - Jinja template analysis

### External Resources
- LLGuidance: https://github.com/guidance-ai/llguidance
- Guidance: https://github.com/guidance-ai/guidance
- llama.cpp: https://github.com/ggml-org/llama.cpp
- GBNF Guide: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md

---

**Document Version:** 1.0
**Last Updated:** 2024
**Status:** Final Recommendations
