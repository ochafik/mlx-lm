# Guidance Library: Comprehensive Analysis

## Overview

Guidance is a high-level Python library for controlling LLM output through structured generation and grammar constraints. It provides a Pythonic API built on top of llguidance, supporting multiple model backends and advanced features like tool calling and token healing.

## Relationship to LLGuidance

```
┌─────────────────────────────────────────┐
│           Guidance (High-Level)         │
│  - Pythonic API                         │
│  - Multi-backend support                │
│  - Tool calling                         │
│  - Role management                      │
│  - Token healing                        │
└──────────────┬──────────────────────────┘
               │ depends on
┌──────────────▼──────────────────────────┐
│         LLGuidance (Low-Level)          │
│  - Grammar compilation                  │
│  - Token masking                        │
│  - Parser implementation                │
└─────────────────────────────────────────┘
```

## Core Architecture

### Key Components

1. **AST Nodes**: Grammar representation
2. **Model Objects**: Immutable state containers
3. **Interpreter**: Executes grammar nodes
4. **Backend Adapters**: Integration with LLM providers

### Node Hierarchy

```
Node
├── LiteralNode     # Fixed text
├── RegexNode       # Regular expression
├── SelectNode      # Choice between options
├── RuleNode        # Grammar rules
├── RepeatNode      # Repetition (*/+)
├── JsonNode        # JSON schema constraints
├── ToolCallNode    # Tool/function calling
├── RoleNode        # Chat roles (system/user/assistant)
└── CaptureNode     # Variable capture
```

## Core API

### 1. gen() - Text Generation

```python
from guidance import gen

# Basic generation
lm += gen(max_tokens=50)

# With regex constraint
lm += gen(regex=r'\d{3}-\d{3}-\d{4}', name='phone')

# With stop sequence
lm += gen(stop='###', max_tokens=100)

# With temperature
lm += gen(temperature=0.7, top_p=0.9, max_tokens=50)

# Sampling options
lm += gen(
    max_tokens=100,
    temperature=0.5,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1
)
```

### 2. select() - Choice Selection

```python
from guidance import select

# Simple choice
lm += select(['Yes', 'No', 'Maybe'], name='answer')

# Weighted choice (via logit bias)
lm += select(
    ['Option A', 'Option B', 'Option C'],
    weights=[2.0, 1.0, 0.5],
    name='choice'
)

# Multi-select
lm += select(
    ['apple', 'banana', 'cherry'],
    name='fruits',
    list_append=True  # Allow multiple selections
)
```

### 3. string() and regex() - Fixed Patterns

```python
from guidance import string, regex

# Literal string
lm += string("Hello, ")
lm += gen(regex='[A-Z][a-z]+')
lm += string("!")

# Complex regex pattern
lm += regex(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
```

### 4. Grammar Combinators

```python
from guidance import one_or_more, zero_or_more, sequence

# One or more items
lm += one_or_more(gen(regex=r'\w+'))

# Zero or more items
lm += zero_or_more(gen(regex=r'\d+'))

# Exact sequence
lm += sequence([
    "Name: ",
    gen(name='name', regex=r'[A-Z][a-z]+'),
    ", Age: ",
    gen(name='age', regex=r'\d{2}')
])

# Optional
lm += string("Optional prefix ") + gen(name='content')
```

### 5. json() - JSON Generation

```python
from guidance import json
from pydantic import BaseModel, Field

# Using Pydantic model
class User(BaseModel):
    name: str = Field(max_length=50)
    age: int = Field(ge=0, le=150)
    email: str = Field(pattern=r'.+@.+')
    active: bool = True

lm += json(name='user', schema=User)

# Access generated data
user_data = lm['user']
print(user_data['name'])  # Access by key
```

### 6. Role Management

```python
from guidance import system, user, assistant

# Define roles
with system():
    lm += "You are a helpful assistant specialized in Python."

with user():
    lm += "How do I parse JSON in Python?"

with assistant():
    lm += gen(max_tokens=200)
```

## Tool Calling

### Tool Definition

```python
from guidance import Tool

# Define tool from function
def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city"""
    # Implementation
    return f"Weather in {city}: 22°C"

# Create tool
weather_tool = Tool.from_callable(
    get_weather,
    name='get_weather',
    description='Get current weather for a city'
)

# Alternative: manual definition
calculator_tool = Tool(
    name='calculator',
    parameters={
        'type': 'object',
        'properties': {
            'expression': {'type': 'string'},
            'precision': {'type': 'integer', 'default': 2}
        },
        'required': ['expression']
    },
    function=lambda expr, prec=2: str(eval(round(float(expr), prec)))
)
```

### Tool Usage

```python
# Require specific tool
lm += gen(
    name='result',
    tools=[weather_tool],
    tool_choice='required'  # 'auto', 'required', 'none'
)

# Auto tool selection
lm += gen(
    name='action',
    tools=[weather_tool, calculator_tool],
    tool_choice='auto'  # Model decides
)

# Multiple tools in sequence
lm += gen(name='step1', tools=[weather_tool], max_tokens=100)
lm += gen(name='step2', tools=[calculator_tool], max_tokens=100)
```

### Tool Results

```python
# Access tool call and result
result = lm['result']
print(result['tool'])      # 'get_weather'
print(result['arguments']) # {'city': 'Paris', 'units': 'celsius'}
print(result['output'])    # 'Weather in Paris: 22°C'
```

## Advanced Features

### Token Healing

Token healing optimizes generation by "healing" token boundaries:

```python
# Without token healing:
# Prompt: "The capital of France is"
# Model might generate: " Paris" (with space)

# With token healing:
# Guidance backs up and regenerates boundary tokens
# Model generates: "Paris" (without extra space)

# Automatic in Guidance, but can be controlled:
lm += gen(max_tokens=50, heal_token=True)
```

### Stateless Functions

```python
from guidance import guidance

# Define reusable grammar component
@guidance(stateless=True)
def generate_address(lm):
    lm += string("Address: ")
    lm += gen(name='street', regex=r'\d+ \w+ (\w+ )?')
    lm += string("\nCity: ")
    lm += gen(name='city', regex=r'[A-Z][a-z]+')
    lm += string(", ")
    lm += gen(name='state', regex=r'[A-Z]{2}')
    lm += string(" ")
    lm += gen(name='zip', regex=r'\d{5}')
    return lm

# Use in multiple contexts
lm += generate_address() + string("\n\n")
lm += generate_address()
```

### Capture and Variables

```python
from guidance import capture

# Capture generation to variable
lm += capture(gen(regex=r'\d+', name='number'), 'my_number')

# Use captured value
lm += string("\nThe number is: ")
lm += string(str(lm['my_number']))

# Conditional based on capture
if int(lm['my_number']) > 100:
    lm += string(" (large number)")
else:
    lm += string(" (small number)")
```

## Model Integration

### Backend Support

```python
# Transformers
from guidance import models
transformers_lm = models.Transformers(model='gpt2')

# Llama.cpp
llamacpp_lm = models.LlamaCpp(model_path='./models/llama-2-7b.gguf')

# OpenAI
openai_lm = models.OpenAI(model='gpt-4')

# Vertex AI
vertex_lm = models.VertexAI(model='gemini-pro')

# Azure OpenAI
azure_lm = models.AzureOpenAI(
    deployment='my-deployment',
    api_key='...'
)
```

### Custom Backend

```python
from guidance import Model

class MLXModel(Model):
    def __init__(self, model_path: str):
        import mlx_lm
        self.model, self.tokenizer = mlx_lm.load(model_path)

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompt, max_tokens, **kwargs):
        # Implement generation
        tokens = self.tokenizer.encode(prompt)
        output = mlx_lm.generate(
            self.model,
            prompt=tokens,
            max_tokens=max_tokens,
            **kwargs
        )
        return self.tokenizer.decode(output)

# Use with Guidance
mlx_lm = MLXModel('/path/to/model')
lm = mlx_lm + gen(max_tokens=100)
```

## Grammar Composition

### Complex Example

```python
from guidance import guidance, gen, select, string, one_or_more, zero_or_more

@guidance(stateless=True)
def html_tag(lm, tag_name, content_gen):
    lm += string(f"<{tag_name}>")
    lm += content_gen
    lm += string(f"</{tag_name}>")
    return lm

@guidance(stateless=True)
def html_text(lm):
    return lm + gen(regex=r'[^<>]+')

@guidance(stateless=True)
def html_paragraph(lm):
    lm += string("<p>")
    lm += one_or_more(
        select([
            html_text(),
            string("<strong>") + html_text() + string("</strong>"),
            string("<em>") + html_text() + string("</em>")
        ])
    )
    lm += string("</p>\n")
    return lm

@guidance(stateless=True)
def html_document(lm):
    lm += string("<html><body>\n")
    lm += one_or_more(html_paragraph())
    lm += string("</body></html>")
    return lm

# Use the composed grammar
lm += html_document(name='my_html')
```

## Performance Considerations

### Token Healing Overhead

- **Pros**: Better tokenization, reduced hallucination
- **Cons**: 1-2 extra forward passes per generation
- **Trade-off**: Enable for quality-critical applications

### Grammar Compilation

```python
# Grammar compilation is one-time
# Compile once, reuse many times

# Bad: Recompile each time
for i in range(100):
    lm += json(schema=User)  # Compiles 100 times

# Good: Compile once
user_json_gen = json(schema=User)  # Compile once
for i in range(100):
    lm += user_json_gen  # Reuse
```

### Streaming

```python
# Guidance supports streaming
for chunk in lm.stream_gen(max_tokens=100):
    print(chunk, end='', flush=True)
```

## MLX-LM Integration Strategy

### Approach 1: Use Guidance Directly

```python
# Create MLX backend for Guidance
from guidance import Model
import mlx_lm

class MLXGuidanceModel(Model):
    def __init__(self, model_path: str):
        self.model, self.tokenizer = mlx_lm.load(model_path)
        self._grammar_state = None

    def get_tokenizer(self):
        return self.tokenizer

    def generate(self, prompt, max_tokens, temperature=0.0, **kwargs):
        # Generate with mlx-lm
        from mlx_lm import generate

        # Apply grammar constraints if present
        if hasattr(self, '_grammar_constraint'):
            # Integration with llguidance
            pass

        text = generate(
            self.model,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            **kwargs
        )
        return text

# Use with Guidance
mlx_guidance = MLXGuidanceModel('/path/to/model')
lm = mlx_guidance + gen(max_tokens=100)
```

### Approach 2: Borrow Patterns from Guidance

```python
# Implement Guidance-like API in mlx-lm
# In mlx_lm/guidance.py

from typing import Optional, List, Callable
import mx.ndarray as mx

class GrammarConstraint:
    """Base class for grammar constraints"""
    def compute_mask(self, logits: mx.array) -> mx.array:
        raise NotImplementedError

class RegexConstraint(GrammarConstraint):
    def __init__(self, pattern: str, tokenizer):
        import re
        self.pattern = re.compile(pattern)
        self.tokenizer = tokenizer

    def compute_mask(self, logits: mx.array) -> mx.array:
        # Get vocab size
        vocab_size = logits.shape[-1]
        mask = mx.zeros((vocab_size,), dtype=mx.bool_)

        # Find tokens matching regex
        for token_id in range(vocab_size):
            token_str = self.tokenizer.decode([token_id])
            if self.pattern.match(token_str):
                mask[token_id] = True

        return mask

class JSONConstraint(GrammarConstraint):
    def __init__(self, schema: dict, tokenizer):
        # Use llguidance for JSON schema
        from llguidance import grammar_from, LLMatcher, LLTokenizer
        self.ll_tokenizer = LLTokenizer.from_hf_tokenizer(tokenizer)
        grammar = grammar_from("json_schema", json.dumps(schema))
        self.matcher = LLMatcher(self.ll_tokenizer, grammar)

    def compute_mask(self, logits: mx.array) -> mx.array:
        import numpy as np
        mask_np = self.matcher.get_token_mask()
        return mx.array(mask_np)

# In generate_step
def generate_step(..., grammar_constraint: Optional[GrammarConstraint] = None):
    # ... existing code ...

    if grammar_constraint:
        mask = grammar_constraint.compute_mask(logits)
        logits = mx.where(mask, logits, -mx.inf)

    # Sample token
    token = sample(logits, temperature, top_p, ...)

    # Update grammar state
    if grammar_constraint and hasattr(grammar_constraint, 'update'):
        grammar_constraint.update(token)

    return token
```

## Dependencies

```txt
guidance>=0.1.0
guidance-stitch==0.1.5
llguidance==1.4.0
jinja2>=3.0
numpy
pydantic>=2.0
requests
psutil
```

## Open Questions

1. **Stateless Function Compilation**: How does Guidance cache compiled grammars?
2. **Tool Call Chaining**: How to chain multiple tool calls efficiently?
3. **Error Recovery**: What happens when grammar cannot be satisfied?
4. **Backward Compatibility**: How to handle models without tool calling training?

## References

- Guidance GitHub: https://github.com/guidance-ai/guidance
- Guidance Documentation: https://guidance-ai.github.io/guidance/
- LLGuidance: https://github.com/guidance-ai/llguidance
- Pydantic: https://docs.pydantic.dev/
