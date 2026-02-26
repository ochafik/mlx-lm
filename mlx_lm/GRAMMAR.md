# Grammar-Constrained Generation

Grammar-constrained generation forces model output to conform to a specific
structure: JSON schemas, regular expressions, or fixed choices. This is
powered by [llguidance](https://github.com/guidance-ai/llguidance), a
Rust-based grammar engine that computes token masks in microseconds.

## Installation

```bash
pip install mlx-lm[grammar]
# or just: pip install llguidance
```

## CLI Usage

All grammar options are available on `generate` and `chat` subcommands.
They are mutually exclusive (pick one).

### JSON Schema

Constrain output to match a JSON schema:

```bash
mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --json-schema '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}' \
  -p "Return a person named Alice age 25"
```

Output: `{"name":"Alice","age":25}`

Load a schema from a file with `@`:

```bash
mlx_lm generate --json-schema @schema.json -p "Generate data"
```

### Regular Expression

Constrain output to match a regex:

```bash
mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --regex "(yes|no)" \
  -p "Is the sky blue?"
```

Output: `yes`

### Choices

Constrain output to one of several strings:

```bash
mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --choices red green blue \
  -p "Pick a color"
```

Output: `blue`

### Lark Grammar

Pass a raw [Lark grammar](https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md)
(via `llguidance.grammar_from`):

```bash
mlx_lm generate --grammar '<your lark grammar string>' -p "..."
```

### Chat Mode

Grammar constraints persist across the session (reset each turn):

```bash
mlx_lm chat \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --json-schema '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}'
```

## Server (OpenAI-Compatible API)

Start the server:

```bash
mlx_lm server --model mlx-community/Llama-3.2-3B-Instruct-4bit --port 8080
```

### `response_format` with JSON Schema

Follows the [OpenAI structured outputs](https://platform.openai.com/docs/guides/structured-outputs)
API format:

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Give me a person named Bob age 30"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "schema": {
          "type": "object",
          "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
          },
          "required": ["name", "age"],
          "additionalProperties": false
        }
      }
    }
  }'
```

### `response_format` with JSON Object

Simple JSON constraint (any valid JSON object):

```bash
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Return some JSON"}],
    "response_format": {"type": "json_object"}
  }'
```

### `response_format` with Regex / Choices

Non-standard extensions:

```bash
# Regex
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "yes or no?"}],
    "response_format": {"type": "regex", "regex": "(yes|no)"}
  }'

# Choices
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Pick a fruit"}],
    "response_format": {"type": "choices", "choices": ["apple", "banana", "cherry"]}
  }'
```

## Python API

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Create a grammar processor (pick one)
processor = make_grammar_logits_processor(
    tokenizer,
    json_schema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    # regex="(yes|no)",
    # choices=["a", "b", "c"],
)

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Give me a name as JSON"}],
    tokenize=False, add_generation_prompt=True,
)
tokens = tokenizer.encode(prompt, add_special_tokens=False)

response = generate(
    model, tokenizer, tokens,
    max_tokens=100,
    sampler=make_sampler(0.0),
    logits_processors=[processor],
    verbose=False,
)
print(response)  # {"name":"Aria"}
```

## How It Works

1. A `GrammarState` (backed by llguidance's `LLMatcher`) tracks the current
   position in the grammar.
2. At each generation step, `compute_bitmask()` returns which tokens are valid.
3. The `GrammarLogitsProcessor` sets invalid token logits to `-inf` so the
   sampler only picks valid tokens.
4. When the grammar is fully satisfied, all logits except EOS are set to `-inf`,
   stopping generation.

## Architecture

```
mlx_lm/grammar/
  __init__.py              # Public API, lazy loading
  base.py                  # Abstract GrammarState interface
  llguidance_adapter.py    # LLGuidanceState (LLMatcher wrapper)
  mask_ops.py              # MLX mask operations
  jinja_analysis.py        # Auto-detect tool call format from chat templates
  tool_schema.py           # Convert tool definitions to Lark grammars

mlx_lm/sample_utils.py    # GrammarLogitsProcessor, make_grammar_logits_processor()
```

## Tool Calling (Experimental)

Grammar-constrained tool calling forces the model to produce syntactically valid
tool calls that conform to the provided tool definitions. The format is
auto-detected from the model's chat template.

### CLI

Pass tool definitions as JSON (or `@file.json`):

```bash
mlx_lm generate \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --tools '[{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]' \
  -p "What is the weather in London?"
```

Or from a file:

```bash
mlx_lm generate --tools @tools.json -p "What is the weather in London?"
```

### Server (OpenAI-Compatible API)

Use `tool_choice` to enable grammar-constrained tool calling:

```bash
# tool_choice: "required" — force a valid tool call
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "required"
  }'

# tool_choice: specific function — force a specific tool
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Weather in Paris?"}],
    "tools": [...],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
  }'
```

**`tool_choice` values:**

| Value | Grammar Constraint | Behavior |
|-------|-------------------|----------|
| `"auto"` (default) | No | Model decides freely whether to call tools |
| `"required"` | Yes | Forces valid tool call output |
| `{"type":"function","function":{"name":"..."}}` | Yes | Forces specific tool |
| `"none"` | No | Tools ignored |

### Python API

```python
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

processor = make_grammar_logits_processor(tokenizer, tools=tools)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Weather in London?"}],
    tokenize=False, add_generation_prompt=True, tools=tools,
)
tokens = tokenizer.encode(prompt, add_special_tokens=False)

response = generate(
    model, tokenizer, tokens,
    max_tokens=100, sampler=make_sampler(0.0),
    logits_processors=[processor], verbose=False,
)
print(response)
```

### How It Works

1. `jinja_analysis.py` analyzes the model's chat template to detect the tool
   call format (JSON, XML, or Python-style).
2. `tool_schema.py` converts tool definitions into a Lark grammar matching the
   detected format.
3. The grammar constrains generation to produce only valid tool calls.
