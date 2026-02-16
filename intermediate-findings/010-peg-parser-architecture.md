# PEG Parser Architecture Analysis (llama.cpp PR #17136)

## Overview

This document provides a comprehensive analysis of the PEG (Parsing Expression Grammar) parser architecture introduced in llama.cpp PR #17136. This parser forms the foundation for automatic grammar generation and tool calling in llama.cpp.

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     common_peg_parser_builder                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   common_peg_arena                         │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         std::variant<parser_types> (18 types)        │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         std::unordered_map<rules>                    │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              common_peg_ast_arena                          │ │
│  │  ┌──────────────────────────────────────────────────────┐  │ │
│  │  │         std::vector<common_peg_ast_node>             │  │ │
│  │  └──────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Builder Construction → Arena → Parse Execution → AST → GBNF Generation
        ↓                  ↓          ↓              ↓            ↓
   Parser Types      Stored Parsers   Result    Semantic     Grammar
                                      Code      Analysis       String
```

## Parser Types (18 Variants)

### 1. Fundamental Parsers

| Parser Type | Description | PEG Notation | Example |
|-------------|-------------|--------------|---------|
| `epsilon_parser` | Always succeeds, matches nothing | ε | Empty string |
| `start_parser` | Matches start of input | ^ | Beginning anchor |
| `end_parser` | Matches end of input | $ | End anchor |
| `literal_parser` | Exact string match | "hello" | Literal text |
| `any_parser` | Matches any single UTF-8 codepoint | . | Wildcard |

### 2. Combinator Parsers

| Parser Type | Description | PEG Notation | Example |
|-------------|-------------|--------------|---------|
| `sequence_parser` | Sequential composition | A B C | All must match |
| `choice_parser` | Ordered choice (first wins) | A \| B \| C | First match |
| `repetition_parser` | Bounded repetition | A{min,max} | min ≤ count ≤ max |
| `and_parser` | Positive lookahead (consume nothing) | &A | Assert A matches |
| `not_parser` | Negative lookahead (consume nothing) | !A | Assert A fails |

### 3. Character Class Parsers

| Parser Type | Description | PEG Notation | Example |
|-------------|-------------|--------------|---------|
| `chars_parser` | Character class with range support | [a-zA-Z] | Letter |
| `space_parser` | Whitespace (space, tab, newline) | [ \\t\\n]* | Spaces |
| `json_string_parser` | JSON string content (without quotes) | specialized | Escaped strings |

### 4. Special Parsers

| Parser Type | Description | Purpose |
|-------------|-------------|---------|
| `until_parser` | Match until one of delimiters found | Efficient exclusion |
| `schema_parser` | JSON schema metadata wrapper | Schema-to-grammar |
| `rule_parser` | Named rule (with optional trigger) | Recursion + lazy gen |
| `ref_parser` | Forward reference to rule (resolved later) | Recursive grammars |
| `atomic_parser` | No partial AST nodes on NEED_MORE_INPUT | Clean streaming |
| `tag_parser` | Semantic tag for AST nodes | Analysis |

## Key Design Patterns

### 1. Arena-Based Allocation

```cpp
class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_ = COMMON_PEG_INVALID_PARSER_ID;
};
```

**Benefits:**
- Stable references (indices don't change)
- Efficient memory layout
- Easy serialization/deserialization

### 2. Builder Pattern with Fluent API

```cpp
// Operator overloading for elegant composition
p1 + p2        // Sequence
p1 | p2        // Choice
p1 << p2       // Sequence with space
"lit" + p      // Literal + parser
```

**Example Usage:**
```cpp
auto json = builder.rule("json", [&]() {
    return builder.choice({
        builder.json_object(),
        builder.json_array(),
        builder.json_string(),
        // ...
    });
});
```

### 3. Visitor-Based Execution

```cpp
struct parser_executor {
    const common_peg_arena & arena;
    common_peg_parse_context & ctx;
    size_t start_pos;

    common_peg_parse_result operator()(const common_peg_literal_parser & p);
    common_peg_parse_result operator()(const common_peg_sequence_parser & p);
    // ... one for each parser type
};
```

**Benefits:**
- Type-safe dispatch via `std::visit`
- Each parser type has isolated logic
- Easy to add new parser types

### 4. Trie-Based Delimiter Matching

```cpp
struct trie {
    struct node {
        std::map<unsigned char, size_t> children;
        bool is_word;
    };
    std::vector<node> nodes;

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };
    match_result check_at(std::string_view sv, size_t start_pos) const;
};
```

**Used in:**
- `until_parser`: Efficient multi-delimiter scanning
- `gbnf_excluding_pattern()`: Generate exclusion grammars

**Benefits:**
- O(k) where k = delimiter length (not O(n×m))
- Single pass through input
- Generates optimal exclusion patterns

## Parse Execution

### Result Types

```cpp
enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
};
```

### Partial Input Support

```cpp
struct common_peg_parse_context {
    std::string input;
    bool is_partial;           // Enable partial parsing
    common_peg_ast_arena ast;  // AST node storage
    int parse_depth;           // Recursion depth tracking
};
```

**Key Insight:** `is_partial` allows streaming parsing where incomplete input returns `NEED_MORE_INPUT` instead of failing.

### Atomic Parsers

```cpp
common_peg_parse_result operator()(const common_peg_atomic_parser & p) {
    auto result = arena.parse(p.child, ctx, start_pos);
    if (result.need_more_input()) {
        // Clear nodes so they don't propagate up
        result.nodes.clear();
    }
    return result;
}
```

**Purpose:** Prevent incomplete nodes from appearing in AST during streaming.

## GBNF Generation

### Conversion Process

```cpp
void common_peg_arena::build_grammar(
    const common_grammar_builder & builder,
    bool lazy  // Only emit trigger rules and descendants
) const {
    // 1. Collect reachable rules (from root or triggers)
    // 2. Convert each parser variant to GBNF syntax
    // 3. Emit rules in order
}
```

### Parser to GBNF Mapping

| PEG Parser | GBNF Output |
|------------|------------|
| `literal("hello")` | `"hello"` |
| `sequence([a,b,c])` | `a b c` |
| `choice([a,b,c])` | `a \| b \| c` |
| `repeat(p, 0, -1)` | `p*` |
| `repeat(p, 1, -1)` | `p+` |
| `repeat(p, 0, 1)` | `p?` |
| `repeat(p, m, n)` | `p{m,n}` |
| `chars("[a-z]")` | `[a-z]` |
| `until(["</tag>"])` | `([^<] \| "<" [^/] ...)*` |
| `schema(p, name, schema)` | `<json-schema-grammar>` |

### Lazy Grammar Generation

```cpp
// Trigger rules mark entry points for lazy generation
builder.trigger_rule("tool-call", tool_parser);

// Only trigger rules and descendants are emitted
arena.build_grammar(grammar_builder, lazy=true);
```

**Use Case:** Chat templates where tool calls are optional - only include tool grammar when actually needed.

## Chat Template Integration

### Native Tool Calling Format

```cpp
class common_chat_peg_native_builder : public common_chat_peg_builder {
  public:
    common_peg_parser tool(const common_peg_parser & p);
    common_peg_parser tool_open(const common_peg_parser & p);
    common_peg_parser tool_close(const common_peg_parser & p);
    common_peg_parser tool_id(const common_peg_parser & p);
    common_peg_parser tool_name(const common_peg_parser & p);
    common_peg_parser tool_args(const common_peg_parser & p);
};
```

### AST Mapping

```cpp
class common_chat_peg_native_mapper : public common_chat_peg_mapper {
    common_chat_tool_call * current_tool;

  public:
    void map(const common_peg_ast_node & node) override;
};
```

**Flow:**
1. Parse input with PEG parser → AST
2. Visit AST nodes with mapper
3. Extract semantic information (tool name, args, etc.)

## Serialization

### JSON Format

```cpp
// Save parser to JSON
std::string data = arena.save();

// Load parser from JSON
arena.load(data);
```

**Schema:**
```json
{
  "parsers": [
    {"type": "literal", "literal": "hello"},
    {"type": "sequence", "children": [0, 1]},
    // ...
  ],
  "rules": {
    "greeting": 0,
    "json-value": 42
  },
  "root": 42
}
```

**Benefits:**
- Pre-compiled grammars can be cached
- Cross-process parser sharing
- Debugging and inspection

## Performance Characteristics

### Parser Execution

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Literal match | O(n) | n = literal length |
| Sequence | O(Σ children) | Short-circuit on fail |
| Choice | O(first match) | Ordered choice |
| Repetition | O(n×child) | n = input length |
| Chars match | O(n) | UTF-8 codepoint iteration |
| Until (trie) | O(n) | Single pass |

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Parser variant | ~32 bytes | std::variant overhead |
| AST node | ~64 bytes | string_view + children |
| Trie node | ~40 bytes | map + bool |

### GBNF Generation

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Rule collection | O(V+E) | Graph traversal |
| GBNF conversion | O(V×avg_children) | Recursive |
| Exclusion pattern | O(k×d) | k = delimiters, d = max depth |

## Key Innovations

### 1. UTF-8 Awareness

```cpp
auto result = parse_utf8_codepoint(ctx.input, pos);
// Returns: {status, codepoint, bytes_consumed}
```

All character operations work with codepoints, not bytes.

### 2. Reference Resolution

```cpp
void common_peg_arena::resolve_refs() {
    // Replace all ref_parser with actual rule IDs
    // Enable forward references in recursive grammars
}
```

**Enables:**
```cpp
auto expr = builder.rule("expr", [&]() {
    return builder.choice({
        term,
        builder.sequence({expr, "+", term}),  // Forward ref
    });
});
```

### 3. Trigger-Based Lazy Generation

```cpp
// Mark entry points
builder.trigger_rule("tool", tool_parser);
builder.trigger_rule("content", content_parser);

// Emit only what's needed
arena.build_grammar(builder, lazy=true);
```

**Result:**
- Smaller grammars
- Faster token mask computation
- Dynamic parser switching

## Comparison: PEG vs CFG

| Property | PEG | CFG |
|----------|-----|-----|
| Semantics | Ordered choice | Unordered choice |
| Ambiguity | Never ambiguous | Can be ambiguous |
| Recursion | Direct and indirect | Direct and indirect |
| Lookahead | Explicit (&A, !A) | Implicit in LR/GLR |
| Implementation | Recursive descent | LALR/Earley/CYK |
| Performance | Predictable | Variable |

**Why PEG for llama.cpp:**
1. Deterministic behavior (no ambiguity)
2. Simple implementation (no table generation)
3. Easy composition (fluent API)
4. Direct mapping to GBNF

## Implementation Details

### Sequence Flattening

```cpp
common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested sequences: (A (B C) D) → (A B C D)
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        if (auto seq = std::get_if<common_peg_sequence_parser>(&parser)) {
            flattened.insert(flattened.end(), seq->children.begin(), seq->children.end());
        } else {
            flattened.push_back(p);
        }
    }
}
```

**Benefit:** Reduces parser depth, improves performance.

### Choice Flattening

Similar optimization for choice parsers to reduce nesting.

### Regex-Like Character Classes

```cpp
// Supports: [a-zA-Z], [^0-9], [\n\t], [\\x00-\\x7F]
auto letter = builder.chars("[a-zA-Z]", 1, -1);  // One or more letters
auto not_digit = builder.chars("[^0-9]");       // Any non-digit
```

**Internals:**
- Parse character ranges
- Support negation
- Handle escape sequences (\n, \t, \xHH, \uHHHH)

## Python Equivalent Considerations

### Core Requirements for Python Port

1. **Variant Type**: Use `Union` or dataclasses with discriminators
2. **Visitor Pattern**: Use `functools.singledispatch` or match statements (Python 3.10+)
3. **Arena Storage**: Simple list/dict-based arena
4. **Builder Pattern**: Fluent API with method chaining
5. **UTF-8 Support**: Python strings are Unicode by default
6. **Trie**: Custom trie implementation or use existing library

### Potential Python Libraries

| Library | Approach | Pros | Cons |
|---------|----------|------|------|
| **parsimonious** | PEG + packrat | Fast (~30% faster than Lark), simple API | Less active development |
| **arpeggio** | PEG + packrat | Good documentation, Python 3 | Less performant than parsimonious |
| **lark** | Earley/LALR | Very popular, multiple backends | Not pure PEG (can emulate) |
| **pyPEG2** | PEG | Minimal, pure Python | Less feature-rich |
| **pyparsing** | Not PEG | Very popular, feature-rich | CFG-based, slower |
| ** TatSu/Grako** | Parser generator | Can generate from grammar | Code generation overhead |

### Recommendation: Custom Implementation

Given llama.cpp's specific requirements:
- Direct GBNF generation
- Lazy grammar generation
- Trie-based delimiters
- AST mapping for chat templates
- JSON schema integration

**Best approach:** Create a lightweight PEG library inspired by parsimonious but with:
1. Direct GBNF output
2. Trigger-based lazy generation
3. MLX-LM specific optimizations
4. Minimal dependencies

## Integration with MLX-LM

### Proposed Architecture

```python
# mlx_lm/grammar/peg_arena.py
class PegParserVariant(TypedDict):
    type: str
    # ... type-specific fields

class PegArena:
    parsers: List[PegParserVariant]
    rules: Dict[str, int]
    root: Optional[int]

    def parse(self, input: str) -> PegParseResult:
        pass

    def to_gbnf(self, lazy: bool = False) -> str:
        pass

# mlx_lm/grammar/peg_builder.py
class PegBuilder:
    def literal(self, text: str) -> PegParser:
        pass

    def sequence(self, *parsers: PegParser) -> PegParser:
        pass

    def choice(self, *parsers: PegParser) -> PegParser:
        pass

    def rule(self, name: str, parser: PegParser, trigger: bool = False) -> PegParser:
        pass

    def json(self) -> PegParser:
        pass

    def schema(self, parser: PegParser, name: str, schema: dict) -> PegParser:
        pass
```

### Tool Calling Example

```python
# Define tool call grammar using PEG
builder = PegBuilder()

get_weather = builder.rule("get-weather", builder.sequence([
    builder.literal('{"name": "get_weather", "arguments":'),
    builder.json_object(),
    builder.literal('}')
]), trigger=True)

get_time = builder.rule("get-time", builder.sequence([
    builder.literal('{"name": "get_time", "arguments":'),
    builder.json_object(),
    builder.literal('}')
]), trigger=True)

tool_call = builder.choice([get_weather, get_time])

# Build arena
arena = builder.build()

# Generate GBNF (lazy - only triggers and descendants)
gbnf = arena.to_gbnf(lazy=True)

# Use for constrained generation
grammar_state = LLGuidanceState.from_gbnf(tokenizer, gbnf)
```

## Performance Optimizations

### 1. Memoization (Packrat Parsing)

```cpp
// Not shown in llama.cpp, but could be added:
std::unordered_map<size_t, common_peg_parse_result> memo_;

common_peg_parse_result parse(common_peg_parser_id id, size_t pos) {
    size_t key = (id << 32) | pos;
    if (memo_.find(key) != memo_.end()) {
        return memo_[key];
    }
    auto result = /* actual parse */;
    memo_[key] = result;
    return result;
}
```

**Benefit:** Linear-time parsing for most grammars.

### 2. Left-Factoring

Automatic optimization of sequences:
```python
# Before:
choice([
    sequence([literal("if"), condition]),
    sequence([literal("if"), condition, literal("else"), body]),
])

# After left-factoring:
sequence([
    literal("if"),
    condition,
    choice([
        eps(),
        sequence([literal("else"), body]),
    ]),
])
```

### 3. Grammar Compilation

Pre-compile frequently used grammars:
- JSON schema → GBNF
- Tool definitions → GBNF
- Chat templates → PEG parser

## Limitations and Trade-offs

### 1. No Left Recursion

```python
# This won't work:
expr = rule("expr", choice([
    expr + "+" + term,  # Left recursion!
    term
]))
```

**Solution:** Right recursion or iteration:
```python
expr = rule("expr", sequence([
    term,
    zero_or_more(sequence([literal("+"), term]))
]))
```

### 2. Ordered Choice Semantics

```python
# First match wins, rest ignored
choice([
    literal("if"),
    literal("ifelse"),  # Never reached!
])
```

**Solution:** Order by specificity (longest first).

### 3. No Backtracking in Choice

Once a choice branch succeeds, no going back:
```python
choice([
    sequence([literal("a"), literal("b")]),
    sequence([literal("a"), literal("c")]),
])
# For input "ac": First branch fails at "b", but doesn't try second
```

**Correction:** Actually, PEG DOES backtrack within choice. The first failing branch allows trying the next. This is correct behavior.

## References

### Code Locations

- `/Users/ochafik/github/llama.cpp/common/peg-parser.h` - Header definitions
- `/Users/ochafik/github/llama.cpp/common/peg-parser.cpp` - Implementation
- `/Users/ochafik/github/llama.cpp/common/chat-peg-parser.h` - Chat templates

### External Resources

- PR #17136: https://github.com/ggml-org/llama.cpp/pull/17136
- PEG Paper: "Parsing Expression Grammars: A Recognition-Based Syntactic Foundation" (Ford, 2004)
- Packrat Parsing: http://www.vpri.org/html/trips/ifn/pegs/pegs.txt

---

**Document Version:** 1.0
**Last Updated:** 2025
**Status:** Complete Architecture Analysis
