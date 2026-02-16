# GBNF (Guidance BNF) Grammar Format: Complete Reference

## Overview

GBNF (Guidance Backus-Naur Form) is an extended BNF grammar format designed specifically for constraining language model outputs. It was developed for the llama.cpp project and provides efficient token-level constraints through grammar compilation.

## Core Syntax

### Rule Definition

```gbnf
rule_name ::= expression
```

- Rules start at the beginning of a line
- Use `::=` for definition
- Rule names are alphanumeric with underscores
- The `root` rule defines the complete output

### Basic Elements

#### 1. String Literals
```gbnf
# Exact text matching
greeting ::= "Hello, world!"
newline  ::= "\n"
tab      ::= "\t"
quote    ::= "\""
```

#### 2. Character Classes
```gbnf
# Single characters or ranges
letter   ::= [a-zA-Z]
digit    ::= [0-9]
hex_digit ::= [0-9a-fA-F]
not_digit ::= [^0-9]        # Negation
any_char ::= .              # Any character
```

#### 3. Alternatives (|)
```gbnf
color ::= "red" | "green" | "blue"
answer ::= "yes" | "no" | "maybe"
```

#### 4. Sequences (juxtaposition)
```gbnf
date ::= year "-" month "-" day
year ::= [0-9]{4}
month ::= [0-9]{2}
day ::= [0-9]{2}
```

#### 5. Repetition Operators
```gbnf
# Zero or more (*)
list ::= item*

# One or more (+)
non_empty_list ::= item+

# Optional (?)
maybe_item ::= item?

# Specific range ({min,max})
digits_2_to_5 ::= [0-9]{2,5}
digits_at_least_3 ::= [0-9]{3,}
digits_at_most_10 ::= [0-9]{0,10}
exactly_5 ::= [0-9]{5}
```

## Advanced Syntax

### Token References

```gbnf
# Reference token by ID
root ::= <[1000]> thinking <[1001]>

# Reference token by literal string
root ::= "<|tool_call|>" content "<|/tool_call|>"
```

### Negation

```gbnf
# Character negation
not_quote ::= [^"]
not_bracket ::= [^\\]]

# Token negation
anything_except_eos ::= !<[2]>

# Pattern negation
not_function ::= !"function"
```

### Whitespace Convention

By convention, `ws` rules are applied after literals when whitespace is allowed:

```gbnf
ws ::= | " " | "\n" [ \t]{0,20}

# Usage in JSON
value ::= object | array | string | number | ("true" | "false" | "null") ws
```

## Complete Grammar Examples

### 1. JSON Grammar

```gbnf
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
```

### 2. Tool Call Grammar

```gbnf
# Generic tool call format
root ::= "{" ws
           "\"name\"" ":" ws string "," ws
           "\"arguments\"" ":" ws parameters
         "}" ws

parameters ::= "{" ws
                (string ":" ws value ("," ws string ":" ws value)*)?
              "}" ws

string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\"" ws

value ::= object | array | string | number | "true" ws | "false" ws | "null" ws

object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [0-9]{0,15})? ws

ws ::= | " " | "\n" [ \t]{0,20}
```

### 3. Chess Move Notation

```gbnf
root    ::= "1. " move " " move "\n" ([1-9] [0-9]? ". " move " " move "\n")+
move    ::= (pawn | nonpawn | castle) [+#]?

nonpawn ::= [NBKQR] [a-h]? [1-8]? "x"? [a-h] [1-8]
pawn    ::= ([a-h] "x")? [a-h] [1-8] ("=" [NBKQR])?
castle  ::= "O-O" "-O"?
```

### 4. Arithmetic Expressions

```gbnf
root  ::= expr "=" ws term "\n"
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
```

### 5. Email Address

```gbnf
root ::= local "@" domain
local ::= [a-zA-Z0-9._%+-]+
domain ::= [a-zA-Z0-9.-]+ "." [a-zA-Z]{2,}
```

## Escape Sequences

Supported escape sequences in string literals:

| Escape | Meaning | Unicode |
|--------|---------|---------|
| `\\` | Backslash | U+005C |
| `\"` | Double quote | U+0022 |
| `\t` | Tab | U+0009 |
| `\n` | Newline | U+000A |
| `\r` | Carriage return | U+000D |
| `\xHH` | Hex byte (1 byte) | U+0000-U+00FF |
| `\uHHHH` | Unicode BMP (2 bytes) | U+0000-U+FFFF |
| `\UHHHHHHHH` | Unicode full (4 bytes) | U+0000-U+10FFFF |

## Comments

Comments start with `#` and continue to end of line:

```gbnf
# This is a comment
root ::= "hello"  # This is an inline comment
```

## Grammar Rules

### Rule Composition

Rules can reference other rules:

```gbnf
root ::= greeting name
greeting ::= "Hello, " | "Hi, "
name ::= [A-Z][a-z]+
```

### Recursive Rules

Rules can be recursive:

```gbnf
root ::= list
list ::= "[" ws (value ("," ws value)*)? "]"
value ::= list | string | number
```

### Left Recursion (Not Supported)

GBNF does not support left recursion. Use right recursion instead:

```gbnf
# BAD: Left recursion
expr ::= expr "+" term | term

# GOOD: Right recursion
expr ::= term ("+" term)*
```

## Performance Considerations

### Efficient Patterns

```gbnf
# Efficient: Use ranges
digit ::= [0-9]

# Efficient: Use specific repetition
digits ::= [0-9]{1,10}

# Avoid: Too many alternatives
letter ::= "a" | "b" | "c" | ... | "z"  # Bad
letter ::= [a-z]                        # Good
```

### Maximum Repetition Threshold

The llama.cpp implementation has `MAX_REPETITION_THRESHOLD = 2000`:

```gbnf
# Acceptable
items ::= item{0,1000}

# May cause issues (exceeds threshold)
items ::= item{0,5000}
```

### Character Class Optimization

```gbnf
# Optimized internally
letter ::= [a-zA-Z]     # Stored as ranges, not individual characters

# Negation is also optimized
not_digit ::= [^0-9]    # Efficient bitmask representation
```

## Common Patterns

### Optional Whitespace

```gbnf
# Minimal whitespace
ws ::= " "?

# Flexible whitespace
ws ::= [ \t\n\r]*

# Convention: applied after literals
value ::= number ws
```

### Comma-Separated Lists

```gbnf
list ::= "[" ws item ("," ws item)* "]"
```

### Quoted Strings

```gbnf
# Simple strings
string ::= "\"" [^"]* "\""

# With escape sequences
string ::= "\"" ([^"\\] | "\\" [\"\\nrt])* "\""

# JSON-style strings
string ::= "\"" ([^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
```

### Numbers

```gbnf
# Integer
integer ::= [0-9]+

# Signed integer
signed ::= "-"? [0-9]+

# Decimal
decimal ::= [0-9]+ "." [0-9]+

# Scientific notation
scientific ::= [0-9]+ ("." [0-9]+)? [eE] [-+]? [0-9]+
```

## Validation

### Grammar Validation

Common validation rules:

1. **Root rule required**: Every grammar must define a `root` rule
2. **No undefined references**: All referenced rules must be defined
3. **No left recursion**: Left-recursive rules will cause errors
4. **Termination**: Recursive rules must have a base case

### Testing

Test grammars with sample inputs:

```bash
# Using llama.cpp
./main -m model.gguf --grammar-file grammar.gbnf -p "Generate"
```

## Conversion Tools

### JSON Schema to GBNF

```bash
python examples/json_schema_to_grammar.py schema.json
```

### Regex to GBNF

```bash
python examples/regex_to_grammar.py "^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$"
```

### Pydantic to GBNF

```bash
python examples/pydantic_models_to_grammar.py my_models.py
```

## Integration with MLX-LM

### Approach 1: Use llguidance

```python
from llguidance import grammar_from

# Convert GBNF to llguidance grammar
grammar = grammar_from("lark", gbnf_string)

# Use for constrained generation
matcher = LLMatcher(tokenizer, grammar)
```

### Approach 2: Custom Parser

```python
# Parse GBNF to AST
class GBNFParser:
    def __init__(self, grammar_string: str):
        self.grammar_string = grammar_string
        self.rules = {}
        self.parse()

    def parse(self):
        """Parse GBNF grammar"""
        for line in self.grammar_string.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, expr = line.split('::=')
            self.rules[name.strip()] = self.parse_expression(expr.strip())

    def parse_expression(self, expr: str) -> Any:
        """Parse grammar expression"""
        # Implementation depends on requirements
        pass
```

## Best Practices

1. **Start with root rule**: Always define what complete output looks like
2. **Use comments**: Document complex rules
3. **Test incrementally**: Build and test grammars piece by piece
4. **Profile performance**: Test with actual model generation
5. **Reuse patterns**: Use common patterns (ws, string, number) as building blocks
6. **Avoid over-constraint**: Too strict grammars can cause generation issues

## References

- llama.cpp GBNF Guide: https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md
- GBNF Examples: https://github.com/ggml-org/llama.cpp/tree/master/grammars
- Constrained Decoding Guide: https://www.aidancooper.co.uk/constrained-decoding/
