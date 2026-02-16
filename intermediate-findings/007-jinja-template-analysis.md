# Jinja Template Differential Analysis for Tool Calling

## Overview

The llama.cpp approach uses differential analysis of Jinja templates to automatically generate grammars for tool calling. This document explains how this system works and how to implement it.

## Concept

Differential analysis means:
1. Parse the Jinja template to find tool call placeholders
2. Identify the format specification from template variables
3. Generate grammar matching that specific format
4. Apply constraints only during tool call generation

## Jinja Template Structure

### Tool Call Template Format

```jinja2
{# Example: Llama-3.1 style #}
{% set tool_present = true %}
{% set message_count = 4 %}
{% set tool_call_count = 1 %}
{%- for message in messages %}
    {{- '<|start_of_role|>' + message['role'] + '<|end_of_role|>' }}
    {%- if message['content'] %}
        {{- message['content'] + '<|end_of_message|>' }}
    {%- endif %}
    {%- if message['tool_calls'] %}
        {%- for tool_call in message['tool_calls'] %}
            {{- '<|tool_call|>' + '\n' }}
            {{- '{"name": "' + tool_call['name'] + '", "arguments": ' }}
            {{- json.dumps(tool_call['arguments']) + '}' + '\n' }}
            {{- '<|tool_call_end|>' }}
        {%- endfor %}
    {%- endif %}
{%- endfor %}
{{- '<|start_of_role|>assistant<|end_of_role|>' }}
```

### Template Analysis

The key is to identify:
1. **Tool call markers**: Special tokens that delimit tool calls
2. **Format patterns**: How tool calls are serialized
3. **Variable interpolation**: What data is inserted

## Automatic Detection Algorithm

### Step 1: Parse Template

```python
from jinja2 import Environment, BaseLoader, meta

def parse_jinja_template(template_str: str) -> meta.Ast:
    """Parse Jinja template into AST"""
    env = Environment(loader=BaseLoader())
    return env.parse(template_str)
```

### Step 2: Find Tool Call Patterns

```python
import re
from typing import List, Dict, Optional

class ToolCallPattern:
    """Detected tool call pattern in template"""

    def __init__(
        self,
        start_marker: Optional[str],
        end_marker: Optional[str],
        format_type: str,
        structure: Dict
    ):
        self.start_marker = start_marker  # e.g., "<|tool_call|>"
        self.end_marker = end_marker      # e.g., "<|tool_call_end|>"
        self.format_type = format_type    # "json", "xml", "custom"
        self.structure = structure         # Parsed structure

def detect_tool_call_patterns(template_str: str) -> List[ToolCallPattern]:
    """Detect tool call patterns in Jinja template"""

    patterns = []

    # Look for common patterns
    if '<|tool_call|>' in template_str:
        patterns.append(ToolCallPattern(
            start_marker='<|tool_call|>',
            end_marker='<|tool_call_end|>',
            format_type='json',
            structure='llama3_style'
        ))

    if 'tool_calls' in template_str and 'for tool_call in' in template_str:
        # Look at the structure inside the loop
        structure = analyze_tool_call_loop(template_str)
        patterns.append(ToolCallPattern(
            start_marker=None,
            end_marker=None,
            format_type=structure['format'],
            structure=structure
        ))

    return patterns

def analyze_tool_call_loop(template_str: str) -> Dict:
    """Analyze the structure of tool call rendering"""

    # Extract the tool_call loop content
    match = re.search(
        r'{%-?\s*for\s+tool_call\s+in\s+message\[.tool_calls.\](.*?)endfor\s*-?%}',
        template_str,
        re.DOTALL
    )

    if not match:
        return {'format': 'unknown'}

    loop_content = match.group(1)

    # Detect format
    if '"name":' in loop_content and '"arguments":' in loop_content:
        return {
            'format': 'json',
            'structure': {
                'name_field': 'name',
                'arguments_field': 'arguments',
                'requires_quotes': True
            }
        }
    elif '<function=' in loop_content:
        return {
            'format': 'xml',
            'structure': {
                'tag_format': '<function={name}>...</function>'
            }
        }

    return {'format': 'custom'}
```

### Step 3: Generate Grammar from Pattern

```python
from typing import List, Dict, Any

def generate_grammar_from_pattern(
    pattern: ToolCallPattern,
    tools: List[Dict[str, Any]]
) -> str:
    """Generate GBNF grammar from detected pattern"""

    if pattern.format_type == 'json':
        return generate_json_tool_grammar(tools, pattern.structure)
    elif pattern.format_type == 'xml':
        return generate_xml_tool_grammar(tools, pattern.structure)
    else:
        return generate_custom_tool_grammar(tools, pattern.structure)

def generate_json_tool_grammar(
    tools: List[Dict[str, Any]],
    structure: Dict
) -> str:
    """Generate JSON grammar for tool calls"""

    # Build tool alternatives
    tool_alternatives = []
    for tool in tools:
        name = tool['name']
        parameters = tool.get('parameters', {})

        # Generate parameter grammar
        param_grammar = generate_json_schema_grammar(parameters)

        tool_alternatives.append(f'''
    {{
      "name": "{name}",
      "arguments": {param_grammar}
    }}''')

    alternatives_str = ' |\\\n    '.join(tool_alternatives)

    return f'''root ::= tool_call
tool_call ::= {alternatives_str}
'''

def generate_json_schema_grammar(schema: Dict) -> str:
    """Generate grammar for JSON schema"""
    schema_type = schema.get('type')

    if schema_type == 'object':
        props = schema.get('properties', {})
        required = schema.get('required', [])

        if not props:
            return '{{}}'

        prop_rules = []
        for i, (prop_name, prop_schema) in enumerate(props.items()):
            is_last = i == len(props) - 1
            prop_grammar = generate_json_schema_grammar(prop_schema)

            separator = '' if is_last else ','
            prop_rules.append(f'"\\"{prop_name}\\"": {prop_grammar}{separator}')

        props_str = ' '.join(prop_rules)
        return f'{{{{{props_str}}}}}'

    elif schema_type == 'array':
        items_schema = schema.get('items', {})
        item_grammar = generate_json_schema_grammar(items_schema)
        min_items = schema.get('minItems', 0)
        max_items = schema.get('maxItems', -1)

        if min_items == 0:
            return f'[{item_grammar}({"," {item_grammar}})*]'
        else:
            return f'[{item_grammar}({"," {item_grammar}})*]'

    elif schema_type == 'string':
        pattern = schema.get('pattern')
        if pattern:
            return regex_to_gbnf(pattern)
        return '"string"'

    elif schema_type in ['number', 'integer']:
        return 'number'

    elif schema_type == 'boolean':
        return '"true" | "false"'

    elif 'enum' in schema:
        alternatives = [f'"{v}"' for v in schema['enum']]
        return ' | '.join(alternatives)

    return 'any'

def regex_to_gbnf(regex_pattern: str) -> str:
    """Convert regex pattern to GBNF character class"""
    # Simplified conversion - full implementation would be more complex
    if regex_pattern == '^[a-zA-Z]+$':
        return '[a-zA-Z]+'
    elif regex_pattern == '^[0-9]+$':
        return '[0-9]+'
    return '.*'
```

## Complete Implementation

### Template Analyzer

```python
class JinjaTemplateAnalyzer:
    """Analyze Jinja templates for tool calling patterns"""

    def __init__(self):
        self.env = Environment(loader=BaseLoader())

    def analyze(self, template_str: str) -> Dict[str, Any]:
        """Complete analysis of template"""

        # Parse template
        ast = self.env.parse(template_str)

        # Find patterns
        patterns = detect_tool_call_patterns(template_str)

        # Extract variables
        variables = self.extract_variables(ast)

        # Build format specification
        format_spec = self.build_format_spec(patterns, variables)

        return {
            'patterns': patterns,
            'variables': variables,
            'format_spec': format_spec,
            'has_tool_calls': len(patterns) > 0
        }

    def extract_variables(self, ast) -> List[str]:
        """Extract all variables used in template"""
        variables = set()

        for node in ast.find_all(meta.Name):
            variables.add(node.name)

        return list(variables)

    def build_format_spec(
        self,
        patterns: List[ToolCallPattern],
        variables: List[str]
    ) -> Dict:
        """Build format specification from patterns"""

        if not patterns:
            return {}

        primary_pattern = patterns[0]

        return {
            'type': primary_pattern.format_type,
            'structure': primary_pattern.structure,
            'markers': {
                'start': primary_pattern.start_marker,
                'end': primary_pattern.end_marker
            },
            'variables': variables
        }
```

### Grammar Generator

```python
class ToolCallGrammarGenerator:
    """Generate grammars from Jinja template analysis"""

    def __init__(self):
        self.analyzer = JinjaTemplateAnalyzer()

    def generate(
        self,
        template_str: str,
        tools: List[Dict[str, Any]]
    ) -> str:
        """Generate complete grammar from template and tools"""

        # Analyze template
        analysis = self.analyzer.analyze(template_str)

        if not analysis['has_tool_calls']:
            return None

        # Generate grammar based on format
        format_type = analysis['format_spec']['type']

        if format_type == 'json':
            return self._generate_json_grammar(
                analysis['format_spec'],
                tools
            )
        elif format_type == 'xml':
            return self._generate_xml_grammar(
                analysis['format_spec'],
                tools
            )
        else:
            return self._generate_custom_grammar(
                analysis['format_spec'],
                tools
            )

    def _generate_json_grammar(
        self,
        format_spec: Dict,
        tools: List[Dict[str, Any]]
    ) -> str:
        """Generate JSON-style tool call grammar"""

        markers = format_spec.get('markers', {})

        # Build root with optional markers
        parts = []
        if markers.get('start'):
            parts.append(f'"{markers["start"]}"')

        parts.append('tool_call')

        if markers.get('end'):
            parts.append(f'"{markers["end"]}"')

        root = ' '.join(parts)

        # Generate tool alternatives
        tool_alts = []
        for tool in tools:
            tool_grammar = self._generate_tool_grammar(tool)
            tool_alts.append(tool_grammar)

        tools_rule = ' |\\\n    '.join(tool_alts)

        return f'''root ::= {root}
tool_call ::= {tools_rule}
'''

    def _generate_tool_grammar(self, tool: Dict[str, Any]) -> str:
        """Generate grammar for single tool"""
        name = tool['name']
        parameters = tool.get('parameters', {})

        param_grammar = self._generate_schema_grammar(parameters)

        return f'''{{"name": "{name}", "arguments": {param_grammar}}}'''

    def _generate_schema_grammar(self, schema: Dict) -> str:
        """Generate grammar from JSON schema"""
        # Implementation similar to generate_json_schema_grammar above
        # This is a simplified version
        schema_type = schema.get('type', 'string')

        if schema_type == 'object':
            return '{' + self._generate_object_grammar(schema) + '}'
        elif schema_type == 'array':
            return '[' + self._generate_array_grammar(schema) + ']'
        elif schema_type == 'string':
            return '"string"'
        elif schema_type in ['number', 'integer']:
            return 'number'
        elif schema_type == 'boolean':
            return '"true" | "false"'
        elif 'enum' in schema:
            return ' | '.join(f'"{v}"' for v in schema['enum'])

        return 'any'

    def _generate_object_grammar(self, schema: Dict) -> str:
        """Generate object grammar"""
        props = schema.get('properties', {})
        required = schema.get('required', [])

        if not props:
            return ''

        parts = []
        for i, (prop_name, prop_schema) in enumerate(props.items()):
            prop_grammar = self._generate_schema_grammar(prop_schema)
            is_required = prop_name in required
            is_last = i == len(props) - 1

            if is_required:
                parts.append(f'"\\"{prop_name}\\"": {prop_grammar}')
            else:
                parts.append(f'("\\"{prop_name}\\"": {prop_grammar})?')

            if not is_last:
                parts.append('","')

        return ' '.join(parts)

    def _generate_array_grammar(self, schema: Dict) -> str:
        """Generate array grammar"""
        items = schema.get('items', {})
        item_grammar = self._generate_schema_grammar(items)

        return f'{item_grammar}({"," {item_grammar})*'
```

## MLX-LM Integration

### Template-Based Grammar Selection

```python
# In mlx_lm/tokenizer_utils.py

def auto_detect_tool_grammar(
    tokenizer,
    tools: List[Dict[str, Any]]
) -> Optional[str]:
    """Auto-detect and generate tool call grammar from tokenizer template"""

    if not hasattr(tokenizer, 'chat_template'):
        return None

    chat_template = tokenizer.chat_template
    if not chat_template:
        return None

    # Generate grammar from template
    generator = ToolCallGrammarGenerator()
    grammar = generator.generate(chat_template, tools)

    return grammar

def get_tool_call_grammar(
    tokenizer,
    tools: List[Dict[str, Any]],
    grammar_override: Optional[str] = None
) -> Optional[str]:
    """Get tool call grammar, with override support"""

    if grammar_override:
        return grammar_override

    return auto_detect_tool_grammar(tokenizer, tools)
```

### Unified Tool Parser with Grammar

```python
# In mlx_lm/tool_parsers/unified_grammar.py

from typing import List, Dict, Any, Optional
import json

class GrammarToolParser:
    """Unified tool parser using grammar constraints"""

    def __init__(
        self,
        tokenizer,
        tools: List[Dict[str, Any]],
        grammar: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.tools = tools

        # Auto-detect or use provided grammar
        if grammar is None:
            grammar = auto_detect_tool_grammar(tokenizer, tools)

        if grammar:
            from mlx_lm.grammar import LLGuidanceState
            self.grammar_state = LLGuidanceState(tokenizer, grammar)
        else:
            self.grammar_state = None

    def get_token_mask(self) -> Optional[np.ndarray]:
        """Get token mask for current position"""
        if self.grammar_state is None:
            return None

        mask = self.grammar_state.compute_token_mask()
        return mask.to_bool_array()

    def update(self, token: int) -> bool:
        """Update grammar state"""
        if self.grammar_state is None:
            return True

        return self.grammar_state.update(token)

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse completed tool call"""
        try:
            tool_call = json.loads(text.strip())

            # Validate against tools
            name = tool_call.get('name')
            if not self._validate_tool_name(name):
                return None

            return {
                'name': name,
                'arguments': tool_call.get('arguments', {})
            }
        except (json.JSONDecodeError, AttributeError):
            return None

    def _validate_tool_name(self, name: str) -> bool:
        """Validate tool name"""
        return any(tool['name'] == name for tool in self.tools)
```

## Usage Examples

### Example 1: Llama-3.1 Style

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"enum": ["celsius", "fahrenheit"]}
            },
            "required": ["city"]
        }
    }
]

# Auto-detect and generate grammar
generator = ToolCallGrammarGenerator()
grammar = generator.generate(tokenizer.chat_template, tools)

print(grammar)
# Output:
# root ::= <|tool_call|> tool_call <|tool_call_end|>
# tool_call ::= {"name": "get_weather", "arguments": {"city": string, ("units": "celsius" | "fahrenheit")?}}
```

### Example 2: Custom Format

```python
# Custom Jinja template
custom_template = '''
{% for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{% if message['tool_calls'] %}
{% for tool_call in message['tool_calls'] }}
TOOL: {{ tool_call['name'] }}({{ tool_call['arguments'] }})
{% endfor %}
{% endif %}
{% endfor %}
'''

tools = [
    {
        "name": "calculator",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"}
            },
            "required": ["expression"]
        }
    }
]

# Generate grammar for custom format
generator = ToolCallGrammarGenerator()
grammar = generator.generate(custom_template, tools)

print(grammar)
# Output: Custom grammar based on detected format
```

## Best Practices

1. **Provide explicit grammar**: When possible, provide explicit grammar for reliability
2. **Validate detection**: Always validate auto-detected patterns
3. **Handle multiple formats**: Support multiple tool call formats in same model
4. **Cache grammars**: Cache generated grammars for performance
5. **Fallback to parsing**: Always have fallback to post-generation parsing

## Limitations

1. **Complex templates**: May not handle highly complex Jinja logic
2. **Custom functions**: Doesn't understand custom Jinja filters/functions
3. **Conditional rendering**: May misinterpret conditional tool call rendering
4. **Template changes**: Regenerating grammar when template changes

## References

- Jinja2 Documentation: https://jinja.palletsprojects.com/
- Llama.cpp Chat Templates: https://github.com/ggml-org/llama.cpp/tree/master/examples
- HuggingFace Chat Templates: https://huggingface.co/docs/transformers/chat_templating
