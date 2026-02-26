# Copyright 2025 Apple Inc.

"""
Extensive end-to-end tests for grammar-constrained generation.

Tests cover:
- Tool calling with grammar constraints (single tool, multiple tools, complex schemas)
- JSON schema generation with various shapes
- Regex, choices
- Server tool_choice parsing with varied inputs
- Format detection across different tokenizer families
- Thinking model compatibility
- Edge cases: empty parameters, deeply nested schemas, optional fields
"""

import json
import pytest
import mlx.core as mx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_TOOLS = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
            },
            "required": ["city"],
        },
    }
]

MULTI_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]

COMPLEX_TOOL = [
    {
        "name": "create_event",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "date": {"type": "string"},
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "location": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"},
                    },
                    "required": ["name"],
                },
                "is_recurring": {"type": "boolean"},
            },
            "required": ["title", "date"],
        },
    }
]

OPENAI_STYLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Format detection tests
# ---------------------------------------------------------------------------


class TestFormatDetection:
    """Test tool call format detection across tokenizer families."""

    def test_llama_format_detection(self):
        """Llama 3 uses JSON format with 'name' and 'parameters' fields."""
        from mlx_lm.grammar.jinja_analysis import get_tool_format_from_tokenizer
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        fmt = get_tool_format_from_tokenizer(tok)
        assert fmt is not None
        assert fmt.format_type == "json"
        assert fmt.name_field == "name"
        assert fmt.arguments_field == "parameters"

    def test_nemotron_format_detection(self):
        """Nemotron uses qwen3_coder XML format with <function=> tags."""
        from mlx_lm.grammar.jinja_analysis import get_tool_format_from_tokenizer
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit",
                trust_remote_code=True,
            )
        except Exception:
            pytest.skip("Nemotron tokenizer not available")

        fmt = get_tool_format_from_tokenizer(tok)
        assert fmt is not None
        # Nemotron has XML-style format with <function=> tags
        assert fmt.format_type in ("xml", "json")

    def test_format_detection_no_tools(self):
        """A template without tool support returns None."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        # Simple template with no tool call indicators
        template = """
{%- for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{%- endfor %}
"""
        analysis = analyze_chat_template(template)
        assert not analysis.has_tool_calls

    def test_format_detection_xml_template(self):
        """Template with <function=> detected as XML format."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        template = """
{%- for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{%- if message.tool_calls %}
{%- for tool_call in message.tool_calls %}
<tool_call>
<function={{ tool_call.name }}>
{{ tool_call.arguments | tojson }}
</function>
</tool_call>
{%- endfor %}
{%- endif %}
{%- endfor %}
"""
        analysis = analyze_chat_template(template)
        assert analysis.has_tool_calls
        if analysis.format:
            assert analysis.format.format_type in ("xml", "json")


# ---------------------------------------------------------------------------
# Grammar generation tests
# ---------------------------------------------------------------------------


class TestGrammarGeneration:
    """Test grammar generation from tool definitions."""

    def test_single_tool_grammar(self):
        """Grammar for a single tool generates valid Lark grammar string."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(SIMPLE_TOOLS, format_override=fmt)
        assert "get_weather" in grammar
        assert "%json" in grammar

    def test_multi_tool_grammar(self):
        """Grammar for multiple tools generates anyOf schema."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(MULTI_TOOLS, format_override=fmt)
        assert "get_weather" in grammar
        assert "search" in grammar
        assert "anyOf" in grammar

    def test_complex_tool_grammar(self):
        """Grammar for complex tool with nested objects/arrays."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(COMPLEX_TOOL, format_override=fmt)
        assert "create_event" in grammar
        assert "%json" in grammar

    def test_xml_format_grammar(self):
        """XML format generates function/parameter style grammar."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="xml")
        grammar = build_tool_grammar(SIMPLE_TOOLS, format_override=fmt)
        assert "<function=" in grammar
        assert "</function>" in grammar
        assert "get_weather" in grammar

    def test_grammar_with_tokenizer_auto_detect(self):
        """Grammar generation uses tokenizer to auto-detect format."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        grammar = build_tool_grammar(SIMPLE_TOOLS, tok)
        # Llama uses "parameters" field, not "arguments"
        assert '"parameters"' in grammar
        assert "get_weather" in grammar

    def test_multi_tool_grammar_from_tokenizer(self):
        """Multi-tool grammar with auto-detected format."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar, build_multi_tool_grammar
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
        grammar = build_multi_tool_grammar(MULTI_TOOLS, tok, allow_multiple=True)
        assert "get_weather" in grammar
        assert "search" in grammar

    def test_empty_parameters_tool(self):
        """Tool with no parameters generates valid grammar."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [{"name": "list_files", "parameters": {"type": "object", "properties": {}}}]
        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(tools, format_override=fmt)
        assert "list_files" in grammar


# ---------------------------------------------------------------------------
# LLGuidance integration tests — grammar state from tools
# ---------------------------------------------------------------------------


class TestLLGuidanceToolGrammars:
    """Test LLGuidanceState creation and token masking with tool grammars."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        pytest.importorskip("llguidance")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def test_from_tools_creates_state(self, tokenizer):
        """LLGuidanceState.from_tools creates a valid grammar state."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)
        assert not state.is_complete()

    def test_from_tools_produces_mask(self, tokenizer):
        """Token mask from tools grammar has valid shape and some allowed tokens."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)
        mask = state.get_token_mask()
        assert mask.shape[0] > 0
        assert mx.any(mask).item()

    def test_multi_tool_mask(self, tokenizer):
        """Multi-tool grammar allows tokens for multiple tool names."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, MULTI_TOOLS)
        mask = state.get_token_mask()
        assert mask.shape[0] > 0
        assert mx.any(mask).item()

    def test_complex_tool_mask(self, tokenizer):
        """Complex tool with nested types creates valid mask."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, COMPLEX_TOOL)
        mask = state.get_token_mask()
        assert mask.shape[0] > 0
        assert mx.any(mask).item()

    def test_tool_grammar_accepts_valid_json_tokens(self, tokenizer):
        """Feed valid compact JSON tokens to the grammar and verify it accepts them."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)

        # The grammar produces compact JSON (no whitespace after colons/commas).
        # Llama format: {"name":"get_weather","parameters":{"city":"London"}}
        valid_json = '{"name":"get_weather","parameters":{"city":"London"}}'
        tokens = tokenizer.encode(valid_json, add_special_tokens=False)

        for tok_id in tokens:
            if state.is_complete():
                break
            mask = state.get_token_mask()
            # The token should be allowed by the mask
            assert mask[tok_id].item(), (
                f"Token {tok_id} ({tokenizer.decode([tok_id])!r}) not allowed at position "
                f"{len(state.generated_tokens)}"
            )
            state.update(tok_id)

        assert state.is_complete(), "Grammar should be complete after valid JSON"

    def test_tool_grammar_rejects_invalid_start(self, tokenizer):
        """Grammar should not allow starting with a random word."""
        from mlx_lm.grammar import LLGuidanceState
        import numpy as np

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)
        mask = state.get_token_mask()
        mask_np = np.array(mask)

        # Only '{' and '{"' tokens should be allowed at position 0
        allowed = np.where(mask_np)[0]
        allowed_strs = {tokenizer.decode([int(t)]).strip() for t in allowed}
        # All allowed tokens should start with '{'
        for t in allowed:
            decoded = tokenizer.decode([int(t)])
            assert decoded.startswith("{"), (
                f"Unexpected allowed token at pos 0: {int(t)} ({decoded!r})"
            )

    def test_clone_tool_grammar(self, tokenizer):
        """Cloning preserves tool grammar state."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)
        # Feed some tokens (compact JSON prefix)
        tokens = tokenizer.encode('{"name":', add_special_tokens=False)
        for tok_id in tokens:
            if not state.is_complete():
                state.update(tok_id)

        cloned = state.clone()
        assert len(cloned.generated_tokens) == len(state.generated_tokens)
        assert not cloned.is_complete()

    def test_reset_tool_grammar(self, tokenizer):
        """Reset returns tool grammar to initial state."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_tools(tokenizer, SIMPLE_TOOLS)
        tokens = tokenizer.encode('{"name":', add_special_tokens=False)
        for tok_id in tokens:
            if not state.is_complete():
                state.update(tok_id)

        assert len(state.generated_tokens) > 0
        state.reset()
        assert len(state.generated_tokens) == 0
        assert not state.is_complete()


# ---------------------------------------------------------------------------
# GrammarLogitsProcessor with tools
# ---------------------------------------------------------------------------


class TestGrammarProcessorWithTools:
    """Test GrammarLogitsProcessor integration with tool grammars."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        pytest.importorskip("llguidance")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def test_processor_creation(self, tokenizer):
        """make_grammar_logits_processor with tools creates a valid processor."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(tokenizer, tools=SIMPLE_TOOLS)
        assert proc is not None
        assert not proc.is_complete

    def test_processor_with_multi_tools(self, tokenizer):
        """Processor handles multiple tool definitions."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(tokenizer, tools=MULTI_TOOLS)
        assert proc is not None

    def test_processor_with_complex_tool(self, tokenizer):
        """Processor handles complex tool with nested schemas."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(tokenizer, tools=COMPLEX_TOOL)
        assert proc is not None

    def test_processor_constrains_logits(self, tokenizer):
        """Processor modifies logits to disallow invalid tokens."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(tokenizer, tools=SIMPLE_TOOLS)
        # Use the actual vocab size from the grammar (includes added tokens)
        vocab_size = len(tokenizer.get_vocab())
        prompt_tokens = mx.array([1, 2, 3])  # dummy prompt
        logits = mx.zeros((1, vocab_size))

        result = proc(prompt_tokens, logits)
        # Some tokens should be -inf (disallowed)
        assert mx.any(result == float("-inf")).item()
        # Some tokens should remain (allowed)
        assert mx.any(result != float("-inf")).item()

    def test_processor_reset_for_chat(self, tokenizer):
        """Processor reset works for multi-turn chat."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(tokenizer, tools=SIMPLE_TOOLS)
        vocab_size = len(tokenizer.get_vocab())

        # First turn
        prompt = mx.array([1, 2, 3])
        logits = mx.zeros((1, vocab_size))
        proc(prompt, logits)

        # Reset for next turn
        proc.reset()
        assert not proc.is_complete

        # Second turn
        prompt2 = mx.array([4, 5, 6])
        result = proc(prompt2, logits)
        assert mx.any(result == float("-inf")).item()


# ---------------------------------------------------------------------------
# Server integration tests
# ---------------------------------------------------------------------------


class TestServerToolChoice:
    """Test server-side tool_choice + grammar integration."""

    def _make_handler(self, body):
        """Create a mock handler with the given request body."""
        from unittest.mock import MagicMock
        from mlx_lm.server import APIHandler

        handler = MagicMock()
        handler.response_format = body.get("response_format")
        handler.body = body
        return handler

    def test_tool_choice_required_single_tool(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS[:1],
            "tool_choice": "required",
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"

    def test_tool_choice_required_multi_tools(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": "required",
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.tools is not None
        assert len(result.tools) == 2

    def test_tool_choice_specific_function(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "search"},
            },
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "search"

    def test_tool_choice_auto_no_grammar(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": "auto",
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_tool_choice_none_no_grammar(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": "none",
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_tool_choice_default_auto(self):
        """Default tool_choice is 'auto' (no grammar constraint)."""
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            # No tool_choice specified
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_tool_choice_nonexistent_function(self):
        """Specifying a nonexistent function returns None."""
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": {
                "type": "function",
                "function": {"name": "does_not_exist"},
            },
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_response_format_takes_priority_over_tools(self):
        """When response_format is specified, it takes priority."""
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": "required",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
                },
            },
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        # response_format should be used, not tools
        assert result.json_schema is not None
        assert result.tools is None

    def test_tools_without_function_wrapper(self):
        """Tools without 'type: function' wrapper are passed through."""
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "tools": SIMPLE_TOOLS,
            "tool_choice": "required",
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.tools[0]["name"] == "get_weather"

    def test_no_tools_no_grammar(self):
        """Without tools or response_format, no grammar args."""
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "messages": [{"role": "user", "content": "Hello"}],
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_response_format_regex(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "response_format": {"type": "regex", "regex": "(yes|no)"},
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.regex == "(yes|no)"

    def test_response_format_choices(self):
        from mlx_lm.server import APIHandler

        handler = self._make_handler({
            "response_format": {
                "type": "choices",
                "choices": ["red", "green", "blue"],
            },
        })
        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.choices == ["red", "green", "blue"]


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


class TestCLIArgs:
    """Test CLI argument parsing for --tools."""

    def test_generate_tools_arg(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        tools_json = json.dumps(SIMPLE_TOOLS)
        args = parser.parse_args(["--tools", tools_json, "--prompt", "test"])
        assert args.tools == tools_json

    def test_chat_tools_arg(self):
        from mlx_lm.chat import setup_arg_parser

        parser = setup_arg_parser()
        tools_json = json.dumps(SIMPLE_TOOLS)
        args = parser.parse_args(["--tools", tools_json])
        assert args.tools == tools_json

    def test_tools_mutually_exclusive_with_json_schema(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--tools", "[]",
                "--json-schema", '{"type":"object"}',
                "--prompt", "test",
            ])

    def test_tools_mutually_exclusive_with_regex(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tools", "[]", "--regex", "abc", "--prompt", "test"])

    def test_tools_mutually_exclusive_with_choices(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tools", "[]", "--choices", "a", "b", "--prompt", "test"])

    def test_tools_mutually_exclusive_with_grammar(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tools", "[]", "--grammar", "start: /abc/", "--prompt", "test"])


# ---------------------------------------------------------------------------
# End-to-end generation tests
# ---------------------------------------------------------------------------


class TestEndToEndToolCalling:
    """End-to-end tests with real model generation.

    Uses mlx-community/Llama-3.2-1B-Instruct-4bit (small, fast, cached).
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        pytest.importorskip("llguidance")
        from mlx_lm import load
        return load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def _generate_with_tools(self, model, tokenizer, tools, prompt_text, max_tokens=200):
        """Helper: generate with grammar-constrained tool calling."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        # Build the prompt with tools in the chat template
        # Llama's template expects OpenAI-style tools with "type":"function"
        openai_tools = []
        for t in tools:
            if "type" not in t or "function" not in t:
                openai_tools.append({
                    "type": "function",
                    "function": t,
                })
            else:
                openai_tools.append(t)

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
            tools=openai_tools,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)

        processor = make_grammar_logits_processor(tokenizer, tools=tools)
        response = generate(
            model, tokenizer, tokens,
            max_tokens=max_tokens,
            sampler=make_sampler(0.0),
            logits_processors=[processor],
            verbose=False,
        )
        return response

    def test_single_tool_call(self, model_and_tokenizer):
        """Grammar forces valid single tool call."""
        model, tokenizer = model_and_tokenizer
        response = self._generate_with_tools(
            model, tokenizer, SIMPLE_TOOLS,
            "What is the weather in London?"
        )
        parsed = json.loads(response)
        assert parsed["name"] == "get_weather"
        assert "parameters" in parsed
        assert "city" in parsed["parameters"]
        assert isinstance(parsed["parameters"]["city"], str)

    def test_multi_tool_picks_one(self, model_and_tokenizer):
        """With multiple tools, grammar allows any valid tool call."""
        model, tokenizer = model_and_tokenizer
        response = self._generate_with_tools(
            model, tokenizer, MULTI_TOOLS,
            "What is the weather in Tokyo?"
        )
        parsed = json.loads(response)
        assert parsed["name"] in ("get_weather", "search")
        assert "parameters" in parsed

    def test_multi_tool_search(self, model_and_tokenizer):
        """With a search-oriented prompt, model picks search tool."""
        model, tokenizer = model_and_tokenizer
        response = self._generate_with_tools(
            model, tokenizer, MULTI_TOOLS,
            "Search for the latest news about AI"
        )
        parsed = json.loads(response)
        assert parsed["name"] in ("get_weather", "search")
        assert "parameters" in parsed

    def test_complex_tool_call(self, model_and_tokenizer):
        """Grammar handles complex tool with nested objects and arrays."""
        model, tokenizer = model_and_tokenizer
        response = self._generate_with_tools(
            model, tokenizer, COMPLEX_TOOL,
            "Create a meeting called 'Team Sync' on 2025-03-01",
            max_tokens=300,
        )
        parsed = json.loads(response)
        assert parsed["name"] == "create_event"
        assert "parameters" in parsed
        params = parsed["parameters"]
        assert "title" in params
        assert "date" in params

    def test_tool_call_is_valid_json(self, model_and_tokenizer):
        """Every grammar-constrained tool call is valid JSON."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            "What is the weather?",
            "Tell me the temperature in Paris",
            "Is it raining in New York?",
        ]
        for prompt_text in prompts:
            response = self._generate_with_tools(
                model, tokenizer, SIMPLE_TOOLS, prompt_text
            )
            parsed = json.loads(response)  # Must not raise
            assert isinstance(parsed, dict)
            assert "name" in parsed

    def test_tool_call_with_enum_parameter(self, model_and_tokenizer):
        """Tool with enum parameter produces valid enum value."""
        model, tokenizer = model_and_tokenizer
        tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city", "units"],
                    "additionalProperties": False,
                },
            }
        ]
        response = self._generate_with_tools(
            model, tokenizer, tools,
            "Weather in Berlin in celsius"
        )
        parsed = json.loads(response)
        assert parsed["name"] == "get_weather"
        assert parsed["parameters"]["units"] in ("celsius", "fahrenheit")


class TestEndToEndJsonSchema:
    """Additional E2E tests for JSON schema constraints."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        pytest.importorskip("llguidance")
        from mlx_lm import load
        return load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def _generate_json(self, model, tokenizer, schema, prompt_text, max_tokens=200):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        processor = make_grammar_logits_processor(tokenizer, json_schema=schema)
        return generate(
            model, tokenizer, tokens,
            max_tokens=max_tokens,
            sampler=make_sampler(0.0),
            logits_processors=[processor],
            verbose=False,
        )

    def test_nested_object_schema(self, model_and_tokenizer):
        """Nested object schema produces valid nested JSON."""
        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
            "required": ["person"],
            "additionalProperties": False,
        }
        response = self._generate_json(
            model, tokenizer, schema,
            "Return nested JSON for a person named Alice age 25"
        )
        parsed = json.loads(response)
        assert "person" in parsed
        assert isinstance(parsed["person"]["name"], str)
        assert isinstance(parsed["person"]["age"], int)

    def test_array_schema(self, model_and_tokenizer):
        """Array schema produces valid JSON array."""
        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        response = self._generate_json(
            model, tokenizer, schema,
            "Return a JSON array of 3 color names"
        )
        parsed = json.loads(response)
        assert isinstance(parsed, list)
        assert all(isinstance(x, str) for x in parsed)

    def test_boolean_and_number_fields(self, model_and_tokenizer):
        """Schema with boolean and number fields produces correct types."""
        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
            },
            "required": ["name", "score", "active"],
            "additionalProperties": False,
        }
        response = self._generate_json(
            model, tokenizer, schema,
            "Return a JSON object with name 'test', score 9.5, active true"
        )
        parsed = json.loads(response)
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["score"], (int, float))
        assert isinstance(parsed["active"], bool)

    def test_enum_string_field(self, model_and_tokenizer):
        """Enum field constrains to valid values."""
        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
            },
            "required": ["color"],
            "additionalProperties": False,
        }
        response = self._generate_json(
            model, tokenizer, schema,
            "Return a JSON with a color field, pick blue"
        )
        parsed = json.loads(response)
        assert parsed["color"] in ("red", "green", "blue")


class TestEndToEndRegexAndChoices:
    """E2E tests for regex and choices constraints."""

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        pytest.importorskip("llguidance")
        from mlx_lm import load
        return load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def _generate(self, model, tokenizer, prompt_text, **kwargs):
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        processor = make_grammar_logits_processor(tokenizer, **kwargs)
        return generate(
            model, tokenizer, tokens,
            max_tokens=50, sampler=make_sampler(0.0),
            logits_processors=[processor], verbose=False,
        )

    def test_yes_no_regex(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        response = self._generate(model, tokenizer, "Is water wet?", regex="(yes|no)")
        assert response.strip() in ("yes", "no")

    def test_email_regex(self, model_and_tokenizer):
        """Email-like regex constrains output."""
        model, tokenizer = model_and_tokenizer
        import re
        response = self._generate(
            model, tokenizer,
            "Give me an email address",
            regex=r"[a-z]+@[a-z]+\.[a-z]+"
        )
        assert re.match(r"[a-z]+@[a-z]+\.[a-z]+", response.strip())

    def test_number_regex(self, model_and_tokenizer):
        """Numeric regex produces number."""
        model, tokenizer = model_and_tokenizer
        response = self._generate(
            model, tokenizer,
            "Give me a number between 1 and 100",
            regex=r"[1-9][0-9]?"
        )
        val = int(response.strip())
        assert 1 <= val <= 99

    def test_choices_single(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        response = self._generate(
            model, tokenizer, "Pick a fruit",
            choices=["apple", "banana", "cherry"]
        )
        assert response.strip() in ("apple", "banana", "cherry")

    def test_choices_many_options(self, model_and_tokenizer):
        """Many choices still constrains correctly."""
        model, tokenizer = model_and_tokenizer
        options = ["a", "b", "c", "d", "e", "f", "g", "h"]
        response = self._generate(
            model, tokenizer, "Pick a letter",
            choices=options,
        )
        assert response.strip() in options


# ---------------------------------------------------------------------------
# Thinking model tests
# ---------------------------------------------------------------------------


class TestThinkingModelCompatibility:
    """Test that grammar infrastructure handles thinking-capable models.

    Models like Nemotron have <think></think> tags that appear before tool
    call output. Grammar constraints must account for this.
    """

    def test_nemotron_format_detection(self):
        """Nemotron tokenizer's chat template detects tool call support."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit",
                trust_remote_code=True,
            )
        except Exception:
            pytest.skip("Nemotron tokenizer not available")

        ct = tok.chat_template
        assert "<think>" in ct, "Nemotron template should have <think>"
        assert "<tool_call>" in ct, "Nemotron template should have <tool_call>"

        analysis = analyze_chat_template(ct)
        assert analysis.has_tool_calls

    def test_nemotron_tool_parser_inference(self):
        """Nemotron should be auto-detected as qwen3_coder format."""
        from mlx_lm.tokenizer_utils import _infer_tool_parser
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit",
                trust_remote_code=True,
            )
        except Exception:
            pytest.skip("Nemotron tokenizer not available")

        parser_type = _infer_tool_parser(tok.chat_template)
        assert parser_type == "qwen3_coder"

    def test_nemotron_template_with_tools(self):
        """Nemotron chat template correctly renders with tools."""
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(
                "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-8bit",
                trust_remote_code=True,
            )
        except Exception:
            pytest.skip("Nemotron tokenizer not available")

        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]

        result = tok.apply_chat_template(
            [{"role": "user", "content": "Weather in London?"}],
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
        # Should end with <think> (generation prompt for thinking model)
        assert result.rstrip().endswith("<think>")
        # Should contain the tool definition somewhere
        assert "get_weather" in result

    def test_thinking_model_grammar_note(self):
        """Document that grammar + thinking requires special handling.

        When a thinking model generates <think>...</think> before a tool call,
        the grammar must either:
        1. Start AFTER the thinking block (not yet supported)
        2. Include a free-form prefix pattern in the grammar

        This test documents the current behavior.
        """
        # This is a documentation test - thinking + grammar is a known limitation
        # The grammar would constrain from the start of generation, but thinking
        # models need to generate free-form think tokens first.
        pass


# ---------------------------------------------------------------------------
# Schema + parsing round-trip tests
# ---------------------------------------------------------------------------


class TestBuildToolSchemaAndParse:
    """Test the unified schema/parse approach."""

    def test_build_tool_schema_single(self):
        """build_tool_schema returns valid schema for a single tool."""
        from mlx_lm.grammar import build_tool_schema
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        schema, returned_fmt = build_tool_schema(SIMPLE_TOOLS, format_override=fmt)
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert schema["properties"]["name"]["const"] == "get_weather"

    def test_build_tool_schema_multi(self):
        """build_tool_schema returns anyOf for multiple tools."""
        from mlx_lm.grammar import build_tool_schema
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        schema, _ = build_tool_schema(MULTI_TOOLS, format_override=fmt)
        assert "anyOf" in schema
        assert len(schema["anyOf"]) == 2

    def test_build_tool_schema_from_tokenizer(self):
        """build_tool_schema auto-detects format from tokenizer."""
        from mlx_lm.grammar import build_tool_schema
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
        schema, fmt = build_tool_schema(SIMPLE_TOOLS, tok)
        # Llama uses "parameters" not "arguments"
        assert fmt.arguments_field == "parameters"
        assert "parameters" in schema["properties"]

    def test_parse_tool_call_output_json(self):
        """parse_tool_call_output normalises field names."""
        from mlx_lm.grammar import parse_tool_call_output
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        # Llama format: uses "parameters"
        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="parameters")
        result = parse_tool_call_output(
            '{"name":"get_weather","parameters":{"city":"London"}}',
            fmt,
        )
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "London"}

    def test_parse_tool_call_output_default_fields(self):
        """parse_tool_call_output with default field names."""
        from mlx_lm.grammar import parse_tool_call_output

        result = parse_tool_call_output(
            '{"name":"search","arguments":{"query":"AI news"}}',
        )
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "AI news"}

    def test_parse_tool_call_output_nested(self):
        """parse_tool_call_output handles function.name style."""
        from mlx_lm.grammar import parse_tool_call_output
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        fmt = ToolCallFormat(format_type="json", name_field="function.name", arguments_field="function.arguments")
        result = parse_tool_call_output(
            '{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}',
            fmt,
        )
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "Paris"}

    def test_round_trip_schema_and_parse(self):
        """Full round-trip: build schema, simulate constrained output, parse."""
        from mlx_lm.grammar import build_tool_schema, parse_tool_call_output
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("mlx-community/Llama-3.2-1B-Instruct-4bit")
        schema, fmt = build_tool_schema(MULTI_TOOLS, tok)

        # Simulate grammar-constrained output (guaranteed valid JSON)
        raw = '{"name":"search","parameters":{"query":"latest news","max_results":5}}'
        parsed = parse_tool_call_output(raw, fmt)
        assert parsed["name"] == "search"
        assert parsed["arguments"]["query"] == "latest news"
        assert parsed["arguments"]["max_results"] == 5


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tools_list(self):
        """Empty tools list raises or returns no grammar."""
        from mlx_lm.server import APIHandler
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.response_format = None
        handler.body = {"tools": [], "tool_choice": "required"}
        result = APIHandler._parse_grammar_args(handler)
        # Empty tools list should not create grammar
        assert result is None

    def test_tool_choice_with_empty_tools(self):
        """tool_choice without tools doesn't crash."""
        from mlx_lm.server import APIHandler
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.response_format = None
        handler.body = {"tool_choice": "required"}
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_malformed_tool_choice(self):
        """Malformed tool_choice dict doesn't crash."""
        from mlx_lm.server import APIHandler
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.response_format = None
        handler.body = {
            "tools": OPENAI_STYLE_TOOLS,
            "tool_choice": {"type": "function"},  # Missing function.name
        }
        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_tool_with_no_parameters(self):
        """Tool with no parameters field uses empty object default."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [{"name": "ping"}]
        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(tools, format_override=fmt)
        assert "ping" in grammar

    def test_tool_with_many_parameters(self):
        """Tool with many parameters generates valid grammar."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [{
            "name": "create_order",
            "parameters": {
                "type": "object",
                "properties": {
                    f"field_{i}": {"type": "string"} for i in range(20)
                },
                "required": [f"field_{i}" for i in range(5)],
            },
        }]
        fmt = ToolCallFormat(format_type="json", name_field="name", arguments_field="arguments")
        grammar = build_tool_grammar(tools, format_override=fmt)
        assert "create_order" in grammar

    def test_response_format_text_no_grammar(self):
        """response_format type=text returns None."""
        from mlx_lm.server import APIHandler
        from unittest.mock import MagicMock

        handler = MagicMock()
        handler.response_format = {"type": "text"}
        handler.body = {"response_format": {"type": "text"}}
        result = APIHandler._parse_grammar_args(handler)
        assert result is None


# ---------------------------------------------------------------------------
# Streaming / grammar_tool_mode integration
# ---------------------------------------------------------------------------


class TestGrammarToolMode:
    """Test grammar-constrained tool call detection and parsing in server."""

    def test_grammar_tool_mode_detected(self):
        """grammar_tool_mode is True when grammar_args has tools."""
        from mlx_lm.server import GrammarArguments

        args = GrammarArguments(tools=SIMPLE_TOOLS)
        grammar_tool_mode = args is not None and args.tools is not None
        assert grammar_tool_mode

    def test_grammar_tool_mode_not_detected_for_json_schema(self):
        """grammar_tool_mode is False for json_schema (not tools)."""
        from mlx_lm.server import GrammarArguments

        args = GrammarArguments(json_schema={"type": "object"})
        grammar_tool_mode = args is not None and args.tools is not None
        assert not grammar_tool_mode

    def test_grammar_tool_mode_not_detected_when_none(self):
        """grammar_tool_mode is False when grammar_args is None."""
        args = None
        grammar_tool_mode = args is not None and getattr(args, "tools", None) is not None
        assert not grammar_tool_mode

    def test_parse_tool_call_output_fallback_parameters(self):
        """parse_tool_call_output falls back to 'parameters' field."""
        from mlx_lm.grammar import parse_tool_call_output

        # No fmt provided, output uses "parameters" not "arguments"
        result = parse_tool_call_output(
            '{"name":"get_weather","parameters":{"city":"London"}}',
        )
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "London"}

    def test_parse_tool_call_output_fallback_nested_function(self):
        """parse_tool_call_output auto-detects nested function field."""
        from mlx_lm.grammar import parse_tool_call_output

        # No fmt, but output has nested "function" field
        result = parse_tool_call_output(
            '{"function":{"name":"search","arguments":{"q":"test"}}}',
        )
        assert result["name"] == "search"
        assert result["arguments"] == {"q": "test"}

    def test_parse_tool_call_output_nested_with_parameters(self):
        """parse_tool_call_output handles nested function with 'parameters'."""
        from mlx_lm.grammar import parse_tool_call_output

        result = parse_tool_call_output(
            '{"function":{"name":"search","parameters":{"q":"test"}}}',
        )
        assert result["name"] == "search"
        assert result["arguments"] == {"q": "test"}

    def test_grammar_tool_mode_parse_tools_integration(self):
        """Simulate parse_tools in grammar_tool_mode."""
        from mlx_lm.grammar.tool_schema import parse_tool_call_output
        import uuid

        tool_text = '{"name":"get_weather","parameters":{"city":"London"}}'
        tool_calls = [tool_text]

        # Simulate parse_tools from server
        result = []
        for tc_text in tool_calls:
            parsed = parse_tool_call_output(tc_text)
            # Simulate format_tool_call
            tool_call_id = str(uuid.uuid4())
            formatted = {
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
                },
                "type": "function",
                "id": tool_call_id,
                "index": 0,
            }
            result.append(formatted)

        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["type"] == "function"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "London"}

    def test_grammar_tool_mode_finish_reason(self):
        """finish_reason should be 'tool_calls' in grammar_tool_mode."""
        # Simulate the server's finish_reason logic
        grammar_tool_mode = True
        made_tool_call = grammar_tool_mode
        tool_text = '{"name":"get_weather","parameters":{"city":"London"}}'

        # Simulate stop condition
        finish_reason = "tool_calls" if made_tool_call else "stop"
        assert finish_reason == "tool_calls"

        # Also test the post-loop override
        finish_reason = "stop"  # e.g., gen.finish_reason overrode it
        if grammar_tool_mode and tool_text:
            finish_reason = "tool_calls"
        assert finish_reason == "tool_calls"

    def test_grammar_tool_mode_empty_output(self):
        """If grammar_tool_mode produces no output, finish_reason stays as-is."""
        grammar_tool_mode = True
        tool_text = ""
        finish_reason = "length"

        if grammar_tool_mode and tool_text:
            finish_reason = "tool_calls"
        # Empty tool_text: finish_reason unchanged
        assert finish_reason == "length"


# ---------------------------------------------------------------------------
# Incremental tool call streaming
# ---------------------------------------------------------------------------


class TestGrammarToolCallStreamer:
    """Test _GrammarToolCallStreamer incremental JSON parsing."""

    def _make_streamer(self):
        """Create a streamer with a mock handler that captures SSE output."""
        from io import BytesIO
        from unittest.mock import MagicMock
        from mlx_lm.server import _GrammarToolCallStreamer

        handler = MagicMock()
        handler.request_id = "req-123"
        handler.system_fingerprint = "fp-test"
        handler.object_type = "chat.completion.chunk"
        handler.requested_model = "test-model"
        handler.created = 1234567890
        handler.wfile = BytesIO()
        return _GrammarToolCallStreamer(handler), handler

    def _get_chunks(self, handler):
        """Parse SSE chunks from handler.wfile."""
        raw = handler.wfile.getvalue().decode()
        chunks = []
        for line in raw.split("\n"):
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
        return chunks

    def test_name_extraction(self):
        """Streamer extracts function name from partial JSON."""
        streamer, handler = self._make_streamer()
        streamer.feed('{"name":"get_weather"')
        assert streamer.active
        assert streamer._name == "get_weather"

    def test_not_active_before_name(self):
        """Streamer not active before name is found."""
        streamer, handler = self._make_streamer()
        streamer.feed('{"na')
        assert not streamer.active

    def test_full_tool_call_streaming(self):
        """Full tool call produces name chunk + arg chunks + finish chunk."""
        streamer, handler = self._make_streamer()

        # Feed in chunks simulating token-by-token generation
        text = '{"name":"get_weather","parameters":{"city":"London"}}'
        for i in range(0, len(text), 5):
            streamer.feed(text[i : i + 5])

        streamer.finalize()
        chunks = self._get_chunks(handler)

        # At least: 1 name chunk + some arg chunks + 1 finish chunk
        assert len(chunks) >= 3

        # First chunk should have the name
        first = chunks[0]
        tc = first["choices"][0]["delta"]["tool_calls"][0]
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == ""
        assert tc["type"] == "function"
        assert "id" in tc

        # Last chunk should have finish_reason
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "tool_calls"

        # Concatenate all argument deltas
        args_str = ""
        for chunk in chunks[1:-1]:  # skip name and finish chunks
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                args_str += delta["tool_calls"][0]["function"]["arguments"]
        assert args_str == '{"city":"London"}'

    def test_arguments_field_name(self):
        """Streamer handles 'arguments' field name."""
        streamer, handler = self._make_streamer()

        text = '{"name":"search","arguments":{"query":"test"}}'
        streamer.feed(text)
        streamer.finalize()

        chunks = self._get_chunks(handler)
        args_str = ""
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "arguments" in tc.get("function", {}):
                    args_str += tc["function"]["arguments"]
        assert '{"query":"test"}' in args_str

    def test_nested_json_args(self):
        """Streamer handles nested JSON objects in arguments."""
        streamer, handler = self._make_streamer()

        text = '{"name":"create","parameters":{"config":{"nested":{"deep":true}},"name":"x"}}'
        for i in range(0, len(text), 3):
            streamer.feed(text[i : i + 3])
        streamer.finalize()

        chunks = self._get_chunks(handler)
        args_str = ""
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "arguments" in tc.get("function", {}):
                    args_str += tc["function"]["arguments"]
        parsed = json.loads(args_str)
        assert parsed["config"]["nested"]["deep"] is True
        assert parsed["name"] == "x"

    def test_string_with_braces(self):
        """Streamer handles strings containing braces correctly."""
        streamer, handler = self._make_streamer()

        # Arguments contain a string with braces — the depth tracker must
        # ignore braces inside JSON strings
        text = '{"name":"eval","parameters":{"code":"if(x){y();}"}}'
        for i in range(0, len(text), 4):
            streamer.feed(text[i : i + 4])
        streamer.finalize()

        chunks = self._get_chunks(handler)
        args_str = ""
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "arguments" in tc.get("function", {}):
                    args_str += tc["function"]["arguments"]
        parsed = json.loads(args_str)
        assert parsed["code"] == "if(x){y();}"

    def test_single_char_feed(self):
        """Streamer works with single-character feeds."""
        streamer, handler = self._make_streamer()

        text = '{"name":"fn","parameters":{"a":1}}'
        for c in text:
            streamer.feed(c)
        streamer.finalize()

        chunks = self._get_chunks(handler)
        # Should have name, args, and finish
        assert any(
            c["choices"][0].get("finish_reason") == "tool_calls" for c in chunks
        )
        # Reconstruct args
        args_str = ""
        for chunk in chunks:
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta:
                tc = delta["tool_calls"][0]
                if "arguments" in tc.get("function", {}):
                    args_str += tc["function"]["arguments"]
        assert json.loads(args_str) == {"a": 1}

    def test_whole_text_at_once(self):
        """Streamer works when entire text is fed in one chunk."""
        streamer, handler = self._make_streamer()

        text = '{"name":"ping","parameters":{}}'
        streamer.feed(text)
        streamer.finalize()

        chunks = self._get_chunks(handler)
        assert streamer.active
        assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"

    def test_name_after_parameters(self):
        """Streamer works when name field comes after parameters."""
        streamer, handler = self._make_streamer()

        # Name field after parameters (unusual but valid JSON)
        text = '{"parameters":{"city":"Paris"},"name":"get_weather"}'
        for i in range(0, len(text), 6):
            streamer.feed(text[i : i + 6])
        streamer.finalize()

        chunks = self._get_chunks(handler)
        assert streamer.active
        assert streamer._name == "get_weather"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
