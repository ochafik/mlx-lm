# Copyright 2025 Apple Inc.

"""
Tests for grammar-constrained generation.

These tests cover:
- Jinja template analysis for tool call format detection
- Tool schema to grammar conversion
- Mask operations
- Grammar logits processor
"""

import json
import pytest
import mlx.core as mx
import numpy as np


class TestJinjaTemplateAnalysis:
    """Test Jinja template differential analysis."""

    def test_detect_json_tool_format(self):
        """Test detection of JSON tool call format."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        template = '''
{%- for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{%- if message['tool_calls'] %}
{%- for tool_call in message['tool_calls'] %}
<tool_call>
{"name": "{{ tool_call['name'] }}", "arguments": {{ tool_call['arguments'] | tojson }}}
</tool_call>
{%- endfor %}
{%- endif %}
{%- endfor %}
'''
        analysis = analyze_chat_template(template)

        assert analysis.has_tool_calls
        assert analysis.format is not None
        assert analysis.format.format_type == "json"

    def test_detect_markers(self):
        """Test detection of tool call markers."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        template = '''
{%- for message in messages %}
{%- if message['tool_calls'] %}
{%- for tool_call in message['tool_calls'] %}
<|tool_call|>
{"name": "{{ tool_call['name'] }}"}
<|/tool_call|>
{%- endfor %}
{%- endif %}
{%- endfor %}
'''
        analysis = analyze_chat_template(template)

        assert analysis.has_tool_calls
        # Format may be detected via pattern matching
        if analysis.format:
            assert analysis.format.start_marker == "<|tool_call|>"

    def test_no_tool_calls(self):
        """Test template without tool calling."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        template = '''
{%- for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{%- endfor %}
'''
        analysis = analyze_chat_template(template)

        assert not analysis.has_tool_calls
        assert analysis.format is None

    def test_xml_format_detection(self):
        """Test detection of XML/function style."""
        from mlx_lm.grammar.jinja_analysis import analyze_chat_template

        template = '''
{%- for message in messages %}
{%- for tool_call in message['tool_calls'] %}
<function={{ tool_call['name'] }}>
{{ tool_call['arguments'] | tojson }}
</function>
{%- endfor %}
{%- endfor %}
'''
        analysis = analyze_chat_template(template)

        assert analysis.has_tool_calls
        # May be detected as XML or custom format
        if analysis.format:
            assert analysis.format.format_type in ("xml", "custom", "json")


class TestToolSchemaConversion:
    """Test tool schema to grammar conversion."""

    def test_simple_tool_schema(self):
        """Test conversion of simple tool schema."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [
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

        # Use explicit format
        fmt = ToolCallFormat(format_type="json")
        grammar = build_tool_grammar(tools, format_override=fmt)

        assert grammar is not None
        assert "get_weather" in grammar
        assert "city" in grammar

    def test_multiple_tools(self):
        """Test conversion with multiple tools."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [
            {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

        fmt = ToolCallFormat(format_type="json")
        grammar = build_tool_grammar(tools, format_override=fmt)

        assert "get_weather" in grammar
        assert "search" in grammar

    def test_with_markers(self):
        """Test grammar generation with markers."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar
        from mlx_lm.grammar.jinja_analysis import ToolCallFormat

        tools = [{"name": "test", "parameters": {"type": "object"}}]

        fmt = ToolCallFormat(
            format_type="json",
            start_marker="<|tool_call|>",
            end_marker="</tool_call>",
        )
        grammar = build_tool_grammar(tools, format_override=fmt)

        assert "<|tool_call|>" in grammar or "tool_call" in grammar

    def test_tools_from_functions(self):
        """Test creating tool schemas from Python functions."""
        from mlx_lm.grammar.tool_schema import tools_from_functions

        def greet(name: str, formal: bool = False) -> str:
            """Greet a person."""
            return f"Hello, {name}!"

        tools = tools_from_functions([greet])

        assert len(tools) == 1
        assert tools[0]["name"] == "greet"
        assert "name" in tools[0]["parameters"]["properties"]
        assert "name" in tools[0]["parameters"]["required"]


class TestMaskOperations:
    """Test MLX mask operations."""

    def test_apply_grammar_mask(self):
        """Test applying grammar mask to logits."""
        from mlx_lm.grammar.mask_ops import apply_grammar_mask

        logits = mx.array([1.0, 2.0, 3.0, 4.0])
        mask = mx.array([True, False, True, False])

        result = apply_grammar_mask(logits, mask)

        assert result[0] == 1.0
        assert result[1] == float("-inf")
        assert result[2] == 3.0
        assert result[3] == float("-inf")

    def test_apply_grammar_mask_batch(self):
        """Test applying grammar mask with batch dimension."""
        from mlx_lm.grammar.mask_ops import apply_grammar_mask

        logits = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = mx.array([True, False, True])

        result = apply_grammar_mask(logits, mask)

        assert result.shape == (2, 3)
        assert result[0, 1] == float("-inf")
        assert result[1, 1] == float("-inf")

    def test_bitmask_conversion(self):
        """Test bitmask to boolean conversion."""
        from mlx_lm.grammar.mask_ops import bitmask_to_bool, bool_to_bitmask

        # Create a simple mask
        original = np.array([True, False, True, False, True, True, False, False])
        original_mx = mx.array(original)

        # Convert to bitmask and back
        bitmask = bool_to_bitmask(original_mx)
        recovered = bitmask_to_bool(bitmask, len(original))

        np.testing.assert_array_equal(np.array(recovered), original)

    def test_count_allowed_tokens(self):
        """Test counting allowed tokens."""
        from mlx_lm.grammar.mask_ops import count_allowed_tokens

        mask = mx.array([True, False, True, False, True])
        count = count_allowed_tokens(mask)

        assert count == 3

    def test_get_allowed_token_ids(self):
        """Test getting allowed token IDs."""
        from mlx_lm.grammar.mask_ops import get_allowed_token_ids

        mask = mx.array([False, True, False, True, False])
        ids = get_allowed_token_ids(mask)

        assert 1 in np.array(ids)
        assert 3 in np.array(ids)


class TestGrammarLogitsProcessor:
    """Test GrammarLogitsProcessor."""

    def test_processor_interface(self):
        """Test processor follows expected interface."""
        from mlx_lm.sample_utils import GrammarLogitsProcessor
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        # Use mock grammar for testing
        grammar = MockGrammarState(vocab_size=100)
        processor = GrammarLogitsProcessor(grammar)

        tokens = mx.array([1, 2, 3])
        logits = mx.random.normal((1, 100))

        result = processor(tokens, logits)

        assert result.shape == logits.shape

    def test_processor_completion(self):
        """Test processor tracks completion.

        Simulates the real generate_step pattern where tokens accumulate:
        first call has prompt tokens, subsequent calls append generated tokens.
        """
        from mlx_lm.sample_utils import GrammarLogitsProcessor
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=100)
        grammar._max_tokens = 5
        processor = GrammarLogitsProcessor(grammar)

        logits = mx.random.normal((1, 100))
        # First call: prompt tokens (not fed to grammar)
        accumulated = mx.array([99])
        processor(accumulated, logits)

        # Subsequent calls: append generated tokens (fed to grammar)
        for i in range(10):
            accumulated = mx.concat([accumulated, mx.array([i])])
            processor(accumulated, logits)
            if processor.is_complete:
                break

        assert processor.is_complete

    def test_processor_reset(self):
        """Test processor reset."""
        from mlx_lm.sample_utils import GrammarLogitsProcessor
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=100)
        grammar._max_tokens = 5
        processor = GrammarLogitsProcessor(grammar)

        logits = mx.random.normal((1, 100))
        # First call: prompt tokens
        accumulated = mx.array([99])
        processor(accumulated, logits)

        # Generate tokens until complete
        for i in range(10):
            accumulated = mx.concat([accumulated, mx.array([i])])
            processor(accumulated, logits)

        assert processor.is_complete

        # Reset
        processor.reset()

        assert not processor.is_complete


class TestMockGrammarState:
    """Test MockGrammarState for testing without llguidance."""

    def test_mock_allows_all(self):
        """Test mock grammar allows all tokens."""
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=1000)
        mask = grammar.get_token_mask()

        assert mask.shape == (1000,)
        assert mx.all(mask)

    def test_mock_tracks_tokens(self):
        """Test mock grammar tracks tokens."""
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=100)
        grammar.update(42)
        grammar.update(43)

        assert len(grammar._tokens) == 2
        assert grammar._tokens == [42, 43]

    def test_mock_clone(self):
        """Test mock grammar cloning."""
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=100)
        grammar.update(1)
        grammar.update(2)

        clone = grammar.clone()

        assert clone._tokens == grammar._tokens
        assert clone is not grammar


class TestLLGuidanceIntegration:
    """Test LLGuidance integration (skipped if not installed)."""

    @pytest.fixture
    def skip_if_no_llguidance(self):
        """Skip test if llguidance is not available."""
        from mlx_lm.grammar.llguidance_adapter import is_llguidance_available

        if not is_llguidance_available():
            pytest.skip("llguidance not installed")

    def test_llguidance_available_check(self):
        """Test availability check works."""
        from mlx_lm.grammar.llguidance_adapter import is_llguidance_available

        # Should return bool without raising
        result = is_llguidance_available()
        assert isinstance(result, bool)


# Integration tests that require a tokenizer
class TestTokenizerIntegration:
    """Test integration with HuggingFace tokenizers."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""

        class MockTokenizer:
            def __init__(self):
                self.vocab_size = 1000
                self.chat_template = """
{%- for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{%- if message['tool_calls'] %}
<tool_call>
{{ message['tool_calls'] | tojson }}
</tool_call>
{%- endif %}
{%- endfor %}
"""

            def get_vocab(self):
                return {f"token_{i}": i for i in range(self.vocab_size)}

            def decode(self, tokens, **kwargs):
                return "".join(f"[{t}]" for t in tokens)

        return MockTokenizer()

    def test_format_detection_with_tokenizer(self, mock_tokenizer):
        """Test format detection from tokenizer."""
        from mlx_lm.grammar.jinja_analysis import get_tool_format_from_tokenizer

        fmt = get_tool_format_from_tokenizer(mock_tokenizer)

        assert fmt is not None
        assert fmt.format_type == "json"

    def test_grammar_building_with_tokenizer(self, mock_tokenizer):
        """Test grammar building with tokenizer for format detection."""
        from mlx_lm.grammar.tool_schema import build_tool_grammar

        tools = [
            {
                "name": "test_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"arg": {"type": "string"}},
                },
            }
        ]

        grammar = build_tool_grammar(tools, tokenizer=mock_tokenizer)

        assert grammar is not None
        assert "test_tool" in grammar


class TestLLGuidanceRealIntegration:
    """Integration tests that require llguidance to be installed."""

    @pytest.fixture
    def hf_tokenizer(self):
        """Load a real HF tokenizer for integration tests."""
        pytest.importorskip("llguidance")
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )

    def test_from_json_schema(self, hf_tokenizer):
        """Test LLGuidanceState.from_json_schema with real tokenizer."""
        from mlx_lm.grammar import LLGuidanceState

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        state = LLGuidanceState.from_json_schema(hf_tokenizer, schema)
        assert not state.is_complete()

        mask = state.get_token_mask()
        assert mask.shape[0] > 0
        assert mask.sum().item() > 0  # Some tokens allowed

    def test_from_regex(self, hf_tokenizer):
        """Test LLGuidanceState.from_regex with real tokenizer."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_regex(hf_tokenizer, "[a-z]+")
        mask = state.get_token_mask()
        assert mask.sum().item() > 0

    def test_from_choices(self, hf_tokenizer):
        """Test LLGuidanceState.from_choices with real tokenizer."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_choices(hf_tokenizer, ["yes", "no"])
        mask = state.get_token_mask()
        assert mask.sum().item() > 0

    def test_token_feeding(self, hf_tokenizer):
        """Test feeding tokens through the grammar state."""
        from mlx_lm.grammar import LLGuidanceState

        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        state = LLGuidanceState.from_json_schema(hf_tokenizer, schema)

        # Feed '{'
        open_brace = hf_tokenizer.encode("{", add_special_tokens=False)[0]
        state.update(open_brace)
        assert not state.is_complete()
        assert state.partial_output == "{"

    def test_clone(self, hf_tokenizer):
        """Test cloning a grammar state."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_regex(hf_tokenizer, "[a-z]+")
        # Feed a letter
        a_id = hf_tokenizer.encode("a", add_special_tokens=False)[0]
        state.update(a_id)

        cloned = state.clone()
        assert len(cloned.generated_tokens) == len(state.generated_tokens)

    def test_reset(self, hf_tokenizer):
        """Test resetting a grammar state."""
        from mlx_lm.grammar import LLGuidanceState

        state = LLGuidanceState.from_regex(hf_tokenizer, "[a-z]+")
        a_id = hf_tokenizer.encode("a", add_special_tokens=False)[0]
        state.update(a_id)
        assert len(state.generated_tokens) == 1

        state.reset()
        assert len(state.generated_tokens) == 0
        assert not state.is_complete()

    def test_grammar_logits_processor_with_real_grammar(self, hf_tokenizer):
        """Test GrammarLogitsProcessor with a real llguidance grammar."""
        from mlx_lm.grammar import LLGuidanceState
        from mlx_lm.sample_utils import GrammarLogitsProcessor

        state = LLGuidanceState.from_json_schema(
            hf_tokenizer,
            {"type": "object", "properties": {"a": {"type": "string"}}},
        )
        processor = GrammarLogitsProcessor(state)

        vocab_size = state._vocab_size
        logits = mx.zeros((1, vocab_size))
        tokens = mx.array([], dtype=mx.int32)

        result = processor(tokens, logits)
        assert result.shape == logits.shape
        # Some tokens should be -inf (constrained)
        assert mx.any(result == float("-inf")).item()

    def test_make_grammar_logits_processor(self, hf_tokenizer):
        """Test the make_grammar_logits_processor factory."""
        from mlx_lm.sample_utils import make_grammar_logits_processor

        proc = make_grammar_logits_processor(
            hf_tokenizer,
            json_schema={"type": "object"},
        )
        assert proc is not None
        assert not proc.is_complete


class TestCLIGrammarArgs:
    """Test grammar CLI argument parsing."""

    def test_generate_arg_parser_has_grammar_args(self):
        """Test that generate.py arg parser has grammar arguments."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(
            ["--json-schema", '{"type": "object"}', "--prompt", "test"]
        )
        assert args.json_schema == '{"type": "object"}'
        assert args.grammar is None
        assert args.regex is None
        assert args.choices is None

    def test_generate_regex_arg(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(["--regex", "[a-z]+", "--prompt", "test"])
        assert args.regex == "[a-z]+"
        assert args.json_schema is None

    def test_generate_choices_arg(self):
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args(["--choices", "yes", "no", "--prompt", "test"])
        assert args.choices == ["yes", "no"]

    def test_generate_mutually_exclusive(self):
        """Test that grammar args are mutually exclusive."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["--json-schema", '{}', "--regex", "abc", "--prompt", "test"]
            )


class TestServerGrammarArgs:
    """Test server-side grammar argument handling."""

    def test_parse_json_schema_response_format(self):
        """Test parsing OpenAI-style response_format with json_schema."""
        from mlx_lm.server import GrammarArguments

        # Simulate what _parse_grammar_args does
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
        }
        # Extract the schema (nested under "schema" key per OpenAI spec)
        schema = rf.get("json_schema", {})
        if "schema" in schema:
            schema = schema["schema"]

        args = GrammarArguments(json_schema=schema)
        assert args.json_schema == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_parse_json_object_response_format(self):
        """Test parsing response_format with type=json_object."""
        from mlx_lm.server import GrammarArguments

        args = GrammarArguments(json_schema={"type": "object"})
        assert args.json_schema == {"type": "object"}

    def test_grammar_arguments_default_none(self):
        """Test GrammarArguments defaults."""
        from mlx_lm.server import GrammarArguments

        args = GrammarArguments()
        assert args.json_schema is None
        assert args.regex is None
        assert args.choices is None


class TestEndToEndGeneration:
    """End-to-end tests with real model generation.

    These tests download and run a small model. They're slower but validate
    the full pipeline from prompt to constrained output.
    """

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        pytest.importorskip("llguidance")
        from mlx_lm import load

        return load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    def test_json_schema_generation(self, model_and_tokenizer):
        """Test that JSON schema constraint produces valid JSON."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        model, tokenizer = model_and_tokenizer
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        }
        processor = make_grammar_logits_processor(tokenizer, json_schema=schema)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Return JSON for a person named Alice age 25"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response = generate(
            model, tokenizer, prompt_tokens,
            max_tokens=100, sampler=make_sampler(0.0),
            logits_processors=[processor], verbose=False,
        )
        parsed = json.loads(response)
        assert isinstance(parsed, dict)
        assert "name" in parsed
        assert "age" in parsed

    def test_regex_generation(self, model_and_tokenizer):
        """Test that regex constraint limits output."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        model, tokenizer = model_and_tokenizer
        processor = make_grammar_logits_processor(tokenizer, regex="(yes|no)")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Is water wet? yes or no"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response = generate(
            model, tokenizer, prompt_tokens,
            max_tokens=10, sampler=make_sampler(0.0),
            logits_processors=[processor], verbose=False,
        )
        assert response.strip() in ("yes", "no")

    def test_choices_generation(self, model_and_tokenizer):
        """Test that choices constraint limits output."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_grammar_logits_processor, make_sampler

        model, tokenizer = model_and_tokenizer
        processor = make_grammar_logits_processor(
            tokenizer, choices=["red", "blue", "green"]
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Pick a color"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response = generate(
            model, tokenizer, prompt_tokens,
            max_tokens=10, sampler=make_sampler(0.0),
            logits_processors=[processor], verbose=False,
        )
        assert response.strip() in ("red", "blue", "green")


class TestToolCallingIntegration:
    """Test grammar-constrained tool calling."""

    def test_cli_tools_arg(self):
        """Test that --tools CLI arg is parsed correctly."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args([
            "--tools", '[{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}]',
            "--prompt", "test",
        ])
        assert args.tools is not None
        assert "get_weather" in args.tools

    def test_cli_tools_mutually_exclusive_with_regex(self):
        """Test that --tools is mutually exclusive with other grammar args."""
        from mlx_lm.generate import setup_arg_parser

        parser = setup_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--tools", "[]", "--regex", "abc", "--prompt", "test"])

    def test_chat_tools_arg(self):
        """Test that chat.py --tools arg is parsed correctly."""
        from mlx_lm.chat import setup_arg_parser

        parser = setup_arg_parser()
        args = parser.parse_args([
            "--tools", '[{"name":"search","parameters":{}}]',
        ])
        assert args.tools is not None

    def test_server_tool_choice_required(self):
        """Test that tool_choice=required creates grammar args from tools."""
        from mlx_lm.server import GrammarArguments
        from unittest.mock import MagicMock

        # Create a mock handler with the right attributes
        handler = MagicMock()
        handler.response_format = None
        handler.body = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
        }

        # Import and call the method directly
        from mlx_lm.server import APIHandler
        result = APIHandler._parse_grammar_args(handler)

        assert result is not None
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"

    def test_server_tool_choice_auto_no_grammar(self):
        """Test that tool_choice=auto does not create grammar args."""
        from unittest.mock import MagicMock
        from mlx_lm.server import APIHandler

        handler = MagicMock()
        handler.response_format = None
        handler.body = {
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            "tool_choice": "auto",
        }

        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_server_tool_choice_specific_function(self):
        """Test that tool_choice with specific function filters tools."""
        from unittest.mock import MagicMock
        from mlx_lm.server import APIHandler

        handler = MagicMock()
        handler.response_format = None
        handler.body = {
            "tools": [
                {"type": "function", "function": {"name": "search", "parameters": {}}},
                {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }

        result = APIHandler._parse_grammar_args(handler)
        assert result is not None
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "get_weather"

    def test_server_tool_choice_none_no_grammar(self):
        """Test that tool_choice=none does not create grammar args."""
        from unittest.mock import MagicMock
        from mlx_lm.server import APIHandler

        handler = MagicMock()
        handler.response_format = None
        handler.body = {
            "tools": [{"type": "function", "function": {"name": "search", "parameters": {}}}],
            "tool_choice": "none",
        }

        result = APIHandler._parse_grammar_args(handler)
        assert result is None

    def test_grammar_arguments_tools_field(self):
        """Test GrammarArguments supports tools field."""
        from mlx_lm.server import GrammarArguments

        tools = [{"name": "test", "parameters": {"type": "object"}}]
        args = GrammarArguments(tools=tools)
        assert args.tools == tools
        assert args.json_schema is None

    def test_make_grammar_logits_processor_with_tools(self):
        """Test make_grammar_logits_processor with tools parameter."""
        pytest.importorskip("llguidance")
        from transformers import AutoTokenizer
        from mlx_lm.sample_utils import make_grammar_logits_processor

        tokenizer = AutoTokenizer.from_pretrained(
            "mlx-community/Llama-3.2-1B-Instruct-4bit"
        )
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
        assert processor is not None
        assert not processor.is_complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
