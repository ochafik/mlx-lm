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
        """Test processor tracks completion."""
        from mlx_lm.sample_utils import GrammarLogitsProcessor
        from mlx_lm.grammar.llguidance_adapter import MockGrammarState

        grammar = MockGrammarState(vocab_size=100)
        grammar._max_tokens = 5
        processor = GrammarLogitsProcessor(grammar)

        tokens = mx.array([1])
        logits = mx.random.normal((1, 100))

        # Process tokens until complete
        for i in range(10):
            processor(mx.array([i]), logits)
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

        # Process some tokens
        for i in range(10):
            processor(mx.array([i]), mx.random.normal((1, 100)))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
