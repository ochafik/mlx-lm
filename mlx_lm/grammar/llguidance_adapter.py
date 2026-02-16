# Copyright 2025 Apple Inc.

"""
LLGuidance integration for grammar-constrained generation.

This module provides the LLGuidanceState class that wraps llguidance's
grammar engine for use with mlx-lm's generation pipeline.
"""

import json
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import numpy as np

from .base import GrammarState

# Lazy import to handle optional dependency
_llguidance = None
_llguidance_import_error = None


def _get_llguidance():
    """Lazily import llguidance."""
    global _llguidance, _llguidance_import_error

    if _llguidance is not None:
        return _llguidance

    if _llguidance_import_error is not None:
        raise _llguidance_import_error

    try:
        import llguidance

        _llguidance = llguidance
        return _llguidance
    except ImportError as e:
        _llguidance_import_error = ImportError(
            "llguidance is required for grammar-constrained generation. "
            "Install with: pip install llguidance"
        )
        raise _llguidance_import_error from e


def is_llguidance_available() -> bool:
    """Check if llguidance is available."""
    try:
        _get_llguidance()
        return True
    except ImportError:
        return False


class LLGuidanceState(GrammarState):
    """
    Grammar state implementation using LLGuidance.

    LLGuidance provides a Rust-based grammar engine with Lark grammar
    support and efficient token mask computation (~1-10μs per token).

    Example::

        from mlx_lm.grammar import LLGuidanceState

        # From JSON schema
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        grammar = LLGuidanceState.from_json_schema(tokenizer, schema)

        # From regex
        grammar = LLGuidanceState.from_regex(tokenizer, r"[a-z]+@[a-z]+\\.com")

        # From Lark grammar
        grammar = LLGuidanceState(tokenizer, 'start: "hello" NAME')
    """

    def __init__(
        self,
        tokenizer,
        grammar: str,
        *,
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize from a Lark grammar string.

        Args:
            tokenizer: HuggingFace tokenizer (or TokenizerWrapper).
            grammar: Lark grammar string.
            vocab_size: Optional vocabulary size override.
        """
        llg = _get_llguidance()

        # Handle TokenizerWrapper
        self._hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)

        # Create llguidance tokenizer
        self._ll_tokenizer = llg.LLTokenizer.from_hf_tokenizer(self._hf_tokenizer)

        # Create matcher
        self._matcher = llg.LLMatcher(self._ll_tokenizer, grammar)

        # Store vocab size
        if vocab_size is not None:
            self._vocab_size = vocab_size
        elif hasattr(self._hf_tokenizer, "vocab_size"):
            self._vocab_size = self._hf_tokenizer.vocab_size
        else:
            # Fallback: get from vocab
            self._vocab_size = len(self._hf_tokenizer.get_vocab())

        self._completed = False
        self._grammar_str = grammar
        self._generated_tokens: List[int] = []

    @classmethod
    def from_json_schema(
        cls,
        tokenizer,
        schema: Union[str, Dict[str, Any]],
        **kwargs,
    ) -> "LLGuidanceState":
        """
        Create grammar state from JSON schema.

        Args:
            tokenizer: HuggingFace tokenizer.
            schema: JSON schema as dict or JSON string.
            **kwargs: Additional arguments to constructor.

        Returns:
            LLGuidanceState configured for the schema.

        Example::

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0}
                },
                "required": ["name"]
            }
            grammar = LLGuidanceState.from_json_schema(tokenizer, schema)
        """
        llg = _get_llguidance()

        if isinstance(schema, dict):
            schema = json.dumps(schema)

        grammar = llg.grammar_from("json_schema", schema)
        return cls(tokenizer, grammar, **kwargs)

    @classmethod
    def from_regex(
        cls,
        tokenizer,
        pattern: str,
        **kwargs,
    ) -> "LLGuidanceState":
        """
        Create grammar state from regex pattern.

        Args:
            tokenizer: HuggingFace tokenizer.
            pattern: Regular expression pattern.
            **kwargs: Additional arguments to constructor.

        Returns:
            LLGuidanceState configured for the pattern.

        Example::

            # Email pattern
            grammar = LLGuidanceState.from_regex(
                tokenizer,
                r"[a-z]+@[a-z]+\\.[a-z]+"
            )
        """
        llg = _get_llguidance()
        grammar = llg.grammar_from("regex", pattern)
        return cls(tokenizer, grammar, **kwargs)

    @classmethod
    def from_choices(
        cls,
        tokenizer,
        choices: List[str],
        **kwargs,
    ) -> "LLGuidanceState":
        """
        Create grammar state for selecting from choices.

        Args:
            tokenizer: HuggingFace tokenizer.
            choices: List of allowed string values.
            **kwargs: Additional arguments to constructor.

        Returns:
            LLGuidanceState that only allows the given choices.

        Example::

            grammar = LLGuidanceState.from_choices(
                tokenizer,
                ["yes", "no", "maybe"]
            )
        """
        llg = _get_llguidance()
        grammar = llg.grammar_from("choice", json.dumps(choices))
        return cls(tokenizer, grammar, **kwargs)

    @classmethod
    def from_tools(
        cls,
        tokenizer,
        tools: List[Dict[str, Any]],
        **kwargs,
    ) -> "LLGuidanceState":
        """
        Create grammar state for tool calling.

        Automatically detects the tool call format from the tokenizer's
        chat template and generates an appropriate grammar.

        Args:
            tokenizer: HuggingFace tokenizer with chat_template.
            tools: List of tool definitions.
            **kwargs: Additional arguments to constructor.

        Returns:
            LLGuidanceState configured for tool calling.

        Example::

            tools = [
                {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"]
                    }
                }
            ]
            grammar = LLGuidanceState.from_tools(tokenizer, tools)
        """
        from .tool_schema import build_tool_grammar

        grammar = build_tool_grammar(tools, tokenizer)
        return cls(tokenizer, grammar, **kwargs)

    def get_token_mask(self) -> mx.array:
        """
        Get boolean mask of allowed tokens for current grammar position.

        Returns:
            mx.array of shape (vocab_size,) with True for allowed tokens.
        """
        if self._completed:
            # If complete, disallow all tokens (generation should stop)
            return mx.zeros((self._vocab_size,), dtype=mx.bool_)

        # Get mask from llguidance
        mask_np = self._matcher.get_token_mask()

        # Convert to MLX boolean array
        # llguidance returns numpy bool array
        return mx.array(mask_np.astype(np.bool_))

    def update(self, token_id: int) -> None:
        """
        Update grammar state with the generated token.

        Args:
            token_id: The token ID that was generated.
        """
        if self._completed:
            return

        self._generated_tokens.append(token_id)
        self._matcher.commit_token(token_id)
        self._completed = self._matcher.is_terminated()

    def is_complete(self) -> bool:
        """
        Check if the grammar has been fully satisfied.

        Returns:
            True if grammar is complete and generation should stop.
        """
        return self._completed

    @property
    def partial_output(self) -> str:
        """
        Get the partial UTF-8 output generated so far.

        Returns:
            The text generated so far.
        """
        try:
            return self._matcher.get_partial_utf8()
        except Exception:
            # Fallback to decoding tokens
            return self._hf_tokenizer.decode(
                self._generated_tokens, skip_special_tokens=False
            )

    def reset(self) -> None:
        """
        Reset the grammar state to its initial position.
        """
        llg = _get_llguidance()
        self._matcher = llg.LLMatcher(self._ll_tokenizer, self._grammar_str)
        self._completed = False
        self._generated_tokens = []

    def clone(self) -> "LLGuidanceState":
        """
        Create a copy of this grammar state.

        Returns:
            A new LLGuidanceState at the same position.
        """
        # Create new instance with same grammar
        new_state = LLGuidanceState(
            self._hf_tokenizer,
            self._grammar_str,
            vocab_size=self._vocab_size,
        )

        # Replay tokens to reach same state
        for token in self._generated_tokens:
            new_state.update(token)

        return new_state

    @property
    def generated_tokens(self) -> List[int]:
        """Get list of generated token IDs."""
        return self._generated_tokens.copy()

    @property
    def grammar(self) -> str:
        """Get the grammar string."""
        return self._grammar_str

    def __repr__(self) -> str:
        status = "complete" if self._completed else "in_progress"
        n_tokens = len(self._generated_tokens)
        return f"LLGuidanceState({status}, {n_tokens} tokens)"


class MockGrammarState(GrammarState):
    """
    Mock grammar state for testing without llguidance.

    Always allows all tokens - useful for testing the integration
    without the llguidance dependency.
    """

    def __init__(self, vocab_size: int = 32000):
        self._vocab_size = vocab_size
        self._completed = False
        self._tokens: List[int] = []
        self._max_tokens = 100

    def get_token_mask(self) -> mx.array:
        """Allow all tokens."""
        return mx.ones((self._vocab_size,), dtype=mx.bool_)

    def update(self, token_id: int) -> None:
        """Track token but don't constrain."""
        self._tokens.append(token_id)
        if len(self._tokens) >= self._max_tokens:
            self._completed = True

    def is_complete(self) -> bool:
        return self._completed

    @property
    def partial_output(self) -> str:
        return f"<{len(self._tokens)} tokens>"

    def reset(self) -> None:
        self._tokens = []
        self._completed = False

    def clone(self) -> "MockGrammarState":
        new_state = MockGrammarState(self._vocab_size)
        new_state._tokens = self._tokens.copy()
        new_state._completed = self._completed
        return new_state
