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


def _make_ll_tokenizer(tokenizer):
    """
    Create an LLTokenizer from a HuggingFace tokenizer.

    llguidance requires a TokenizerWrapper with specific attributes
    (.tokens, .eos_token_id, .bos_token_id, .special_token_ids, __call__).
    This bridge adapts a HuggingFace tokenizer to that interface.
    """
    llg = _get_llguidance()

    # Keep the outer HF tokenizer for metadata (eos_token_id, vocab, etc.)
    # It might be wrapped in an mlx-lm TokenizerWrapper that has ._tokenizer
    # pointing to a fast tokenizers.Tokenizer - we want the PreTrainedTokenizer
    hf_tok = tokenizer
    # If it's an mlx-lm TokenizerWrapper, get the underlying PreTrainedTokenizer
    if hasattr(tokenizer, "_tokenizer") and hasattr(tokenizer, "encode"):
        hf_tok = tokenizer

    class _HFBridge:
        def __init__(self, hf_tok):
            self.eos_token_id = hf_tok.eos_token_id
            self.bos_token_id = getattr(hf_tok, "bos_token_id", None)

            # Build tokens list: bytes for each token id
            vocab = hf_tok.get_vocab()
            max_id = max(vocab.values()) if vocab else 0
            self.tokens = [b""] * (max_id + 1)
            for token_str, token_id in vocab.items():
                self.tokens[token_id] = token_str.encode("utf-8", errors="replace")

            self.special_token_ids = list(
                getattr(hf_tok, "all_special_ids", [])
            )
            self._hf_tok = hf_tok

        def __call__(self, text):
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            return self._hf_tok.encode(text, add_special_tokens=False)

    bridge = _HFBridge(hf_tok)
    wrapper = llg.TokenizerWrapper(bridge)
    return llg.LLTokenizer(wrapper), hf_tok


class LLGuidanceState(GrammarState):
    """
    Grammar state implementation using LLGuidance.

    LLGuidance provides a Rust-based grammar engine with Lark grammar
    support and efficient token mask computation.

    Example::

        from mlx_lm.grammar import LLGuidanceState

        # From JSON schema
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        grammar = LLGuidanceState.from_json_schema(tokenizer, schema)

        # From regex
        grammar = LLGuidanceState.from_regex(tokenizer, r"[a-z]+@[a-z]+\\.com")
    """

    def __init__(
        self,
        tokenizer,
        grammar: str,
        *,
        vocab_size: Optional[int] = None,
    ):
        """
        Initialize from an llguidance grammar definition string.

        Args:
            tokenizer: HuggingFace tokenizer (or mlx-lm TokenizerWrapper).
            grammar: llguidance grammar definition string (from grammar_from()).
            vocab_size: Optional vocabulary size override.
        """
        llg = _get_llguidance()

        self._ll_tokenizer, self._hf_tokenizer = _make_ll_tokenizer(tokenizer)
        self._matcher = llg.LLMatcher(self._ll_tokenizer, grammar)

        if vocab_size is not None:
            self._vocab_size = vocab_size
        else:
            self._vocab_size = self._ll_tokenizer.vocab_size

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

        Uses :func:`build_tool_schema` to build a JSON schema and then
        ``grammar_from("json_schema", ...)`` for reliable constraining.

        Args:
            tokenizer: HuggingFace tokenizer with chat_template.
            tools: List of tool definitions.
        """
        from .tool_schema import build_tool_schema

        llg = _get_llguidance()
        schema, _fmt = build_tool_schema(tools, tokenizer)
        grammar = llg.grammar_from("json_schema", json.dumps(schema))
        return cls(tokenizer, grammar, **kwargs)

    def get_token_mask(self) -> mx.array:
        """
        Get boolean mask of allowed tokens for current grammar position.

        Returns:
            mx.array of shape (vocab_size,) with True for allowed tokens.
        """
        if self._completed:
            return mx.zeros((self._vocab_size,), dtype=mx.bool_)

        # compute_bitmask() returns bytes (packed bits, little-endian)
        mask_bytes = self._matcher.compute_bitmask()
        mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
        bool_mask = np.unpackbits(mask_np, bitorder="little")[: self._vocab_size]
        return mx.array(bool_mask.astype(np.bool_))

    def update(self, token_id: int) -> None:
        """
        Update grammar state with the generated token.

        Args:
            token_id: The token ID that was generated.
        """
        if self._completed:
            return

        self._generated_tokens.append(token_id)
        self._matcher.consume_token(token_id)
        self._completed = self._matcher.is_stopped()

    def is_complete(self) -> bool:
        """Check if the grammar has been fully satisfied."""
        return self._completed

    @property
    def partial_output(self) -> str:
        """Get the partial output generated so far."""
        return self._hf_tokenizer.decode(
            self._generated_tokens, skip_special_tokens=False
        )

    def reset(self) -> None:
        """Reset the grammar state to its initial position."""
        llg = _get_llguidance()
        self._matcher = llg.LLMatcher(self._ll_tokenizer, self._grammar_str)
        self._completed = False
        self._generated_tokens = []

    def clone(self) -> "LLGuidanceState":
        """Create a copy of this grammar state."""
        new_state = LLGuidanceState.__new__(LLGuidanceState)
        new_state._ll_tokenizer = self._ll_tokenizer
        new_state._hf_tokenizer = self._hf_tokenizer
        new_state._vocab_size = self._vocab_size
        new_state._grammar_str = self._grammar_str
        new_state._generated_tokens = self._generated_tokens.copy()
        new_state._completed = self._completed

        # Use deep_copy if available, otherwise replay tokens
        if hasattr(self._matcher, "deep_copy"):
            new_state._matcher = self._matcher.deep_copy()
        else:
            llg = _get_llguidance()
            new_state._matcher = llg.LLMatcher(
                self._ll_tokenizer, self._grammar_str
            )
            for token in self._generated_tokens:
                new_state._matcher.consume_token(token)

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
