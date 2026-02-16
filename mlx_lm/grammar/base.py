# Copyright 2025 Apple Inc.

"""
Base classes for grammar-constrained generation.

This module provides the abstract interface for grammar state tracking
during token generation.
"""

from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx


class GrammarState(ABC):
    """
    Abstract base class for grammar-constrained generation state.

    A GrammarState tracks the current position in a grammar during token
    generation and provides token masks to constrain the model's output.
    The grammar state is updated after each token is generated, and provides
    a boolean mask indicating which tokens are valid at the current position.

    Example usage::

        grammar = SomeGrammarState(tokenizer, grammar_spec)

        while not grammar.is_complete():
            mask = grammar.get_token_mask()
            constrained_logits = apply_grammar_mask(logits, mask)
            token = sample(constrained_logits)
            grammar.update(token)
    """

    @abstractmethod
    def get_token_mask(self) -> mx.array:
        """
        Get boolean mask of allowed tokens for current grammar position.

        Returns:
            mx.array: Boolean array of shape (vocab_size,) where True
                indicates the token is allowed at the current position.
        """
        pass

    @abstractmethod
    def update(self, token_id: int) -> None:
        """
        Update grammar state with the generated token.

        This should be called after each token is generated and before
        the next call to get_token_mask().

        Args:
            token_id: The token ID that was generated.
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if the grammar has been fully satisfied.

        Returns:
            bool: True if the grammar is complete and generation should stop.
        """
        pass

    @property
    def partial_output(self) -> str:
        """
        Get the partial UTF-8 output generated so far.

        This is useful for debugging and for getting intermediate results.

        Returns:
            str: The text generated so far.
        """
        return ""

    def reset(self) -> None:
        """
        Reset the grammar state to its initial position.

        This allows reusing the same grammar state for multiple generations.
        """
        pass

    def clone(self) -> "GrammarState":
        """
        Create a copy of this grammar state.

        This is useful for speculative decoding or beam search where
        multiple generation paths need independent grammar states.

        Returns:
            GrammarState: A copy of this grammar state.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support cloning"
        )
