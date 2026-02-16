# Copyright 2025 Apple Inc.

"""
MLX-optimized operations for grammar token masks.

This module provides efficient operations for applying token masks
to logits during grammar-constrained generation.
"""

from typing import Union

import mlx.core as mx
import numpy as np


def apply_grammar_mask(
    logits: mx.array,
    mask: mx.array,
    fill_value: float = float("-inf"),
) -> mx.array:
    """
    Apply boolean token mask to logits.

    Tokens that are not allowed by the grammar have their logits set to
    fill_value (typically -inf) so they have zero probability after softmax.

    Args:
        logits: Logits array of shape (..., vocab_size).
        mask: Boolean mask of shape (vocab_size,) where True means allowed.
        fill_value: Value to use for disallowed tokens. Default: -inf.

    Returns:
        mx.array: Constrained logits with same shape as input.

    Example::

        mask = grammar.get_token_mask()  # [True, False, True, ...]
        constrained = apply_grammar_mask(logits, mask)
        # Tokens where mask is False now have -inf logits
    """
    # Ensure mask is boolean
    if mask.dtype != mx.bool_:
        mask = mask.astype(mx.bool_)

    # Use where for efficient masking
    # Broadcasting handles batch dimension automatically
    return mx.where(mask, logits, fill_value)


def bitmask_to_bool(
    bitmask: Union[np.ndarray, mx.array],
    vocab_size: int,
) -> mx.array:
    """
    Convert compressed bitmask to boolean array.

    Some grammar engines (like LLGuidance) return bitmasks as uint32 arrays
    for memory efficiency. This expands them to full boolean arrays.

    Args:
        bitmask: Compressed mask of shape (ceil(vocab_size/32),) as uint32.
        vocab_size: Total vocabulary size.

    Returns:
        mx.array: Boolean array of shape (vocab_size,).

    Note:
        Each uint32 in the bitmask encodes 32 token IDs. Bit i in word j
        corresponds to token ID (j * 32 + i).
    """
    # Convert to numpy if needed
    if isinstance(bitmask, mx.array):
        bitmask = np.array(bitmask)

    # Expand bitmask to boolean using vectorized operations
    # Create array of bit positions
    n_words = len(bitmask)
    bit_positions = np.arange(32, dtype=np.uint32)

    # Expand each word
    result = np.zeros(n_words * 32, dtype=np.bool_)
    for i, word in enumerate(bitmask):
        word = np.uint32(word)
        for bit in range(32):
            token_id = i * 32 + bit
            if token_id < vocab_size:
                result[token_id] = bool(word & (np.uint32(1) << bit))

    # Truncate to vocab size
    return mx.array(result[:vocab_size])


def bool_to_bitmask(
    mask: Union[np.ndarray, mx.array],
) -> np.ndarray:
    """
    Convert boolean array to compressed bitmask.

    This is the inverse of bitmask_to_bool.

    Args:
        mask: Boolean array of shape (vocab_size,).

    Returns:
        np.ndarray: Compressed mask of shape (ceil(vocab_size/32),) as uint32.
    """
    # Convert to numpy if needed
    if isinstance(mask, mx.array):
        mask = np.array(mask)

    vocab_size = len(mask)
    n_words = (vocab_size + 31) // 32
    result = np.zeros(n_words, dtype=np.uint32)

    for token_id, allowed in enumerate(mask):
        if allowed:
            word_idx = token_id // 32
            bit_idx = token_id % 32
            result[word_idx] |= np.uint32(1) << bit_idx

    return result


def count_allowed_tokens(mask: mx.array) -> int:
    """
    Count the number of allowed tokens in a mask.

    Args:
        mask: Boolean mask of shape (vocab_size,).

    Returns:
        int: Number of True values in the mask.
    """
    return int(mx.sum(mask.astype(mx.int32)).item())


def get_allowed_token_ids(mask: mx.array) -> mx.array:
    """
    Get the token IDs that are allowed by the mask.

    Args:
        mask: Boolean mask of shape (vocab_size,).

    Returns:
        mx.array: Array of allowed token IDs.
    """
    # Get indices where mask is True
    # Use numpy for boolean indexing since MLX doesn't support it yet
    mask_np = np.array(mask)
    indices = np.arange(len(mask_np))
    allowed = indices[mask_np]
    return mx.array(allowed)
