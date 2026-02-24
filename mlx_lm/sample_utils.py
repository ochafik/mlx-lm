# Copyright © 2023-2024 Apple Inc.

import math
from functools import partial
from typing import Callable, Dict, List, Optional

import mlx.core as mx


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.0,
    xtc_special_tokens: List[int] = [],
) -> Callable[[mx.array], mx.array]:
    """
    Make a sampler function for use with ``generate_step``.

    Args:
        temp (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        top_p (float, optional): Nulceus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): The minimum value (scaled by the top token's
          probability) that a token probability must have to be considered.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
          be filtered by min_p sampling.
        top_k (int, optional): The top k tokens ranked by probability to constrain
          the sampling to.
        xtc_probability (float, optional): The probability of applying XTC
            sampling.
        xtc_threshold (float, optional): The threshold the probs need to reach
            for being sampled.
        xtc_special_tokens (list(int), optional): List of special tokens IDs to
            be excluded from XTC sampling.


    Returns:
        Callable[mx.array, mx.array]:
            A sampler which takes log-probabilities and returns tokens.
    """
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)

    # Create sampler chain
    sampling_methods = []
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    # Apply the sampling methods
    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        # Return the sampled token
        return categorical_sampling(logprobs, temp)

    return sampler


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
):
    """
    Make logits processors for use with ``generate_step``.

    Args:
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        logit_bias (dictionary, optional): Additive logit bias.

    Returns:
        List[Callable[[mx.array, mx.array], mx.array]]:
            A list of logits processors. Each processor in the list is a
            callable which takes an array of tokens and an array of logits
            and returns the updated logits.
    """
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            logits[:, indices] += values
            return logits

        logits_processors.append(logit_bias_processor)

    if repetition_penalty and repetition_penalty != 0.0:
        logits_processors.append(
            make_repetition_penalty(repetition_penalty, repetition_context_size)
        )
    return logits_processors


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(
    logprobs: mx.array,
    top_k: int,
) -> mx.array:
    """
    Sample from only the top K tokens ranked by probability.

    Args:
        logprobs: A vector of log probabilities.
        top_k (int): Top k tokens to sample from.
    """
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}] interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    """
    Apply min-p sampling to the logprobs.

    Min-p keeps all tokens that are above a minimum probability, scaled by the
    probability of the most likely token. As a result, the filter is more
    aggressive given a very high-probability token.

    Args:
        logprobs: A vector of log probabilities.
        min_p (float): Minimum token probability. Typical values are in the
            0.01-0.2 range, comparably selective as setting `top_p` in the
            0.99-0.8 range.
        min_tokens_to_keep (int, optional): Minimum number of tokens that cannot
            be filtered. Default: ``1``.

    """
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
        raise ValueError(
            f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}"
        )
    # reference implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L531-L605

    # Indices sorted in decreasing order
    sorted_indices = mx.argsort(-logprobs, axis=-1)
    sorted_logprobs = mx.take_along_axis(logprobs, sorted_indices, axis=-1)

    # Top probability
    top_logprobs = sorted_logprobs[:, 0:1]

    # Calculate the min_p threshold
    scaled_min_p = top_logprobs + math.log(min_p)

    # Mask tokens that have a probability less than the scaled min_p
    tokens_to_remove = sorted_logprobs < scaled_min_p
    tokens_to_remove[..., :min_tokens_to_keep] = False

    # Create pool of tokens with probability less than scaled min_p
    selected_logprobs = mx.where(tokens_to_remove, -float("inf"), sorted_logprobs)

    # Create a mapping to rearrange back to original indices
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )

    # Rearrange selected_logprobs back to original order
    original_order_logprobs = mx.take_along_axis(
        selected_logprobs, inverse_indices, axis=-1
    )

    return original_order_logprobs


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logprobs: A vector of log probabilities.
        top_p: The cumulative probability threshold for top-p filtering.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.exp(logprobs)
    # sort in ascending order
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # Rearrange cumulative probs back to original order
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)

    # select tokens with cumulative probs below threshold
    return mx.where(
        cumulative_probs > 1 - top_p,
        logprobs,
        -float("inf"),
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> mx.array:
    """
    Apply XTC sampling to the logits.

    Args:
        logits: The logits from the model's output.
        xtc_probability (float): Probability of XTC sampling to happen for each token
        xtc_threshold (float): The threshold the probs need to reach for being sampled.
        special_tokens_ids (list(int)): List of special tokens IDs to be excluded from XTC sampling.
    """
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`threshold` has to be a float in the [0, 0.5] interval, but is {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`probability` has to be a float in the [0, 1] interval, but is {xtc_probability}"
        )

    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min()
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


def make_repetition_penalty(penalty: float, context_size: int = 20):
    """
    Make repetition penalty processor.

    Paper: https://arxiv.org/abs/1909.05858

    Args:
        penalty (float): The repetition penalty factor to be applied.
        context_size (int): The number of previous tokens to use.
            Default: ``20``.

    Returns:
        Callable[[mx.array, List[int]], mx.array]:
            The repetition penalty processor.
    """
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor


class GrammarLogitsProcessor:
    """
    Logits processor that applies grammar constraints during generation.

    This processor integrates with mlx-lm's existing logits_processors mechanism
    to apply grammar constraints from llguidance or other grammar engines.

    Example::

        from mlx_lm.grammar import LLGuidanceState
        from mlx_lm.sample_utils import GrammarLogitsProcessor

        # Create grammar state
        grammar = LLGuidanceState.from_json_schema(tokenizer, schema)

        # Create processor
        processor = GrammarLogitsProcessor(grammar)

        # Use in generation
        response = generate(
            model, tokenizer, prompt,
            logits_processors=[processor]
        )

    Args:
        grammar_state: Grammar state object implementing get_token_mask().
        warn_on_empty_mask: If True, warn when no tokens are allowed.
    """

    def __init__(self, grammar_state, *, warn_on_empty_mask: bool = True, eos_token_ids=None):
        self.grammar = grammar_state
        self._warn_on_empty = warn_on_empty_mask
        self._warned = False
        self._token_callback = None
        self._prev_tokens_len = 0
        if isinstance(eos_token_ids, int):
            self._eos_token_ids = {eos_token_ids}
        else:
            self._eos_token_ids = set(eos_token_ids or [])

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """
        Apply grammar constraint to logits.

        Args:
            tokens: Accumulated tokens so far (prompt + generated).
            logits: Logits from the model (shape: (batch, vocab_size) or (vocab_size,)).

        Returns:
            Constrained logits with -inf for disallowed tokens.
        """
        # The generate_step function passes accumulated tokens (prompt tail + generated).
        # On the first call, tokens contains prompt tokens - skip those.
        # On subsequent calls, update grammar with each newly appended token.
        tokens_flat = tokens.reshape(-1)
        current_len = tokens_flat.size

        if self._prev_tokens_len == 0:
            # First call: these are prompt tokens, skip update
            self._prev_tokens_len = current_len
        else:
            # Feed newly appended tokens to the grammar
            for i in range(self._prev_tokens_len, current_len):
                token_id = int(tokens_flat[i].item())
                self.grammar.update(token_id)
            self._prev_tokens_len = current_len

        # Check if complete — force EOS so generation stops
        if self.grammar.is_complete():
            if self._eos_token_ids:
                neg_inf = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
                eos_mask = mx.zeros(logits.shape, dtype=mx.bool_)
                for eos_id in self._eos_token_ids:
                    eos_mask = eos_mask | (mx.arange(logits.shape[-1]) == eos_id)
                return mx.where(eos_mask, logits, neg_inf)
            return logits

        # Get token mask from grammar
        mask = self.grammar.get_token_mask()

        # Check if any tokens are allowed
        if self._warn_on_empty and not self._warned:
            if not mx.any(mask):
                import warnings
                warnings.warn(
                    "Grammar constraint has no valid tokens at current position. "
                    "This may indicate a grammar that cannot be satisfied."
                )
                self._warned = True
                return logits

        # Apply constraint: set disallowed tokens to -inf
        constrained = mx.where(
            mask,
            logits,
            mx.full(logits.shape, float("-inf"), dtype=logits.dtype),
        )

        return constrained

    @property
    def is_complete(self) -> bool:
        """Check if grammar generation is complete."""
        return self.grammar.is_complete()

    @property
    def partial_output(self) -> str:
        """Get the partial output generated so far."""
        return self.grammar.partial_output

    def reset(self):
        """Reset the grammar state for reuse."""
        self.grammar.reset()
        self._prev_tokens_len = 0
        self._warned = False


def make_grammar_logits_processor(
    tokenizer,
    *,
    json_schema: Optional[Dict] = None,
    regex: Optional[str] = None,
    choices: Optional[List[str]] = None,
    tools: Optional[List[Dict]] = None,
    grammar: Optional[str] = None,
):
    """
    Create a grammar logits processor from various constraint types.

    This is a convenience function that creates the appropriate grammar
    state and wraps it in a GrammarLogitsProcessor.

    Args:
        tokenizer: HuggingFace tokenizer.
        json_schema: JSON schema dict to constrain output.
        regex: Regular expression pattern to match.
        choices: List of allowed string values.
        tools: List of tool definitions for tool calling.
        grammar: Raw Lark grammar string.

    Returns:
        GrammarLogitsProcessor configured with the constraint.

    Raises:
        ImportError: If llguidance is not installed.
        ValueError: If no constraint type is specified.

    Example::

        # JSON schema constraint
        processor = make_grammar_logits_processor(
            tokenizer,
            json_schema={"type": "object", "properties": {"name": {"type": "string"}}}
        )

        # Regex constraint
        processor = make_grammar_logits_processor(
            tokenizer,
            regex=r"[a-z]+@[a-z]+\\.com"
        )

        # Choice constraint
        processor = make_grammar_logits_processor(
            tokenizer,
            choices=["yes", "no", "maybe"]
        )
    """
    # Import grammar module (handles llguidance lazy loading)
    from mlx_lm.grammar import LLGuidanceState

    # Create appropriate grammar state
    if json_schema is not None:
        grammar_state = LLGuidanceState.from_json_schema(tokenizer, json_schema)
    elif regex is not None:
        grammar_state = LLGuidanceState.from_regex(tokenizer, regex)
    elif choices is not None:
        grammar_state = LLGuidanceState.from_choices(tokenizer, choices)
    elif tools is not None:
        grammar_state = LLGuidanceState.from_tools(tokenizer, tools)
    elif grammar is not None:
        grammar_state = LLGuidanceState(tokenizer, grammar)
    else:
        raise ValueError(
            "Must specify one of: json_schema, regex, choices, tools, or grammar"
        )

    # Get EOS token IDs so the processor can force stop when grammar completes
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_id, int):
            eos_ids = {eos_id}
        elif eos_id is not None:
            eos_ids = set(eos_id)
        else:
            eos_ids = set()

    return GrammarLogitsProcessor(grammar_state, eos_token_ids=eos_ids)
