"""
Micro-benchmark: Compare mx.gather_qmm vs explicit per-expert matmuls.

This answers the key question: does gather_qmm already batch tokens per expert
into grouped matmuls, or can we do better with explicit Python-level batching?
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import numpy as np

# Qwen3.5-35B-A3B parameters
NUM_EXPERTS = 256
TOP_K = 8
HIDDEN_DIM = 2048
EXPERT_INTERMEDIATE = 1024  # moe_intermediate_size
BITS = 4
GROUP_SIZE = 64

def create_quantized_weights(num_experts, out_dim, in_dim, bits=4, group_size=64):
    """Create fake quantized expert weights."""
    # Quantized weight shape: (num_experts, out_dim, in_dim * bits / 32)
    w = mx.random.normal((num_experts, out_dim, in_dim))
    weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return weight, scales, biases

def create_routing(batch_size, seq_len, num_experts, top_k):
    """Create random top-k routing indices."""
    # Simulate softmax + argpartition routing
    gate_logits = mx.random.normal((batch_size, seq_len, num_experts))
    probs = mx.softmax(gate_logits, axis=-1)
    inds = mx.argpartition(probs, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(probs, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores

print("Creating weights...")
weight, scales, biases = create_quantized_weights(NUM_EXPERTS, EXPERT_INTERMEDIATE, HIDDEN_DIM)
mx.eval(weight, scales, biases)
print(f"  weight shape: {weight.shape}")
print(f"  scales shape: {scales.shape}")
print(f"  biases shape: {biases.shape}")

for seq_len in [128, 512, 1024, 2048]:
    print(f"\n{'='*80}")
    print(f"seq_len={seq_len}, num_experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"{'='*80}")

    # Create input and routing
    x = mx.random.normal((1, seq_len, HIDDEN_DIM))
    inds, scores = create_routing(1, seq_len, NUM_EXPERTS, TOP_K)
    mx.eval(x, inds, scores)

    # ==================== Method 1: gather_qmm (current approach) ====================
    print("\n--- Method 1: gather_qmm (current mlx-lm approach) ---")

    # Prepare: expand dims and sort (matching SwitchGLU behavior)
    x_expanded = mx.expand_dims(x, (-2, -3))

    def gather_sort(x_in, indices):
        *_, M = indices.shape
        indices_flat = indices.flatten()
        order = mx.argsort(indices_flat)
        inv_order = mx.argsort(order)
        return x_in.flatten(0, -3)[order // M], indices_flat[order], inv_order

    x_sorted, idx_sorted, inv_order = gather_sort(x_expanded, inds)
    mx.eval(x_sorted, idx_sorted, inv_order)

    # Warmup
    for _ in range(3):
        out = mx.gather_qmm(
            x_sorted, weight, scales, biases,
            rhs_indices=idx_sorted,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
            sorted_indices=True,
        )
        mx.eval(out)

    times_gather = []
    for _ in range(20):
        start = time.perf_counter()
        out = mx.gather_qmm(
            x_sorted, weight, scales, biases,
            rhs_indices=idx_sorted,
            transpose=True, group_size=GROUP_SIZE, bits=BITS,
            sorted_indices=True,
        )
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times_gather.append(elapsed)

    gather_ms = np.mean(times_gather) * 1000
    print(f"  Time: {gather_ms:.2f} ms +/- {np.std(times_gather)*1000:.2f} ms")

    # ==================== Method 2: Explicit per-expert matmuls ====================
    print("\n--- Method 2: Explicit per-expert loop ---")

    # Group tokens by expert
    inds_flat = np.array(inds[0].flatten())  # [seq_len * top_k]
    x_flat = x[0]  # [seq_len, hidden_dim]

    # For each expert, find which tokens go to it and do a single matmul
    # This is the "explicit batching" approach

    def explicit_per_expert_numpy(x_in, indices_np, w, s, b):
        """Process each expert explicitly with a loop (using numpy for indexing)."""
        results = []
        for e in range(NUM_EXPERTS):
            token_ids = np.where(indices_np == e)[0]
            if len(token_ids) == 0:
                continue
            # Get unique source tokens (each token-expert pair maps back)
            src_tokens = token_ids // TOP_K
            x_expert = x_in[mx.array(src_tokens)]  # [n_tokens, hidden_dim]
            out_expert = mx.quantized_matmul(
                x_expert, w[e], s[e], b[e],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            results.append(out_expert)
        return results

    # Warmup
    for _ in range(3):
        r = explicit_per_expert_numpy(x_flat, inds_flat, weight, scales, biases)
        mx.eval(r[-1] if r else mx.array(0))

    times_explicit = []
    for _ in range(5):
        start = time.perf_counter()
        r = explicit_per_expert_numpy(x_flat, inds_flat, weight, scales, biases)
        mx.eval(r[-1] if r else mx.array(0))
        elapsed = time.perf_counter() - start
        times_explicit.append(elapsed)

    explicit_ms = np.mean(times_explicit) * 1000
    print(f"  Time: {explicit_ms:.2f} ms +/- {np.std(times_explicit)*1000:.2f} ms")

    # ==================== Method 3: Explicit per-expert with mx.vmap (if possible) ====================
    # Skip this for now - vmap may not work well with quantized ops

    # ==================== Method 4: Grouped matmuls using contiguous chunks ====================
    print("\n--- Method 3: Grouped contiguous matmuls (sorted indices, explicit chunks) ---")

    def grouped_matmuls(x_sorted_in, idx_sorted_in, w, s, b):
        """Process contiguous expert groups as separate matmuls."""
        idx_np = np.array(idx_sorted_in)
        # Find contiguous group boundaries
        changes = np.where(np.diff(idx_np) != 0)[0] + 1
        boundaries = np.concatenate([[0], changes, [len(idx_np)]])

        results = []
        for i in range(len(boundaries) - 1):
            si = int(boundaries[i])
            ei = int(boundaries[i + 1])
            expert_id = int(idx_np[si])
            # x_sorted_in has shape (N, 1, 1, hidden_dim) after gather_sort
            chunk = x_sorted_in[si:ei].reshape(ei - si, -1)  # [n_tokens, hidden_dim]
            out = mx.quantized_matmul(
                chunk, w[expert_id], s[expert_id], b[expert_id],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            results.append(out)
        return results

    # Warmup
    for _ in range(3):
        r = grouped_matmuls(x_sorted, idx_sorted, weight, scales, biases)
        mx.eval(r[-1] if r else mx.array(0))

    times_grouped = []
    for _ in range(10):
        start = time.perf_counter()
        r = grouped_matmuls(x_sorted, idx_sorted, weight, scales, biases)
        mx.eval(r[-1] if r else mx.array(0))
        elapsed = time.perf_counter() - start
        times_grouped.append(elapsed)

    grouped_ms = np.mean(times_grouped) * 1000
    print(f"  Time: {grouped_ms:.2f} ms +/- {np.std(times_grouped)*1000:.2f} ms")

    # ==================== Summary ====================
    print(f"\n--- Summary for seq_len={seq_len} ---")
    print(f"  gather_qmm (sorted):     {gather_ms:8.2f} ms")
    print(f"  Explicit per-expert:     {explicit_ms:8.2f} ms")
    print(f"  Grouped contiguous:      {grouped_ms:8.2f} ms")
    if gather_ms > 0:
        print(f"  Explicit/gather ratio:   {explicit_ms/gather_ms:.2f}x")
        print(f"  Grouped/gather ratio:    {grouped_ms/gather_ms:.2f}x")

print("\nDone!")
