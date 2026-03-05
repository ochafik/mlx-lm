"""
Micro-benchmark v2: Compare gather_qmm vs explicit per-expert matmuls.
Fixed: properly eval ALL results for fair comparison.
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
EXPERT_INTERMEDIATE = 1024
BITS = 4
GROUP_SIZE = 64

def create_quantized_weights(num_experts, out_dim, in_dim, bits=4, group_size=64):
    w = mx.random.normal((num_experts, out_dim, in_dim))
    weight, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return weight, scales, biases

print("Creating weights...")
weight, scales, biases = create_quantized_weights(NUM_EXPERTS, EXPERT_INTERMEDIATE, HIDDEN_DIM)
mx.eval(weight, scales, biases)
print(f"  weight: {weight.shape}, scales: {scales.shape}")

def gather_sort(x_in, indices):
    *_, M = indices.shape
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    return x_in.flatten(0, -3)[order // M], indices_flat[order], inv_order

for seq_len in [128, 256, 512, 1024, 2048, 4096]:
    print(f"\n{'='*80}")
    print(f"seq_len={seq_len}, num_experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"  token-expert pairs: {seq_len * TOP_K}")
    print(f"  avg tokens/expert: {seq_len * TOP_K / NUM_EXPERTS:.1f}")
    print(f"{'='*80}")

    x = mx.random.normal((1, seq_len, HIDDEN_DIM))
    gate_logits = mx.random.normal((1, seq_len, NUM_EXPERTS))
    probs = mx.softmax(gate_logits, axis=-1)
    inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    mx.eval(x, inds)

    x_expanded = mx.expand_dims(x, (-2, -3))
    x_sorted, idx_sorted, inv_order = gather_sort(x_expanded, inds)
    mx.eval(x_sorted, idx_sorted, inv_order)

    # ==================== Method 1: gather_qmm ====================
    # Warmup
    for _ in range(5):
        out = mx.gather_qmm(
            x_sorted, weight, scales, biases,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        mx.eval(out)

    times_gather = []
    for _ in range(20):
        start = time.perf_counter()
        out = mx.gather_qmm(
            x_sorted, weight, scales, biases,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times_gather.append(elapsed)
    gather_ms = np.mean(times_gather) * 1000

    # ==================== Method 2: Explicit grouped matmuls ====================
    idx_np = np.array(idx_sorted)
    changes = np.where(np.diff(idx_np) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [len(idx_np)]])
    num_groups = len(boundaries) - 1

    # Pre-compute group info
    group_starts = [int(boundaries[i]) for i in range(num_groups)]
    group_ends = [int(boundaries[i+1]) for i in range(num_groups)]
    group_expert_ids = [int(idx_np[group_starts[i]]) for i in range(num_groups)]

    def run_grouped():
        results = []
        for i in range(num_groups):
            chunk = x_sorted[group_starts[i]:group_ends[i]].reshape(
                group_ends[i] - group_starts[i], -1
            )
            out = mx.quantized_matmul(
                chunk, weight[group_expert_ids[i]], scales[group_expert_ids[i]],
                biases[group_expert_ids[i]],
                transpose=True, group_size=GROUP_SIZE, bits=BITS,
            )
            results.append(out)
        return results

    # Warmup
    for _ in range(5):
        r = run_grouped()
        mx.eval(*r)  # Eval ALL results

    times_grouped = []
    for _ in range(20):
        start = time.perf_counter()
        r = run_grouped()
        mx.eval(*r)  # Eval ALL results
        elapsed = time.perf_counter() - start
        times_grouped.append(elapsed)
    grouped_ms = np.mean(times_grouped) * 1000

    # ==================== Method 3: gather_qmm unsorted for comparison ====================
    # Prepare unsorted input matching gather_sort output shape but without sorting
    x_unsorted = mx.expand_dims(x, (-2, -3))  # [1, seq_len, 1, 1, hidden]
    inds_flat_unsorted = inds.flatten()  # [seq_len * top_k]
    # Need to replicate x for each top_k selection
    x_flat_unsorted = x_unsorted.flatten(0, -3)  # [seq_len, 1, hidden]
    # Replicate each token top_k times
    x_rep = mx.repeat(x_flat_unsorted, TOP_K, axis=0)  # [seq_len*top_k, 1, hidden]

    times_unsorted = []
    for _ in range(5):
        out = mx.gather_qmm(
            x_rep, weight, scales, biases,
            rhs_indices=inds_flat_unsorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=False,
        )
        mx.eval(out)

    for _ in range(20):
        start = time.perf_counter()
        out = mx.gather_qmm(
            x_rep, weight, scales, biases,
            rhs_indices=inds_flat_unsorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=False,
        )
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times_unsorted.append(elapsed)
    unsorted_ms = np.mean(times_unsorted) * 1000

    # ==================== Summary ====================
    print(f"  gather_qmm (sorted):     {gather_ms:8.2f} ms")
    print(f"  gather_qmm (unsorted):   {unsorted_ms:8.2f} ms")
    print(f"  Grouped explicit:        {grouped_ms:8.2f} ms  ({num_groups} groups)")
    print(f"  Speedup (grouped/sorted): {gather_ms/grouped_ms:.2f}x")
    print(f"  Speedup (grouped/unsorted): {unsorted_ms/grouped_ms:.2f}x")

    # Group size distribution
    group_sizes = np.array([group_ends[i] - group_starts[i] for i in range(num_groups)])
    print(f"  Group sizes: mean={group_sizes.mean():.1f}, min={group_sizes.min()}, max={group_sizes.max()}")

print("\nDone!")
