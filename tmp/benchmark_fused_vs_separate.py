"""
Benchmark: Fused gate+up (2 gather_qmm) vs separate gate/up (3 gather_qmm).

Compares the actual MoE block performance with fused vs separate projections.
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Qwen3.5-35B-A3B-4bit parameters
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


def gather_sort(x_in, indices):
    *_, M = indices.shape
    indices_flat = indices.flatten()
    order = mx.argsort(indices_flat)
    inv_order = mx.argsort(order)
    return x_in.flatten(0, -3)[order // M], indices_flat[order], inv_order


print("Creating weights...")
# Separate gate and up weights
gate_w, gate_s, gate_b = create_quantized_weights(NUM_EXPERTS, EXPERT_INTERMEDIATE, HIDDEN_DIM)
up_w, up_s, up_b = create_quantized_weights(NUM_EXPERTS, EXPERT_INTERMEDIATE, HIDDEN_DIM)
down_w, down_s, down_b = create_quantized_weights(NUM_EXPERTS, HIDDEN_DIM, EXPERT_INTERMEDIATE)

# Fused gate+up weights (concatenate along output dim)
fused_w = mx.concatenate([gate_w, up_w], axis=1)
fused_s = mx.concatenate([gate_s, up_s], axis=1)
fused_b = mx.concatenate([gate_b, up_b], axis=1)

mx.eval(gate_w, gate_s, gate_b, up_w, up_s, up_b, down_w, down_s, down_b, fused_w, fused_s, fused_b)
print(f"  Separate: gate={gate_w.shape}, up={up_w.shape}")
print(f"  Fused: gate_up={fused_w.shape}")
print(f"  Down: {down_w.shape}")

for seq_len in [128, 256, 512, 1024, 2048]:
    print(f"\n{'='*70}")
    print(f"seq_len={seq_len}, experts={NUM_EXPERTS}, top_k={TOP_K}")
    print(f"{'='*70}")

    x = mx.random.normal((1, seq_len, HIDDEN_DIM))
    gate_logits = mx.random.normal((1, seq_len, NUM_EXPERTS))
    probs = mx.softmax(gate_logits, axis=-1)
    inds = mx.argpartition(probs, kth=-TOP_K, axis=-1)[..., -TOP_K:]
    mx.eval(x, inds)

    x_expanded = mx.expand_dims(x, (-2, -3))
    x_sorted, idx_sorted, inv_order = gather_sort(x_expanded, inds)
    mx.eval(x_sorted, idx_sorted, inv_order)

    # ==================== Method 1: Separate gate + up (3 gather_qmm) ====================
    def run_separate():
        x_gate = mx.gather_qmm(
            x_sorted, gate_w, gate_s, gate_b,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        x_up = mx.gather_qmm(
            x_sorted, up_w, up_s, up_b,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        x_act = nn.silu(x_gate) * x_up
        x_out = mx.gather_qmm(
            x_act, down_w, down_s, down_b,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        return x_out

    # Warmup
    for _ in range(5):
        mx.eval(run_separate())

    times_separate = []
    for _ in range(20):
        start = time.perf_counter()
        mx.eval(run_separate())
        elapsed = time.perf_counter() - start
        times_separate.append(elapsed)
    separate_ms = np.mean(times_separate) * 1000

    # ==================== Method 2: Fused gate+up (2 gather_qmm) ====================
    def run_fused():
        gate_up = mx.gather_qmm(
            x_sorted, fused_w, fused_s, fused_b,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        x_gate, x_up = mx.split(gate_up, 2, axis=-1)
        x_act = nn.silu(x_gate) * x_up
        x_out = mx.gather_qmm(
            x_act, down_w, down_s, down_b,
            rhs_indices=idx_sorted, transpose=True,
            group_size=GROUP_SIZE, bits=BITS, sorted_indices=True,
        )
        return x_out

    # Warmup
    for _ in range(5):
        mx.eval(run_fused())

    times_fused = []
    for _ in range(20):
        start = time.perf_counter()
        mx.eval(run_fused())
        elapsed = time.perf_counter() - start
        times_fused.append(elapsed)
    fused_ms = np.mean(times_fused) * 1000

    speedup = separate_ms / fused_ms
    saved = separate_ms - fused_ms
    print(f"  Separate (3 gather_qmm): {separate_ms:8.2f} ms")
    print(f"  Fused    (2 gather_qmm): {fused_ms:8.2f} ms")
    print(f"  Speedup: {speedup:.2f}x ({saved:.2f} ms saved)")

print("\nDone!")
