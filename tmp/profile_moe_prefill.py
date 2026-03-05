"""
Profile MoE expert computation during prefill for Qwen3.5-35B-A3B-4bit.
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

print(f"Loading model: {MODEL_NAME}")
from mlx_lm.utils import load_model, load_tokenizer, _download
model_path = _download(MODEL_NAME)
model, config = load_model(model_path, lazy=False, strict=False)
tokenizer = load_tokenizer(model_path)

# Navigate to the language model
lm = model
if hasattr(model, 'language_model'):
    lm = model.language_model

layers = lm.layers if hasattr(lm, 'layers') else lm.model.layers

# Find MoE layers
moe_layers = []
for i, layer in enumerate(layers):
    mlp = layer.mlp
    if hasattr(mlp, 'switch_mlp'):
        moe_layers.append((i, mlp))

print(f"Found {len(moe_layers)} MoE layers")
if moe_layers:
    first_moe = moe_layers[0][1]
    print(f"  num_experts: {first_moe.num_experts}")
    print(f"  top_k: {first_moe.top_k}")
    gp = first_moe.switch_mlp.gate_proj
    if hasattr(gp, 'scales'):
        print(f"  quantized: bits={gp.bits}, group_size={gp.group_size}")
    g = first_moe.gate
    print(f"  gate weight shape: {g.weight.shape}")
    if hasattr(g, 'scales'):
        actual_input = g.scales.shape[-1] * g.group_size
        print(f"  gate actual input_dims: {actual_input}")

# Profile with different sequence lengths
print("\n" + "="*80)
print("PROFILING MoE BLOCK (single layer)")
print("="*80)

moe_block = moe_layers[0][1]
# Get the actual hidden_size (gate may be quantized)
if hasattr(moe_block.gate, 'input_dims'):
    hidden_size = moe_block.gate.input_dims
else:
    hidden_size = moe_block.gate.weight.shape[1]
# If quantized, the weight shape is compressed; compute actual input dims
gate_w = moe_block.gate
if hasattr(gate_w, 'scales'):
    hidden_size = gate_w.scales.shape[-1] * gate_w.group_size
print(f"  hidden_size (computed): {hidden_size}")

for seq_len in [1, 16, 64, 128, 256, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, hidden_size))
    mx.eval(x)

    # Warmup
    for _ in range(3):
        y = moe_block(x)
        mx.eval(y)

    # Timed runs
    times = []
    for _ in range(10):
        start = time.perf_counter()
        y = moe_block(x)
        mx.eval(y)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    tokens_per_sec = seq_len / np.mean(times)

    print(f"  seq_len={seq_len:5d}: {mean_ms:8.2f} ms +/- {std_ms:5.2f} ms  "
          f"({tokens_per_sec:10.0f} tok/s)")

# Now profile just the switch_mlp part vs routing
print("\n" + "="*80)
print("PROFILING BREAKDOWN: Routing vs Expert Computation vs Shared Expert")
print("="*80)

for seq_len in [128, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, hidden_size))
    mx.eval(x)

    # Profile routing (gate + argpartition + scoring)
    times_routing = []
    for _ in range(10):
        start = time.perf_counter()
        gates = moe_block.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = moe_block.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if moe_block.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        mx.eval(inds, scores)
        elapsed = time.perf_counter() - start
        times_routing.append(elapsed)

    # Profile expert MLP (switch_mlp only)
    gates = moe_block.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = moe_block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if moe_block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    mx.eval(inds, scores)

    times_expert = []
    for _ in range(10):
        start = time.perf_counter()
        y = moe_block.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        mx.eval(y)
        elapsed = time.perf_counter() - start
        times_expert.append(elapsed)

    # Profile shared expert
    times_shared = []
    for _ in range(10):
        start = time.perf_counter()
        shared_y = moe_block.shared_expert(x)
        shared_y = mx.sigmoid(moe_block.shared_expert_gate(x)) * shared_y
        mx.eval(shared_y)
        elapsed = time.perf_counter() - start
        times_shared.append(elapsed)

    r_ms = np.mean(times_routing) * 1000
    e_ms = np.mean(times_expert) * 1000
    s_ms = np.mean(times_shared) * 1000
    total = r_ms + e_ms + s_ms

    print(f"\n  seq_len={seq_len}:")
    print(f"    Routing:       {r_ms:8.2f} ms ({r_ms/total*100:5.1f}%)")
    print(f"    Expert MLP:    {e_ms:8.2f} ms ({e_ms/total*100:5.1f}%)")
    print(f"    Shared Expert: {s_ms:8.2f} ms ({s_ms/total*100:5.1f}%)")
    print(f"    Total:         {total:8.2f} ms")

# Token distribution analysis
print("\n" + "="*80)
print("TOKEN DISTRIBUTION ACROSS EXPERTS")
print("="*80)

for seq_len in [128, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, hidden_size))
    gates = moe_block.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = moe_block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    mx.eval(inds)

    inds_np = np.array(inds[0])
    expert_counts = np.bincount(inds_np.flatten(), minlength=moe_block.num_experts)

    print(f"\n  seq_len={seq_len}, top_k={k}, num_experts={moe_block.num_experts}:")
    print(f"    Token-expert pairs: {inds_np.size}")
    print(f"    Mean/Std tokens/expert: {expert_counts.mean():.1f} / {expert_counts.std():.1f}")
    print(f"    Min/Max: {expert_counts.min()} / {expert_counts.max()}")
    print(f"    Experts with 0 tokens: {(expert_counts == 0).sum()}")
    print(f"    p50/p90/p99: {np.percentile(expert_counts, 50):.0f} / {np.percentile(expert_counts, 90):.0f} / {np.percentile(expert_counts, 99):.0f}")

    # How well would expert-batching work? Show group sizes
    sorted_inds = np.sort(inds_np.flatten())
    changes = np.where(np.diff(sorted_inds) != 0)[0]
    group_sizes = np.diff(np.concatenate([[0], changes + 1, [len(sorted_inds)]]))
    print(f"    Expert group sizes (after sort): mean={group_sizes.mean():.1f}, std={group_sizes.std():.1f}")
    print(f"    Group size p10/p50/p90: {np.percentile(group_sizes, 10):.0f} / {np.percentile(group_sizes, 50):.0f} / {np.percentile(group_sizes, 90):.0f}")

print("\nDone!")
