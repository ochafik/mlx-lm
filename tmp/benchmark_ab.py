"""Interleaved A/B benchmark: fused vs unfused gate+up in actual MoE block.

For each seq_len, alternates between fused and unfused calls to control
for thermal drift and system load variation.
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load_model, _download
from mlx_lm.models.switch_layers import (
    SwitchGLU, _gather_sort, _scatter_unsort
)

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

print("Loading model...")
model_path = _download(MODEL_NAME)
model, _ = load_model(model_path, lazy=False, strict=False)
print("Model loaded!")

# Find a MoE layer
moe_layer = None
for i, layer in enumerate(model.language_model.model.layers):
    if hasattr(layer.mlp, 'switch_mlp'):
        moe_layer = layer.mlp
        switch = moe_layer.switch_mlp
        print(f"Using MoE layer {i}")
        break

assert moe_layer is not None
assert hasattr(switch, 'gate_up_proj')

# Extract weights
fused_w = switch.gate_up_proj.weight
fused_s = switch.gate_up_proj.scales
fused_b = switch.gate_up_proj.get("biases", None)
half_w = fused_w.shape[1] // 2
half_s = fused_s.shape[1] // 2

gate_weight = fused_w[:, :half_w, :]
up_weight = fused_w[:, half_w:, :]
gate_scales = fused_s[:, :half_s, :]
up_scales = fused_s[:, half_s:, :]
gate_biases = up_biases = None
if fused_b is not None:
    half_b = fused_b.shape[1] // 2
    gate_biases = fused_b[:, :half_b, :]
    up_biases = fused_b[:, half_b:, :]

mx.eval(gate_weight, up_weight, gate_scales, up_scales)
if gate_biases is not None:
    mx.eval(gate_biases, up_biases)

group_size = switch.gate_up_proj.group_size
bits = switch.gate_up_proj.bits
mode_q = getattr(switch.gate_up_proj, 'mode', 'affine')
down_proj = switch.down_proj
activation = switch.activation


def call_fused(self, x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    gate_up = mx.gather_qmm(
        x, fused_w, fused_s, fused_b,
        rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, mode=mode_q,
        sorted_indices=do_sort,
    )
    x_gate, x_up = mx.split(gate_up, 2, axis=-1)
    x = down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


def call_unfused(self, x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    x_up = mx.gather_qmm(
        x, up_weight, up_scales, up_biases,
        rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, mode=mode_q,
        sorted_indices=do_sort,
    )
    x_gate = mx.gather_qmm(
        x, gate_weight, gate_scales, gate_biases,
        rhs_indices=idx, transpose=True,
        group_size=group_size, bits=bits, mode=mode_q,
        sorted_indices=do_sort,
    )
    x = down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


N_WARMUP = 10
N_ITER = 40

print(f"\nInterleaved benchmark: {N_WARMUP} warmup + {N_ITER} iterations per mode per seq_len")
print("=" * 75)
print(f"{'seq_len':>8} {'unfused(3)':>12} {'fused(2)':>12} {'change':>10} {'note':>20}")
print("-" * 75)

for seq_len in [1, 64, 128, 256, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, 2048))
    mx.eval(x)

    # Warmup both modes
    for fn in [call_fused, call_unfused]:
        SwitchGLU.__call__ = fn
        for _ in range(N_WARMUP):
            out = moe_layer(x)
            mx.eval(out)

    # Interleaved measurement
    fused_times = []
    unfused_times = []

    for i in range(N_ITER):
        # Alternate which goes first to avoid ordering bias
        if i % 2 == 0:
            order = [(call_fused, fused_times), (call_unfused, unfused_times)]
        else:
            order = [(call_unfused, unfused_times), (call_fused, fused_times)]

        for fn, times_list in order:
            SwitchGLU.__call__ = fn
            start = time.perf_counter()
            out = moe_layer(x)
            mx.eval(out)
            elapsed = time.perf_counter() - start
            times_list.append(elapsed)

    # Stats (trimmed mean, exclude top/bottom 5)
    fused_times.sort()
    unfused_times.sort()
    trim = 5
    f_trimmed = fused_times[trim:-trim]
    u_trimmed = unfused_times[trim:-trim]

    f_median = np.median(fused_times) * 1000
    u_median = np.median(unfused_times) * 1000
    f_mean = np.mean(f_trimmed) * 1000
    u_mean = np.mean(u_trimmed) * 1000
    f_std = np.std(f_trimmed) * 1000
    u_std = np.std(u_trimmed) * 1000

    change = (f_median - u_median) / u_median * 100

    # Check if change is statistically significant (rough: > 2*combined_std)
    combined_std = (f_std**2 + u_std**2)**0.5
    sig = abs(f_mean - u_mean) > 2 * combined_std
    note = "significant" if sig else "within noise"

    print(f"{seq_len:8d} {u_median:10.2f}ms {f_median:10.2f}ms {change:+9.1f}% {note:>20}")
    print(f"{'':8s} {'':>3s}(±{u_std:.2f})    {'':>3s}(±{f_std:.2f})")

print("=" * 75)
print("\nDone!")
