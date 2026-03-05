"""Quick interleaved A/B benchmark - focused on key seq_lens.

Runs many more iterations at fewer seq_lens for better statistics.
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model, _download
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

print("Loading model...")
model_path = _download(MODEL_NAME)
model, _ = load_model(model_path, lazy=False, strict=False)
print("Model loaded!")

moe_layer = None
for i, layer in enumerate(model.language_model.model.layers):
    if hasattr(layer.mlp, 'switch_mlp'):
        moe_layer = layer.mlp
        switch = moe_layer.switch_mlp
        print(f"Using MoE layer {i}")
        break

assert moe_layer is not None

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


def run_fused(x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    gate_up = mx.gather_qmm(x, fused_w, fused_s, fused_b,
        rhs_indices=idx, transpose=True, group_size=group_size, bits=bits, mode=mode_q, sorted_indices=do_sort)
    x_gate, x_up = mx.split(gate_up, 2, axis=-1)
    x = down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


def run_unfused(x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    x_up = mx.gather_qmm(x, up_weight, up_scales, up_biases,
        rhs_indices=idx, transpose=True, group_size=group_size, bits=bits, mode=mode_q, sorted_indices=do_sort)
    x_gate = mx.gather_qmm(x, gate_weight, gate_scales, gate_biases,
        rhs_indices=idx, transpose=True, group_size=group_size, bits=bits, mode=mode_q, sorted_indices=do_sort)
    x = down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


N_WARMUP = 15
N_ITER = 60

print(f"\nInterleaved benchmark: {N_WARMUP} warmup + {N_ITER} iterations")
print("NOTE: Other GPU processes running - numbers will be noisy")
print("=" * 75)

for seq_len in [128, 256, 512, 1024]:
    x = mx.random.normal((1, seq_len, 2048))
    gates = mx.random.normal((1, seq_len, 256))
    inds = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
    mx.eval(x, inds)

    # Warmup both
    for _ in range(N_WARMUP):
        out = run_fused(x, inds); mx.eval(out)
        out = run_unfused(x, inds); mx.eval(out)

    fused_times = []
    unfused_times = []

    for i in range(N_ITER):
        if i % 2 == 0:
            start = time.perf_counter(); out = run_fused(x, inds); mx.eval(out)
            fused_times.append(time.perf_counter() - start)
            start = time.perf_counter(); out = run_unfused(x, inds); mx.eval(out)
            unfused_times.append(time.perf_counter() - start)
        else:
            start = time.perf_counter(); out = run_unfused(x, inds); mx.eval(out)
            unfused_times.append(time.perf_counter() - start)
            start = time.perf_counter(); out = run_fused(x, inds); mx.eval(out)
            fused_times.append(time.perf_counter() - start)

    # Use p25 as a more stable estimator under contention
    f_p25 = np.percentile(fused_times, 25) * 1000
    u_p25 = np.percentile(unfused_times, 25) * 1000
    f_med = np.median(fused_times) * 1000
    u_med = np.median(unfused_times) * 1000
    f_min = np.min(fused_times) * 1000
    u_min = np.min(unfused_times) * 1000

    change_p25 = (f_p25 - u_p25) / u_p25 * 100
    change_med = (f_med - u_med) / u_med * 100
    change_min = (f_min - u_min) / u_min * 100

    print(f"\nseq_len={seq_len}")
    print(f"  {'':12s} {'unfused(3)':>12} {'fused(2)':>12} {'change':>10}")
    print(f"  {'min':12s} {u_min:10.2f}ms {f_min:10.2f}ms {change_min:+9.1f}%")
    print(f"  {'p25':12s} {u_p25:10.2f}ms {f_p25:10.2f}ms {change_p25:+9.1f}%")
    print(f"  {'median':12s} {u_med:10.2f}ms {f_med:10.2f}ms {change_med:+9.1f}%")

print("\n" + "=" * 75)
print("Done!")
