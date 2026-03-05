"""Benchmark different sort thresholds for gather_qmm sorted_indices.

Current threshold: indices.size >= 64
Test: What threshold gives best performance across seq_lens?

For Qwen3.5: top_k=8, num_experts=256
- seq_len=1: indices.size = 8 (no sort)
- seq_len=8: indices.size = 64 (threshold!)
- seq_len=16: indices.size = 128
- seq_len=64: indices.size = 512
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

# Save original __call__
original_call = SwitchGLU.__call__

N_WARMUP = 10
N_ITER = 40

# Test thresholds: never sort, sort at 64 (current), sort at 128, sort at 256, always sort
thresholds = [
    ("never", float('inf')),
    (">=64", 64),
    (">=128", 128),
    (">=256", 256),
    (">=512", 512),
    ("always", 0),
]

# Patch __call__ to use a configurable threshold
def make_call(threshold):
    def patched_call(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= threshold
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)

        if hasattr(self, "gate_up_proj"):
            gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
            x_gate, x_up = mx.split(gate_up, 2, axis=-1)
        else:
            x_up = self.up_proj(x, idx, sorted_indices=do_sort)
            x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)

        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
    return patched_call


print(f"\nSort threshold benchmark: {N_WARMUP} warmup + {N_ITER} iterations")
print(f"Qwen3.5: 256 experts, top_k=8")
print("=" * 90)

# Header
header = f"{'seq_len':>8} {'idx.size':>8}"
for name, _ in thresholds:
    header += f" {name:>10}"
print(header)
print("-" * 90)

for seq_len in [1, 4, 8, 16, 32, 64, 128, 256, 512]:
    x = mx.random.normal((1, seq_len, 2048))
    gates = mx.random.normal((1, seq_len, 256))
    inds = mx.argpartition(gates, kth=-8, axis=-1)[..., -8:]
    mx.eval(x, inds)

    idx_size = inds.size
    row = f"{seq_len:8d} {idx_size:8d}"

    for name, threshold in thresholds:
        SwitchGLU.__call__ = make_call(threshold)

        # Warmup
        for _ in range(N_WARMUP):
            out = moe_layer(x)
            mx.eval(out)

        # Interleave is hard with many thresholds, use simple sequential
        times = []
        for _ in range(N_ITER):
            start = time.perf_counter()
            out = moe_layer(x)
            mx.eval(out)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        p25_ms = np.percentile(times, 25) * 1000
        row += f" {p25_ms:8.2f}ms"

    print(row)

# Restore
SwitchGLU.__call__ = original_call

print("=" * 90)
print("\nNote: 'never' = no sorting, 'always' = sort even 1 token.")
print("Sort overhead = 2x argsort. Benefits kick in when gather_qmm batched kernel")
print("can group contiguous experts (requires B>=16 and B/E>=4).")
print("For 256 experts: B/E>=4 requires B>=1024 tokens, i.e. seq_len>=128 (top_k=8).")
print("\nDone!")
