#!/usr/bin/env python3
"""Compare interleaved vs batched benchmarking to check if interleaving masks speedup."""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import random, time
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model, _download
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort

def get_moe_layer(model):
    layers = model.layers if hasattr(model, 'layers') else model.language_model.model.layers
    for layer in layers:
        if hasattr(layer.mlp, 'switch_mlp'):
            return layer.mlp
    raise RuntimeError("No MoE layer found")

def get_hidden_size(model):
    if hasattr(model, 'args') and hasattr(model.args, 'hidden_size'):
        return model.args.hidden_size
    moe = get_moe_layer(model)
    s = moe.switch_mlp
    return s.gate_up_proj.input_dims if hasattr(s, 'gate_up_proj') else s.down_proj.output_dims

def _fused_call(self, x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx, inv_order = indices, None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
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

def _unfused_call(self, x, indices):
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx, inv_order = indices, None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    if hasattr(self, "gate_up_proj"):
        proj = self.gate_up_proj
        w, s = proj["weight"], proj["scales"]
        b = proj.get("biases")
        gs, bits = proj.group_size, proj.bits
        mode = getattr(proj, 'mode', 'affine')
        hw, hs = w.shape[1]//2, s.shape[1]//2
        gb, ub = (b[:,:b.shape[1]//2,:], b[:,b.shape[1]//2:,:]) if b is not None else (None, None)
        x_gate = mx.gather_qmm(x, w[:,:hw,:], s[:,:hs,:], gb, rhs_indices=idx, transpose=True, group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort)
        x_up = mx.gather_qmm(x, w[:,hw:,:], s[:,hs:,:], ub, rhs_indices=idx, transpose=True, group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort)
    else:
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
    x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen3.5-122B-A10B-4bit"

print(f"Loading {MODEL}...", file=sys.stderr, flush=True)
model_path = _download(MODEL)
model, _ = load_model(model_path, lazy=False, strict=False)
moe = get_moe_layer(model)
hidden = get_hidden_size(model)
print(f"Loaded. hidden_size={hidden}", file=sys.stderr, flush=True)

N_WARMUP = 20
N_ITER = 80

print(f"\n{'=' * 90}")
print(f"  Interleaved vs Batched comparison | {MODEL}")
print(f"  {N_WARMUP} warmup + {N_ITER} iterations per mode")
print(f"{'=' * 90}")

for seq_len in [128, 256, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, hidden))
    mx.eval(x)

    # --- Batched: all fused, then all unfused ---
    SwitchGLU.__call__ = _fused_call
    for _ in range(N_WARMUP):
        mx.eval(moe(x))
    batched_fused = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        mx.eval(moe(x))
        batched_fused.append((time.perf_counter() - t0) * 1000)

    SwitchGLU.__call__ = _unfused_call
    for _ in range(N_WARMUP):
        mx.eval(moe(x))
    batched_unfused = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        mx.eval(moe(x))
        batched_unfused.append((time.perf_counter() - t0) * 1000)

    # --- Interleaved: alternate every iteration ---
    for fn in [_fused_call, _unfused_call]:
        SwitchGLU.__call__ = fn
        for _ in range(N_WARMUP):
            mx.eval(moe(x))

    inter_fused, inter_unfused = [], []
    for i in range(N_ITER):
        if i % 2 == 0:
            pairs = [(_fused_call, inter_fused), (_unfused_call, inter_unfused)]
        else:
            pairs = [(_unfused_call, inter_unfused), (_fused_call, inter_fused)]
        for fn, tlist in pairs:
            SwitchGLU.__call__ = fn
            t0 = time.perf_counter()
            mx.eval(moe(x))
            tlist.append((time.perf_counter() - t0) * 1000)

    bf = np.median(batched_fused)
    bu = np.median(batched_unfused)
    intf = np.median(inter_fused)
    intu = np.median(inter_unfused)

    sp_batched = (bu - bf) / bu * 100
    sp_inter = (intu - intf) / intu * 100

    print(f"\n  seq_len={seq_len}")
    print(f"  {'':14s} {'unfused':>10} {'fused':>10} {'speedup':>10}")
    print(f"  {'batched':14s} {bu:8.2f}ms {bf:8.2f}ms {sp_batched:+8.1f}%")
    print(f"  {'interleaved':14s} {intu:8.2f}ms {intf:8.2f}ms {sp_inter:+8.1f}%")

print(f"\n{'=' * 90}")
