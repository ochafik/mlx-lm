#!/usr/bin/env python3
"""Batched (non-interleaved) MoE block benchmark with error bars.

Runs all-fused then all-unfused (batched), which reflects real-world
performance where the GPU cache is warm for one path.
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
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
        hw, hs = w.shape[1] // 2, s.shape[1] // 2
        gb, ub = (b[:, :b.shape[1]//2, :], b[:, b.shape[1]//2:, :]) if b is not None else (None, None)
        x_gate = mx.gather_qmm(x, w[:, :hw, :], s[:, :hs, :], gb,
                                rhs_indices=idx, transpose=True,
                                group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort)
        x_up = mx.gather_qmm(x, w[:, hw:, :], s[:, hs:, :], ub,
                              rhs_indices=idx, transpose=True,
                              group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort)
    else:
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
    x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


def run_batched(moe, x, fn, n_warmup, n_iter):
    SwitchGLU.__call__ = fn
    for _ in range(n_warmup):
        mx.eval(moe(x))
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        mx.eval(moe(x))
        times.append((time.perf_counter() - t0) * 1000)
    return times


def stats(times):
    a = np.array(times)
    return {
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "min": float(np.min(a)),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[1, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    parser.add_argument("-n", "--iterations", type=int, default=60)
    parser.add_argument("-w", "--warmup", type=int, default=15)
    args = parser.parse_args()

    print(f"Loading {args.model}...", file=sys.stderr, flush=True)
    model_path = _download(args.model)
    model, _ = load_model(model_path, lazy=False, strict=False)
    moe = get_moe_layer(model)
    hidden = get_hidden_size(model)
    print(f"Loaded. hidden_size={hidden}", file=sys.stderr, flush=True)

    # Verify correctness
    x_test = mx.random.normal((1, 16, hidden)); mx.eval(x_test)
    SwitchGLU.__call__ = _fused_call
    out_f = moe(x_test); mx.eval(out_f)
    SwitchGLU.__call__ = _unfused_call
    out_u = moe(x_test); mx.eval(out_u)
    diff = mx.abs(out_f - out_u).max().item()
    print(f"Correctness: max diff = {diff:.2e}", file=sys.stderr, flush=True)
    assert diff < 1e-2

    short_model = args.model.split("/")[-1]
    print(f"\n{'=' * 95}")
    print(f"  Batched MoE Block Benchmark (GPU-cache-warm, {args.iterations} iter, {args.warmup} warmup)")
    print(f"  Model: {args.model}  (hidden_size={hidden})")
    print(f"{'=' * 95}")
    print(f"{'seq_len':>8} │ {'unfused median':>15} {'fused median':>15} │ {'speedup':>8} │ {'unfused σ':>10} {'fused σ':>10}")
    print(f"{'─' * 8}─┼─{'─' * 15}─{'─' * 15}─┼─{'─' * 8}─┼─{'─' * 10}─{'─' * 10}")

    for seq_len in args.seq_lens:
        x = mx.random.normal((1, seq_len, hidden))
        mx.eval(x)

        # unfused first, then fused, then unfused again (A-B-A for drift check)
        u1 = run_batched(moe, x, _unfused_call, args.warmup, args.iterations)
        f_ = run_batched(moe, x, _fused_call, args.warmup, args.iterations)
        u2 = run_batched(moe, x, _unfused_call, args.warmup, args.iterations)

        su1, sf, su2 = stats(u1), stats(f_), stats(u2)

        # Average both unfused runs for baseline
        u_med = (su1["median"] + su2["median"]) / 2
        u_std = (su1["std"] + su2["std"]) / 2
        f_med = sf["median"]
        f_std = sf["std"]
        drift = abs(su1["median"] - su2["median"]) / su1["median"] * 100
        speedup = (u_med - f_med) / u_med * 100

        flag = " *" if drift > 10 else ""
        print(f"{seq_len:8d} │ {u_med:11.2f} ms {f_med:11.2f} ms │ {speedup:+7.1f}% │ {u_std:7.2f} ms {f_std:7.2f} ms{flag}")

    print(f"{'─' * 95}")
    print(f"  Positive % = fused is faster. σ = standard deviation. * = drift > 10% (unstable).")
    print()


if __name__ == "__main__":
    main()
