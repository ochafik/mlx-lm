#!/usr/bin/env python3
"""Rigorous MoE gate+up fusion benchmark.

Two benchmarking modes:
  --level block   : Single MoE block, interleaved fused/unfused (fast, low noise)
  --level model   : Full model forward pass, single mode per invocation (use with hyperfine)

For full model, use hyperfine:
  hyperfine --warmup 1 \
    'python bench_moe_prefill.py --level model --mode unfused --seq-len 1024' \
    'python bench_moe_prefill.py --level model --mode fused --seq-len 1024'

Unfused mode splits fused weights on the fly (zero extra memory).
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import argparse
import json
import random
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.utils import load_model, _download
from mlx_lm.models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"


# ---- SwitchGLU __call__ variants ----

def _fused_call(self, x, indices):
    """Standard fused path: 2 gather_qmm calls."""
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
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


def _unfused_call(self, x, indices):
    """Unfused path: 3 gather_qmm calls. Splits fused weights on the fly."""
    x = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx = indices
    inv_order = None
    if do_sort:
        x, idx, inv_order = _gather_sort(x, indices)
    if self.training:
        idx = mx.stop_gradient(idx)

    if hasattr(self, "gate_up_proj"):
        # Split fused weights into 2 separate gather_qmm calls (3 total with down)
        proj = self.gate_up_proj
        w = proj["weight"]
        s = proj["scales"]
        b = proj.get("biases")
        gs = proj.group_size
        bits = proj.bits
        mode = getattr(proj, 'mode', 'affine')
        half_w = w.shape[1] // 2
        half_s = s.shape[1] // 2

        if b is not None:
            half_b = b.shape[1] // 2
            gate_b = b[:, :half_b, :]
            up_b = b[:, half_b:, :]
        else:
            gate_b = up_b = None

        x_gate = mx.gather_qmm(
            x, w[:, :half_w, :], s[:, :half_s, :], gate_b,
            rhs_indices=idx, transpose=True,
            group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort,
        )
        x_up = mx.gather_qmm(
            x, w[:, half_w:, :], s[:, half_s:, :], up_b,
            rhs_indices=idx, transpose=True,
            group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort,
        )
    else:
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)

    x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
    if do_sort:
        x = _scatter_unsort(x, inv_order, indices.shape)
    return x.squeeze(-2)


# ---- Benchmark helpers ----

def setup_model():
    model_path = _download(MODEL_NAME)
    model, _ = load_model(model_path, lazy=False, strict=False)
    return model


def get_moe_layer(model):
    for layer in model.language_model.model.layers:
        if hasattr(layer.mlp, 'switch_mlp'):
            return layer.mlp
    raise RuntimeError("No MoE layer found")


def percentile_stats(times_ms):
    a = np.array(times_ms)
    return {
        "min": float(np.min(a)),
        "p5": float(np.percentile(a, 5)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "n": len(a),
    }


def verify_correctness(model):
    """Verify fused and unfused produce identical output."""
    tokens = mx.array([[1, 2, 3, 4, 5]])

    SwitchGLU.__call__ = _fused_call
    out_f = model(tokens)
    mx.eval(out_f)

    SwitchGLU.__call__ = _unfused_call
    out_u = model(tokens)
    mx.eval(out_u)

    diff = mx.abs(out_f - out_u).max().item()
    assert diff < 1e-2, f"Outputs differ: max diff = {diff}"
    return diff


# ---- Block-level benchmark (interleaved) ----

def bench_block(model, seq_lens, n_warmup, n_iter):
    """Single MoE block, interleaved fused/unfused per iteration."""
    moe_layer = get_moe_layer(model)
    results = {}

    for seq_len in seq_lens:
        x = mx.random.normal((1, seq_len, 2048))
        mx.eval(x)

        # Warmup both paths
        for fn in [_fused_call, _unfused_call]:
            SwitchGLU.__call__ = fn
            for _ in range(n_warmup):
                mx.eval(moe_layer(x))

        # Interleaved measurement
        fused_times, unfused_times = [], []
        for i in range(n_iter):
            if random.random() < 0.5:
                pairs = [(_fused_call, fused_times), (_unfused_call, unfused_times)]
            else:
                pairs = [(_unfused_call, unfused_times), (_fused_call, fused_times)]

            for fn, tlist in pairs:
                SwitchGLU.__call__ = fn
                start = time.perf_counter()
                mx.eval(moe_layer(x))
                tlist.append((time.perf_counter() - start) * 1000)

        f_stats = percentile_stats(fused_times)
        u_stats = percentile_stats(unfused_times)
        speedup = (u_stats["median"] - f_stats["median"]) / u_stats["median"] * 100

        results[seq_len] = {"fused": f_stats, "unfused": u_stats, "speedup_pct": round(speedup, 1)}

    return results


# ---- Model-level benchmark (single mode, for hyperfine) ----

def bench_model(model, mode, seq_len, n_warmup, n_iter):
    """Full model forward pass, single mode. Designed for hyperfine wrapping."""
    if mode != "original":
        fn = _fused_call if mode == "fused" else _unfused_call
        SwitchGLU.__call__ = fn

    tokens = mx.random.randint(0, 10000, (1, seq_len))
    mx.eval(tokens)

    # Warmup
    for _ in range(n_warmup):
        mx.eval(model(tokens))

    # Timed iterations
    times = []
    for _ in range(n_iter):
        start = time.perf_counter()
        mx.eval(model(tokens))
        times.append((time.perf_counter() - start) * 1000)

    stats = percentile_stats(times)
    return stats


# ---- Output ----

def print_block_results(results):
    print(f"\n{'=' * 85}")
    print(f"  Single MoE Block: fused (2 calls) vs unfused (3 calls)")
    print(f"  Model: {MODEL_NAME}")
    print(f"{'=' * 85}")
    print(f"{'seq_len':>8} │ {'unfused':>12} {'fused':>12} │ {'speedup':>8} │ {'IQR u/f':>16}")
    print(f"{'─' * 8}─┼─{'─' * 12}─{'─' * 12}─┼─{'─' * 8}─┼─{'─' * 16}")

    for sl in sorted(results.keys()):
        r = results[sl]
        u, f = r["unfused"], r["fused"]
        sp = r["speedup_pct"]
        print(f"{sl:8d} │ {u['median']:9.2f} ms {f['median']:9.2f} ms │ {sp:+7.1f}% │ {u['iqr']:.2f} / {f['iqr']:.2f}")

    print(f"{'─' * 85}")
    print(f"  Positive % = fused is faster. IQR = interquartile range (noise).\n")


def print_model_result(mode, seq_len, stats):
    print(f"\n  Full model forward pass | mode={mode} | seq_len={seq_len}")
    print(f"  median={stats['median']:.1f}ms  p25={stats['p25']:.1f}ms  p75={stats['p75']:.1f}ms  "
          f"IQR={stats['iqr']:.1f}ms  min={stats['min']:.1f}ms  n={stats['n']}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=["block", "model"], default="block")
    parser.add_argument("--mode", choices=["fused", "unfused", "original"], default="fused",
                        help="For --level model: fused=patched 2-call, unfused=patched 3-call, original=no patching")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="For --level model (single seq_len)")
    parser.add_argument("--iterations", "-n", type=int, default=80)
    parser.add_argument("--warmup", "-w", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    print(f"Loading {MODEL_NAME}...", file=sys.stderr, flush=True)
    model = setup_model()
    print("Loaded.", file=sys.stderr, flush=True)

    diff = verify_correctness(model)
    print(f"Correctness: max diff = {diff:.2e}", file=sys.stderr, flush=True)

    if args.level == "block":
        seq_lens = args.seq_lens or [1, 64, 128, 256, 512, 1024, 2048, 4096]
        results = bench_block(model, seq_lens, args.warmup, args.iterations)
        if args.json:
            print(json.dumps({str(k): v for k, v in results.items()}, indent=2))
        else:
            print_block_results(results)

    elif args.level == "model":
        stats = bench_model(model, args.mode, args.seq_len, args.warmup, args.iterations)
        if args.json:
            print(json.dumps({"mode": args.mode, "seq_len": args.seq_len, **stats}, indent=2))
        else:
            print_model_result(args.mode, args.seq_len, stats)


if __name__ == "__main__":
    main()
