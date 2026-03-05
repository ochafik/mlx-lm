#!/usr/bin/env python3
"""Benchmark full model forward pass. No monkey-patching.

Uses the model exactly as loaded (fused or unfused, depending on code state).
Designed for comparing git HEAD (fused) vs git stash (unfused) via hyperfine.

Usage:
  python bench_forward.py --seq-len 512 -n 20 -w 5
  hyperfine 'python bench_forward.py --seq-len 1024 -n 15 -w 3'
"""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import argparse
import json
import time

import mlx.core as mx
import numpy as np

from mlx_lm.utils import load_model, _download

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("-n", "--iterations", type=int, default=15)
    parser.add_argument("-w", "--warmup", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mx.random.seed(args.seed)

    print(f"Loading {MODEL_NAME}...", file=sys.stderr, flush=True)
    model_path = _download(MODEL_NAME)
    model, _ = load_model(model_path, lazy=False, strict=False)

    # Check if fused
    has_fused = False
    for layer in model.language_model.model.layers:
        if hasattr(layer.mlp, 'switch_mlp'):
            switch = layer.mlp.switch_mlp
            has_fused = hasattr(switch, 'gate_up_proj')
            break
    mode = "fused" if has_fused else "unfused"
    print(f"Mode: {mode}", file=sys.stderr, flush=True)

    tokens = mx.random.randint(0, 10000, (1, args.seq_len))
    mx.eval(tokens)

    # Warmup
    for _ in range(args.warmup):
        mx.eval(model(tokens))

    # Timed iterations
    times = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        mx.eval(model(tokens))
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    a = np.array(times)
    stats = {
        "mode": mode,
        "seq_len": args.seq_len,
        "n": len(a),
        "min": float(np.min(a)),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
    }

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(f"\n  Full model forward | {mode} | seq_len={args.seq_len} | n={len(a)}")
        print(f"  median={stats['median']:.1f}ms  p25={stats['p25']:.1f}ms  "
              f"p75={stats['p75']:.1f}ms  IQR={stats['iqr']:.1f}ms  min={stats['min']:.1f}ms")


if __name__ == "__main__":
    main()
