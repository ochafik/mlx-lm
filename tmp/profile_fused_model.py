"""Profile actual MoE block with fused gate+up weights."""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import time
import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model, _download

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

print("Loading model...")
model_path = _download(MODEL_NAME)
model, _ = load_model(model_path, lazy=False, strict=False)
print("Model loaded!")

# Get a representative MoE layer
moe_layer = None
for i, layer in enumerate(model.language_model.model.layers):
    if hasattr(layer.mlp, 'switch_mlp'):
        moe_layer = layer.mlp
        print(f"Using MoE layer {i}")
        break

assert moe_layer is not None

# Profile at different sequence lengths
for seq_len in [1, 16, 64, 128, 256, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, 2048))
    mx.eval(x)

    # Warmup
    for _ in range(3):
        out = moe_layer(x)
        mx.eval(out)

    # Benchmark
    times = []
    for _ in range(10):
        start = time.perf_counter()
        out = moe_layer(x)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    tok_per_s = seq_len / (mean_ms / 1000)
    print(f"  seq_len={seq_len:5d}: {mean_ms:8.2f} ± {std_ms:.2f} ms  ({tok_per_s:,.0f} tok/s)")

print("\nDone!")
