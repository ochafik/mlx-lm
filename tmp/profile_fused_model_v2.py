"""Profile actual MoE block - more warmup, more iterations."""

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

# Get a representative MoE layer
for i, layer in enumerate(model.language_model.model.layers):
    if hasattr(layer.mlp, 'switch_mlp'):
        moe_layer = layer.mlp
        switch = moe_layer.switch_mlp
        has_fused = hasattr(switch, 'gate_up_proj')
        print(f"Layer {i}: fused={has_fused}")
        if has_fused:
            print(f"  gate_up_proj weight: {switch.gate_up_proj.weight.shape}")
        break

for seq_len in [1, 64, 128, 256, 512, 1024, 2048]:
    x = mx.random.normal((1, seq_len, 2048))
    mx.eval(x)

    # Extended warmup
    for _ in range(10):
        out = moe_layer(x)
        mx.eval(out)

    # Benchmark with more iterations
    times = []
    for _ in range(30):
        start = time.perf_counter()
        out = moe_layer(x)
        mx.eval(out)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times.sort()
    # Use median and trim outliers
    trimmed = times[3:-3]  # Remove 3 fastest and 3 slowest
    mean_ms = np.mean(trimmed) * 1000
    std_ms = np.std(trimmed) * 1000
    median_ms = np.median(times) * 1000
    print(f"  seq_len={seq_len:5d}: median={median_ms:7.2f} ms, mean={mean_ms:7.2f} ± {std_ms:.2f} ms")

print("\nDone!")
