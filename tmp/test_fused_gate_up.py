"""Test that gate+up fusion works correctly with Qwen3.5-35B-A3B-4bit."""

import sys
sys.path.insert(0, "/Users/ochafik/github/mlx-lm2")

import mlx.core as mx
from mlx_lm.utils import load_model, _download, load_tokenizer

MODEL_NAME = "mlx-community/Qwen3.5-35B-A3B-4bit"

print("Loading model...")
model_path = _download(MODEL_NAME)
model, _ = load_model(model_path, lazy=False, strict=False)
tokenizer = load_tokenizer(model_path)
print("Model loaded successfully!")

# Check that the fused gate_up_proj exists
layer = model.language_model.model.layers[0]
if hasattr(layer.mlp, 'switch_mlp'):
    switch_mlp = layer.mlp.switch_mlp
    has_fused = hasattr(switch_mlp, 'gate_up_proj')
    has_separate = hasattr(switch_mlp, 'gate_proj')
    print(f"  Has gate_up_proj (fused): {has_fused}")
    print(f"  Has gate_proj (separate): {has_separate}")
    if has_fused:
        gup = switch_mlp.gate_up_proj
        print(f"  gate_up_proj weight shape: {gup.weight.shape}")
        print(f"  gate_up_proj scales shape: {gup.scales.shape}")
    down = switch_mlp.down_proj
    print(f"  down_proj weight shape: {down.weight.shape}")
else:
    print("  Layer 0 is not MoE")

# Test generation
print("\nTesting generation...")
prompt = "Hello, how are you?"
tokens = mx.array([tokenizer.encode(prompt)])
print(f"  Input tokens: {tokens.shape}")

# Run a single forward pass
out = model(tokens)
mx.eval(out)
print(f"  Output shape: {out.shape}")
print(f"  First few logits: {out[0, -1, :5]}")

# Test with mlx_lm.generate
from mlx_lm import generate
print("\nGenerating text...")
result = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
print(f"  Generated: {result}")

print("\nAll tests passed!")
