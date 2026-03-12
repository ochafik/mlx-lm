#!/usr/bin/env python3
"""Download MTP weights from source Qwen3.5 models and patch them into
quantized mlx-community models that had MTP stripped during conversion.

Usage:
    python benchmarks/patch_mtp_weights.py --source Qwen/Qwen3.5-27B \
        --target mlx-community/Qwen3.5-27B-4bit

    python benchmarks/patch_mtp_weights.py --source Qwen/Qwen3.5-35B-A3B \
        --target mlx-community/Qwen3.5-35B-A3B-4bit --quantize
"""
import argparse
import json
import os
import re
import shutil

import mlx.core as mx
from huggingface_hub import hf_hub_download, snapshot_download


def get_mtp_weights(source_repo: str) -> dict:
    """Download and extract MTP weights from a source HuggingFace model."""
    # Get the weight index to find which shards contain MTP weights
    index_path = hf_hub_download(source_repo, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Find shards containing MTP weights
    mtp_shards = {}
    for key, shard in index["weight_map"].items():
        if "mtp" in key:
            mtp_shards.setdefault(shard, []).append(key)

    print(f"MTP weights spread across {len(mtp_shards)} shard(s)")
    for shard, keys in mtp_shards.items():
        print(f"  {shard}: {len(keys)} keys")

    # Download shards and extract MTP weights
    mtp_weights = {}
    for shard_name, keys in mtp_shards.items():
        print(f"Downloading {shard_name}...")
        shard_path = hf_hub_download(source_repo, shard_name)
        weights = mx.load(shard_path)
        for key in keys:
            if key in weights:
                mtp_weights[key] = weights[key]
        del weights

    print(f"Extracted {len(mtp_weights)} MTP weight tensors")
    return mtp_weights


def convert_moe_experts_to_switch(mtp_weights: dict) -> dict:
    """Convert individual expert weights to SwitchGLU stacked format.

    Input format:  mtp.layers.0.mlp.experts.N.{gate,up,down}_proj.weight
    Output format: mtp.layers.0.mlp.switch_mlp.{gate,up,down}_proj.weight
                   (stacked as [num_experts, ...])

    Also handles gate_up_proj fused format if present.
    """
    # Check if there are individual expert weights
    expert_pattern = re.compile(
        r"mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    )

    # Group by (layer, proj_type)
    expert_groups = {}
    non_expert_weights = {}

    for key, value in mtp_weights.items():
        m = expert_pattern.match(key)
        if m:
            layer_idx, expert_idx, proj_type = m.group(1), int(m.group(2)), m.group(3)
            group_key = (layer_idx, proj_type)
            expert_groups.setdefault(group_key, {})[expert_idx] = value
        else:
            non_expert_weights[key] = value

    if not expert_groups:
        return mtp_weights  # No expert weights to convert

    result = dict(non_expert_weights)
    for (layer_idx, proj_type), experts in expert_groups.items():
        num_experts = max(experts.keys()) + 1
        # Stack experts into [num_experts, ...] tensor
        stacked = mx.stack([experts[i] for i in range(num_experts)])
        key = f"mtp.layers.{layer_idx}.mlp.switch_mlp.{proj_type}.weight"
        result[key] = stacked
        print(f"  Stacked {num_experts} experts -> {key}: {stacked.shape}")

    return result


def convert_to_mlx_format(mtp_weights: dict) -> dict:
    """Convert raw HF weight keys to MLX format (post-sanitize).

    - Prefix with 'language_model.' for the model tree
    - Apply +1.0 correction to RMSNorm weights (MLX convention)
    """
    norm_suffixes = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
        "mtp.norm.weight",
        ".pre_fc_norm_embedding.weight",
        ".pre_fc_norm_hidden.weight",
    )

    result = {}
    for key, value in mtp_weights.items():
        new_key = f"language_model.{key}"
        if any(key.endswith(sfx) for sfx in norm_suffixes):
            if value.ndim == 1:
                value = value + 1.0
                print(f"  Norm correction: {key}")
        result[new_key] = value
    return result


def quantize_mtp_weights(mtp_weights: dict, group_size: int = 64, bits: int = 4) -> dict:
    """Quantize MTP linear weights to match model quantization.

    Quantizes weight matrices (2D tensors) using MLX's affine quantization.
    Skips norms, embeddings, biases, and gate (router) weights.
    """
    import mlx.nn as nn

    skip_suffixes = (".weight",)  # norm weights are 1D, will be skipped by shape check
    gate_patterns = (".gate.weight",)  # MoE router — keep full precision

    quantized = {}
    n_quantized = 0
    n_skipped = 0

    for key, value in mtp_weights.items():
        # Only quantize weight matrices (2D linear or 3D stacked experts)
        is_gate = any(key.endswith(p) for p in gate_patterns)
        if value.ndim >= 2 and key.endswith(".weight") and not is_gate:
            if value.ndim == 3:
                # Stacked expert weights [num_experts, out, in] — quantize each
                ws, ss, bs = [], [], []
                for i in range(value.shape[0]):
                    w, s, b = mx.quantize(value[i], group_size=group_size, bits=bits)
                    ws.append(w); ss.append(s); bs.append(b)
                w_stacked = mx.stack(ws)
                s_stacked = mx.stack(ss)
                b_stacked = mx.stack(bs)
                base = key.rsplit(".weight", 1)[0]
                quantized[base + ".weight"] = w_stacked
                quantized[base + ".scales"] = s_stacked
                quantized[base + ".biases"] = b_stacked
                n_quantized += 1
            else:
                w, scales, biases = mx.quantize(value, group_size=group_size, bits=bits)
                base = key.rsplit(".weight", 1)[0]
                quantized[base + ".weight"] = w
                quantized[base + ".scales"] = scales
                quantized[base + ".biases"] = biases
                n_quantized += 1
        else:
            quantized[key] = value
            n_skipped += 1

    print(f"  Quantized {n_quantized} weight matrices, skipped {n_skipped} tensors")
    return quantized


def patch_target_model(target_repo: str, mtp_weights: dict, quantize: bool = False):
    """Add MTP weights to a quantized model's local cache."""
    # Download the target model (should be cached already)
    target_path = snapshot_download(target_repo)

    # Convert to MLX format (add prefix, norm correction)
    mtp_weights = convert_to_mlx_format(mtp_weights)

    # Optionally quantize
    if quantize:
        # Force evaluation of all lazy arrays before quantizing
        mx.eval(*mtp_weights.values())

        # Read target model's quantization config
        config_path = os.path.join(target_path, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        quant_cfg = config.get("quantization", {})
        group_size = quant_cfg.get("group_size", 64)
        bits = quant_cfg.get("bits", 4)
        print(f"\nQuantizing MTP weights to {bits}-bit (group_size={group_size})...")
        mtp_weights = quantize_mtp_weights(mtp_weights, group_size=group_size, bits=bits)
        mx.eval(*mtp_weights.values())

    # Save MTP weights as a new safetensors file
    mtp_file = os.path.join(target_path, "model-mtp.safetensors")
    mx.save_safetensors(mtp_file, mtp_weights)
    print(f"Saved MTP weights to {mtp_file}")

    # Update the weight index if it exists
    index_file = os.path.join(target_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        for key in mtp_weights:
            index["weight_map"][key] = "model-mtp.safetensors"
        with open(index_file, "w") as f:
            json.dump(index, f, indent=2)
        print(f"Updated weight index with {len(mtp_weights)} MTP entries")
    else:
        # Single-file model — create an index
        # List existing safetensors files
        existing = [
            f for f in os.listdir(target_path) if f.endswith(".safetensors") and f != "model-mtp.safetensors"
        ]
        if len(existing) == 1:
            # Load existing weights to get their keys
            existing_path = os.path.join(target_path, existing[0])
            existing_weights = mx.load(existing_path)
            weight_map = {k: existing[0] for k in existing_weights}
            weight_map.update({k: "model-mtp.safetensors" for k in mtp_weights})
            index = {"metadata": {}, "weight_map": weight_map}
            with open(index_file, "w") as f:
                json.dump(index, f, indent=2)
            print(f"Created weight index with {len(weight_map)} entries")

    return target_path


def main():
    parser = argparse.ArgumentParser(description="Patch MTP weights into quantized models")
    parser.add_argument("--source", required=True, help="Source HF model with MTP weights")
    parser.add_argument("--target", required=True, help="Target quantized HF model")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize MTP weights to match target model's quantization")
    args = parser.parse_args()

    print(f"Source: {args.source}")
    print(f"Target: {args.target}")

    # Download MTP weights from source
    mtp_weights = get_mtp_weights(args.source)

    # Check if this is an MoE model (has individual expert weights)
    has_experts = any("experts." in k for k in mtp_weights)
    if has_experts:
        print("\nConverting MoE expert weights to SwitchGLU format...")
        mtp_weights = convert_moe_experts_to_switch(mtp_weights)

    # Patch into target model
    print(f"\nPatching into {args.target}...")
    target_path = patch_target_model(args.target, mtp_weights, quantize=args.quantize)
    print(f"\nDone! Model at: {target_path}")


if __name__ == "__main__":
    main()
