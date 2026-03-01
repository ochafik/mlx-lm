#!/usr/bin/env python3
# Copyright (c) 2025 Apple Inc.
#
# Converts raw RWKV v7 .pth weights (from BlinkDL/rwkv7-g1 etc.)
# to the MLX safetensors format compatible with mlx_lm/models/rwkv7.py.
#
# Usage:
#   python -m mlx_lm.models.rwkv7_convert \
#       --input /path/to/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \
#       --output ./rwkv7-0.1B-mlx \
#       --tokenizer fla-hub/rwkv7-0.1B-g1
#

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np


def load_pth(pth_path: str) -> dict:
    """Load a raw RWKV .pth file using torch and return as numpy dict."""
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required to load .pth files. Install with: pip install torch")
        sys.exit(1)

    print(f"Loading weights from {pth_path} ...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    print(f"  Found {len(state_dict)} weight tensors")
    return state_dict


def infer_config(state_dict: dict) -> dict:
    """Infer model config from weight shapes."""

    # hidden_size from embedding
    emb_weight = state_dict["emb.weight"]
    vocab_size = emb_weight.shape[0]
    hidden_size = emb_weight.shape[1]

    # Count layers
    layer_indices = set()
    for k in state_dict:
        m = re.match(r"blocks\.(\d+)\.", k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_hidden_layers = max(layer_indices) + 1

    # intermediate_size from FFN key weight
    ffn_key = state_dict["blocks.0.ffn.key.weight"]
    intermediate_size = ffn_key.shape[0]

    # LoRA dimensions
    decay_low_rank_dim = state_dict["blocks.0.att.w1"].shape[1]
    a_low_rank_dim = state_dict["blocks.0.att.a1"].shape[1]
    gate_low_rank_dim = state_dict["blocks.0.att.g1"].shape[1]

    # v_low_rank_dim from layer 1 (layer 0 has v_lora but it's unused)
    if num_hidden_layers > 1:
        v_low_rank_dim = state_dict["blocks.1.att.v1"].shape[1]
    else:
        # Fallback: use layer 0's v1 if only 1 layer
        v_low_rank_dim = state_dict["blocks.0.att.v1"].shape[1]

    config = {
        "model_type": "rwkv7",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "head_dim": 64,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "norm_eps": 1e-5,
        "a_low_rank_dim": a_low_rank_dim,
        "v_low_rank_dim": v_low_rank_dim,
        "gate_low_rank_dim": gate_low_rank_dim,
        "decay_low_rank_dim": decay_low_rank_dim,
        "tie_word_embeddings": False,
    }

    print(f"  Inferred config:")
    for k, v in config.items():
        print(f"    {k}: {v}")

    return config


def convert_weights(state_dict: dict, config: dict) -> dict:
    """
    Convert raw RWKV v7 weight names and shapes to MLX rwkv7.py conventions.

    Returns a dict of {mlx_name: numpy_array}.
    """
    import torch

    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]

    mlx_weights = {}

    def to_numpy(t):
        """Convert a torch tensor to numpy, handling bfloat16."""
        if isinstance(t, torch.Tensor):
            if t.dtype == torch.bfloat16:
                return t.to(torch.float32).numpy()
            return t.numpy()
        return np.array(t)

    # Global weights
    # emb.weight -> model.embeddings.weight
    mlx_weights["model.embeddings.weight"] = to_numpy(state_dict["emb.weight"])

    # ln_out -> model.norm
    mlx_weights["model.norm.weight"] = to_numpy(state_dict["ln_out.weight"])
    mlx_weights["model.norm.bias"] = to_numpy(state_dict["ln_out.bias"])

    # head.weight -> lm_head.weight (no transpose; both PyTorch and MLX nn.Linear store (out, in))
    mlx_weights["lm_head.weight"] = to_numpy(state_dict["head.weight"])

    for i in range(num_layers):
        prefix_raw = f"blocks.{i}"
        prefix_mlx = f"model.layers.{i}"

        # Pre-norm (layer 0 only)
        if i == 0:
            mlx_weights[f"{prefix_mlx}.pre_norm.weight"] = to_numpy(
                state_dict[f"{prefix_raw}.ln0.weight"]
            )
            mlx_weights[f"{prefix_mlx}.pre_norm.bias"] = to_numpy(
                state_dict[f"{prefix_raw}.ln0.bias"]
            )

        # Attention norm (ln1)
        mlx_weights[f"{prefix_mlx}.attn_norm.weight"] = to_numpy(
            state_dict[f"{prefix_raw}.ln1.weight"]
        )
        mlx_weights[f"{prefix_mlx}.attn_norm.bias"] = to_numpy(
            state_dict[f"{prefix_raw}.ln1.bias"]
        )

        # FFN norm (ln2)
        mlx_weights[f"{prefix_mlx}.ffn_norm.weight"] = to_numpy(
            state_dict[f"{prefix_raw}.ln2.weight"]
        )
        mlx_weights[f"{prefix_mlx}.ffn_norm.bias"] = to_numpy(
            state_dict[f"{prefix_raw}.ln2.bias"]
        )

        # Time mixing parameters: x_r, x_w, x_k, x_v, x_a, x_g
        # Raw shape is (C,) or (1,1,C); MLX expects (1,1,C)
        for param in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"]:
            raw_key = f"{prefix_raw}.att.{param}"
            arr = to_numpy(state_dict[raw_key])
            # Reshape to (1, 1, C) if needed
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, -1)
            elif arr.ndim == 2:
                arr = arr.reshape(1, 1, -1)
            mlx_weights[f"{prefix_mlx}.attn.{param}"] = arr

        # k_k, k_a: raw shape is (C,), MLX stores as (C,) and sanitize reshapes to (H, N)
        for param in ["k_k", "k_a"]:
            raw_key = f"{prefix_raw}.att.{param}"
            arr = to_numpy(state_dict[raw_key])
            # Store as 1D; sanitize in rwkv7.py will reshape to (num_heads, head_dim)
            mlx_weights[f"{prefix_mlx}.attn.{param}"] = arr.ravel()

        # r_k: raw shape is (H, N), keep as-is
        raw_key = f"{prefix_raw}.att.r_k"
        mlx_weights[f"{prefix_mlx}.attn.r_k"] = to_numpy(state_dict[raw_key])

        # Linear projections: receptance, key, value, output
        # Raw: (out_dim, in_dim) via PyTorch nn.Linear. MLX nn.Linear stores (out, in). Keep as-is.
        proj_map = {
            "receptance": "r_proj",
            "key": "k_proj",
            "value": "v_proj",
            "output": "o_proj",
        }
        for raw_name, mlx_name in proj_map.items():
            raw_key = f"{prefix_raw}.att.{raw_name}.weight"
            mlx_weights[f"{prefix_mlx}.attn.{mlx_name}.weight"] = to_numpy(
                state_dict[raw_key]
            )

        # Group norm (ln_x): weight and bias, stored as (C,), sanitize reshapes to (H, N)
        mlx_weights[f"{prefix_mlx}.attn.g_norm.weight"] = to_numpy(
            state_dict[f"{prefix_raw}.att.ln_x.weight"]
        ).ravel()
        mlx_weights[f"{prefix_mlx}.attn.g_norm.bias"] = to_numpy(
            state_dict[f"{prefix_raw}.att.ln_x.bias"]
        ).ravel()

        # --- LoRA conversions ---
        # w_lora: decay (w0=bias, w1=down, w2=up)
        #   w1: raw (C, D) -> nn.Linear(C, D) stores (D, C) -> transpose to (D, C)
        #   w2: raw (D, C) -> nn.Linear(D, C) stores (C, D) -> transpose to (C, D)
        #   w0: bias for lora.2, squeeze to (C,)
        w1 = to_numpy(state_dict[f"{prefix_raw}.att.w1"])  # (C, D)
        w2 = to_numpy(state_dict[f"{prefix_raw}.att.w2"])  # (D, C)
        w0 = to_numpy(state_dict[f"{prefix_raw}.att.w0"]).ravel()  # (C,)
        mlx_weights[f"{prefix_mlx}.attn.w_lora.lora.0.weight"] = w1.T  # (D, C)
        mlx_weights[f"{prefix_mlx}.attn.w_lora.lora.2.weight"] = w2.T  # (C, D)
        mlx_weights[f"{prefix_mlx}.attn.w_lora.lora.2.bias"] = w0  # (C,)

        # a_lora: iclr (a0=bias, a1=down, a2=up)
        a1 = to_numpy(state_dict[f"{prefix_raw}.att.a1"])  # (C, D)
        a2 = to_numpy(state_dict[f"{prefix_raw}.att.a2"])  # (D, C)
        a0 = to_numpy(state_dict[f"{prefix_raw}.att.a0"]).ravel()  # (C,)
        mlx_weights[f"{prefix_mlx}.attn.a_lora.lora.0.weight"] = a1.T  # (D, C)
        mlx_weights[f"{prefix_mlx}.attn.a_lora.lora.2.weight"] = a2.T  # (C, D)
        mlx_weights[f"{prefix_mlx}.attn.a_lora.lora.2.bias"] = a0  # (C,)

        # v_lora: value residual (v0=bias, v1=down, v2=up) — layer 0 skipped
        if i > 0:
            v1 = to_numpy(state_dict[f"{prefix_raw}.att.v1"])  # (C, D)
            v2 = to_numpy(state_dict[f"{prefix_raw}.att.v2"])  # (D, C)
            v0 = to_numpy(state_dict[f"{prefix_raw}.att.v0"]).ravel()  # (C,)
            mlx_weights[f"{prefix_mlx}.attn.v_lora.lora.0.weight"] = v1.T  # (D, C)
            mlx_weights[f"{prefix_mlx}.attn.v_lora.lora.2.weight"] = v2.T  # (C, D)
            mlx_weights[f"{prefix_mlx}.attn.v_lora.lora.2.bias"] = v0  # (C,)

        # g_lora: gate (g1=down, g2=up, NO bias)
        g1 = to_numpy(state_dict[f"{prefix_raw}.att.g1"])  # (C, D)
        g2 = to_numpy(state_dict[f"{prefix_raw}.att.g2"])  # (D, C)
        mlx_weights[f"{prefix_mlx}.attn.g_lora.lora.0.weight"] = g1.T  # (D, C)
        mlx_weights[f"{prefix_mlx}.attn.g_lora.lora.2.weight"] = g2.T  # (C, D)

        # --- FFN (Channel Mixing) ---
        # x_k: time shift for FFN, shape (C,) — keep squeezed
        ffn_xk = to_numpy(state_dict[f"{prefix_raw}.ffn.x_k"]).ravel()
        mlx_weights[f"{prefix_mlx}.ffn.x_k"] = ffn_xk

        # FFN key and value projections (no transpose, same convention)
        mlx_weights[f"{prefix_mlx}.ffn.key.weight"] = to_numpy(
            state_dict[f"{prefix_raw}.ffn.key.weight"]
        )
        mlx_weights[f"{prefix_mlx}.ffn.value.weight"] = to_numpy(
            state_dict[f"{prefix_raw}.ffn.value.weight"]
        )

        print(f"  Converted layer {i}/{num_layers - 1}")

    print(f"  Total MLX weights: {len(mlx_weights)}")
    return mlx_weights


def save_mlx_weights(mlx_weights: dict, output_dir: Path, max_shard_size_gb: float = 5.0):
    """Save weights as safetensors shards using mlx."""
    import mlx.core as mx

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays to mlx arrays
    mx_weights = {}
    for k, v in mlx_weights.items():
        mx_weights[k] = mx.array(v)

    total_size = sum(v.nbytes for v in mx_weights.values())
    max_shard_bytes = int(max_shard_size_gb * 1024**3)

    # Shard the weights
    shards = []
    current_shard = {}
    current_size = 0

    for k in sorted(mx_weights.keys()):
        v = mx_weights[k]
        nbytes = v.nbytes
        if current_size + nbytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[k] = v
        current_size += nbytes

    if current_shard:
        shards.append(current_shard)

    # Write shards
    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": {},
    }

    if len(shards) == 1:
        shard_name = "model.safetensors"
        shard_path = output_dir / shard_name
        print(f"  Writing {shard_path} ...")
        mx.save_safetensors(str(shard_path), shards[0], metadata={"format": "mlx"})
        for weight_name in shards[0]:
            index_data["weight_map"][weight_name] = shard_name
    else:
        for i, shard in enumerate(shards):
            shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            shard_path = output_dir / shard_name
            print(f"  Writing {shard_path} ...")
            mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
            for weight_name in shard:
                index_data["weight_map"][weight_name] = shard_name

    # Sort weight map
    index_data["weight_map"] = {
        k: index_data["weight_map"][k]
        for k in sorted(index_data["weight_map"])
    }

    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)
    print(f"  Wrote weight index to {index_path}")


def setup_tokenizer(tokenizer_source: str, output_dir: Path):
    """
    Copy tokenizer files to output directory.

    tokenizer_source can be:
    - A HuggingFace model ID (e.g., "fla-hub/rwkv7-0.1B-g1") — downloads tokenizer files
    - A local directory containing tokenizer files
    - A path to a specific vocab file (rwkv_vocab_v20230424.txt)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(tokenizer_source)

    if source_path.is_file():
        # Single vocab file — copy it and create a tokenizer config
        print(f"  Copying vocab file {source_path} to {output_dir}")
        shutil.copy2(source_path, output_dir / source_path.name)
        # Write a minimal tokenizer config pointing at the vocab file
        tokenizer_config = {
            "tokenizer_class": "RWKV5Tokenizer",
            "vocab_file": source_path.name,
        }
        with open(output_dir / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"  Wrote tokenizer_config.json")
        return

    if source_path.is_dir():
        # Local directory — copy all tokenizer-related files
        print(f"  Copying tokenizer files from {source_path}")
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "rwkv_vocab_v20230424.txt",
            "tokenizer.model",
            "added_tokens.json",
        ]
        copied = 0
        for fname in tokenizer_files:
            src = source_path / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)
                copied += 1
        # Also copy any .txt or .json tokenizer-related files
        for f in source_path.glob("*tokenizer*"):
            if f.is_file():
                dst = output_dir / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    copied += 1
        print(f"  Copied {copied} tokenizer file(s)")
        return

    # Assume it's a HuggingFace model ID — use transformers to download tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "WARNING: transformers is required to download tokenizer from HuggingFace. "
            "Install with: pip install transformers"
        )
        print("  Skipping tokenizer setup.")
        return

    print(f"  Downloading tokenizer from {tokenizer_source} ...")
    try:
        from huggingface_hub import snapshot_download

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))

        # Copy any vocab files that save_pretrained may have missed/renamed
        vocab_files = getattr(tokenizer, "vocab_files_names", {})
        if vocab_files:
            src_dir = Path(
                snapshot_download(tokenizer_source, allow_patterns=["*.txt", "*.json", "*.py"])
            )
            for vocab_file in vocab_files.values():
                src_vocab = src_dir / vocab_file
                dst_vocab = output_dir / vocab_file
                if src_vocab.exists() and not dst_vocab.exists():
                    shutil.copy2(str(src_vocab), str(dst_vocab))
                    print(f"  Copied {vocab_file}")

        print(f"  Saved tokenizer to {output_dir}")
    except Exception as e:
        print(f"WARNING: Failed to download tokenizer: {e}")
        print("  You may need to manually copy tokenizer files to the output directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw RWKV v7 .pth weights to MLX safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with HuggingFace tokenizer
  python -m mlx_lm.models.rwkv7_convert \\
      --input RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth \\
      --output ./rwkv7-0.1B-mlx \\
      --tokenizer fla-hub/rwkv7-0.1B-g1

  # With local tokenizer directory
  python -m mlx_lm.models.rwkv7_convert \\
      --input model.pth \\
      --output ./rwkv7-mlx \\
      --tokenizer /path/to/tokenizer/dir

  # With specific vocab file
  python -m mlx_lm.models.rwkv7_convert \\
      --input model.pth \\
      --output ./rwkv7-mlx \\
      --tokenizer rwkv_vocab_v20230424.txt
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw RWKV v7 .pth file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for MLX model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help=(
            "Tokenizer source: HuggingFace model ID (e.g. fla-hub/rwkv7-0.1B-g1), "
            "local directory with tokenizer files, or path to vocab .txt file"
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for saved weights (default: float16)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Step 1: Load raw weights
    state_dict = load_pth(str(input_path))

    # Step 2: Infer config
    config = infer_config(state_dict)

    # Step 3: Convert weight names and shapes
    print("Converting weights ...")
    mlx_weights = convert_weights(state_dict, config)

    # Free the raw state dict
    del state_dict

    # Step 4: Cast to target dtype
    dtype_map = {
        "float16": np.float16,
        "bfloat16": np.float32,  # numpy doesn't have bfloat16; we'll use mlx for the cast
        "float32": np.float32,
    }

    if args.dtype == "bfloat16":
        # Cast via mlx since numpy doesn't support bfloat16
        import mlx.core as mx

        print(f"Casting weights to bfloat16 ...")
        for k in mlx_weights:
            mlx_weights[k] = np.array(mx.array(mlx_weights[k]).astype(mx.bfloat16))
    elif args.dtype == "float16":
        print(f"Casting weights to float16 ...")
        for k in mlx_weights:
            if mlx_weights[k].dtype in (np.float32, np.float64):
                mlx_weights[k] = mlx_weights[k].astype(np.float16)
    else:
        print(f"Keeping weights as float32 ...")
        for k in mlx_weights:
            if mlx_weights[k].dtype == np.float64:
                mlx_weights[k] = mlx_weights[k].astype(np.float32)

    # Step 5: Save weights
    print(f"Saving MLX weights to {output_dir} ...")
    save_mlx_weights(mlx_weights, output_dir)

    # Free converted weights
    del mlx_weights

    # Step 6: Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Step 7: Setup tokenizer
    if args.tokenizer:
        print(f"Setting up tokenizer ...")
        setup_tokenizer(args.tokenizer, output_dir)
    else:
        print(
            "No tokenizer specified. You can add one later by copying tokenizer files "
            "to the output directory or re-running with --tokenizer."
        )

    print(f"\nConversion complete! Output directory: {output_dir}")
    print(f"  To load with mlx_lm:")
    print(f'    from mlx_lm import load, generate')
    print(f'    model, tokenizer = load("{output_dir}")')
    print(f'    generate(model, tokenizer, prompt="Hello")')


if __name__ == "__main__":
    main()
