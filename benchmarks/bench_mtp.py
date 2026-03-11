#!/usr/bin/env python3
"""Benchmark MTP speculative decoding vs baseline for Qwen3.5 models.

Usage:
    python benchmarks/bench_mtp.py --model mlx-community/Qwen3.5-9B-MLX-4bit
    python benchmarks/bench_mtp.py --model mlx-community/Qwen3.5-35B-A3B-4bit --sweep-ndt
    python benchmarks/bench_mtp.py --model mlx-community/Qwen3.5-122B-A10B-4bit --max-tokens 50
"""
import argparse
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate


PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to merge two sorted lists.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
    "Write a haiku about artificial intelligence.",
    "What is the difference between a compiler and an interpreter?",
    "Explain how a hash table works.",
    "What are the SOLID principles in software engineering?",
    "Describe the water cycle in detail.",
    "Write a short story about a robot learning to paint.",
    "Explain the concept of recursion with an example.",
    "What are the advantages of microservices over monolithic architecture?",
    "How does public key cryptography work?",
]


def bench_config(model, tokenizer, prompts, max_tokens, mtp, ndt, num_prompts):
    prompts = prompts[:num_prompts]
    total_toks = 0
    total_gen_time = 0
    accepted = 0
    cycles = 0

    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        prompt_text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        gen_toks = 0
        from_draft_count = 0
        non_draft_count = 0

        for resp in stream_generate(
            model,
            tokenizer,
            prompt_text,
            max_tokens=max_tokens,
            mtp=mtp,
            num_draft_tokens=ndt,
        ):
            gen_toks = resp.generation_tokens
            gen_tps = resp.generation_tps
            if resp.from_draft:
                from_draft_count += 1
            else:
                non_draft_count += 1

        total_toks += gen_toks
        if gen_tps > 0:
            total_gen_time += gen_toks / gen_tps
        accepted += from_draft_count
        cycles += non_draft_count

    avg_tps = total_toks / total_gen_time if total_gen_time > 0 else 0
    accept_rate = accepted / (accepted + cycles) if (accepted + cycles) > 0 else 0
    return avg_tps, total_toks, total_gen_time, accept_rate


def main():
    parser = argparse.ArgumentParser(description="Benchmark MTP speculative decoding")
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--sweep-ndt", action="store_true", help="Sweep NDT 0-3")
    parser.add_argument("--ndt", type=int, default=1, help="Num draft tokens (default 1)")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    text_model = model.language_model if hasattr(model, "language_model") else model
    has_mtp = getattr(text_model, "has_mtp", False)
    print(f"  has_mtp: {has_mtp}")
    print(f"  max_tokens: {args.max_tokens}, num_prompts: {args.num_prompts}")
    print(f"  peak memory: {mx.get_peak_memory() / 1e9:.1f} GB")
    print()

    # Warmup
    for resp in stream_generate(model, tokenizer, "Hello", max_tokens=5):
        pass

    # Baseline
    print("Running baseline...")
    base_tps, base_toks, base_time, _ = bench_config(
        model, tokenizer, PROMPTS, args.max_tokens, False, 0, args.num_prompts
    )
    print(f"  Baseline: {base_tps:.1f} tok/s ({base_toks} tokens in {base_time:.1f}s)")
    print()

    if not has_mtp:
        print("Model does not have MTP heads. Skipping MTP benchmarks.")
        return

    if args.sweep_ndt:
        ndts = [1, 2, 3]
    else:
        ndts = [args.ndt]

    for ndt in ndts:
        print(f"Running MTP NDT={ndt}...")
        mtp_tps, mtp_toks, mtp_time, accept_rate = bench_config(
            model, tokenizer, PROMPTS, args.max_tokens, True, ndt, args.num_prompts
        )
        speedup = mtp_tps / base_tps if base_tps > 0 else 0
        print(
            f"  MTP NDT={ndt}: {mtp_tps:.1f} tok/s ({mtp_toks} tokens in {mtp_time:.1f}s) "
            f"speedup={speedup:.2f}x accept={accept_rate:.0%}"
        )

    print(f"\nPeak memory: {mx.get_peak_memory() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
