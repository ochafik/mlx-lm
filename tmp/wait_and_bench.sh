#!/bin/bash
# Wait for GPU contention to clear, then run comprehensive benchmarks.
# Usage: bash tmp/wait_and_bench.sh [model_name]

set -e
cd /Users/ochafik/github/mlx-lm2

MODEL="${1:-mlx-community/Qwen3.5-122B-A10B-4bit}"
MAX_WAIT=3600  # 1 hour
CHECK_INTERVAL=30
OUTFILE="tmp/bench_results_$(echo "$MODEL" | tr '/' '_').txt"

echo "=== MoE Gate+Up Fusion Benchmark ==="
echo "Model: $MODEL"
echo "Output: $OUTFILE"
echo "Waiting up to ${MAX_WAIT}s for GPU contention to clear..."
echo

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    # Check for heavy python processes using GPU (exclude this script's children)
    contention=$(ps aux | grep -E "python.*(decloud|mlx_vlm|mlx_lm|generate)" | grep -v grep | grep -v "wait_and_bench" | grep -v bench_moe | grep -v bench_forward | awk '$3 > 20 {print $0}')
    if [ -z "$contention" ]; then
        echo "[$(date +%H:%M:%S)] GPU looks idle. Starting benchmarks."
        break
    else
        echo "[$(date +%H:%M:%S)] GPU busy (waited ${elapsed}s):"
        echo "$contention" | head -3 | awk '{printf "  PID=%s CPU=%.0f%% %s %s\n", $2, $3, $11, $12}'
        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))
    fi
done

if [ $elapsed -ge $MAX_WAIT ]; then
    echo "Timed out waiting for GPU. Running anyway..."
fi

# Run benchmarks and tee to output file
{
    echo "============================================================"
    echo "  MoE Gate+Up Fusion Benchmark"
    echo "  Model: $MODEL"
    echo "  Date:  $(date)"
    echo "  Host:  $(hostname) / $(sysctl -n hw.model 2>/dev/null || echo unknown)"
    echo "  MLX:   $(python3 -c 'import mlx.core; print(mlx.core.__version__)' 2>/dev/null || echo unknown)"
    echo "============================================================"
    echo

    echo ">>> Step 1: Single MoE Block (interleaved A/B, 80 iterations)"
    echo
    python tmp/bench_moe_prefill.py \
        --model "$MODEL" \
        --level block \
        --iterations 80 \
        --warmup 20 \
        --seq-lens 1 64 128 256 512 1024 2048 4096
    echo

    echo ">>> Step 2: Full model forward pass (fused, no patching)"
    echo
    for sl in 256 512 1024 2048; do
        echo "--- seq_len=$sl ---"
        python tmp/bench_forward.py --model "$MODEL" --seq-len $sl -n 10 -w 3
        echo
    done

    echo ">>> Step 3: Full model forward pass (unfused via git stash)"
    echo
    git stash -q
    for sl in 256 512 1024 2048; do
        echo "--- seq_len=$sl ---"
        python tmp/bench_forward.py --model "$MODEL" --seq-len $sl -n 10 -w 3
        echo
    done
    git stash pop -q

    echo ">>> Step 4: Hyperfine comparison at key seq_lens"
    echo
    for sl in 512 1024 2048; do
        echo "--- seq_len=$sl ---"
        hyperfine \
            --warmup 1 \
            --runs 3 \
            -n "fused(sl=$sl)" \
            "python tmp/bench_forward.py --model '$MODEL' --seq-len $sl -n 8 -w 3" \
            -n "unfused(sl=$sl)" \
            "git stash -q && python tmp/bench_forward.py --model '$MODEL' --seq-len $sl -n 8 -w 3; git stash pop -q"
        echo
    done

    echo "============================================================"
    echo "  DONE at $(date)"
    echo "============================================================"
} 2>&1 | tee "$OUTFILE"

echo
echo "Results saved to: $OUTFILE"
