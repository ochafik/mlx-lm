#!/bin/bash
# Full model A/B benchmark via hyperfine + git stash
# Runs each seq_len with both fused (current) and unfused (stashed) code.

set -e
cd /Users/ochafik/github/mlx-lm2

SEQ_LENS="${1:-256 512 1024 2048}"
ITERS="${2:-10}"
WARMUP="${3:-3}"
RUNS="${4:-3}"

echo "============================================================"
echo "  Full model A/B: fused vs unfused via git stash + hyperfine"
echo "  seq_lens: $SEQ_LENS"
echo "  per-run: $WARMUP warmup + $ITERS iterations"
echo "  hyperfine runs: $RUNS"
echo "============================================================"
echo

for sl in $SEQ_LENS; do
    echo "--- seq_len=$sl ---"
    hyperfine \
        --warmup 1 \
        --runs $RUNS \
        --export-json "/tmp/bench_sl${sl}.json" \
        -n "fused(sl=$sl)" \
        "python tmp/bench_forward.py --seq-len $sl -n $ITERS -w $WARMUP" \
        -n "unfused(sl=$sl)" \
        "git stash -q && python tmp/bench_forward.py --seq-len $sl -n $ITERS -w $WARMUP; git stash pop -q"
    echo
done

echo "============================================================"
echo "  Summary (from hyperfine JSON)"
echo "============================================================"
for sl in $SEQ_LENS; do
    python3 -c "
import json
with open('/tmp/bench_sl${sl}.json') as f:
    data = json.load(f)
for r in data['results']:
    name = r['command']
    mean = r['mean']
    stddev = r['stddev']
    print(f'  {name[:30]:30s}: {mean:.2f}s ± {stddev:.2f}s')
" 2>/dev/null || true
done
