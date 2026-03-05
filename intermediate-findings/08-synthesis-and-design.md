# Synthesis & Design: Expert-Batched Prefill for mlx-lm MoE

## Executive Summary

After thorough investigation of mlx-lm's MoE implementation, MLX's gather_mm/gather_qmm
Metal kernels, and expert-parallel strategies from vLLM, SGLang, and KTransformers, here are
the key conclusions:

### What mlx-lm Already Does Well

**MLX's gather_qmm with sorted_indices=True IS already a grouped GEMM** at the Metal kernel level.
The Metal shader (`affine_gather_qmm_rhs` in quantized.h) physically scans sorted indices,
finds contiguous expert groups, and performs one GEMM per group. This was added in MLX PRs #2040
and #2078 (May 2025).

**Benchmarks confirm this**: When evaluating all results fairly, `gather_qmm(sorted)` is
2-3.5x faster than launching 256 separate `quantized_matmul` calls. The single-dispatch
approach eliminates Metal kernel launch overhead.

### Where Optimization Opportunities Remain

Despite the already-efficient low-level kernel, there are several Python-level and
algorithmic improvements that can improve MoE prefill performance:

1. **Fuse gate_proj + up_proj weights** (estimated 1.3-1.5x speedup on expert MLP)
2. **Overlap shared expert with routed experts** (estimated 1.05-1.1x on total MoE block)
3. **Counting sort instead of argsort** (estimated 1.1-1.2x on sort+routing overhead)
4. **Better sort threshold** (fix the anomaly at seq_len=128)
5. **mx.compile the routing logic** (reduce Python/graph overhead)

---

## Detailed Analysis

### Current Performance Baseline (Qwen3.5-35B-A3B-4bit)

Model config: 256 experts, top-8, hidden_size=2048, moe_intermediate_size=1024, 4-bit quantized

| seq_len | Total MoE block | Expert MLP (89%) | Routing (3%) | Shared (8%) |
|---------|-----------------|-------------------|--------------|-------------|
| 512     | 10.33 ms        | 8.56 ms           | 0.33 ms      | 0.75 ms     |
| 1024    | 14.17 ms        | 12.81 ms          | 0.46 ms      | 1.15 ms     |
| 2048    | 23.71 ms        | 21.50 ms          | 0.62 ms      | 2.00 ms     |

### Optimization 1: Fuse gate_proj + up_proj (HIGHEST IMPACT)

**Current flow** (3 gather_qmm calls):
```python
x_up   = self.up_proj(x, idx, sorted_indices=True)    # gather_qmm: x @ W_up
x_gate = self.gate_proj(x, idx, sorted_indices=True)   # gather_qmm: x @ W_gate
x_act  = silu(x_gate) * x_up                           # activation
x_out  = self.down_proj(x_act, idx, sorted_indices=True)# gather_qmm: x_act @ W_down
```

**Proposed flow** (2 gather_qmm calls):
```python
# Concatenate gate+up weights: W_gate_up = [W_gate; W_up] shape (E, 2*I, H)
x_gate_up = self.gate_up_proj(x, idx, sorted_indices=True)  # gather_qmm: x @ W_gate_up
x_gate, x_up = mx.split(x_gate_up, 2, axis=-1)
x_act = silu(x_gate) * x_up
x_out = self.down_proj(x_act, idx, sorted_indices=True)     # gather_qmm: x_act @ W_down
```

**Why this helps**: Each gather_qmm call has fixed overhead (kernel launch, index scanning,
BM-tile boundary processing). Reducing from 3 to 2 calls eliminates ~33% of this overhead.
The fused matmul is also more efficient because the input x is read from memory once instead
of twice (better cache utilization).

**Implementation notes**:
- Qwen3.5 already stores gate+up fused in the checkpoint: `experts.gate_up_proj` is a
  `(num_experts, 2*intermediate, hidden)` tensor that gets split during sanitize()
- We can avoid the split and keep them concatenated
- For quantized models, need a `QuantizedSwitchLinear` with output_dims = 2*intermediate
- The split happens after the matmul (cheap)

**What vLLM/SGLang do**: Both frameworks fuse gate+up as "w1" (or "w13") into a single
GEMM, then apply activation, then do "w2" (down projection) as a second GEMM. This is
the standard 2-GEMM approach used universally.

### Optimization 2: Overlap Shared Expert with Routed Experts

**Current flow** (sequential):
```python
y = self.switch_mlp(x, inds)                    # routed experts (89% of time)
y = (y * scores[..., None]).sum(axis=-2)
shared_y = self.shared_expert(x)                 # shared expert (8% of time)
shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
return y + shared_y
```

**Proposed flow** (overlapped via lazy evaluation):
```python
# Launch both computations before evaluating either
y = self.switch_mlp(x, inds)
shared_y = self.shared_expert(x)
shared_gate = mx.sigmoid(self.shared_expert_gate(x))

# Now evaluate together - MLX's lazy evaluation should overlap them
y_weighted = (y * scores[..., None]).sum(axis=-2)
return y_weighted + shared_gate * shared_y
```

MLX's lazy evaluation graph should naturally overlap these independent computations.
However, they may already be overlapped to some extent. The benefit depends on whether
the Metal command buffer can schedule both concurrently.

**What SGLang does**: SGLang's Single-Batch Overlap (SBO) explicitly overlaps shared expert
computation with DeepEP communication. The concept is the same - run independent work in
parallel.

### Optimization 3: Counting Sort Instead of Argsort

**Current flow**:
```python
def _gather_sort(x, indices):
    *_, M = indices.shape
    indices = indices.flatten()          # [B*S*top_k]
    order = mx.argsort(indices)          # O(n log n) comparison sort
    inv_order = mx.argsort(order)        # Another O(n log n) sort
    return x.flatten(0, -3)[order // M], indices[order], inv_order
```

**Proposed flow** (counting sort):
```python
def _counting_sort(x, indices, num_experts):
    """O(n + E) counting sort, also produces expert counts for free."""
    *_, M = indices.shape
    indices_flat = indices.flatten()

    # Count tokens per expert
    counts = mx.zeros(num_experts, dtype=mx.int32)
    # ... or use scatter_add / bincount
    expert_counts = mx.bincount(indices_flat, minlength=num_experts)

    # Compute offsets (prefix sum)
    offsets = mx.cumsum(expert_counts)  # [num_experts]
    offsets = mx.concatenate([mx.array([0]), offsets[:-1]])

    # Place each token at its sorted position
    # This requires atomic increment or sequential placement
    # On GPU, this is done with a histogram + scatter pattern
    ...
```

**Challenge**: MLX may not have efficient `bincount` or atomic scatter operations.
The argsort approach, while O(n log n), is a single well-optimized Metal kernel call.
A Python-level counting sort might actually be slower due to multiple kernel launches.

**Alternative**: Use `mx.compile` to fuse the sort operations, or explore whether
MLX's argsort is already using radix sort internally (which would be O(n * word_size)).

### Optimization 4: Sort Threshold Tuning

Current threshold: `do_sort = indices.size >= 64`

The benchmark shows an anomaly at seq_len=128 (slower than seq_len=256). The gather_qmm
batched kernel requires `B >= 16` and `B/E >= 4`. For Qwen3.5 with 256 experts:
- seq_len=128, top_k=8: B=1024, B/E=4.0 → barely meets threshold
- seq_len=64, top_k=8: B=512, B/E=2.0 → does NOT meet B/E>=4 threshold

The sort adds overhead (two argsort calls) that may not be recouped at small sizes.
Better threshold: `do_sort = indices.size >= 128` or dynamically compute based on
B/E ratio and kernel thresholds.

### Optimization 5: Compile Routing Logic

The routing computation (gate → softmax → argpartition → scoring) involves multiple
small operations. Using `@mx.compile` can fuse these into a single kernel:

```python
@mx.compile
def route(x, gate_weight, top_k, norm_topk_prob):
    gates = mx.softmax(gate_weight @ x.T, axis=0)  # simplified
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores
```

This is already done in some models (deepseek_v3, glm4_moe_lite) but not in qwen3_next.

---

## Implementation Plan

### Phase 1: Gate+Up Fusion (Highest Impact, Moderate Complexity)

1. Modify `SwitchGLU` to accept a fused gate+up projection:
   - Add `SwitchGLUFused` class with a single `gate_up_proj` instead of separate `gate_proj` + `up_proj`
   - The `gate_up_proj` has output_dims = 2 * hidden_dims
   - After gather_qmm, split the output along the last dimension

2. Update `qwen3_5_moe.py` sanitize() to keep gate_up fused instead of splitting

3. Update `qwen3_next.py` to use `SwitchGLUFused` when gate+up weights are available

4. Ensure backward compatibility (old format with separate weights still works)

### Phase 2: Routing Optimizations (Low Complexity)

1. Add `@mx.compile` to `Qwen3NextSparseMoeBlock.__call__` routing section
2. Tune the sort threshold based on model config (num_experts, top_k)
3. Consider precomputing expert_counts during routing for diagnostics

### Phase 3: Shared Expert Overlap (Low Complexity)

1. Reorder operations in `Qwen3NextSparseMoeBlock.__call__` to launch shared expert
   computation before waiting for routed expert results
2. Verify with profiling that MLX's lazy eval actually overlaps them

### Phase 4: Metal Kernel Improvements (High Complexity, Requires MLX Changes)

These would require changes to the MLX framework itself:

1. **Fused sort+gather_qmm**: Combine the argsort + gather_qmm into a single Metal kernel
   that does counting sort and grouped GEMM in one dispatch
2. **Tile-aligned padding**: Pad per-expert token counts to BM multiples within the kernel
3. **Adaptive tile sizes**: Choose BM based on avg tokens/expert (already done in NAX path)

---

## Comparison with Other Frameworks

| Feature | mlx-lm (current) | vLLM | SGLang | KTransformers |
|---------|-------------------|------|--------|---------------|
| Grouped GEMM | gather_qmm (Metal) | Triton fused_moe | Triton + DeepGEMM | Per-expert loop |
| Token sorting | argsort + sorted_indices | moe_align_block_size | moe_align_block_size | argsort |
| Gate+Up fusion | Separate | Fused (w13) | Fused (w13) | Separate |
| Shared expert overlap | No | No | SBO (with DeepEP) | No |
| Quantized experts | 4-bit gather_qmm | FP8/INT8/FP4 | FP8/INT8/FP4 | BF16/FP16 |
| Tile-aligned padding | No | Yes (BLOCK_SIZE_M) | Yes (BLOCK_SIZE_M) | No |
| Prefill/decode split | No | Adaptive | DeepEP Normal/LL | Yes (CPU offload) |

---

## Implementation Results (Phase 1: Gate+Up Fusion)

### Changes Made
1. **switch_layers.py**: Added `fuse_gate_up` parameter to `SwitchGLU`, plus `fuse_gate_up_weights()` utility for backward compat
2. **qwen3_next.py**: `Qwen3NextSparseMoeBlock` now uses `SwitchGLU(fuse_gate_up=True)`, sanitize fuses per-expert weights
3. **qwen3_5_moe.py**: Sanitize keeps gate_up fused instead of splitting
4. **qwen3_moe.py**: Same pattern as qwen3_next
5. **qwen3_vl_moe.py**: Same pattern, keeps gate_up fused with swapaxes

### Backward Compatibility
- `fuse_gate_up_weights()` concatenates quantized gate+up weights along output dimension
- Works correctly for already-quantized models (quantization is per-row)
- All existing tests pass (50 model subtests including qwen3_next, qwen3_5, qwen3_5_moe, qwen3_vl_moe)

### Actual Performance (Qwen3.5-35B-A3B-4bit, single MoE block)

| seq_len | Baseline (separate) | Fused (new) | Change |
|---------|-------------------|-------------|--------|
| 1       | 1.16 ms           | 1.40 ms     | +21%   |
| 16      | 2.39 ms           | 2.84 ms     | +19%   |
| 64      | 4.22 ms           | 3.82 ms     | -9%    |
| 128     | 10.35 ms          | 5.51 ms     | -47%   |
| 256     | 9.24 ms           | 6.66 ms     | -28%   |
| 512     | 10.33 ms          | 9.01 ms     | -13%   |
| 1024    | 14.17 ms          | 13.96 ms    | -1%    |
| 2048    | 23.71 ms          | 23.72 ms    | 0%     |

Note: Baseline numbers from earlier session; direct comparison may have system variation.

## Expected Impact (Remaining Phases)

For Qwen3.5-35B-A3B-4bit prefill at seq_len=1024:
- Phase 2 (routing opt): additional 3-5%
- Phase 3 (shared overlap): additional 3-5%

---

## Key Insight

The most surprising finding is that **MLX already implements grouped GEMM at the Metal kernel
level**. The gather_qmm batched kernel (PR #2078) was specifically designed for MoE workloads
and already does what vLLM/SGLang's Triton fused_moe kernels do — but at the hardware level
in Metal. The Python-level sort (gather_sort with argsort) feeds sorted indices to the kernel,
which then identifies contiguous expert groups and batches the matmuls.

This means the optimization opportunity is NOT in "implementing expert batching" (already done)
but in **reducing the number of kernel calls** (gate+up fusion), **eliminating unnecessary
overhead** (better sort thresholds, compiled routing), and **overlapping independent work**
(shared expert parallelism).
