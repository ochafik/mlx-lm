# MLX `gather_mm` and `gather_qmm` Internals

## Summary

**Yes, MLX already batches tokens going to the same expert into a single matmul when `sorted_indices=True`.** This is a true grouped-GEMM primitive at the Metal kernel level -- not just better memory access patterns. The kernel physically iterates over contiguous groups of identical indices within a threadgroup tile, performing one matmul per expert group rather than one per token. This was added in PRs #2040 (gather_mm, May 2025) and #2078 (gather_qmm, May 2025).

## Key Findings

### 1. Three Dispatch Paths in `GatherMM::eval_gpu`

File: `/tmp/mlx-source/mlx/backend/metal/matmul.cpp`, lines 2363-2410

```cpp
void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  // ...
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  // PATH 1: Batched RHS kernel (the fast path for MoE)
  // Activated when: M==1 (each "input" is a single row/token) AND sorted_indices==true
  if (M == 1 && right_sorted_ == true) {
    if (metal::is_nax_available() && ...) {
      return gather_mm_rhs_nax(a, b, rhs_indices, out, d, s);  // Apple M5+ NAX path
    }
    gather_mm_rhs(a, b, rhs_indices, out, d, s);  // Standard batched path
    return;
  }

  // PATH 2: Vector-matrix (gemv) path
  if (M == 1) {
    gather_mv(b, a, rhs_indices, lhs_indices, out, N, K, false, d, s);
    return;
  }
  // ...

  // PATH 3: General gather_mm (per-batch-element dispatch via tid.z)
  gather_mm(a, b, lhs_indices, rhs_indices, out, M, N, K, d, s);
}
```

The critical insight: **Path 1 (`gather_mm_rhs`) is only used when `M==1` AND `right_sorted_==true`.** In the MoE context, `M==1` means each token is a 1-row matrix (a vector), which is the normal case since each token is processed independently against its assigned expert's weight matrix.

### 2. The Batched Kernel Strategy: One Matmul Per Expert Group

File: `/tmp/mlx-source/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h`, lines 20-239

The `gather_mm_rhs` Metal kernel implements **true expert-level batching**. Here is the critical loop structure:

```metal
[[kernel]] void gather_mm_rhs(
    const device T* A,
    const device T* B,
    const device uint32_t* rhs_indices,
    device T* C,
    const constant GEMMParams* params,
    uint simd_lane_id, uint simd_group_id, uint3 tid) {

  // Each threadgroup handles a BM x BN tile of the output
  // BM = 16, BN = 64
  const int c_row = tid.y * BM;

  // CRITICAL: Scan through the sorted indices within this tile
  // to find contiguous groups with the same expert index
  uint32_t index_next = rhs_indices[c_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;          // Current expert index
    offset_next = tgp_bm;

    // Find where the next different expert starts
    for (; n < tgp_bm; n++) {
      if (rhs_indices[c_row + n] != index) {
        offset_next = n;
        index_next = rhs_indices[c_row + n];
        break;
      }
    }

    // Perform ONE matmul for all tokens [offset, offset_next) that share this expert
    // B is indexed by the expert: B + index * batch_stride_b
    loader_b_t loader_b(B + index * params->batch_stride_b, ...);

    // Full GEMM loop for this group
    for (int k = 0; k < gemm_k_iterations; k++) {
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }

    // Store only the relevant slice of the output
    if (offset_next - offset == BM) {
      mma_op.store_result(C, params->ldd);
    } else {
      mma_op.store_result_slice(C, params->ldd, short2(0, offset), short2(BN, offset_next));
    }
  }
}
```

**This is a proper grouped GEMM.** Within each BM-sized tile (BM=16 for the standard kernel, BM=64 for NAX), the kernel:
1. Scans the sorted indices to find contiguous runs of the same expert index
2. Loads the expert's weight matrix (`B + index * batch_stride_b`) once
3. Performs the matmul for all tokens in that group simultaneously
4. Writes only the relevant rows of the output using `store_result_slice`

### 3. Threadgroup Dispatch Strategy

```cpp
// C++ dispatch in gather_mm_rhs (matmul.cpp:1950-1953)
MTL::Size group_dims = MTL::Size(32, wn, wm);  // 32 * 2 * 1 = 64 threads per threadgroup
MTL::Size grid_dims = MTL::Size(params.tiles_n, params.tiles_m, 1);
// tiles_m = ceil(M / BM), tiles_n = ceil(N / BN)
// Only 1 in the Z dimension -- no batch dimension in the grid!
```

**The grid is 2D (tiles_n x tiles_m), NOT 3D.** All tokens are laid out flat along M. There is no per-token dispatch. Instead, each threadgroup handles BM=16 consecutive tokens (rows) and a BN=64 slice of the output columns. Within that tile, the kernel internally loops over expert groups.

This contrasts with the general `gather_mm` kernel which uses `tid.z` for the batch dimension:
```cpp
// General gather_mm dispatch (matmul.cpp:2340-2342)
MTL::Size grid_dims = MTL::Size(params.tiles_n, params.tiles_m, batch_size_out);
// Each tid.z handles one batch element independently
```

### 4. Block Sizes and Adaptive Tuning

The standard `gather_mm_rhs` uses fixed tile sizes:
- BM=16, BN=64, BK=16 (small tiles suited for few tokens per expert)

The NAX variant (`gather_mm_rhs_nax`) adapts tile sizes based on tokens-per-expert:
```cpp
// matmul.cpp:2001-2011
int E = b.shape(0);
if (M / E > 48) {
  bm = 64; wm = 2;   // Many tokens per expert: use large tiles
} else if (M / E > 24) {
  bm = 32; wm = 1;    // Moderate
} else {
  bm = 16; wm = 1;    // Few tokens per expert: small tiles
}
bn = 128; bk = 128;   // Large K and N tiles for NAX
```

This is important: when there are many tokens per expert (large batch/prompt), the kernel uses larger tiles to get better GPU utilization.

### 5. `gather_qmm` -- Identical Strategy for Quantized Weights

File: `/tmp/mlx-source/mlx/backend/metal/quantized.cpp`, lines 1355-1423

`GatherQMM::eval_gpu` follows the exact same pattern but with additional constraints:

```cpp
void GatherQMM::eval_gpu(...) {
  // PATH 1: Batched RHS kernel for quantized
  // Requires: M==1, B>=16, sorted_indices==true, B/E>=4
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(x, w, scales, biases, rhs_indices, out, ...);
    return;
  }

  // PATH 2: General gather_qmm (per-batch dispatch)
  if (M >= vector_limit) {
    gather_qmm(...);
    return;
  }

  // PATH 3: gather_qmv (vector-matrix for small M)
  gather_qmv(...);
}
```

**Additional requirements for the batched quantized path:**
- `B >= 16`: Need at least 16 total tokens (batch elements)
- `B / E >= 4`: Need at least 4 tokens per expert on average

These thresholds prevent using the batched kernel when there are too few tokens to benefit from grouping.

The Metal kernel (`affine_gather_qmm_rhs` in `quantized.h:2157-2341`) uses the identical scan-and-group loop:

```metal
// quantized.h:2233-2250
uint32_t index_next = indices[y_row];
short offset_next = 0;
int n = 0;
while (n < tgp_bm) {
  // ... same grouping logic as gather_mm_rhs ...
  // Load quantized weights for expert `index`
  loader_w_t loader_w(wl + index * stride_w, scales + index * stride_s, biases + index * stride_s, ...);
  // Full GEMM loop
  gemm_loop_aligned(Xs, Ws, mma_op, loader_x, loader_w, K_it);
  // Store slice
  mma_op.store_result_slice(y, N, short2(0, offset), short2(BN, offset_next));
}
```

The quantized kernel uses BM=16, BN=32, BK=32 for the standard path, and BM=64, BN=64, BK=64 for the NAX path.

### 6. NAX Kernels (Apple M5+ / macOS 26.2+)

NAX (Neural Accelerator eXtensions) kernels are available on M5+ chips (architecture gen >= 17 for non-P cores, >= 18 for P cores) running macOS 26.2+. These use larger tile sizes and different instruction sets but follow the same batching strategy:

```cpp
// device.h:268-287
inline bool is_nax_available() {
  can_use_nax &= gen >= (arch == 'p' ? 18 : 17);
  return can_use_nax;
}
```

The NAX gather_mm kernel (`gather_mm_rhs_nax` in `steel_gemm_gather_nax.h`) operates per-simdgroup rather than per-threadgroup, with each simdgroup handling an SM x SN sub-tile and independently scanning its own portion of the sorted indices.

### 7. How mlx-lm Uses This

File: `/Users/ochafik/github/mlx-lm2/mlx_lm/models/switch_layers.py`

The `SwitchGLU` and `SwitchMLP` classes already implement sorting:

```python
class SwitchGLU(nn.Module):
    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))

        # Sort when batch is large enough
        do_sort = indices.size >= 64
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        # These calls use sorted_indices=do_sort
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
```

And `QuantizedSwitchLinear.__call__` passes `sorted_indices` through to `mx.gather_qmm`:

```python
def __call__(self, x, indices, sorted_indices=False):
    x = mx.gather_qmm(
        x, self["weight"], self["scales"], self.get("biases"),
        rhs_indices=indices, transpose=True,
        group_size=self.group_size, bits=self.bits, mode=self.mode,
        sorted_indices=sorted_indices,
    )
```

### 8. Performance Impact (From PR Benchmarks)

**PR #2040 (gather_mm) benchmarks on M2 Ultra:**

| N | D | M | E | I | Before unsorted | Before sorted | With batched kernel |
|---|---|---|---|---|-----------------|---------------|---------------------|
| 1024 | 1024 | 1024 | 32 | 4 | 12.7ms | 10.2ms | **1.9ms** (5.4x) |
| 1024 | 1024 | 1024 | 8 | 4 | 10.3ms | 10.2ms | **1.7ms** (6x) |
| 1024 | 4096 | 1024 | 256 | 4 | 171.8ms | 40.6ms | **26.6ms** (1.5x vs sorted) |
| 1024 | 4096 | 1024 | 8 | 4 | 46.9ms | 39.4ms | **5.5ms** (7.2x) |

**PR #2078 (gather_qmm) real-world MoE benchmarks:**

| Model | Prompt size | Before unsorted | Before sorted | With batched kernel |
|-------|-------------|-----------------|---------------|---------------------|
| Mixtral 8x7B | ~500 | 171 tps | 189 tps | **590 tps** (3.1x) |
| Mixtral 8x7B | ~6000 | 179 tps | 196 tps | **681 tps** (3.5x) |
| Qwen 1.5 2.7B | ~500 | 1071 tps | 1239 tps | **2213 tps** (1.8x) |
| DeepSeek V3 | ~450 | - | 112 tps | **154 tps** (1.4x) |
| DeepSeek V3 | ~2000 | - | 114 tps | **210 tps** (1.8x) |

DeepSeek V3 shows smaller gains because it has 256 experts, meaning fewer tokens per expert on average, which reduces the batching benefit.

## Answers to Key Questions

### Q: When sorted_indices=True, does gather_mm dispatch one matmul per contiguous group (expert)?

**Yes, exactly.** The Metal kernel scans the sorted index array within each BM-sized tile, finds contiguous runs of the same expert index, and performs one full GEMM per run. The weight matrix for that expert is loaded once, and all tokens in the run participate in the same matmul. The result is written back using `store_result_slice` to write only the rows belonging to that expert group.

### Q: Or does it still process each token individually, just with better memory access patterns?

**No, it truly batches.** The tokens are processed together in a single matmul. The A (input) matrix contains multiple token rows, and B (weight) matrix is loaded once per expert. The SIMD groups within the threadgroup collaborate on the same matmul operation. This is fundamentally different from running N separate matvecs.

### Q: Is there a "grouped GEMM" primitive underneath?

**Yes.** The `gather_mm_rhs` and `gather_qmm_rhs` kernels ARE grouped GEMM primitives. They implement the grouping logic directly in the Metal shader rather than dispatching separate kernels per group. Within a single kernel dispatch, the threadgroup loops over expert groups and performs separate matmuls for each, sharing threadgroup memory and synchronization.

### Q: What's the Metal kernel strategy -- one threadgroup per token or one per expert group?

**Neither exactly.** The grid is dispatched as `(tiles_n, tiles_m, 1)` where:
- `tiles_m = ceil(total_tokens / BM)` -- tiles along the token (row) dimension
- `tiles_n = ceil(output_dim / BN)` -- tiles along the output (column) dimension
- Z=1, no batch dimension

Each threadgroup handles BM consecutive tokens (rows) and BN output columns. **Within each threadgroup**, it internally loops over expert groups found in those BM tokens. If all BM tokens go to the same expert, it performs one matmul. If they go to 3 different experts (because BM=16 spans a boundary), it performs 3 separate matmuls within the same threadgroup.

### Q: How does gather_qmm differ in its execution strategy?

**The strategy is identical** -- the quantized kernel uses the same scan-and-group loop. The differences are:
1. **Additional dispatch guards**: `B >= 16` and `B / E >= 4` (need enough tokens per expert)
2. **Different tile sizes**: BM=16, BN=32, BK=32 (standard) or BM=64, BN=64, BK=64 (NAX)
3. **Quantized weight loading**: Uses `QuantizedBlockLoader` instead of `BlockLoader`, which handles dequantization of packed integer weights with scales and biases
4. **Weight stride computation**: Accounts for the packed quantized format (`K_w = K * bytes_per_pack / pack_factor`)

## Implications for mlx-lm

1. **mlx-lm already does expert batching during prompt processing.** The `SwitchGLU`/`SwitchMLP` layers sort tokens by expert when `indices.size >= 64`, and the underlying `gather_qmm` kernel batches contiguous groups into single matmuls.

2. **During token generation (batch_size=1), batching does not help.** With only 1 token, there is nothing to batch. The code falls through to `gather_qmv` (quantized matvec) or `gather_mv` (standard matvec) which simply does an index lookup per expert.

3. **Multi-request batching could unlock this for generation.** If multiple user requests are batched together during generation (e.g., continuous batching), tokens going to the same expert could be grouped and benefit from the batched kernel. This requires `B >= 16` and `B/E >= 4` for the quantized path.

4. **The threshold of 64 tokens for sorting** (in `switch_layers.py`) is somewhat arbitrary. The batched kernel starts being beneficial at BM=16 tokens per tile, but sorting overhead makes it only worthwhile with more tokens.

## Source Files Examined

- `/tmp/mlx-source/mlx/backend/metal/matmul.cpp` - C++ dispatch logic for GatherMM
- `/tmp/mlx-source/mlx/backend/metal/quantized.cpp` - C++ dispatch logic for GatherQMM
- `/tmp/mlx-source/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather.h` - Metal kernel for `gather_mm_rhs`
- `/tmp/mlx-source/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_gather_nax.h` - NAX Metal kernel for `gather_mm_rhs_nax`
- `/tmp/mlx-source/mlx/backend/metal/kernels/quantized.h` - Metal kernel for `affine_gather_qmm_rhs`
- `/tmp/mlx-source/mlx/backend/metal/kernels/quantized_nax.h` - NAX Metal kernel for `affine_gather_qmm_rhs_nax`
- `/tmp/mlx-source/mlx/backend/metal/kernels/fp_quantized.h` - Metal kernel for `fp_gather_qmm_rhs`
- `/tmp/mlx-source/mlx/ops.cpp` - Python-to-C++ binding for `gather_mm` / `gather_qmm`
- `/Users/ochafik/github/mlx-lm2/mlx_lm/models/switch_layers.py` - mlx-lm's MoE layer using `gather_qmm`
- `/tmp/mlx-source/benchmarks/python/gather_mm_bench.py` - Benchmark for gather_mm
- `/tmp/mlx-source/benchmarks/python/gather_qmm_bench.py` - Benchmark for gather_qmm
- PR #2040: https://github.com/ml-explore/mlx/pull/2040 (gather_mm batched kernel)
- PR #2078: https://github.com/ml-explore/mlx/pull/2078 (gather_qmm batched kernel)
