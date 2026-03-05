# vLLM MoE (Mixture of Experts) Implementation Analysis

## Repository: vllm-project/vllm (cloned to tmp/vllm)
## Date: 2026-03-05
## Focus: Expert-parallel strategies during prefill, token batching, and grouped matmul kernels

---

## 1. Architecture Overview

vLLM's MoE implementation is highly modular, decomposed into four stages:

```
[Router] -> [Quantize-Dispatch/Prepare] -> [Permute-Experts-Unpermute] -> [Combine/Finalize]
```

### Key source files:
- **`vllm/model_executor/layers/fused_moe/modular_kernel.py`** (1764 lines) - Defines the abstract component interfaces
- **`vllm/model_executor/layers/fused_moe/fused_moe.py`** (2337 lines) - Triton kernels and TritonExperts implementation
- **`vllm/model_executor/layers/fused_moe/fused_batched_moe.py`** (1128 lines) - Batched expert format variant
- **`vllm/model_executor/layers/fused_moe/moe_align_block_size.py`** - Token-to-expert alignment/padding
- **`vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py`** - CUDA-based permute/unpermute ops
- **`vllm/model_executor/layers/fused_moe/deep_gemm_moe.py`** - DeepGemm grouped GEMM backend
- **`vllm/model_executor/layers/fused_moe/deep_gemm_utils.py`** - Scatter/gather Triton kernels for DeepGemm
- **`vllm/model_executor/layers/fused_moe/cutlass_moe.py`** - CUTLASS grouped GEMM backend
- **`vllm/model_executor/layers/fused_moe/layer.py`** (1554 lines) - FusedMoE nn.Module (the user-facing layer)
- **`vllm/model_executor/layers/fused_moe/config.py`** - Configuration dataclasses
- **`vllm/model_executor/layers/fused_moe/router/`** - Router implementations (fused_topk, grouped_topk)
- **`csrc/moe/moe_align_sum_kernels.cu`** - CUDA kernels for alignment and moe_sum

### Two activation formats:

```python
# modular_kernel.py, line 83-92
class FusedMoEActivationFormat(Enum):
    Standard = ("standard",)         # (num_tokens, hidden_dim)
    BatchedExperts = ("batched_experts",)  # (num_experts, max_tokens_per_expert, hidden_dim)
```

- **Standard format**: Used by Triton, DeepGemm, and standard CUTLASS experts. Tokens stay flat; a sorted_token_ids index redirects each Triton program to the correct expert's weight block.
- **BatchedExperts format**: Used by DeepEP low-latency all2all and batched kernels. Tokens are physically rearranged into a 3D [E, max_tokens, K] tensor.

---

## 2. Top-K Routing and Scoring/Weighting

### 2.1 Standard fused_topk (Softmax/Sigmoid)

**File**: `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`

```python
# Line 69-113
def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    scoring_func: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, _ = hidden_states.size()
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indices = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)

    if scoring_func == "softmax":
        ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
    elif scoring_func == "sigmoid":
        ops.topk_sigmoid(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
```

The CUDA kernel (`csrc/moe/topk_softmax_kernels.cu`) fuses softmax + top-k selection into a single kernel pass. It:
1. Computes softmax over all experts per token
2. Selects top-k experts by score
3. Optionally renormalizes the selected weights so they sum to 1

### 2.2 Grouped Top-K (DeepSeek-V2/V3)

**File**: `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`

```python
# Line 84-165 - The torch.compile'd grouped_topk function
def grouped_topk(hidden_states, gating_output, topk, renormalize,
                 num_expert_group=0, topk_group=0, scoring_func="softmax",
                 routed_scaling_factor=1.0, e_score_correction_bias=None):
    # 1. Score computation
    scores = torch.softmax(gating_output, dim=-1)  # or sigmoid

    # 2. Group-level selection: experts are divided into groups
    # Select top groups first, then top experts within selected groups
    group_scores = scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1)[1]

    # 3. Mask out non-selected groups
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = group_mask.unsqueeze(-1).expand(...).reshape(num_token, -1)
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    # 4. Select top-k from remaining (with optional e_score_correction_bias)
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)[1]
    topk_weights = original_scores.gather(1, topk_ids)  # unbiased weights

    # 5. Optional renormalization and routed_scaling_factor
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
```

There is also a fused CUDA kernel (`ops.grouped_topk`) that handles sigmoid scoring with bias correction in one pass (line 32-75).

### 2.3 Router weight application

Weights can be applied either:
- **On input** (before expert computation): `a1.mul_(topk_weights)` -- used for top-k=1 models
- **On output** (after expert computation): multiplied in the second Triton kernel via `MUL_ROUTED_WEIGHT=True`

---

## 3. Token Grouping/Batching by Expert: `moe_align_block_size`

This is the **central mechanism** for organizing tokens for GPU-efficient computation.

### 3.1 The Standard Path

**File**: `vllm/model_executor/layers/fused_moe/moe_align_block_size.py`, line 11-103

```python
def moe_align_block_size(
    topk_ids: torch.Tensor,       # [total_tokens, top_k]
    block_size: int,               # BLOCK_SIZE_M from Triton config
    num_experts: int,              # total global experts
    expert_map: torch.Tensor | None = None,  # global->local mapping for EP
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

**How it works** (from the CUDA kernel in `csrc/moe/moe_align_sum_kernels.cu`):

1. **Count tokens per expert**: Using shared memory atomics, count how many token-expert pairs belong to each expert.
2. **Prefix sum**: Using CUB BlockScan, compute exclusive prefix sum of (token counts rounded up to block_size) to get each expert's start position.
3. **Sort tokens**: Assign each token-expert pair to its position in the sorted array, indexed by expert.
4. **Pad**: Each expert's token count is padded up to a multiple of `block_size` (typically BLOCK_SIZE_M). Padding slots contain sentinel value = `numel` (total tokens), which the Triton kernel detects via `token_mask = offs_token < num_valid_tokens`.
5. **Build expert_ids**: One expert_id per block of BLOCK_SIZE_M tokens, telling the Triton kernel which expert weight matrix to use.

**Example from docstring** (line 59-72):
```
topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]
block_size = 4, num_experts = 4

Flattened: [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]
Each expert gets 3 tokens -> padded to 4

sorted_token_ids = [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
                    ^-expert 1--^  ^-expert 2---^  ^-expert 3---^  ^-expert 4-^
expert_ids = [1, 2, 3, 4]  (one per block)
num_tokens_post_padded = 16
```

### 3.2 Expert Parallel Handling

When expert parallelism is used, `expert_map` (shape `[global_num_experts]`) maps global expert indices to local indices. Experts not on the current rank have value -1.

```python
# moe_align_block_size.py, line 100-102
if expert_map is not None and not ignore_invalid_experts:
    expert_ids = expert_map[expert_ids]
    # expert_ids now contains -1 for non-local experts
```

The Triton kernel checks for this:
```python
# fused_moe.py, line 437-453 (fused_moe_kernel)
off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
if off_experts == -1:
    # Write back zeros to the output when the expert is not
    # in the current expert parallel rank.
    write_zeros_to_output(...)
    return
```

### 3.3 Batched Alignment (for BatchedExperts format)

**File**: `moe_align_block_size.py`, line 106-193

```python
def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,  # [num_experts] - valid tokens per expert
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
```

This is used when tokens are already organized as `[E, max_tokens_per_expert, hidden]`. The CUDA kernel (`batched_moe_align_block_size_kernel` in `csrc/moe/moe_align_sum_kernels.cu`) uses CUB BlockScan prefix sum to compute expert offsets, handling padding per-expert.

### 3.4 Naive Block Assignment (Small Batch Optimization)

**File**: `fused_moe.py`, line 1826-1853

For very sparse routing (few tokens, many experts), vLLM skips the full sort:

```python
SPARSITY_FACTOR = 4
naive_block_assignment = (
    expert_map is None
    and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
    and not (use_int8_w8a16 or use_int4_w4a16) with block_shape
)

if naive_block_assignment:
    # Each token gets its own BLOCK_SIZE_M block
    expert_ids = curr_topk_ids.view(-1)
    sorted_token_ids = None  # kernel uses pid_m directly
```

In the Triton kernel (line 421-429):
```python
if not naive_block_assignment:
    offs_token_id = pid_m * BLOCK_SIZE_M + offs
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
else:
    offs_token = tl.where(
        offs == 0, pid_m,          # first element = pid_m
        num_valid_tokens,          # remaining = sentinel (skip)
    )
```

---

## 4. Expert Computation Backends

### 4.1 TritonExperts (Standard Path)

**File**: `fused_moe.py`, line 1927-2163

The primary compute kernel. The full MoE forward is:

```python
class TritonExperts(mk.FusedMoEExpertsModular):
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, ...):
        # Step 1: Align tokens to blocks
        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, config["BLOCK_SIZE_M"], global_num_experts, expert_map
        )

        # Step 2: First GEMM (gate_up_proj) - hidden_states @ w1
        invoke_fused_moe_triton_kernel(
            hidden_states, w1, intermediate_cache1,
            ..., sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=False, top_k=top_k_num, ...
        )

        # Step 3: Activation (SiLU, GELU, etc.)
        self.activation(activation, intermediate_cache2, intermediate_cache1.view(-1, N))

        # Step 4: Quantize intermediate for second GEMM
        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(...)

        # Step 5: Second GEMM (down_proj) - activated @ w2, with weight multiplication
        invoke_fused_moe_triton_kernel(
            qintermediate_cache2, w2, intermediate_cache3,
            ..., sorted_token_ids, expert_ids, num_tokens_post_padded,
            mul_routed_weight=not apply_router_weight_on_input, top_k=1, ...
        )

        # Step 6: Reduce across top-k dimension
        ops.moe_sum(intermediate_cache3, output)
```

#### The Triton Kernel Itself (`fused_moe_kernel`)

**File**: `fused_moe.py`, line 314-575

Key design:
- **Grid**: `(cdiv(EM, BLOCK_SIZE_M) * cdiv(N, BLOCK_SIZE_N),)` - one program per (M-block, N-block) pair
- **Group ordering for L2 reuse**: Programs are grouped by `GROUP_SIZE_M` adjacent M-blocks to share weight tiles in L2 cache
- **Token lookup**: Each program reads `sorted_token_ids` to find which tokens to process
- **Expert lookup**: `expert_ids[pid_m]` determines which expert weight to use
- **Token division by top_k**: `offs_token[:, None] // top_k * stride_am` - the original token index is recovered by dividing by top_k since each token appears top_k times in the sorted array
- **FP32 accumulation**: Always accumulates in float32, converts at end
- **Router weight multiplication**: Optionally fused into the second GEMM kernel via `MUL_ROUTED_WEIGHT`

```python
# Key addressing (line 457-459):
a_ptrs = a_ptr + (
    offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
)
# offs_token // top_k recovers original token index since sorted_token_ids
# repeats each token top_k times
```

#### Workspace reuse strategy (line 1730-1746):
```python
# Cache1 and Cache3 share memory (different shapes, sequential use)
cache13 = torch.empty(M * top_k_num * max(N, K), ...)
intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)  # GEMM1 output
intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)  # GEMM2 output

# Cache2 is separate (used concurrently with cache1)
intermediate_cache2 = torch.empty((M * top_k_num, activation_out_dim), ...)  # post-activation
```

### 4.2 DeepGemm Experts (FP8 Grouped GEMM)

**File**: `deep_gemm_moe.py`, line 116-315

Uses the DeepGemm library for hardware-accelerated FP8 grouped GEMMs with contiguous layout. Only supports FP8 with 128x128 block quantization.

```python
class DeepGemmExperts(mk.FusedMoEExpertsModular):
    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids, ...):
        # Step 1: Permute tokens into expert-contiguous layout
        a1q, a1q_scale, expert_ids, inv_perm = deepgemm_moe_permute(
            aq=a1q, aq_scale=a1q_scale, topk_ids=topk_ids,
            local_num_experts=local_num_experts, expert_map=expert_map,
            expert_tokens_meta=expert_tokens_meta, aq_out=a1q_perm,
        )

        # Step 2: First grouped GEMM
        m_grouped_fp8_gemm_nt_contiguous(
            (a1q, a1q_scale), (w1, self.w1_scale), mm1_out, expert_ids
        )

        # Step 3: Fused activation + quantize
        a2q, a2q_scale = self._act_mul_quant(input=mm1_out, output=quant_out, activation=activation)

        # Step 4: Second grouped GEMM
        m_grouped_fp8_gemm_nt_contiguous(
            (a2q, a2q_scale), (w2, self.w2_scale), mm2_out, expert_ids
        )

        # Step 5: Unpermute + weighted reduce
        deepgemm_unpermute_and_reduce(a=mm2_out, topk_ids=topk_ids,
            topk_weights=topk_weights, inv_perm=inv_perm, output=output)
```

#### DeepGemm Permutation (ep_scatter/ep_gather)

**File**: `deep_gemm_utils.py`, line 339-427

The scatter kernel (`ep_scatter`) is a two-pass Triton approach:

**Pass 1** (`_fwd_kernel_ep_scatter_1`, line 61-97):
- Computes per-expert start locations using round-up-to-128 alignment
- Fills `m_indices` (expert_ids for DeepGemm) with expert IDs for valid rows, leaving padding rows as -1

**Pass 2** (`_fwd_kernel_ep_scatter_2`, line 100-165):
- For each token, for each of its top-k expert assignments:
  - Uses `atomic_add` on expert_start_loc to get the destination index
  - Copies the token's hidden states and quantization scales to the destination
  - Records the inverse permutation for later gathering

**Alignment**: Each expert's token block is aligned to 128 (DeepGemm requirement):
```python
# deep_gemm_utils.py, line 17-23
def expert_num_tokens_round_up_and_sum(expert_num_tokens, alignment):
    ent = (expert_num_tokens + (alignment - 1)) // alignment * alignment
    return torch.sum(ent).item()
```

The gather kernel (`ep_gather`, line 297-336) performs the unpermute + weighted reduction:
- For each output token, iterates over its top-k experts
- Loads the expert output using the inverse permutation index
- Multiplies by the routing weight and accumulates in float32

### 4.3 CUTLASS Grouped GEMM

**File**: `cutlass_moe.py`, line 51-263

Supports two modes:

**Standard mode** (non-batched):
```python
# Line 192-207 - uses moe_permute to sort tokens by expert
a1q, a1q_scale, expert_first_token_offset, inv_perm, _ = moe_permute(
    a1q, a1q_scale, topk_ids, num_expert, local_E, expert_map,
    permuted_hidden_states=a1q_perm,
)
# Build problem sizes from expert offsets
ops.get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    expert_first_token_offset, problem_sizes1, problem_sizes2, N, K, swap_ab
)
# swap_ab optimization: when M <= 64, transpose to reduce padding
```

**Batched mode**:
```python
# Line 162-178 - tokens already in [E, max_tokens, K] format
ops.get_cutlass_batched_moe_mm_data(
    expert_offsets, problem_sizes1, problem_sizes2,
    expert_num_tokens, local_E, padded_M, N, K,
)
```

After computation, the standard path uses `moe_unpermute` (line 256-262):
```python
moe_unpermute(
    out=output,
    permuted_hidden_states=mm2_out,
    topk_weights=topk_weights,
    inv_permuted_idx=inv_perm,
    expert_first_token_offset=expert_first_token_offset,
)
```

### 4.4 BatchedTritonExperts

**File**: `fused_batched_moe.py`, line 251-376

This Triton kernel operates on the 3D `[E, max_tokens, K]` format:

```python
# batched_triton_kernel (line 252-376)
# Grid: (num_experts, M_blocks * N_blocks)
expert_id = tl.program_id(axis=0)
pid_mn = tl.program_id(axis=1)

e_num_tokens = tl.load(expert_num_tokens + expert_id)
if e_num_tokens == 0:
    return  # Skip empty experts

# Each program handles one (expert, M-block, N-block)
cta_m_start = pid_m * BLOCK_M
if cta_m_start >= e_num_tokens:
    return  # Skip padding

# Offset into [E, max_tokens, K] tensor
a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
```

The `BatchedPrepareAndFinalize` class (line 492-645) reorganizes tokens:
```python
# Line 589-618 - Scatter tokens to expert batches
for expert_id in range(first_expert, last_expert):
    topks = torch.any(topk_ids == expert_id, dim=1).flatten()
    rows = torch.count_nonzero(topks)
    idx = expert_id - first_expert
    tokens_per_expert[idx] = rows
    b_a1[idx, :rows, :] = a1[:topks.numel()][topks]
```

---

## 5. Permute/Unpermute of Tokens

### 5.1 CUDA-based moe_permute/moe_unpermute

**File**: `moe_permute_unpermute.py`

```python
def moe_permute(
    hidden_states: torch.Tensor,    # [n_token, n_hidden]
    a1q_scale: torch.Tensor | None,
    topk_ids: torch.Tensor,         # [n_token, topk]
    n_expert: int,
    n_local_expert: int = -1,
    expert_map: torch.Tensor | None = None,
    permuted_hidden_states: torch.Tensor | None = None,
) -> tuple:
    # Returns:
    # - permuted_hidden_states: [n_token * topk, n_hidden] sorted by expert
    # - expert_first_token_offset: [n_local_expert + 1] cumulative offsets
    # - inv_permuted_idx: for unpermuting back
    # - permuted_idx: mapping from original to permuted positions

    # Calls C++ kernel: torch.ops._moe_C.moe_permute(...)
```

```python
def moe_unpermute(
    out: torch.Tensor,
    permuted_hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    expert_first_token_offset: torch.Tensor | None = None,
) -> None:
    # Unpermutes and applies topk_weights, writing reduced output
    # Calls C++ kernel: torch.ops._moe_C.moe_unpermute(...)
```

This is used by the CUTLASS backend (standard mode) for contiguous expert grouping with offset-based grouped GEMM.

### 5.2 Triton scatter/gather for DeepGemm

**File**: `deep_gemm_utils.py`

Uses `ep_scatter` (two Triton kernels) and `ep_gather` (one Triton kernel) as described in Section 4.2. Key difference from moe_permute: handles FP8 quantization scales alongside hidden states, and uses 128-byte alignment.

---

## 6. Padding Strategies for GPU-Friendly Batch Sizes

### 6.1 Per-expert padding to BLOCK_SIZE_M

The `moe_align_block_size` function pads each expert's token count to be a multiple of BLOCK_SIZE_M:

```python
# moe_align_block_size.py, line 74
max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
```

Worst case padding overhead: `num_experts * (BLOCK_SIZE_M - 1)` extra tokens.

### 6.2 DeepGemm 128-alignment

```python
# deep_gemm_utils.py, line 77
tokens_per_expert = round_up_128(tokens_per_expert)
# BLOCK_E = 128 -- alignment constant
```

Each expert's token block is rounded up to 128 elements. Padding rows have `expert_ids = -1` so DeepGemm skips them.

### 6.3 CUTLASS swap_ab optimization

```python
# cutlass_moe.py, line 202-203
# When few tokens (M <= 64), transpose the GEMM to reduce padding
swap_ab = a1q.size(0) <= 64
```

### 6.4 Adaptive BLOCK_SIZE_M based on batch size

```python
# fused_moe.py, line 1226-1323 (get_default_config)
if M <= 32:
    block_m = 16
elif M <= 96:
    block_m = 32
elif M <= 512:
    block_m = 64
else:
    block_m = 128
```

Smaller batches use smaller block sizes to reduce padding waste.

### 6.5 Chunk-based processing

```python
# fused_moe.py, line 1698-1699
CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE  # default: 16384
M = min(num_tokens, CHUNK_SIZE)
```

For very large prefills, tokens are processed in chunks of 16K to avoid excessive memory allocation. This is critical for prefill since the intermediate caches scale as `M * top_k * max(N, K)`.

---

## 7. Prefill vs Decode MoE Paths

vLLM does **not** have entirely separate prefill and decode MoE kernels. Instead, the same kernel infrastructure adapts to different batch sizes:

### 7.1 Configuration adaptation

The auto-tuning config selection (`try_get_optimal_moe_config` at line 1326) picks the optimal Triton config based on the number of tokens M:

```python
# Config files in fused_moe/configs/ map batch sizes to kernel parameters:
# E.g., E=128,N=768,device_name=NVIDIA_H200.json might contain:
# {
#   "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1},
#   "64": {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 16},
#   "512": {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 32},
# }
```

### 7.2 Prefill characteristics (large M)
- More tokens per expert -> better GPU utilization
- Larger BLOCK_SIZE_M (64-128) with GROUP_SIZE_M > 1 for L2 reuse
- Chunk processing if M > 16384
- Full `moe_align_block_size` sorting is worthwhile

### 7.3 Decode characteristics (small M)
- Very few tokens (often 1 per sequence)
- Smaller BLOCK_SIZE_M (16) to reduce padding
- `GROUP_SIZE_M = 1` (no benefit from grouping)
- Naive block assignment when tokens * top_k * 4 <= num_experts
- EM optimization to skip unnecessary blocks:
  ```python
  # fused_moe.py, line 773-779
  if A.size(0) < config["BLOCK_SIZE_M"]:
      # Fewer tokens than one block -> cap EM to skip invalid blocks
      EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
  ```

### 7.4 The DeepGemm path sensitivity

DeepGemm has a minimum M requirement:
```python
# deep_gemm_moe.py, line 44-46
def _valid_deep_gemm_shape(M, N, K):
    align = get_mk_alignment_for_contiguous_layout()[0]  # typically 128
    return align <= M and N % align == 0 and K % align == 0
```

For decode with M < 128, DeepGemm falls back to Triton (via `TritonOrDeepGemmExperts`). For prefill, DeepGemm is typically used since M > 128.

---

## 8. Expert Parallelism Strategy

### 8.1 Expert Distribution

**File**: `layer.py`, line 67-153

```python
def determine_expert_map(ep_size, ep_rank, global_num_experts,
                         expert_placement_strategy="linear", ...):
    # Linear: contiguous expert assignment
    # start_idx = ep_rank * base_experts + min(ep_rank, remainder)
    # expert_map[start_idx : start_idx + local_num_experts] = torch.arange(...)

    # Round-robin: interleaved assignment (for models with expert groups)
    # local_log_experts = torch.arange(ep_rank, global_num_experts, ep_size)
```

### 8.2 All2All Communication Backends

**File**: `all2all_utils.py` and `config.py`

```python
# config.py, line 925-975 (FusedMoEParallelConfig properties)
use_all2all_kernels = dp_size > 1 and use_ep
use_deepep_ht_kernels = use_all2all_kernels and all2all_backend == "deepep_high_throughput"
use_deepep_ll_kernels = use_all2all_kernels and all2all_backend == "deepep_low_latency"
use_fi_all2allv_kernels = use_all2all_kernels and all2all_backend == "flashinfer_all2allv"
use_naive_all2all_kernels = use_all2all_kernels and all2all_backend in ["naive", "allgather_reducescatter"]
```

Available backends:
- **naive**: Standard PyTorch all-to-all
- **allgather_reducescatter**: Alternative collective approach
- **deepep_high_throughput**: DeepEP library, high throughput mode
- **deepep_low_latency**: DeepEP library, low latency mode (uses BatchedExperts format)
- **flashinfer_all2allv**: FlashInfer all-to-all-v
- **mori**: Mori communication backend

### 8.3 The PrepareAndFinalize abstraction

The `FusedMoEPrepareAndFinalizeModular` class provides the dispatch/combine interface:

```python
# For no-DP/EP (single GPU): MoEPrepareAndFinalizeNoDPEPModular
#   - prepare() just quantizes input, passes through
#   - finalize() applies topk_weights and reduces

# For DeepEP high-throughput: DeepEPHTPrepareAndFinalize
#   - prepare() quantizes, then does all2all dispatch to expert-owning ranks
#   - finalize() does all2all combine, applies weights, reduces

# For DeepEP low-latency: DeepEPLLPrepareAndFinalize
#   - Uses BatchedExperts format
#   - prepare() reorganizes into [E, max_tokens, K]
#   - finalize() gathers results back
```

### 8.4 Async dispatch/compute overlap (DBO)

The modular kernel supports `prepare_async` / `finalize_async` for overlapping communication with computation:

```python
# modular_kernel.py, line 287-343
def supports_async(self) -> bool: ...
def prepare_async(self, a1, topk_weights, topk_ids, ...) -> tuple[Callable, ReceiverType]:
    # Returns a (hook, receiver) pair
    # hook() is lightweight check that recv is complete
    # receiver() waits and returns the data
```

---

## 9. Model Integration Examples

### 9.1 Mixtral

**File**: `vllm/model_executor/models/mixtral.py`, line 74-152

```python
class MixtralMoE(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size, intermediate_size, ...):
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False, ...)
        self.experts = FusedMoE(
            num_experts=num_experts, top_k=top_k,
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            reduce_results=True, renormalize=True, ...
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = self.experts(hidden_states, router_logits)
        return final_hidden_states.view(orig_shape)
```

The FusedMoE layer internally handles routing (via FusedTopKRouter with softmax), token sorting, expert computation, and reduction. Mixtral uses 8 experts with top_k=2.

### 9.2 DeepSeek-V2/V3 (GroupedTopK)

Uses `GroupedTopKRouter` with:
- `num_expert_group` (e.g., 8 groups)
- `topk_group` (e.g., select top 4 groups)
- `e_score_correction_bias` for load balancing
- `routed_scaling_factor` for output scaling
- 256 experts with top_k=8 (DeepSeek-V3)

---

## 10. GPT-OSS Triton Kernels (Advanced Alternative)

**File**: `gpt_oss_triton_kernels_moe.py`

An alternative kernel path using the `triton_kernels` package with a very different approach:

```python
# Uses bitmatrix representation for expert-token mapping
# pack_bitmatrix kernel (line 68-79): packs topk_ids into a compact bit array
# Then uses matmul_ogs (output-gathered-sparse) kernel which natively handles
# the sparse expert-token mapping without explicit sorting
```

This approach uses:
- **Bitmatrix**: Compact representation of which tokens go to which experts
- **matmul_ogs**: A sparse matmul that does the gather/scatter internally
- **Ragged tensor metadata**: For variable-length per-expert batches

---

## 11. Summary of Key Design Patterns

### Token Flow Through MoE:

```
Input: hidden_states [M, K]
  |
  v
Router: gate(hidden_states) -> router_logits [M, E]
  |
  v
TopK Selection: fused_topk/grouped_topk -> topk_weights [M, topk], topk_ids [M, topk]
  |
  v
Token Alignment: moe_align_block_size(topk_ids, BLOCK_SIZE_M, E)
  -> sorted_token_ids, expert_ids, num_tokens_post_padded
  |
  v
GEMM1: fused_moe_kernel(hidden_states, w1, cache1, sorted_token_ids, expert_ids)
  -> cache1 [M, topk, 2*N]  (gate+up projection)
  |
  v
Activation: SiLU(cache1[:N]) * cache1[N:]  -> cache2 [M*topk, N]
  |
  v
Optional Quantize: quantize(cache2) -> qcache2
  |
  v
GEMM2: fused_moe_kernel(qcache2, w2, cache3, sorted_token_ids, expert_ids)
  -> cache3 [M, topk, K]  (with optional weight multiplication)
  |
  v
Reduce: moe_sum(cache3) -> output [M, K]
```

### Key Insight: There is no separate "expert parallel during prefill" path

Instead, vLLM uses a single unified path that adapts:

1. **Routing** is always local (each rank runs the full router on its tokens)
2. **Expert assignment** handles both local and remote experts via `expert_map`
3. **Communication** (all2all) happens in the PrepareAndFinalize layer
4. **Computation** kernels see only local experts; non-local experts are -1 in expert_ids
5. **Batch size adaptation** via auto-tuned configs handles the prefill/decode spectrum

The fundamental design is that tokens are sorted/permuted to be contiguous per expert, then a fused kernel processes all expert-token blocks in one launch, with each Triton program handling one (M-block, N-block) tile. The expert weight is selected by looking up `expert_ids[pid_m]`, and the sorted_token_ids array maps each position back to the original token's hidden state.
