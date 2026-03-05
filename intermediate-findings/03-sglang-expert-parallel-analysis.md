# SGLang MoE Expert-Parallel Analysis

Comprehensive analysis of SGLang's Mixture-of-Experts implementation, focusing on expert-parallel strategies during prefill, token dispatch/combine patterns, grouped GEMM kernels, and the prefill vs. decode dichotomy.

Repository version: shallow clone of `https://github.com/sgl-project/sglang.git` (HEAD as of March 2026).

---

## 1. Architecture Overview

SGLang's MoE implementation follows a modular, extensible pipeline:

```
[input_hidden_states]
        |
        v
   TopK.forward -> select_experts / triton_kernels.routing / bypass
        |
        v
   [TopKOutput]
        |
        v
 FusedMoE.forward -> Dispatcher.dispatch -> DeepEP / Standard / bypass
        |                     |
        |                     v
        |              [DispatchOutput]
        |                     |
        |                     v
        |             quant_method.apply -> MoeRunner.forward
        |                     |              |
        |                     |              v
        |                     | pre-permute + grouped_gemm + post-permute
        |                     |              |
        |                     |--------------
        |                     v
        |               [CombineInput]
        |                     |
        |                     v
        |            Dispatcher.combine -> DeepEP / bypass
        |                     |
        |---------------------
        v
[final_hidden_states]
```

Source: `/python/sglang/srt/layers/moe/fused_moe_triton/layer.py` and `/docs/advanced_features/expert_parallelism.md`

### Key Components

| Component | File | Role |
|-----------|------|------|
| `FusedMoE` | `layers/moe/fused_moe_triton/layer.py:147` | Main MoE layer, unified entry point |
| `DeepEPMoE` | `layers/moe/ep_moe/layer.py:70` | Expert-parallel MoE using DeepEP |
| `TopK` | `layers/moe/topk.py:232` | Top-K router with multi-platform support |
| `StandardDispatcher` | `layers/moe/token_dispatcher/standard.py:81` | Non-EP or hybrid EP dispatch |
| `DeepEPDispatcher` | `layers/moe/token_dispatcher/deepep.py:731` | DeepEP all-to-all dispatch |
| `FlashinferDispatcher` | `layers/moe/token_dispatcher/flashinfer.py` | FlashInfer-based dispatch |
| `MoeRunner` | `layers/moe/moe_runner/runner.py:26` | Orchestrates grouped-GEMM execution |
| `DeepGemmRunnerCore` | `layers/moe/moe_runner/deep_gemm.py:112` | DeepGEMM backend for FP8 grouped GEMMs |
| `TritonRunnerCore` | `layers/moe/moe_runner/triton.py` | Triton-based grouped GEMMs |

---

## 2. Top-K Routing and Scoring/Weighting

### 2.1 Routing Functions

File: `/python/sglang/srt/layers/moe/topk.py`

SGLang supports multiple routing strategies:

**Standard softmax/sigmoid top-k** (`fused_topk`, line 476):
```python
def fused_topk(hidden_states, gating_output, topk, renormalize, ...):
    M, _ = hidden_states.shape
    topk_weights = torch.empty(M, topk, dtype=torch.float32, ...)
    topk_ids = torch.empty(M, topk, dtype=torch.int32, ...)
    if scoring_func == "softmax":
        topk_softmax(topk_weights, topk_ids, gating_output, renormalize)
    elif scoring_func == "sigmoid":
        topk_sigmoid(topk_weights, topk_ids, gating_output, renormalize, correction_bias)
```

Both `topk_softmax` and `topk_sigmoid` are CUDA kernels from `sgl_kernel` (C++ implementations in `sgl-kernel/csrc/moe/moe_topk_softmax_kernels.cu` and `moe_topk_sigmoid_kernels.cu`).

**DeepSeek V2/V3/R1 grouped top-k** (`grouped_topk_gpu`, line 516):
```python
def grouped_topk_gpu(hidden_states, gating_output, topk, renormalize,
                     num_expert_group, topk_group, ...):
    scores = torch.softmax(gating_output, dim=-1)
    group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    # ... mask and select
```

This is decorated with `@torch.compile(dynamic=True)` for performance.

**DeepSeek V3 biased grouped top-k** (`biased_grouped_topk_gpu`, line 740):
Uses sigmoid scoring with correction bias. Has multiple fast paths:
1. `fused_topk_deepseek` from FlashInfer (CUDA kernel)
2. `moe_fused_gate` from sgl_kernel (CUDA kernel)
3. `kimi_k2_moe_fused_gate` for Kimi K2 (384 experts, 1 expert group)
4. `biased_grouped_topk_impl` (torch.compile fallback)

### 2.2 Fused Router Kernel

File: `/python/sglang/srt/layers/moe/router.py`

SGLang has a custom **fused router** that combines the router linear projection with top-k selection in a single kernel:

```python
class FusedMoeRouter:
    def forward_cuda(self, x, autotune=False):
        return fused_moe_router_shim(
            moe_softcapping=self.moe_softcapping,
            hidden_states=x,
            gating_output=self.router_linear.weight,  # NOT logits, raw weights
            topk=self.topk,
            renormalize=False,
        )
```

Two Triton kernels:
- **CUDA-core kernel** (`fused_moe_router_cudacore_kernel`, line 14): Used for small batches (bs < 512) or few experts (<=8). Computes `logits = sum(w_router * x)` per-element, then iterative argmax for top-k.
- **Tensor-core kernel** (`fused_moe_router_tensorcore_kernel`, line 159): Used for large batches (bs >= 512) or many experts (>8). Uses `tl.dot` for matrix multiply, supports top-k <= 2.

Both support **logit softcapping** (tanh-based) and **correction bias** (added after softcapping).

### 2.3 TopKOutput Formats

File: `/python/sglang/srt/layers/moe/topk.py`

Three output formats allow different downstream consumers:

```python
class TopKOutputFormat(IntEnum):
    STANDARD = auto()      # (topk_weights, topk_ids, router_logits)
    TRITON_KERNEL = auto() # (routing_data, gather_indx, scatter_indx)
    BYPASSED = auto()      # Deferred routing for FlashInfer/TRT-LLM
```

The `BYPASSED` format is used when the MoE runner backend handles routing internally (e.g., `flashinfer_trtllm` or `flashinfer_mxfp4`), passing raw hidden states and router logits downstream.

---

## 3. Token Grouping by Expert (moe_align_block_size)

### 3.1 The Core Algorithm

File: `/python/sglang/srt/layers/moe/fused_moe_triton/moe_align_block_size.py`

This is the critical function that groups tokens by expert and pads to GPU-friendly block sizes:

```python
def moe_align_block_size(topk_ids, block_size, num_experts):
    """
    Returns:
    - sorted_token_ids: sorted by expert assignment, padded to block_size
    - expert_ids: which expert each block maps to
    - num_tokens_post_padded: total tokens after padding
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, ...)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, ...)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, ...)
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, ...)

    sgl_moe_align_block_size(topk_ids, num_experts + 1, block_size,
                              sorted_ids, expert_ids, num_tokens_post_pad,
                              cumsum_buffer, True)
```

### 3.2 CUDA Kernel Implementation

File: `/sgl-kernel/csrc/moe/moe_align_kernel.cu`

The CUDA kernel (`moe_align_block_size_kernel`, line 56) does:

1. **Count tokens per expert** using shared memory atomics:
```c++
for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i] + 1;  // +1 because -1 means "filtered"
    atomicAdd(&shared_counts[expert_id], 1);
}
```

2. **Compute padded counts** (round up to block_size):
```c++
int32_t padded_count = (count + block_size - 1) / block_size * block_size;
```

3. **Exclusive prefix sum** (Blelloch scan on CUDA, warp-scan on HIP) to compute expert offsets.

4. **Sort tokens into expert groups**: A separate thread block fills `sorted_token_ids` with the padding value (num_valid_tokens), then atomically assigns each token to its expert's region:
```c++
// In count_and_sort_expert_tokens_kernel:
int32_t expert_id = topk_ids[i] + 1;
int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
sorted_token_ids[rank_post_pad] = i;
```

The padding strategy ensures each expert's token count is a multiple of `block_size` (typically 16, 32, 64, or 128 depending on the Triton config), with padding tokens set to `numel` (an out-of-range index that gets masked in the kernel).

### 3.3 Expert ID -1 for Filtered Experts

In expert-parallel mode, `topk_ids` values of -1 indicate tokens routed to experts on other ranks. The kernel maps -1 to expert index 0 (via `topk_ids[i] + 1`), and the `expert_ids` array uses -1 to signal blocks that should be zeroed out rather than computed:

```python
# In fused_moe_kernel (line 440):
if filter_expert and off_experts == -1:
    write_zeros_to_output(...)
    return
```

---

## 4. Fused MoE Triton Kernel

### 4.1 Main Kernel Structure

File: `/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py`

The kernel (`fused_moe_kernel`, line 324) performs blocked GEMM with expert routing:

```python
@triton.jit
def fused_moe_kernel(
    a_ptr, a_desc, b_ptr, b_desc, bias_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    # ... strides ...
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    GROUP_SIZE_M, MUL_ROUTED_WEIGHT, top_k,
    compute_type, use_fp8_w8a8, use_int8_w8a8,
    ...
):
```

Key design choices:

1. **L2 cache optimization**: Uses grouped ordering (`GROUP_SIZE_M`) for program IDs to promote L2 data reuse across adjacent blocks.

2. **Token lookup via sorted_token_ids**: Each block loads its token indices from `sorted_token_ids`, then computes `offs_token // top_k` to find the original token index in the input:
```python
offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
# Input A is indexed by original token: offs_token // top_k
a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + ...)
```

3. **Expert weight selection**: Each block reads its expert ID from `expert_ids`, then uses it to offset into the stacked weight tensor B `[E, N, K]`:
```python
off_experts = tl.load(expert_ids_ptr + pid_m)
b_ptrs = b_ptr + off_experts * stride_be + ...
```

4. **Router weight multiplication**: Optionally multiplies the output by the routing weight:
```python
if MUL_ROUTED_WEIGHT:
    moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
    accumulator *= moe_weight[:, None]
```

5. **TMA (Tensor Memory Access)**: On SM90+ (Hopper), supports TMA descriptors for both A (input) and B (weights) to accelerate memory access:
```python
if a_desc is not None:
    a = a_desc.load([start_offs_m, k_start])
```

6. **swap_ab optimization**: On SM90 GPUs, when `BLOCK_SIZE_M < 64` and `BLOCK_SIZE_N >= 64`, transposes A and B to get better tensor core utilization.

7. **Fused sum all-reduce**: For EP, can fuse the final reduction across top-k experts with all-reduce using atomic adds:
```python
if FUSE_SUM_ALL_REDUCE:
    offs_token_out = offs_token // ROUTER_TOPK
    tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
```

### 4.2 Two-Stage Execution

File: `/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`

The `fused_experts_impl` function (line 323) executes two fused MoE kernels:

**Stage 1 - Gate-Up Projection (w1):**
```python
sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    curr_topk_ids, config["BLOCK_SIZE_M"], E)

invoke_fused_moe_kernel(
    curr_hidden_states, w1, b1, intermediate_cache1,
    a1_scale, w1_scale, ...,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    apply_router_weight_on_input, topk_ids.shape[1], config, ...)
```

**Activation function** (SiLU, GELU, etc.) applied between stages. Uses a custom Triton kernel `act_and_mul_kernel` that handles expert filtering:
```python
if filter_expert:
    act_and_mul_triton(intermediate_cache1, intermediate_cache2, config,
                       topk_ids, expert_ids, down_moe_use_tma, activation)
```

**Stage 2 - Down Projection (w2):**
```python
invoke_fused_moe_kernel(
    intermediate_cache2, w2, b2,
    intermediate_cache3 if not no_combine else out_hidden_states,
    a2_scale, w2_scale, ...,
    sorted_token_ids, expert_ids, num_tokens_post_padded,
    not apply_router_weight_on_input, 1, down_config or config, ...)
```

**Final reduction** across top-k experts:
```python
# For topk == 1: write directly to output
# For topk == 2: torch.add of two slices
# For topk > 2: moe_sum_reduce (CUDA kernel) or torch.compile fallback
moe_sum_reduce(intermediate_cache3, out_hidden_states, routed_scaling_factor)
```

### 4.3 Chunking Strategy

For very large token counts (>64K), the implementation chunks:
```python
CHUNK_SIZE = 64 * 1024
for chunk in range((num_tokens // CHUNK_SIZE) + 1):
    begin_chunk_idx = chunk * CHUNK_SIZE
    end_chunk_idx = min((chunk + 1) * CHUNK_SIZE, num_tokens)
    # ... process chunk ...
```

---

## 5. Expert Parallelism: Token Dispatch and Combine

### 5.1 Dispatch Architecture

The dispatcher abstraction (`BaseDispatcher`) defines:
- `dispatch(hidden_states, topk_output) -> DispatchOutput`: Sends tokens to the correct EP rank
- `combine(combine_input) -> hidden_states`: Gathers results back

### 5.2 StandardDispatcher (Non-EP or Hybrid EP)

File: `/python/sglang/srt/layers/moe/token_dispatcher/standard.py:81`

For hybrid EP (EP + TP with `ep_size < tp_size`), uses all-reduce or all-gather:

```python
class StandardDispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # Quantize to FP4 before all-gather to reduce communication
            x, x_sf = fp4_quantize_flashinfer(hidden_states, global_scale, ...)
            topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens())

        # Map global expert IDs to local expert IDs
        if self.moe_ep_size > 1:
            self.local_expert_mapping[
                rank * num_local_routed : (rank+1) * num_local_routed
            ] = torch.arange(0, num_local_routed, ...)
            # Experts on other ranks map to -1
            topk_output = topk_output._replace(
                topk_ids=self.local_expert_mapping[topk_output.topk_ids])
```

The combine step just does `reduce_scatterv` if FP4 all-gather was used.

### 5.3 DeepEP Dispatcher (Full EP)

File: `/python/sglang/srt/layers/moe/token_dispatcher/deepep.py`

The DeepEP dispatcher has two modes, automatically selected based on whether the batch contains prefill (extend) or decode tokens:

```python
class DeepEPDispatcher(BaseDispatcher):
    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher   # Prefill path
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher  # Decode path
```

**DeepEPMode.AUTO resolution** (line 113):
```python
def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
    if self != DeepEPMode.AUTO:
        return self
    if is_extend_in_batch:
        return DeepEPMode.NORMAL      # Prefill -> normal (high throughput)
    else:
        return DeepEPMode.LOW_LATENCY  # Decode -> low latency
```

#### 5.3.1 Normal Mode (Prefill)

File: `/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:372`

Two-phase dispatch with async support:

**dispatch_a** (before communication):
```python
def dispatch_a(self, hidden_states, topk_output):
    topk_ids = topk_ids.to(torch.int64)
    if ENABLE_JIT_DEEPGEMM and not BF16_DISPATCH:
        # Quantize to FP8 before dispatch to reduce communication
        hidden_states = sglang_per_token_group_quant_fp8(hidden_states, 128, ...)
    previous_event = Buffer.capture() if self.async_finish else None
    return hidden_states, topk_ids, topk_weights, previous_event
```

**_dispatch_core** (the actual all-to-all):
```python
def _dispatch_core(self, x, topk_ids, topk_weights, previous_event):
    buffer = self._get_buffer()
    # First: compute dispatch layout (which tokens go where)
    (num_tokens_per_rank, num_tokens_per_rdma_rank,
     num_tokens_per_expert, is_token_in_rank, previous_event
    ) = buffer.get_dispatch_layout(topk_ids, self.num_experts, ...)

    # Then: actual all-to-all dispatch
    (recv_x, recv_topk_ids, recv_topk_weights,
     num_recv_tokens_per_expert, self.handle, event
    ) = buffer.dispatch(
        x, topk_idx=topk_ids, topk_weights=topk_weights,
        num_tokens_per_rank=num_tokens_per_rank,
        expert_alignment=128 if ENABLE_JIT_DEEPGEMM else 1,
        config=DeepEPConfig.get_instance().normal_dispatch_config)
```

Key: `expert_alignment=128` ensures received tokens are aligned to 128 for DeepGEMM's block sizes.

**dispatch_b** (after communication):
```python
def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
    (hidden_states, topk_ids, topk_weights,
     num_recv_tokens_per_expert, event
    ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
    event.current_stream_wait()
    return DeepEPNormalDispatchOutput(
        hidden_states, hidden_states_scale, topk_ids, topk_weights,
        num_recv_tokens_per_expert)
```

The output `num_recv_tokens_per_expert` is a `List[int]` telling each expert how many tokens it received.

#### 5.3.2 Low-Latency Mode (Decode)

File: `/python/sglang/srt/layers/moe/token_dispatcher/deepep.py:534`

Optimized for small batches (decode), uses `buffer.low_latency_dispatch`:

```python
def _dispatch_core(self, hidden_states, topk_ids):
    use_fp8 = not SGLANG_DEEPEP_BF16_DISPATCH
    packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
        buffer.low_latency_dispatch(
            hidden_states, topk_ids,
            self.num_max_dispatch_tokens_per_rank,  # <= 1024
            self.num_experts,
            use_fp8=use_fp8,
            round_scale=DEEPGEMM_BLACKWELL,
            use_ue8m0=DEEPGEMM_BLACKWELL,
        )
    )
    return packed_recv_hidden, self.packed_recv_count, event, hook
```

The output is shaped `[num_local_experts, expected_m, hidden_size]` where:
```python
expected_m = (num_tokens * ep_world_size * topk + num_experts) // num_experts
```

And `masked_m` (per-expert actual counts) replaces `num_recv_tokens_per_expert`.

---

## 6. Permute/Unpermute: Token Reordering for Expert Groups

### 6.1 Pre-Permute: Dispatch -> Runner Input

The `PermuteMethodPool` (in `moe_runner/base.py:124`) registers format-specific conversion functions. Key conversions:

#### Standard -> DeepGemm (`pre_permute_standard_to_deep_gemm`, deep_gemm.py:362)

For Standard dispatch (non-EP or hybrid EP):

```python
def pre_permute_standard_to_deep_gemm(dispatch_output, quant_info, runner_config, running_state):
    hidden_states, topk_output = dispatch_output.hidden_states, dispatch_output.topk_output
    topk_weights, topk_ids, _ = topk_output

    # moe_ep_deepgemm_preprocess: sort, compute seg_indptr, scatter into expert groups
    masked_m, expected_m, src2dst, hidden_states, hidden_states_scale = (
        moe_ep_deepgemm_preprocess(topk_ids, num_local_experts, hidden_states, top_k, block_shape))

    return DeepGemmRunnerInput(
        hidden_states=hidden_states,           # [num_experts, m_max, hidden_size], FP8
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,  # [num_experts], actual count per expert
        expected_m=expected_m)
```

#### DeepEP Normal -> DeepGemm (`pre_permute_deepep_normal_to_deep_gemm`, deep_gemm.py:496)

```python
def pre_permute_deepep_normal_to_deep_gemm(dispatch_output, quant_info, runner_config, running_state):
    (hidden_states, hidden_states_scale, topk_ids, topk_weights,
     num_recv_tokens_per_expert) = dispatch_output

    all_tokens = sum(num_recv_tokens_per_expert)
    # Scatter tokens into contiguous per-expert blocks
    ep_scatter(hidden_states, hidden_states_scale, topk_ids,
               num_recv_tokens_per_expert_gpu, expert_start_loc,
               input_tensor, input_tensor_scale, m_indices, output_index, ...)

    return DeepGemmRunnerInput(
        hidden_states=input_tensor,        # [all_tokens, K], contiguous per expert
        hidden_states_scale=input_tensor_scale,
        use_masked_gemm=False,
        m_indices=m_indices)               # expert boundaries for grouped GEMM
```

#### DeepEP Low-Latency -> DeepGemm (`pre_permute_deepep_ll_to_deep_gemm`, deep_gemm.py:454)

Low-latency mode already produces data in the right format:
```python
def pre_permute_deepep_ll_to_deep_gemm(dispatch_output, ...):
    hidden_states, hidden_states_scale, topk_ids, topk_weights, masked_m, expected_m = dispatch_output
    return DeepGemmRunnerInput(
        hidden_states=hidden_states,           # [num_experts, m, hidden_size]
        hidden_states_scale=hidden_states_scale,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=expected_m)
```

### 6.2 The `moe_ep_deepgemm_preprocess` Function

File: `/python/sglang/srt/layers/moe/ep_moe/kernels.py:1041`

This is the critical preprocessing for standard dispatch to DeepGEMM:

```python
def moe_ep_deepgemm_preprocess(topk_ids, num_local_experts, hidden_states, top_k, block_shape):
    # 1. Sort topk_ids to group tokens by expert
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)

    # 2. Binary search for expert boundaries
    compute_seg_indptr_triton_kernel[(num_local_experts + 1,)](
        reorder_topk_ids, seg_indptr, topk_ids.numel())

    # 3. Compute per-expert counts
    compute_masked_m_triton_kernel[(num_local_experts,)](seg_indptr, masked_m)

    # 4. Pad M dimension: m_max = ceil(num_tokens / 256) * 256
    m_max = (hidden_states.size(0) // 256 + 1) * 256

    # 5. Allocate output: [num_experts, m_max, hidden_size]
    gateup_input = torch.empty(
        (num_local_experts, m_max, hidden_states.size(1)),
        device=hidden_states.device, dtype=output_dtype)

    # 6. Compute src2dst mapping with expert-aware offsets
    deepgemm_compute_src2dst_triton_kernel[grid](
        topk_ids, reorder_ids, seg_indptr, src2dst, m_max, ...)

    # 7. Quantize input to FP8
    hidden_states, scale = per_token_group_quant_fp8(hidden_states, block_k)

    # 8. Scatter tokens into expert groups
    fill_gateup_input_triton_kernel[(hidden_states.shape[0],)](
        hidden_states, scale, gateup_input, gateup_input_scale,
        src2dst, topk_ids, top_k, hidden_size, scale_size, ...)

    return masked_m, expected_m, src2dst, gateup_input, gateup_input_scale
```

### 6.3 Post-Permute: Runner Output -> Combine Input

#### DeepGemm -> Standard (`post_permute_deep_gemm_to_standard`, deep_gemm.py:413)

```python
def post_permute_deep_gemm_to_standard(runner_output, quant_info, runner_config, running_state):
    output = torch.empty(hidden_states_shape, ...)
    # Scatter results back to original token order, applying routing weights
    post_reorder_triton_kernel[(hidden_states_shape[0],)](
        runner_output.hidden_states, output, src2dst, topk_ids, topk_weights,
        top_k, hidden_size, BLOCK_SIZE=512)
    if runner_config.routed_scaling_factor is not None:
        output *= runner_config.routed_scaling_factor
    return StandardCombineInput(hidden_states=output)
```

The `post_reorder_triton_kernel` (in ep_moe/kernels.py:105) reverses the scatter, accumulating weighted results:
```python
@triton.jit
def deepep_post_reorder_triton_kernel(...):
    src_idx = tl.program_id(0)
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        sum_vec = tl.zeros([BLOCK_SIZE], dtype=InDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                weigh_scale = tl.load(topk_weights_ptr + idx)
                in_data = tl.load(down_output_ptr + dst_idx * hidden_size + offset)
                sum_vec += in_data * weigh_scale
        tl.store(output_ptr + src_idx * hidden_size + offset, sum_vec)
```

#### DeepGemm -> DeepEP Normal (`post_permute_deep_gemm_to_deepep_normal`, deep_gemm.py:588)

```python
def post_permute_deep_gemm_to_deepep_normal(runner_output, ...):
    # Gather results back into dispatch order
    ep_gather(hidden_states, topk_ids, topk_weights, output_index, gather_out)
    return DeepEPNormalCombineInput(
        hidden_states=gather_out, topk_ids=..., topk_weights=...)
```

#### DeepGemm -> DeepEP Low-Latency (`post_permute_deep_gemm_to_deepep_ll`, deep_gemm.py:480)

No reordering needed; just pass through:
```python
def post_permute_deep_gemm_to_deepep_ll(runner_output, ...):
    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=..., topk_weights=...)
```

---

## 7. DeepGEMM Runner: Contiguous vs. Masked Grouped GEMM

### 7.1 Contiguous Grouped GEMM (Normal Mode Prefill)

File: `/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:134`

Used when tokens have been scattered into contiguous per-expert blocks with `m_indices`:

```python
def _run_contiguous_gemm(self, runner_input, quant_info, running_state):
    # GateUp GEMM: [all_tokens, K] x [E, N, K]^T -> [all_tokens, N]
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (hidden_states, hidden_states_scale),
        w13_weight_fp8,
        gateup_output,
        m_indices)  # m_indices tells which expert each contiguous block belongs to

    # Activation: SiLU + Mul
    silu_and_mul(gateup_output.view(-1, N), down_input)

    # Quantize for next GEMM
    down_input_fp8, down_input_scale = sglang_per_token_group_quant_fp8(down_input, 128, ...)

    # Down GEMM: [all_tokens, N/2] x [E, K, N/2]^T -> [all_tokens, K]
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_contig(
        (down_input_fp8, down_input_scale),
        w2_weight_fp8,
        down_output,
        m_indices)
```

### 7.2 Masked Grouped GEMM (Low-Latency Decode)

File: `/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:217`

Used when data is in `[num_experts, m, hidden_size]` layout with `masked_m` per-expert counts:

```python
def _run_masked_gemm(self, runner_input, quant_info, running_state):
    num_groups, m, k = hidden_states.shape  # [num_experts, max_m, hidden_size]

    # GateUp GEMM with masking
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        (hidden_states, hidden_states_scale),
        (w13_weight, w13_scale),
        gateup_output,      # [num_experts, m, N]
        masked_m,           # [num_experts], actual count per expert
        expected_m)

    # Fused activation + quantization (respects masked_m)
    silu_and_mul_masked_post_quant_fwd(gateup_output, down_input, down_input_scale,
                                       scale_block_size, masked_m)

    # Down GEMM with masking + optional overlap
    deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
        (down_input, down_input_scale),
        (w2_weight, w2_scale),
        down_output,
        masked_m, expected_m,
        **gemm_overlap_args_dict)  # Can overlap with combine communication
```

The masked GEMM only computes rows `[0, masked_m[i])` for expert `i`, skipping padding rows entirely.

---

## 8. Prefill vs. Decode MoE Paths

### 8.1 The Decision Point

File: `/python/sglang/srt/layers/moe/utils.py:101`

```python
class DeepEPMode(Enum):
    NORMAL = "normal"        # High throughput, for prefill
    LOW_LATENCY = "low_latency"  # Low latency, for decode
    AUTO = "auto"            # Automatic selection

    def resolve(self, is_extend_in_batch: bool) -> DeepEPMode:
        if self != DeepEPMode.AUTO:
            return self
        if is_extend_in_batch:
            return DeepEPMode.NORMAL      # Prefill -> NORMAL
        else:
            return DeepEPMode.LOW_LATENCY  # Decode -> LOW_LATENCY
```

### 8.2 Prefill Path (Normal Mode)

**Communication**: DeepEP normal dispatch with configurable `num_sms` for communication.
- Uses RDMA or NVLink-based all-to-all
- Supports async communication via `async_finish=True`
- FP8 quantization before dispatch to halve communication bandwidth
- Expert alignment to 128 for DeepGEMM compatibility

**Computation**: Contiguous grouped GEMM
- Tokens are scattered into contiguous per-expert buffers
- `m_indices` array marks expert boundaries
- `grouped_gemm_nt_f8f8bf16_contig` processes all experts in a single kernel launch
- Higher throughput due to large batch sizes per expert

### 8.3 Decode Path (Low-Latency Mode)

**Communication**: DeepEP low-latency dispatch
- `num_max_dispatch_tokens_per_rank <= 1024`
- Uses FP8 or NVFP4 quantization during dispatch
- Returns data in `[num_experts, expected_m, hidden_size]` layout
- Supports `return_recv_hook` for fine-grained overlap control
- Compatible with CUDA Graphs

**Computation**: Masked grouped GEMM
- Data already in `[num_experts, m, hidden_size]` 3D layout
- `masked_m` per-expert counts allow skipping empty expert slots
- `grouped_gemm_nt_f8f8bf16_masked` only processes valid rows
- Lower latency due to elimination of scatter/gather overhead

### 8.4 Non-EP Path (Standard Triton)

For non-EP or hybrid EP setups, the Triton fused MoE kernel handles both prefill and decode uniformly:
- `moe_align_block_size` groups tokens by expert
- Same `fused_moe_kernel` for both phases
- Difference is only in batch size (large for prefill, small for decode)
- CHUNK_SIZE of 64K prevents excessive memory usage for very large prefills

---

## 9. Padding Strategies for GPU-Friendly Batch Sizes

### 9.1 Block-Size Padding in moe_align_block_size

Each expert's token count is rounded up to the nearest multiple of `BLOCK_SIZE_M`:
```c++
int32_t padded_count = (count + block_size - 1) / block_size * block_size;
```

Typical values: BLOCK_SIZE_M = 16, 32, 64, or 128 depending on the auto-tuned config.

### 9.2 DeepGEMM M Padding

For masked grouped GEMM, the M dimension is padded to a multiple of 256:
```python
m_max = (hidden_states.size(0) // 256 + 1) * 256
```

### 9.3 Expert Alignment in DeepEP

When using DeepGEMM, dispatched tokens are aligned to 128:
```python
buffer.dispatch(..., expert_alignment=128 if ENABLE_JIT_DEEPGEMM else 1, ...)
```

### 9.4 Intermediate Size Padding

For FlashInfer TRT-LLM, intermediate size must be a multiple of 128:
```python
if self.use_flashinfer_trtllm_moe and self.intermediate_size_per_partition % 128 != 0:
    self.intermediate_size_per_partition = round_up(self.intermediate_size_per_partition, 128)
```

### 9.5 Weight Padding (SGLANG_MOE_PADDING)

Optional 128-element padding on weight dimensions for FP8/INT8:
```python
padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0
```

---

## 10. Novel Optimizations Unique to SGLang

### 10.1 DeepEP Integration with Auto Mode Switching

SGLang is (to our knowledge) the first inference engine to deeply integrate DeepEP with automatic mode switching between normal (prefill-optimized) and low-latency (decode-optimized) communication modes:

```python
resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
```

### 10.2 Two-Batch Overlap (TBO)

File: Referenced in `/docs/advanced_features/expert_parallelism.md`

TBO splits requests into micro-batches and interleaves attention computation with dispatch/combine operations, achieving up to 2x throughput:
```python
operations = [
    self._forward_attn,
    YieldOperation(),      # Overlap with dispatch of prior micro-batch
    self._forward_dispatch,
    self._forward_mlp,
    YieldOperation(),      # Overlap with combine
    self._forward_combine,
]
```

### 10.3 Single-Batch Overlap (SBO)

Uses dispatcher hooks to overlap shared expert computation with DeepEP communication:
```python
class DeepEPDispatcher(BaseDispatcher):
    def dispatch(self, hidden_states, topk_output):
        self.dispatch_a(hidden_states, topk_output)    # Start async communication
        self._deepep_dispatch_hooks(self)               # Run shared experts during comm
        ret = self.dispatch_b()                         # Wait for communication
        return ret
```

### 10.4 Fused Router Kernel

The `FusedMoeRouter` (router.py) fuses the router linear projection, softcapping, bias addition, and top-k selection into a single Triton kernel, eliminating intermediate materializations.

### 10.5 Fused Activation + Quantization

For DeepGEMM, the `silu_and_mul_masked_post_quant_fwd` kernel fuses SiLU activation, element-wise multiplication, and FP8 quantization in a single pass, respecting the `masked_m` per-expert counts.

### 10.6 EPLB (Expert Parallelism Load Balancer)

Integration of DeepSeek's EPLB for dynamic expert replication/placement:
```python
# In topk.py, supports logical-to-physical expert mapping:
topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
```

### 10.7 FP4 Quantize-Before-AllGather

For flashinfer_cutlass with modelopt_fp4:
```python
if should_use_flashinfer_cutlass_moe_fp4_allgather():
    x, x_sf = fp4_quantize_flashinfer(hidden_states, global_scale, ...)
    topk_weights, topk_ids, x, x_sf = get_tp_group().all_gatherv(...)
```

This 4x reduces communication volume by quantizing to FP4 before the all-gather.

### 10.8 Kimi K2 Specialized Gate

File: `/python/sglang/srt/layers/moe/topk.py:607`

Specialized routing for Kimi K2 (384 experts, num_expert_group=1):
```python
if num_experts == 384 and num_expert_group == 1:
    return kimi_k2_moe_fused_gate(gating_output, correction_bias, topk=topk, ...)
```

With a dedicated CUDA kernel in `sgl-kernel/csrc/moe/kimi_k2_moe_fused_gate.cu`.

### 10.9 Pluggable Backend Architecture

The `PermuteMethodPool` + `FusedOpPool` system allows registering arbitrary dispatch-to-runner conversions:

```python
@register_pre_permute("deepep_normal", "deep_gemm")
def pre_permute_deepep_normal_to_deep_gemm(...): ...

@register_fused_func("deepep", "deep_gemm")
def fused_deepep_deepgemm(...): ...
```

This enables combining any A2A backend with any runner backend.

---

## 11. CUDA Kernels for Expert Computation

### 11.1 sgl-kernel CUDA Kernels

Directory: `/sgl-kernel/csrc/moe/`

| Kernel | File | Purpose |
|--------|------|---------|
| `moe_align_block_size` | `moe_align_kernel.cu` | Sort tokens by expert, pad to block_size |
| `moe_sum_reduce` | `moe_sum_reduce.cu` | Sum across top-k expert outputs |
| `moe_topk_softmax` | `moe_topk_softmax_kernels.cu` | Fused softmax + top-k |
| `moe_topk_sigmoid` | `moe_topk_sigmoid_kernels.cu` | Fused sigmoid + top-k |
| `moe_fused_gate` | `moe_fused_gate.cu` | DeepSeek-style biased grouped top-k |
| `kimi_k2_moe_fused_gate` | `kimi_k2_moe_fused_gate.cu` | Kimi K2 384-expert gate |
| `prepare_moe_input` | `prepare_moe_input.cu` | Compute problem sizes + permutations for CUTLASS MoE |
| `fp8_blockwise_moe_kernel` | `fp8_blockwise_moe_kernel.cu` | FP8 blockwise MoE GEMM |
| `moe_sum` | `moe_sum.cu` | Simple sum reduction |
| `nvfp4_blockwise_moe` | `nvfp4_blockwise_moe.cu` | NVFP4 MoE kernel |

### 11.2 prepare_moe_input (for CUTLASS MoE)

File: `/sgl-kernel/csrc/moe/prepare_moe_input.cu`

This kernel prepares inputs for CUTLASS grouped GEMMs:

```c++
// 1. Count tokens per expert
__global__ void compute_problem_sizes(
    const int* topk_ids, int32_t* problem_sizes1, int32_t* problem_sizes2,
    int32_t* atomic_buffer, int64_t topk_length, int64_t n, int64_t k) {
    int expert_id = blockIdx.x;
    int occurrences = 0;
    for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT)
        occurrences += (topk_ids[i] == expert_id);
    atomicAdd(&atomic_buffer[expert_id], occurrences);
    // Store as (M, 2N, K) for gate-up and (M, K, N) for down
    problem_sizes1[expert_id * 3] = final_occurrences;
    problem_sizes1[expert_id * 3 + 1] = 2 * n;
    problem_sizes1[expert_id * 3 + 2] = k;
}

// 2. Compute expert offsets via prefix sum
__global__ void compute_expert_offsets(...) {
    expert_offsets[0] = 0;
    for (int i = 0; i < num_experts; ++i) {
        atomic_buffer[i] = tot_offset;
        tot_offset += problem_sizes1[i * 3];
        expert_offsets[i + 1] = tot_offset;
    }
}

// 3. Compute input/output permutations
__global__ void compute_arg_sorts(
    const int32_t* topk_ids, int32_t* input_permutation,
    int32_t* output_permutation, int32_t* atomic_buffer,
    int64_t topk_length, int64_t topk) {
    int expert_id = blockIdx.x;
    for (int i = threadIdx.x; i < topk_length; i += THREADS_PER_EXPERT) {
        if (topk_ids[i] == expert_id) {
            int start = atomicAdd(&atomic_buffer[expert_id], 1);
            input_permutation[start] = i / topk;   // Original token index
            output_permutation[i] = start;          // Position in expert group
        }
    }
}
```

---

## 12. Weight Sharding for Expert Parallelism

### 12.1 Expert Distribution

File: `/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:206`

```python
self.moe_ep_size = get_moe_expert_parallel_world_size()
self.moe_ep_rank = get_moe_expert_parallel_rank()
# Each rank holds (num_experts - shared) / ep_size + shared experts
self.num_local_experts = (
    num_experts - num_fused_shared_experts
) // self.moe_ep_size + num_fused_shared_experts
```

### 12.2 Weight Loading

The `weight_loader` method maps global expert IDs to local ones:
```python
def _map_global_expert_id_to_local_expert_id(self, expert_id):
    start_idx = self.moe_ep_rank * num_local_routed_experts
    end_idx = (self.moe_ep_rank + 1) * num_local_routed_experts
    if start_idx <= expert_id < end_idx:
        return expert_id - start_idx
    elif self.num_fused_shared_experts > 0 and expert_id >= num_global_routed_experts:
        return expert_id - num_global_routed_experts + num_local_routed_experts
    else:
        return -1  # Not on this rank
```

Shared experts are replicated on all ranks and placed at the end of the local expert list.

### 12.3 EPLB Dynamic Rebalancing

With EPLB enabled, the logical-to-physical expert mapping can change dynamically:
```python
physical_expert_ids = global_expert_location_metadata.logical_to_all_physical(
    self.layer_id, expert_id, require_global_experts)
for physical_expert_id in physical_expert_ids:
    self._weight_loader_physical(param, loaded_weight, weight_name, shard_id, physical_expert_id)
```

---

## 13. Summary of Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `layers/moe/fused_moe_triton/layer.py` | ~1200 | Main FusedMoE layer, weight loading, forward |
| `layers/moe/fused_moe_triton/fused_moe.py` | ~806 | fused_experts_impl, 2-stage GEMM execution |
| `layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` | ~1174 | Triton kernels for fused MoE GEMM |
| `layers/moe/fused_moe_triton/moe_align_block_size.py` | ~87 | Token-to-expert alignment/padding |
| `layers/moe/topk.py` | ~1114 | Top-K routing (softmax, sigmoid, grouped) |
| `layers/moe/router.py` | ~429 | Fused router kernels |
| `layers/moe/token_dispatcher/deepep.py` | ~873 | DeepEP dispatch/combine |
| `layers/moe/token_dispatcher/standard.py` | ~194 | Standard dispatch (hybrid EP) |
| `layers/moe/moe_runner/deep_gemm.py` | ~615 | DeepGEMM runner + permute functions |
| `layers/moe/moe_runner/runner.py` | ~120 | MoeRunner orchestration |
| `layers/moe/moe_runner/base.py` | ~286 | Base classes + permute pool |
| `layers/moe/ep_moe/kernels.py` | ~1200+ | Triton kernels for EP preprocessing |
| `layers/moe/ep_moe/layer.py` | ~400+ | DeepEPMoE layer |
| `layers/moe/utils.py` | ~336 | Config enums and globals |
| `sgl-kernel/csrc/moe/moe_align_kernel.cu` | ~250 | CUDA moe_align_block_size |
| `sgl-kernel/csrc/moe/prepare_moe_input.cu` | ~180 | CUDA prepare_moe_input for CUTLASS |

All file paths are relative to `/Users/ochafik/github/mlx-lm2/tmp/sglang/python/sglang/srt/` unless noted as `sgl-kernel/`.
