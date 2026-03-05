# MoE Prefill Optimization Research: Academic & Industry Landscape

## Table of Contents

1. [The Fundamental MoE Compute Problem](#1-the-fundamental-moe-compute-problem)
2. [Token Grouping & Dispatch Strategies](#2-token-grouping--dispatch-strategies)
3. [MegaBlocks: Block-Sparse & GroupedGEMM](#3-megablocks-block-sparse--groupedgemm)
4. [Grouped GEMM Approaches](#4-grouped-gemm-approaches)
5. [Expert-Parallel vs Expert-Slicing](#5-expert-parallel-vs-expert-slicing)
6. [DeepSpeed-MoE](#6-deepspeed-moe)
7. [Tutel: Adaptive MoE](#7-tutel-adaptive-moe)
8. [SonicMoE: IO & Tile-Aware Optimization](#8-sonicmoe-io--tile-aware-optimization)
9. [MoE-Gen: Single-GPU Module-Based Batching](#9-moe-gen-single-gpu-module-based-batching)
10. [DeepSeek V3: FP8 Grouped GEMM](#10-deepseek-v3-fp8-grouped-gemm)
11. [vLLM & SGLang Fused MoE Kernels](#11-vllm--sglang-fused-moe-kernels)
12. [Expert Choice vs Token Choice Routing](#12-expert-choice-vs-token-choice-routing)
13. [MoE Arithmetic Intensity Analysis](#13-moe-arithmetic-intensity-analysis)
14. [Expert Load Balancing & Compute Efficiency](#14-expert-load-balancing--compute-efficiency)
15. [Padding vs Token Dropping Tradeoffs](#15-padding-vs-token-dropping-tradeoffs)
16. [Apple Silicon & MLX-Specific Considerations](#16-apple-silicon--mlx-specific-considerations)
17. [Multi-Node Expert Parallelism on Apple Silicon](#17-multi-node-expert-parallelism-on-apple-silicon)
18. [Prefill vs Decode: Different Optimization Regimes](#18-prefill-vs-decode-different-optimization-regimes)
19. [Synthesis: What Matters for Single-Device Apple Silicon Inference](#19-synthesis-what-matters-for-single-device-apple-silicon-inference)

---

## 1. The Fundamental MoE Compute Problem

### Problem Statement

In a Mixture-of-Experts (MoE) model, a gating/router network dynamically assigns each token to a subset of K experts (out of N total). This creates a fundamental compute challenge:

- **Variable-size workloads**: Each expert receives a different number of tokens, creating load imbalance.
- **Low arithmetic intensity**: Because the global batch is split across N experts, each expert sees only a fraction of the tokens. For a batch of B tokens with top-K routing over N experts, each expert receives on average `B * K / N` tokens. This small per-expert batch size means each expert's GEMM has low arithmetic intensity (few FLOPs per byte of weight loaded).
- **Naive sequential execution**: Processing experts in a for-loop launches many small kernels, each with poor GPU utilization.
- **Padding waste**: Fixed-capacity approaches pad each expert's batch to the same size, wasting compute on padding tokens.
- **Token dropping**: Alternatively, tokens exceeding capacity are dropped, hurting model quality.

### Quantitative Impact

Research shows that dense model FFN layers achieve arithmetic intensity of ~15.74 FLOP/byte, while MoE layers remain at ~8 FLOP/byte due to routing distributing batches across experts. MoE FFN layers sustain low SM utilization (28%-34%) and high DRAM pressure (>80% at small batch sizes).

**Sources:**
- [A Systematic Characterization of LLM Inference on GPUs (2024)](https://www.arxiv.org/pdf/2512.01644)
- [MoE-Gen (2025)](https://arxiv.org/abs/2503.09716)

---

## 2. Token Grouping & Dispatch Strategies

### The Standard Permute-Compute-Unpermute Pipeline

Nearly all optimized MoE implementations follow this pattern:

1. **Router/Gating**: Compute expert assignments and weights for each token.
2. **Permute (Gather)**: Sort/reorder tokens by expert ID so that all tokens assigned to the same expert are contiguous in memory. This uses argsort on expert IDs.
3. **Compute**: Execute each expert's FFN on its contiguous group of tokens. This can be done via:
   - Sequential for-loop over experts (naive)
   - Grouped/batched GEMM (optimized)
   - Block-sparse matmul (MegaBlocks approach)
4. **Unpermute (Scatter)**: Use argsort on the original indices to restore tokens to their original sequence order.
5. **Combine**: Weight expert outputs by router scores and sum.

### Key Implementation Detail: Argsort-Based Permutation

```
# Pseudocode
expert_ids = router(tokens)                    # [B, K] expert assignments
sorted_indices = argsort(expert_ids.flatten()) # sort by expert
permuted_tokens = tokens[sorted_indices]       # gather: group by expert
# ... compute experts on contiguous groups ...
inverse_indices = argsort(sorted_indices)      # compute inverse permutation
output = expert_outputs[inverse_indices]       # scatter: restore order
```

The permutation step is critical for performance: it transforms scattered token-to-expert assignments into contiguous memory blocks that enable efficient batched computation.

### Performance Cost of Permutation

Research from MegaScale-MoE shows that local rearrangement (permuting tokens before/after expert computation) constitutes a dominant share of overhead -- nearly 69% of shuffle cost within a node and ~25% across nodes. This means MoE performance is constrained not only by compute throughput but also by memory-bound permutation and repacking steps.

**Sources:**
- [Batching Strategies for MoE Inference (apxml.com)](https://apxml.com/courses/mixture-of-experts/chapter-5-moe-inference-optimization-deployment/batching-strategies-moe-inference)
- [MegaScale-MoE (2025)](https://arxiv.org/html/2505.11432v1)
- [MoE Parallelism for Inference (Medium)](https://medium.com/@zdj0712/moe-parallelism-for-inference-tricks-and-pytorch-deep-dive-17fd8ef86db2)

---

## 3. MegaBlocks: Block-Sparse & GroupedGEMM

**Paper**: "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts" (Gale, Narayanan, Young, Zaharia, MLSys 2023)

### Core Optimization Idea

Rather than using batched matrix multiplication or sequential expert execution, MegaBlocks reformulates MoE computation as a **sparse matrix-dense matrix multiplication** where the sparse matrix has **block-diagonal structure**.

The key insight: each expert's computation can be viewed as a block in a larger sparse matrix. When experts receive variable numbers of tokens (load imbalance), this is naturally handled by having variable-sized blocks -- each block is computed as many smaller fixed-size sub-blocks using block-sparse matrix multiplication.

### How Tokens Are Dispatched

1. Router assigns tokens to experts (no token dropping -- all tokens are processed).
2. Tokens are grouped by expert assignment.
3. Instead of padding to uniform capacity, the entire computation is expressed as a block-sparse matmul where each expert's tokens form a variable-size block.
4. Block-sparse GPU kernels handle the variable-length blocks efficiently.

### Two Execution Modes

1. **dMoE with Block-Sparse ops**: Uses custom block-sparse CUDA kernels. The sparse matrix has block-diagonal structure matching expert assignments.
2. **dMoE with Grouped GEMM (recommended for Hopper GPUs)**: Uses grouped GEMM to execute all expert matmuls in a single kernel launch. Install with `megablocks[gg]`.

### Performance

- **Block-sparse kernels**: Achieve 98.6% of cuBLAS throughput on average (std dev 4%, range 91%-104%).
- **End-to-end training speedups**: Up to 40% over MoEs trained with Tutel.
- **vs dense models**: 1.8x-2.4x end-to-end training speedups for same validation loss vs dense Transformers trained with Megatron-LM.
- **No token dropping**: Eliminates quality loss from dropped tokens while maintaining hardware efficiency.

### Relevance to Single-Device Apple Silicon

HIGH. The block-sparse and grouped GEMM approaches are directly applicable to single-device inference:
- No inter-device communication needed.
- Grouped GEMM concept (batching all expert matmuls into one kernel) is the most promising optimization for Apple Silicon.
- The "no padding, no dropping" philosophy maps well to inference where you want to process all tokens.

**Sources:**
- [MegaBlocks Paper (arXiv 2211.15841)](https://arxiv.org/abs/2211.15841)
- [MegaBlocks PDF (Berkeley)](https://people.eecs.berkeley.edu/~matei/papers/2023/mlsys_megablocks.pdf)
- [MegaBlocks GitHub (Databricks)](https://github.com/databricks/megablocks)

---

## 4. Grouped GEMM Approaches

### Concept

A Grouped GEMM executes multiple independent GEMMs together in a single kernel launch. For MoE, each expert's forward pass involves matrix multiplications, and grouped GEMM batches these across all active experts.

Instead of:
```
for expert_i in active_experts:
    output_i = expert_i.weight @ tokens_i    # separate kernel launch each
```

Grouped GEMM does:
```
all_outputs = grouped_gemm(all_weights, all_token_groups)  # single kernel launch
```

### PyTorch/Triton Persistent Cache-Aware Grouped GEMM

Meta and IBM developed a Triton-based persistent cache-aware grouped GEMM kernel optimized for MoE models like DeepSeek V3.

**Key optimizations:**
1. **Persistent kernel design**: CTAs (Cooperative Thread Arrays) stay "alive" and dynamically pick up new tile computations, avoiding repeated kernel launch overhead.
2. **Grouped tile ordering**: Tiles of the GEMM problem are reordered so adjacent tiles share input data, improving L2 cache hit rates. This alone yields 1.33x speedup and +60% improvement in L2 cache hit rate.
3. **Hopper TMA unit**: Leverages Tensor Memory Accelerator for efficient data movement.

**Performance:**
- Up to 1.50x speedup over baseline Triton kernels.
- Up to 2.62x speedup over manual PyTorch loop implementation on H100 GPUs.

### NVIDIA cuBLAS Grouped GEMM APIs

NVIDIA introduced native grouped GEMM APIs in cuBLAS, providing hardware-optimized implementations. These support variable M dimensions (different token counts per expert) with fixed N and K.

### DeepGEMM (DeepSeek)

DeepSeek released DeepGEMM, an FP8 GEMM library supporting both dense and MoE GEMMs. Key design: groups only the M-axis while N and K remain fixed (matching expert weight shapes). For prefilling, tokens are concatenated into a single tensor in "contiguous" layout.

### Relevance to Single-Device Apple Silicon

CRITICAL. Grouped GEMM is the single most impactful optimization for MoE on Apple Silicon:
- Metal Performance Shaders could potentially support a grouped GEMM pattern.
- The core idea (one kernel launch for all experts) eliminates per-expert kernel launch overhead which is especially costly on Metal.
- The "contiguous layout" approach (concatenating all expert token groups into one tensor with offset metadata) could be implemented as a custom Metal kernel or via creative use of MLX operations.
- Apple's unified memory means no host-device transfer overhead, but kernel launch overhead still matters.

**Sources:**
- [PyTorch Blog: Accelerating MoEs with Triton Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)
- [NVIDIA cuBLAS Grouped GEMM](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates)
- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)
- [Ian Barber Blog: Grouped GEMMs and MoE](https://ianbarber.blog/2025/02/11/grouped-gemms-and-moe/)

---

## 5. Expert-Parallel vs Expert-Slicing

### Expert Parallelism (EP)

Each expert is assigned to a different device/GPU. Tokens are dispatched to the appropriate device via All-to-All communication.

**Characteristics:**
- Does not reduce computation granularity of individual operators.
- Leverages aggregate memory bandwidth across devices.
- Requires All-to-All communication for token dispatch and result gathering.
- Standard approach for large-scale distributed MoE inference.

### Expert Slicing (Tensor Parallelism for Experts)

Each expert's weight matrices are sliced (sharded) across multiple devices. All devices participate in computing every expert, each handling a slice.

**Characteristics:**
- Reduces per-device memory for each expert.
- Requires reduce-scatter to combine partial outputs.
- Better for memory-constrained scenarios.
- KTransformers uses this for CPU NUMA-aware expert slicing, achieving 1.63x decoding throughput improvement.

### Relevance to Single-Device Apple Silicon

LOW for distributed variants, but the concepts inform single-device optimization:
- On a single Apple Silicon device, there's no distribution -- all experts are in unified memory.
- The key insight from EP/slicing research is that **expert computation is often memory-bandwidth bound** on small batches.
- For single-device inference, the relevant optimization is making expert computation more compute-efficient (larger effective batch per expert, grouped GEMM), not distributing it.

**Sources:**
- [DeepSpeed-MoE (arXiv 2201.05596)](https://arxiv.org/abs/2201.05596)
- [BentoML LLM Inference Handbook: Parallelism](https://bentoml.com/llm/inference-optimization/data-tensor-pipeline-expert-hybrid-parallelism)
- [Meta Engineering: Scaling LLM Inference](https://engineering.fb.com/2025/10/17/ai-research/scaling-llm-inference-innovations-tensor-parallelism-context-parallelism-expert-parallelism/)
- [KTransformers (SOSP 2025)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

---

## 6. DeepSpeed-MoE

**Paper**: "DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale" (Rajbhandari et al., ICML 2022)

### Core Optimization Ideas

1. **Dense representation & kernel fusion**: Fuses the gating function into a single kernel. Uses a dense token-to-expert mapping table (instead of sparse operations). The gating kernel combines top-k, cumsum, and scatter operations.

2. **Data-layout transformations**: Sorts tokens by expert ID, then back to original ordering, without requiring sparse einsum operations. This enables efficient dense computation for each expert.

3. **Capacity-based dispatch**: Uses a capacity factor to determine maximum tokens per expert. Computes token IDs within each expert based on cumsum of assignments.

### Batching Strategy

- Tokens are gathered into dense mini-batches per expert after gating.
- Each expert receives a contiguous batch of its assigned tokens.
- After expert computation, outputs are scattered back to original positions.
- Kernel fusion reduces overhead of the gather/scatter operations.

### Performance

- Achieves up to 7.3x improvement in MoE inference latency over baseline.
- Supports expert parallelism combined with tensor parallelism.

### Relevance to Single-Device Apple Silicon

MEDIUM. The kernel fusion ideas (fusing gate+sort+dispatch) are applicable. The dense representation approach (avoiding sparse ops) is well-suited to Metal which lacks native sparse operation support. However, DeepSpeed's optimizations are primarily for CUDA GPUs and distributed settings.

**Sources:**
- [DeepSpeed-MoE Paper (arXiv 2201.05596)](https://arxiv.org/abs/2201.05596)
- [DeepSpeed-MoE Tutorial](https://www.deepspeed.ai/tutorials/mixture-of-experts-inference/)
- [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/deepspeed-advancing-moe-inference-and-training-to-power-next-generation-ai-scale/)

---

## 7. Tutel: Adaptive MoE

**Paper**: "Tutel: Adaptive Mixture-of-Experts at Scale" (Hwang et al., MLSys 2023)

### Core Optimization Ideas

1. **Adaptive parallelism**: Dynamically switches between data parallelism and expert parallelism based on runtime workload. Uses an identical layout for model parameters and data that works with all parallelism methods without tensor migration overhead.

2. **Adaptive pipelining**: Overlaps computation and communication dynamically.

3. **2-Dimensional Hierarchical (2DH) All-to-All**: Optimized communication primitive for MoE token dispatch.

4. **Fast encode/decode with sparse computation on GPU**: Optimized sparse operations for token routing.

### Performance

- With 128 GPUs: up to 3.11x MoE-layer speedup.
- 1.55x/2.11x speedup for end-to-end training/inference of SwinV2-MoE.
- Serves as a widely-used baseline in subsequent MoE research.

### Relevance to Single-Device Apple Silicon

LOW for distributed features, but the adaptive execution concept is interesting: on Apple Silicon, you could adaptively choose between sequential expert execution (for decode, where batch per expert is tiny) and grouped execution (for prefill, where batch per expert is larger).

**Sources:**
- [Tutel Paper (arXiv 2206.03382)](https://arxiv.org/abs/2206.03382)
- [Tutel PDF (MLSys 2023)](https://yzygitzh.github.io/assets/papers/hwang_mlsys_2023.pdf)
- [Tutel GitHub (Microsoft)](https://github.com/microsoft/Tutel)

---

## 8. SonicMoE: IO & Tile-Aware Optimization

**Paper**: "SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations" (Guo et al., 2024)

### Core Optimization Ideas

1. **Token Rounding Routing**: A hardware-aware routing method where the routed number of tokens to each expert is always a **multiple of the GEMM tile size**. This eliminates partial tiles (which waste compute) and padding.

   - Starts from standard top-K routing.
   - Performs expert-wise ranking to create a score matrix.
   - Rounds per-expert token counts to multiples of the GEMM tile size (Mtile).
   - Deviation from original routing is bounded by 1 tile per expert.

2. **IO-aware optimizations**: Co-designs memory IO patterns with computation to overlap data movement with compute on Hopper/Blackwell GPUs.

3. **Efficient backward pass**: Derives a more efficient algorithm for MoE backward computation.

### Performance

- Token rounding routing: 16% faster than baseline token-choice routing (when scaling experts 4x from 30B MoE).
- 45% reduction in activation memory.
- 1.86x compute throughput improvement on Hopper GPUs.

### Relevance to Single-Device Apple Silicon

MEDIUM-HIGH. The token rounding concept is highly relevant:
- On Metal, GEMM performance is sensitive to matrix dimensions (tile-aligned sizes perform significantly better).
- Rounding per-expert token counts to Metal-friendly tile sizes could yield similar benefits.
- The IO-aware optimization philosophy (co-designing data movement with computation) is critical for Apple Silicon where memory bandwidth is the primary bottleneck.
- The specific Hopper/Blackwell features (TMA, warp specialization) do not apply, but the *principle* of tile-awareness does.

**Sources:**
- [SonicMoE Paper (arXiv 2512.14080)](https://arxiv.org/abs/2512.14080)
- [SonicMoE GitHub (Dao-AILab)](https://github.com/Dao-AILab/sonic-moe)

---

## 9. MoE-Gen: Single-GPU Module-Based Batching

**Paper**: "MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching" (2025)

### Core Optimization Idea

MoE-Gen identifies that existing inference systems use "model-based batching" (processing a fixed batch through the entire model), which results in tiny per-expert batches in MoE layers. Instead, MoE-Gen introduces **module-based batching**:

1. **Decouple attention and expert batch sizes**: Attention and expert modules have different memory/FLOP profiles. Attention benefits from smaller batches; experts need larger batches for efficiency.

2. **Accumulate tokens in host memory**: After attention computation, tokens are accumulated in host memory across multiple attention batches.

3. **Launch large expert batches**: Once enough tokens are accumulated, a large batch is sent to the GPU for expert computation, maximizing arithmetic intensity.

4. **Batch size optimization per module**: Different optimal batch sizes are computed for attention vs expert modules to fully overlap GPU computation and host-GPU communication.

### Performance

- 8-31x higher throughput vs model-based batching systems (FlexGen, MoE-Lightning, DeepSpeed).
- Even greater improvements over continuous batching systems (vLLM, Ollama) on DeepSeek and Mixtral.

### Relevance to Single-Device Apple Silicon

MEDIUM. The insight about different optimal batch sizes for attention vs experts is valuable, but:
- On Apple Silicon with unified memory, there's no host-GPU transfer bottleneck to overlap.
- The concept of accumulating more tokens before expert computation could still be relevant for prefill (e.g., processing attention in chunks but accumulating a large batch for expert layers).
- The core principle -- increase the effective batch size seen by each expert -- is universally applicable.

**Sources:**
- [MoE-Gen Paper (arXiv 2503.09716)](https://arxiv.org/abs/2503.09716)

---

## 10. DeepSeek V3: FP8 Grouped GEMM

### Core Optimization Ideas

1. **FP8 Mixed Precision**: Activations are cached and dispatched in FP8 to reduce memory and communication overhead. Weight quantization uses fine-grained blockwise and tilewise scaling (each 128x128 weight submatrix and each 1x128 activation subvector scaled separately).

2. **Grouped GEMM for MoE**: DeepGEMM library supports MoE GEMMs where only the M-axis (token count per expert) varies, while N and K (expert weight dimensions) remain fixed. For prefilling, tokens assigned to different experts are concatenated into a single contiguous tensor.

3. **256 experts, 8 active per token**: An extreme MoE configuration where arithmetic intensity per expert is very low, making grouped GEMM essential.

4. **Triton kernel tuning**: Optimizes BLOCK_SIZE_M/N/K, GROUP_SIZE_M (for L2 cache reuse), warps per block, and pipeline stages for prefetching.

### Relevance to Single-Device Apple Silicon

HIGH for the grouped GEMM concept, LOW for FP8 specifics:
- The contiguous layout (concatenating expert token groups into one tensor) is directly applicable.
- The fine-grained quantization approach is interesting but Apple Silicon lacks native FP8 support in current Metal.
- The 256-expert, 8-active configuration makes the arithmetic intensity problem even more acute -- exactly the scenario where grouped GEMM provides the most benefit.

**Sources:**
- [DeepSeek-V3 Technical Report (arXiv 2412.19437)](https://arxiv.org/abs/2412.19437)
- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)

---

## 11. vLLM & SGLang Fused MoE Kernels

### Implementation Approaches

Both vLLM and SGLang implement fused MoE kernels, primarily using Triton:

1. **Fused experts kernel**: A single Triton kernel that:
   - Takes permuted tokens (sorted by expert).
   - Computes all expert FFNs in a single kernel launch.
   - Supports multiple quantization types (FP8, INT8, FP4, etc.).

2. **MoE Align & Sort**: SGLang split this into two kernel launches:
   - **Alignment**: Ensures token counts per expert align with compute tile sizes.
   - **Placement**: Assigns tokens to their sorted positions.

   This was necessary because the original single-kernel implementation was not efficient for large-scale prefill with up to 256 experts.

3. **Multiple backends**: SGLang supports Triton, CUTLASS, DeepGEMM, FlashInfer, and AITER backends for different hardware.

### Performance

SGLang achieved day-0 support for DeepSeek V3 (December 2024) using a Triton-first approach. Performance comparisons between vLLM and SGLang kernels show varying throughput advantages depending on batch size and configuration.

### Relevance to Single-Device Apple Silicon

MEDIUM. The fused MoE kernel concept (single kernel for all experts) is directly applicable as a Metal compute shader design. The align-and-sort preprocessing step is important: on Metal, ensuring tile-aligned token counts per expert would improve GEMM efficiency. The multi-backend approach suggests that a Metal-specific kernel could be optimized differently than CUDA/Triton implementations.

**Sources:**
- [vLLM Fused MoE Kernel Features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
- [SGLang Fused MoE Triton (GitHub)](https://github.com/sgl-project/sglang/tree/main/benchmark/kernels/fused_moe_triton)
- [SGLang MoE Align & Sort (HuggingFace Blog)](https://huggingface.co/blog/yiakwy-xpu-team/efficient-moe-align-sort-design-for-sglang)
- [SGLang MoE System (DeepWiki)](https://deepwiki.com/sgl-project/sglang/9-high-performance-kernel-library-(sgl-kernel))

---

## 12. Expert Choice vs Token Choice Routing

### Token Choice Routing (Traditional)

Each token selects its top-K experts. Problems:
- Load imbalance: popular experts overflow, others are underutilized.
- Requires capacity factor with padding (wasting compute) or token dropping (hurting quality).
- Auxiliary loss needed for load balancing, but tuning it is difficult (too much loss hurts specialization).

### Expert Choice Routing (Google, NeurIPS 2022)

Each expert selects its top tokens (fixed number per expert). Advantages:
- **Perfect load balancing by construction**: each expert processes exactly its capacity.
- **Variable experts per token**: some tokens may be processed by many experts, others by few.
- ~20% training/inference step time reduction vs GLaM.
- 2x faster training convergence for 8B/64E models.

### Relevance to Single-Device Apple Silicon

MEDIUM-HIGH for inference:
- Expert choice routing at inference time is tricky (requires knowing all tokens before routing, which is natural during prefill but not during autoregressive decode).
- For prefill specifically, expert choice could guarantee uniform expert batch sizes, eliminating padding/imbalance -- this is ideal for Metal GEMM which prefers uniform matrix sizes.
- However, most deployed models use token-choice routing, so optimization must handle variable per-expert batch sizes.

**Sources:**
- [Expert Choice Routing (Google Research)](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)
- [Expert Choice Routing Paper (arXiv 2202.09368)](https://arxiv.org/abs/2202.09368)
- [AdaMoE (EMNLP 2024)](https://aclanthology.org/2024.findings-emnlp.361/)

---

## 13. MoE Arithmetic Intensity Analysis

### The Core Problem

Arithmetic intensity = FLOPs / bytes transferred. For GPUs to be compute-bound (efficient), arithmetic intensity must exceed the machine's compute-to-bandwidth ratio (the "ridge point" on the roofline model).

**Dense model FFN**: With batch B and hidden dimension D, a single FFN GEMM has dimensions [B, D] x [D, 4D]. Arithmetic intensity grows with B.

**MoE FFN**: With N experts and top-K routing, each expert's GEMM has dimensions [B*K/N, D] x [D, 4D]. The effective batch per expert is B*K/N -- much smaller than B. This makes each expert's computation **memory-bandwidth bound** rather than compute-bound.

### Quantitative Analysis

| Configuration | Arithmetic Intensity (FLOP/byte) |
|---|---|
| Dense FFN (large batch) | ~15.74 |
| MoE FFN (typical) | ~8.0 |
| MoE FFN (small batch) | <4.0 |

For reference, modern GPU ridge points:
- NVIDIA H100: ~150-300 FLOP/byte (FP16)
- Apple M4 Max GPU: ~20-40 FLOP/byte (estimated, FP16)

Apple Silicon has a much lower ridge point due to high memory bandwidth relative to compute, which means MoE experts can potentially be compute-bound at smaller batch sizes than on NVIDIA GPUs. This is actually favorable for MoE workloads.

### How to Improve MoE Arithmetic Intensity

1. **Increase per-expert batch size**: Accumulate more tokens per expert (MoE-Gen approach).
2. **Grouped GEMM**: Batch expert computations so weight loading is amortized across tiles.
3. **Wider expert parallelism**: Distribute experts so each has less weight-loading pressure.
4. **Quantization**: Reduces bytes per weight, increasing effective arithmetic intensity (FP8, INT4, etc.).
5. **Tile-aware routing**: Ensure per-expert token counts are multiples of GEMM tile size (SonicMoE approach).

**Sources:**
- [NVIDIA: Scaling Large MoE Models with Wide Expert Parallelism](https://developer.nvidia.com/blog/scaling-large-moe-models-with-wide-expert-parallelism-on-nvl72-rack-scale-systems)
- [A Systematic Characterization of LLM Inference on GPUs](https://www.arxiv.org/pdf/2512.01644)
- [MoE-Gen (2025)](https://arxiv.org/abs/2503.09716)

---

## 14. Expert Load Balancing & Compute Efficiency

### The Load Imbalance Problem

Dynamic routing creates "hot" experts that receive disproportionately many tokens. In distributed settings, GPUs hosting hot experts become stragglers determining end-to-end latency. On a single device, load imbalance means some expert computations are large (efficient) while others are tiny (inefficient).

### Impact on Compute Efficiency During Prefill

During prefill, all tokens in the prompt are available simultaneously, allowing:
- Batching all tokens' expert assignments at once.
- Potentially using expert-choice routing for perfect balance.
- Optimizing token-to-expert dispatch globally.

Research shows:
- Expert activation patterns can be determined during prefill and used to optimize decode-phase expert allocation.
- EPS-MoE improves prefill throughput by 52.4% through expert pipeline scheduling.
- LIBRA proposes effective load balancing strategies that maintain both efficiency and model quality.

### Strategies for Balanced Computation

1. **Auxiliary loss during training**: Encourages balanced routing (standard approach).
2. **Expert choice routing**: Guarantees balance by construction.
3. **Dynamic capacity**: Adjust per-expert capacity based on actual assignments (vs fixed capacity factor).
4. **Token rounding**: Round per-expert counts to tile multiples (SonicMoE).
5. **Rectify-Router**: Process dropped tokens via intra-GPU rectification.

**Sources:**
- [HuggingFace Blog: MoE Balance Review](https://huggingface.co/blog/NormalUhr/moe-balance)
- [LIBRA: Effective Yet Efficient Load Balancing](https://openreview.net/pdf?id=WhxNwgGkAS)
- [EPS-MoE (arXiv 2410.12247)](https://arxiv.org/pdf/2410.12247)
- [Toward Efficient Inference for Mixture of Experts (NeurIPS 2024)](https://www.seas.upenn.edu/~leebcc/documents/huang24-neurips.pdf)

---

## 15. Padding vs Token Dropping Tradeoffs

### The Fundamental Tradeoff

MoE implementations must choose how to handle variable per-expert token counts:

**Padding approach:**
- Pad each expert's batch to a fixed capacity (capacity = tokens_per_expert * capacity_factor).
- Wastes compute on padding tokens (typical capacity factors: 1.25x-2.0x).
- Uniform matrix sizes enable efficient batched computation.

**Token dropping approach:**
- When an expert exceeds capacity, overflow tokens are dropped (passed via residual connection only).
- Saves compute but hurts model quality.
- Common in training (GShard, Switch Transformer).

**No-drop, no-pad approach (MegaBlocks):**
- Process all tokens with variable-size expert batches.
- Uses block-sparse matmul or grouped GEMM to handle variable sizes efficiently.
- Best of both worlds but requires specialized kernels.

### Rectify-Router (EMNLP 2024)

- **Intra-GPU Rectification**: Efficiently processes dropped tokens within the same GPU.
- **Fill-in Rectification**: Optimally manages padding tokens.
- Demonstrates that both dropped tokens and padding can be reclaimed.

### Relevance to Single-Device Apple Silicon

HIGH. For inference on Apple Silicon:
- Token dropping is generally unacceptable (all tokens must be processed).
- Padding waste is proportional to load imbalance -- with quantized MoE models having many experts (e.g., 64 or 256), imbalance can cause significant waste.
- The grouped GEMM / block-sparse approach (variable-size expert batches, no padding) is the ideal target.
- If Metal GEMM requires aligned sizes, minimal padding to tile boundaries (SonicMoE-style rounding) is the best compromise.

**Sources:**
- [Turn Waste into Worth: Rectifying Top-k Router of MoE (EMNLP 2024)](https://arxiv.org/html/2402.12399v1)
- [MegaBlocks (MLSys 2023)](https://arxiv.org/abs/2211.15841)

---

## 16. Apple Silicon & MLX-Specific Considerations

### Hardware Characteristics

| Feature | Apple M4 Max | NVIDIA H100 (for comparison) |
|---|---|---|
| Memory Bandwidth | ~546 GB/s | ~3350 GB/s (HBM3) |
| Compute (FP16) | ~13.5 TFLOPS | ~990 TFLOPS |
| Compute/BW Ratio | ~24.7 FLOP/byte | ~295 FLOP/byte |
| Memory | Up to 128 GB unified | 80 GB HBM3 |
| Kernel Launch | Metal command buffer | CUDA kernel launch |

Key implications for MoE:
1. **Lower ridge point**: Apple Silicon becomes compute-bound at lower arithmetic intensity, meaning MoE expert computations may be compute-bound at smaller batch sizes. This is *favorable* for MoE.
2. **Unified memory**: No host-device transfer overhead. Expert weights are always "resident" in accessible memory.
3. **Metal kernel launch overhead**: Metal command buffer encoding has overhead that makes many small kernel launches costly. This makes grouped GEMM (single kernel) even more important.
4. **No native sparse operations**: Metal lacks native block-sparse matmul support. Custom Metal compute shaders would be needed.

### MLX Framework Status

- MLX supports MoE models (Mixtral, DBRX, Qwen, GPT-OSS).
- Uses `QuantizedSwitchLinear` for quantized MoE expert computation (from prior analysis).
- Current implementation likely uses sequential expert execution (for-loop over experts).
- MLX leverages Metal Performance Shaders for GEMM operations.
- M5 chip introduces Neural Accelerators with dedicated matrix-multiplication operations.

### Metal 4 / M5 Neural Accelerators

- Dedicated hardware for matrix multiplication (TensorOps).
- Up to 4x speedup for time-to-first-token vs M4.
- Metal Performance Primitives framework for leveraging Neural Accelerators.
- Potential for specialized MoE kernels using these new capabilities.

**Sources:**
- [Apple MLX Research: Exploring LLMs with MLX and M5](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Apple vs Oranges: M-Series for HPC (arXiv 2502.05317)](https://arxiv.org/html/2502.05317v1)
- [Metal Benchmarks (GitHub)](https://github.com/philipturner/metal-benchmarks)
- [Profiling Apple Silicon for ML Training (arXiv 2501.14925)](https://arxiv.org/pdf/2501.14925)

---

## 17. Multi-Node Expert Parallelism on Apple Silicon

**Paper**: "Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model" (RACS 2024)

### Setup

- Mac Studio cluster with M2 Ultra chips.
- Runs unquantized DBRX model (132B bfloat16 parameters).
- Uses MLX/Metal for GPU-accelerated computation.
- `mx.distributed` for cross-node communication via Ethernet/Thunderbolt.

### Key Findings

1. **Expert computation vs communication time**: Computation time for experts is comparable to communication time for exchanging outputs, making network latency (not bandwidth) the bottleneck.

2. **Communication overhead grows with scale**: 23% at 2 nodes, 29% at 3 nodes, 33% at 4 nodes. Scalability is greatly hindered by communication time.

3. **Cost efficiency**: Mac Studio cluster is 1.15x more cost-efficient than NVIDIA H100 AI supercomputer.

### Relevance to Single-Device Inference

MEDIUM. Key takeaways for single-device:
- On a single device, there is zero communication overhead -- all the "communication" time in the distributed case becomes free, making single-device MoE more efficient per-FLOP.
- The comparison confirms that Apple Silicon GPU compute is competitive for MoE workloads when communication overhead is eliminated.
- For large MoE models that fit in memory (Apple Silicon's advantage with up to 128-192GB unified memory), single-device execution avoids the scalability issues identified in this paper.

**Sources:**
- [Multi-Node Expert Parallelism on Apple Silicon (arXiv 2506.23635)](https://arxiv.org/abs/2506.23635)
- [ACM RACS Proceedings](https://dl.acm.org/doi/10.1145/3649601.3698722)

---

## 18. Prefill vs Decode: Different Optimization Regimes

### Prefill Phase (Compute-Bound)

- Processes entire input sequence at once.
- Large batch of tokens available for routing.
- Each expert receives a meaningful number of tokens (B_prefill * K / N can be large).
- Compute-bound: benefits from high FLOPs (Tensor Parallelism scales well).
- **MoE-specific**: Even during prefill, per-expert batches are smaller than the full batch. With 256 experts and top-8 routing on a 4K token prompt, each expert gets ~125 tokens on average -- still meaningful for compute efficiency.

### Decode Phase (Memory-Bandwidth-Bound)

- Generates one token at a time (or small batch of tokens across requests).
- Very small effective batch per expert: with batch size B and top-K routing over N experts, each expert gets ~B*K/N tokens. For B=1, K=2, N=8: each expert gets 0.25 tokens on average.
- Memory-bandwidth bound: benefits from high memory bandwidth.
- **MoE-specific**: The per-expert batch is even tinier than for dense models. MoE FFNs sustain 28%-34% SM utilization during decode.

### Optimization Strategy Differences

| Aspect | Prefill | Decode |
|---|---|---|
| Bottleneck | Compute | Memory bandwidth |
| Per-expert batch | Moderate-large | Tiny |
| Best optimization | Grouped GEMM, tile-aware routing | Weight caching, quantization |
| Expert parallelism benefit | High (reduces compute) | Low (adds communication) |
| Grouped GEMM benefit | Very high | Limited (batches too small) |

### Implications for Apple Silicon

During **prefill** on Apple Silicon:
- Grouped GEMM / fused expert computation has the most impact.
- Token permutation/grouping by expert enables efficient batched computation.
- Tile-aligned expert batch sizes matter for Metal GEMM performance.

During **decode** on Apple Silicon:
- Expert computation is memory-bandwidth bound regardless.
- Apple Silicon's high bandwidth-to-compute ratio helps.
- The main optimization is minimizing weight loading (quantization, caching).

**Sources:**
- [Prefill vs Decode Bottlenecks (arXiv 2512.22066)](https://arxiv.org/html/2512.22066v1)
- [A Systematic Characterization of LLM Inference on GPUs](https://www.arxiv.org/pdf/2512.01644)
- [MoE-Lens (2025)](https://arxiv.org/html/2504.09345v1)

---

## 19. Synthesis: What Matters for Single-Device Apple Silicon Inference

### Priority 1: Grouped/Fused Expert Computation (CRITICAL for Prefill)

The single most impactful optimization is replacing sequential per-expert kernel launches with a single fused operation. Options:

1. **Grouped GEMM via Metal**: A custom Metal compute shader that processes all expert GEMMs in a single dispatch. Tokens are sorted by expert and concatenated; the kernel uses offset/count metadata to process each expert's sub-batch.

2. **Concatenated expert computation**: Concatenate all active expert weights into a block-diagonal matrix and all sorted token groups into one tensor. Compute as a single (block-sparse) matmul. This is the MegaBlocks approach adapted for Metal.

3. **MLX-level fusion**: Use MLX's `mx.compile` or custom Metal kernel to fuse the permute-compute-unpermute pipeline into a single operation, eliminating intermediate materialization.

### Priority 2: Token Grouping / Permutation Optimization

- Sort tokens by expert ID using argsort (already available in MLX).
- Compute expert boundaries (cumsum of per-expert counts).
- Ensure the permutation is fused with or pipelined alongside expert computation.
- Consider tile-aligned rounding of per-expert counts (SonicMoE-style) if Metal GEMM performance is sensitive to matrix dimensions.

### Priority 3: Arithmetic Intensity Improvement

- **Quantization**: INT4/INT8 expert weights reduce bytes-per-weight, effectively increasing arithmetic intensity. MLX already supports quantized models.
- **Batch accumulation for experts**: During prefill, process attention in chunks but accumulate a larger batch for expert layers (MoE-Gen insight).
- **Apple Silicon advantage**: The lower compute-to-bandwidth ratio means MoE experts may naturally be closer to compute-bound on Apple Silicon than on NVIDIA GPUs.

### Priority 4: Minimize Overhead

- **Avoid Python-level loops over experts**: Each Python-level operation in MLX creates graph nodes and potential synchronization points.
- **Pre-compute routing metadata**: Router indices, expert counts, and offsets can be computed once and reused for both gate1 and gate2 projections within each expert.
- **Memory layout**: Ensure expert weights are stored contiguously in memory for efficient access patterns.

### What Does NOT Matter for Single-Device Apple Silicon

- **All-to-All communication**: Not relevant (single device).
- **Expert parallelism / expert slicing across devices**: Not relevant.
- **Network topology optimization**: Not relevant.
- **Host-device transfer optimization**: Not relevant (unified memory).
- **FP8-specific optimizations**: Apple Silicon lacks native FP8 support (though future hardware may add it).

### Estimated Impact Hierarchy

| Optimization | Estimated Prefill Speedup | Complexity |
|---|---|---|
| Grouped GEMM (single kernel for all experts) | 2-4x | High (custom Metal kernel) |
| Token sort + contiguous expert dispatch | 1.3-2x | Medium (MLX ops) |
| Tile-aligned expert batch rounding | 1.1-1.3x | Low |
| Fused permute-compute-unpermute | 1.2-1.5x | High (custom Metal kernel) |
| Quantization (INT4 experts) | 1.5-2x on bandwidth | Low (MLX supports) |
| Batch accumulation for experts | 1.2-1.5x | Medium |

---

## Appendix: Key Papers and References

| Paper/System | Year | Key Contribution | Link |
|---|---|---|---|
| MegaBlocks | 2023 | Block-sparse MoE, no token dropping | [arXiv 2211.15841](https://arxiv.org/abs/2211.15841) |
| DeepSpeed-MoE | 2022 | Kernel fusion, dense dispatch | [arXiv 2201.05596](https://arxiv.org/abs/2201.05596) |
| Tutel | 2023 | Adaptive parallelism | [arXiv 2206.03382](https://arxiv.org/abs/2206.03382) |
| SonicMoE | 2024 | Tile-aware token rounding | [arXiv 2512.14080](https://arxiv.org/abs/2512.14080) |
| MoE-Gen | 2025 | Module-based batching, single GPU | [arXiv 2503.09716](https://arxiv.org/abs/2503.09716) |
| DeepSeek V3 | 2024 | FP8 grouped GEMM, 256 experts | [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) |
| Expert Choice Routing | 2022 | Expert-selects-tokens routing | [arXiv 2202.09368](https://arxiv.org/abs/2202.09368) |
| FlashMoE | 2025 | Fused dispatch+compute kernel | [arXiv 2506.04667](https://arxiv.org/html/2506.04667v1) |
| Multi-Node EP on Apple Silicon | 2024 | Apple Silicon MoE distributed | [arXiv 2506.23635](https://arxiv.org/abs/2506.23635) |
| KTransformers | 2025 | CPU-GPU hybrid MoE inference | [SOSP 2025](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf) |
| Rectify-Router | 2024 | Reclaim dropped/padded tokens | [arXiv 2402.12399](https://arxiv.org/html/2402.12399v1) |
| MoE-Lightning | 2024 | CPU-GPU-I/O pipelining | [ACM ASPLOS 2025](https://dl.acm.org/doi/10.1145/3669940.3707267) |
| PyTorch Grouped GEMM | 2025 | Persistent cache-aware Triton kernel | [PyTorch Blog](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) |
| vLLM Fused MoE | 2024-25 | Production fused MoE kernels | [vLLM Docs](https://docs.vllm.ai/en/latest/design/moe_kernel_features/) |
| SGLang MoE | 2024-25 | Align & Sort, multi-backend | [SGLang GitHub](https://github.com/sgl-project/sglang) |
| Apple MLX M5 | 2025 | Neural Accelerators for LLM | [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) |
