# KTransformers MoE Inference Analysis

## Overview

KTransformers is a heterogeneous CPU+GPU inference framework developed by KVCache.AI, primarily targeting large MoE models (DeepSeek V2/V3, Qwen MoE, Mixtral) that cannot fit entirely in GPU memory. The framework offloads expert weights to CPU memory while keeping attention, gating, shared experts, and selectively "hot" routed experts on GPU.

The codebase has two main components:
- **kt-sft** (`kt-sft/ktransformers/`): The original framework with model definitions, operator injection, and serving infrastructure.
- **kt-kernel** (`kt-kernel/`): The newer, standalone kernel library with refactored MoE CPU backends, AMX/AVX512 kernels, and a clean Python API.

## 1. Expert Computation Architecture

### 1.1 The Hybrid CPU+GPU Design

KTransformers uses a **module injection** system driven by YAML configuration files that replace standard HuggingFace model components with optimized implementations.

From the YAML optimization rule (`DeepSeek-V3-Chat-fp8-linear-ggml-experts.yaml`):
```yaml
# Gate stays on GPU
- match:
    class: ktransformers.models.modeling_deepseek_v3.MoEGate
  replace:
    class: ktransformers.operators.gate.KMoEGate
    kwargs:
      generate_device: "cuda:0"

# Experts go to CPU for generation, GPU for prefill
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      prefill_device: "cuda"
      prefill_op: "KExpertsTorch"
      generate_device: "cpu"
      generate_op: "KExpertsCPU"
      out_device: "cuda"
  recursive: False
```

**Key architectural decisions:**
- The **gate/router** always runs on GPU (fast, small parameter count)
- **Shared experts** always run on GPU
- **Routed experts** run on CPU during decode, GPU during prefill (configurable)
- Attention (MLA) runs on GPU
- The result is that only the expert FFN weights (the bulk of MoE model parameters) need CPU offloading

### 1.2 Dual-Mode Operation (Prefill vs Decode)

From `KTransformersExperts` (`kt-sft/ktransformers/operators/experts.py`, line 1767):
```python
class KTransformersExpertsV2(BaseInjectedModule, KExpertsBase):
    def forward(self, input_tensor, expert_ids, weights, bsz_tensor, cuda_graph_idx=0):
        if self.mode == InferenceState.GENERATE:
            return self.generate_experts.forward(...)  # CPU path
        elif self.mode == InferenceState.PREFILL:
            return self.prefill_experts.forward(...)    # GPU path
```

- **Prefill**: All experts run on GPU via `KExpertsTorch` (standard PyTorch GEMM)
- **Decode**: Experts run on CPU via `KExpertsCPU` using optimized kernels (AMX/llamafile)
- Mode switching happens at the framework level between prefill and decode phases

### 1.3 The CPU MoE Kernel Pipeline

In the C++ kernel layer (`kt-kernel/operators/amx/moe_base.hpp`), the forward pass follows this pipeline:

**Prefill path** (`forward_prefill`, for qlen > 1):
1. **Token-to-expert mapping**: Count how many tokens route to each expert, build position arrays
2. **Input scatter**: Copy each token's hidden state to per-expert input buffers (parallelized via work-stealing)
3. **Quantize input**: Convert scattered bf16 inputs to AMX tile format (`gate_up_ba_[expert].from_mat(...)`)
4. **Gate+Up GEMM**: Run gate and up projections simultaneously for all activated experts (2x parallelism via `task_id2 / 2` for gate vs up)
5. **Activation**: Apply SiLU activation fused with element-wise multiply: `gate_output = silu(gate) * up`
6. **Quantize intermediate**: Prepare for down projection
7. **Down GEMM**: Run down projection for all activated experts
8. **Weighted scatter-reduce**: Apply expert weights and accumulate results back to token positions using AVX512 FMA

**Decode path** (`forward_decode`, for qlen == 1):
- Same pipeline but simplified for single-token case
- No need for complex token routing -- each expert processes one copy of the single token
- Direct buffer allocation without the parallel scatter step

Key code from `moe_base.hpp` (line 165):
```cpp
void forward(int qlen, int k, const int64_t* expert_ids, const float* weights,
             const void* input, void* output) {
    if (qlen > 1) {
        forward_prefill(qlen, k, expert_ids, weights, input, output);
    } else {
        forward_decode(k, expert_ids, weights, input, output);
    }
}
```

## 2. CPU Offloading Strategy

### 2.1 Per-Expert GPU/CPU Placement

KTransformers uses a **per-expert boolean mask** (`gpu_experts_mask`) to decide which experts run on GPU vs CPU. This mask is per-layer.

From `kt-kernel/operators/common.hpp` (line 230):
```cpp
struct GeneralMOEConfig {
    int num_gpu_experts = 0;
    uint8_t* gpu_experts_mask = nullptr;  // Bool mask: true = expert on GPU

    inline bool should_skip_expert(int64_t expert_id) const {
        return expert_id < 0 || expert_id >= expert_num ||
               (gpu_experts_mask && gpu_experts_mask[expert_id]);
    }
};
```

When the CPU kernel encounters an expert marked as "on GPU" (via `should_skip_expert`), it simply skips that expert. The GPU side handles those experts separately. This means:
- CPU kernels only process experts in their mask
- GPU kernels only process experts in their mask
- The results are merged at the Python layer

### 2.2 Hot Expert Selection Strategies

From the expert scheduling tutorial (`doc/en/kt-kernel/experts-sched-Tutorial.md`) and README, KTransformers supports four expert placement strategies:

| Strategy | Description |
|----------|-------------|
| `uniform` | Distributes GPU experts evenly across all MoE layers |
| `frequency` | Places most frequently activated experts on GPU based on profiled activation statistics |
| `front-loading` | Fills GPU experts from the first layer onwards |
| `random` | Randomly selects experts with fixed seed (42) |

The `generate_gpu_experts_masks` function (`kt-kernel/python/experts_base.py`, line 21) implements frequency-based selection:
```python
def generate_gpu_experts_masks(
    activation_freq: torch.Tensor,  # shape: (num_layers, num_experts)
    num_gpu_experts: int,           # total GPU experts across ALL layers
) -> torch.Tensor:
    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(flat_freq, k=num_gpu_experts, largest=True, sorted=False)
    gpu_experts_masks = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    gpu_experts_masks[top_indices] = True
    return gpu_experts_masks.view(num_layers, num_experts_per_layer)
```

This is a **global** selection across all layers -- the most frequently activated experts across the entire model are placed on GPU. This means some layers may have more GPU experts than others, which is a significant insight.

### 2.3 Dynamic Expert Updates

KTransformers also supports **dynamic expert redistribution** during inference:
- During layerwise prefill, the system collects actual routing statistics
- It then redistributes GPU experts based on observed activation patterns
- Controlled by `--kt-enable-dynamic-expert-update` and `--kt-gpu-prefill-token-threshold`

Performance data from benchmarks (Qwen3-Next-80B-A3B with 4x RTX 4090):
- At 20% GPU expert ratio: static frequency = 61.92 tok/s, dynamic = 74.73 tok/s (21% improvement)
- At 50% GPU expert ratio: static frequency = 76.19 tok/s, dynamic = 81.17 tok/s (7% improvement)
- Dynamic updates are most effective at lower GPU expert ratios

## 3. Expert Batching and Grouping Strategies

### 3.1 Expert-Parallel Execution in Prefill

During prefill, KTransformers groups tokens by their assigned experts and processes all tokens for each expert as a batch. From the `forward_prefill` function in `moe_base.hpp`:

```cpp
// Step 1: Count tokens per expert
for (int i = 0; i < qlen; i++) {
    for (int j = 0; j < k; j++) {
        if (config_.should_skip_expert(expert_ids[i * k + j])) continue;
        m_local_pos_[i][j] = m_local_num_[expert_ids[i * k + j]]++;
    }
}

// Step 2: Scatter inputs to per-expert buffers
pool->do_work_stealing_job(qlen, nullptr, [&](int i) {
    for (int j = 0; j < k; j++) {
        memcpy(m_local_input_ptr_[expert_ids[i * k + j]] + ...,
               (ggml_bf16_t*)input + i * config_.hidden_size, ...);
    }
});

// Step 3: Gate+Up GEMM batched across all activated experts
// Key: gate and up are interleaved for 2x parallelism
pool->do_work_stealing_job(
    nth * activated_expert * 2, [](int _) { T::config(); },
    [this, nth, qlen](int task_id2) {
        int task_id = task_id2 / 2;
        bool do_up = task_id2 % 2;
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
    });
```

**Key batching insight**: KTransformers does NOT batch across experts (no grouped GEMM). Instead it processes experts sequentially but parallelizes the GEMM for each expert across CPU threads using a work-stealing thread pool. The gate and up projections are interleaved (task_id2 % 2) to improve CPU utilization.

### 3.2 NUMA-Aware Tensor Parallelism (TP)

From `kt-kernel/operators/moe-tp.hpp`, KTransformers implements **NUMA-aware tensor parallelism**:
- The intermediate dimension is split across NUMA nodes
- Each NUMA node processes its portion of the gate/up/down projections
- Results are merged with AVX512 reductions across NUMA nodes

```cpp
TP_MOE_Common(GeneralMOEConfig config) : config(config) {
    tp_count = config.pool->config.subpool_count;  // = NUMA node count
    for (auto i = 0; i < tp_count; i++) {
        GeneralMOEConfig tp_config = config;
        tp_config.intermediate_size /= tp_count;  // Split intermediate dim
        tp_configs.push_back(tp_config);
    }
    // Create one MoE instance per NUMA node
    config.pool->dispense_backend()->do_numa_job(
        [this](int i) { tps[i] = new T(tp_configs[i], i); });
}
```

### 3.3 Buffer Pre-allocation and Double Buffering

From `KExpertsCPUBuffer` in `kt-kernel/python/experts_base.py`:
- Pinned memory buffers are pre-allocated for known batch sizes
- **Double buffering** (`buffer_depth = 2`) allows overlapping data transfer with computation
- Buffer slot assignment: `current_slot = layer_idx % buffer_depth`
- This enables pipelining across consecutive MoE layers

```python
class KExpertsCPUBuffer:
    buffer_depth: int = 2  # Double buffering for pipelining

    @classmethod
    def get_buffer(cls, hidden_states, num_experts_per_tok):
        # Returns tuple of (input_cpu, immediate_ids, deferred_ids, weights, output_cpu, bsz, output_gpu)
        # All pinned memory for async GPU-CPU transfer
```

## 4. Deferred Expert Execution (Pipelining)

### 4.1 The Deferred Expert Mechanism

KTransformers implements a novel **deferred expert execution** strategy where some experts for the current layer are executed asynchronously while the next layer's attention/gating runs on GPU.

From `BaseMoEWrapper.submit_forward` (`kt-kernel/python/experts_base.py`, line 299):
```python
def submit_forward(self, hidden_states, topk_ids, topk_weights, cuda_stream):
    if self.max_deferred_experts_per_token > 0:
        protected_k = self.num_experts_per_tok - self.max_deferred_experts_per_token
        immediate_ids, deferred_ids = self.select_deferred_experts(
            topk_ids_long, topk_weights, protected_k)
    else:
        immediate_ids = topk_ids_long
        deferred_ids = None

    # Submit immediate experts (highest-weight experts)
    self.cpu_infer.submit_with_cuda_stream(cuda_stream,
        self.moe.forward_task(bsz, ..., immediate_ids, weights, input, output, incremental))

    # Submit deferred experts (lower-weight experts, can overlap with next layer)
    if deferred_ids is not None:
        self.cpu_infer.submit_with_cuda_stream(cuda_stream,
            self.moe.forward_task(bsz, ..., deferred_ids, weights, input, output_next_slot, False))
        BaseMoEWrapper._layer_has_pending_deferred[self.layer_idx] = True
```

### 4.2 Expert Selection for Deferral

The `select_deferred_experts` method (`kt-kernel/python/experts_base.py`, line 269):
```python
def select_deferred_experts(self, expert_ids, expert_scores, protected_k):
    # Select top-protected_k experts by score as "immediate" (must-compute-now)
    topk_result = torch.topk(expert_scores, k=protected_k, dim=-1, largest=True, sorted=False)
    protected_indices = topk_result.indices

    # immediate_ids: experts with highest weights (computed synchronously)
    immediate_ids = expert_ids.clone().masked_fill(~protected_mask, -1)
    # deferred_ids: lower-weight experts (can overlap with next layer)
    deferred_ids = expert_ids.clone().masked_fill(protected_mask, -1)
    return immediate_ids, deferred_ids
```

The idea: For DeepSeek V3 with top-8 routing, if `max_deferred_experts_per_token=2`, then the 6 highest-weight experts are computed immediately, and the 2 lowest-weight experts are deferred. The deferred experts' computation overlaps with the next layer's attention, with their results accumulated incrementally.

### 4.3 Cross-Layer Pipelining

```python
# When syncing results, check if previous layer had deferred work
incremental = BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False)
# If incremental=True, the output includes accumulated results from deferred experts

def sync_forward(self, hidden_states, cuda_stream):
    allow_pending = 1 if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx, False) else 0
    self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
    output_gpu[current_slot].copy_(output_cpu[current_slot], non_blocking=True)
```

## 5. Custom CPU Kernels

### 5.1 AMX Kernels (Intel Advanced Matrix Extensions)

KTransformers provides multiple AMX-based MoE implementations:
- **AMXInt4_MOE**: INT4 quantized weights, highest performance on AMX-capable CPUs (Sapphire Rapids+)
- **AMXInt8_MOE**: INT8 quantized weights, better accuracy than INT4
- **AMXBF16_MOE**: BF16 weights, no quantization loss
- **AMXFP8_MOE**: FP8 weights, good balance of performance and accuracy
- **AMXInt4_KGroup_MOE**: INT4 with K-group quantization (native INT4 weight format shared with GPU)

The AMX kernel architecture (from `moe_base.hpp`):
1. Uses AMX tile registers for matrix multiply (TMUL instructions)
2. Implements CRTP (Curiously Recurring Template Pattern) for zero-cost polymorphism
3. Buffer management: BufferA (input tiles), BufferB (weight tiles), BufferC (output tiles)
4. Work distribution: `do_work_stealing_job` for load-balanced parallelism across cores

Key vectorization patterns:
```cpp
// AVX512 bf16-to-fp32 conversion + FMA for weighted accumulation
avx512_32xbf16_to_32xfp32((__m512i*)(down_output_ptr), &down_output0, &down_output1);
x0 = _mm512_fmadd_ps(down_output0, weight, x0);
x1 = _mm512_fmadd_ps(down_output1, weight, x1);
```

### 5.2 Llamafile Backend

The `LLAMA_MOE_TP` class (`kt-kernel/operators/llamafile/moe.hpp`) uses llamafile's SGEMM implementation:
- Supports all GGML quantization types (Q4_K_M, Q5_K_M, Q6_K, etc.)
- Uses llamafile's optimized `flag_sgemm` for matrix multiplication
- Input/output quantization via GGML type traits system
- TP splitting aligned to QK_K (256 elements) block boundaries

### 5.3 Work Distribution

The CPU thread pool uses work-stealing with NUMA awareness:
```cpp
// Thread pool configuration
WorkerPoolConfig:
    subpool_count     // Number of NUMA nodes
    subpool_numa_map  // NUMA node mapping
    subpool_thread_count  // Threads per NUMA node

// Work distribution for GEMM
int nth = T::recommended_nth(config_.intermediate_size);  // Threads per expert
pool->do_work_stealing_job(
    nth * activated_expert * 2,  // Total tasks = threads * experts * 2 (gate+up)
    [](int _) { T::config(); },  // Per-thread AMX tile config
    [this, nth](int task_id2) {   // Actual work
        int task_id = task_id2 / 2;
        bool do_up = task_id2 % 2;
        int expert_idx = m_expert_id_map_[task_id / nth];
        int ith = task_id % nth;
        derived()->do_gate_up_gemm(do_up, expert_idx, ith, nth, qlen);
    });
```

## 6. Async GPU-CPU Communication

### 6.1 Submit/Sync Pattern

KTransformers uses a non-blocking submit/sync pattern for CPU expert computation:

```python
# In KDeepseekV3MoE.forward():
if sequence_length == 1 and torch.cuda.is_current_stream_capturing():
    # Async submit to CPU
    self.experts.generate_experts.submit_for_one_decode(hidden_states, topk_idx, topk_weight)

    # While CPU computes experts, GPU computes shared experts
    y_ = self.shared_experts(identity)

    # Sync and get CPU results
    y = self.experts.generate_experts.sync_for_one_decode()
    y += y_
```

This overlaps:
- **CPU**: Routed expert computation (the slow part)
- **GPU**: Shared expert computation, attention for the next token, etc.

### 6.2 Pinned Memory and CUDA Stream Synchronization

From `KExpertsCPUBuffer.get_buffer`:
```python
# All CPU buffers use pinned memory for fast DMA transfers
input_tensor_cpu = torch.zeros(..., device="cpu", pin_memory=True, dtype=torch.bfloat16)
output_cpu = torch.zeros(..., device="cpu", pin_memory=True, dtype=torch.bfloat16)

# CUDA stream synchronization
self.cpu_infer.submit_with_cuda_stream(cuda_stream, self.moe.forward_task(...))
self.cpu_infer.sync_with_cuda_stream(cuda_stream, allow_pending)
```

## 7. Performance Results

### 7.1 Expert Placement Strategy Comparison

From the benchmark data (Qwen3-Next-80B-A3B, 4x RTX 4090):

| GPU Expert Ratio | random | uniform | frequency | dynamic |
|------------------|--------|---------|-----------|---------|
| 0% | 53.0 | 53.0 | 52.7 | 53.4 |
| 10% | 56.6 | 56.6 | 58.6 | 70.2 |
| 30% | 62.9 | 62.1 | 66.5 | 75.6 |
| 50% | 70.4 | 65.3 | 76.2 | 81.2 |
| 70% | 74.4 | 76.2 | 89.4 | 88.7 |
| 100% | 112.6 | 112.3 | 114.3 | 113.0 |

Key observations:
- `frequency` strategy dominates `uniform` by 10-30% at mid-range GPU expert ratios
- `dynamic-expert-update` provides the biggest gains at low GPU ratios (10-40%)
- At 100% GPU experts, all strategies converge (no CPU bottleneck)

### 7.2 Hardware Requirements

- **Minimum**: RTX 4090 24GB + x86 CPU with AVX512, 256GB RAM
- **Tested**: 4x RTX 4090 + Intel Xeon Gold 6454S (Sapphire Rapids with AMX), 512GB DDR5
- AMX-capable CPUs (Sapphire Rapids+) are strongly preferred for INT4/INT8 kernels

## 8. Comparison with MLX Approach

### 8.1 Fundamental Differences

| Aspect | KTransformers | MLX (mlx-lm) |
|--------|--------------|--------------|
| **Target hardware** | x86 CPU (AVX512/AMX) + NVIDIA GPU | Apple Silicon (unified memory) |
| **Memory model** | Discrete: CPU RAM + GPU VRAM | Unified: shared CPU/GPU memory |
| **Expert placement** | Explicit GPU/CPU mask per expert | All experts accessible by GPU via unified memory |
| **Data transfer** | Pinned memory + CUDA streams | Zero-copy (same physical memory) |
| **Expert computation** | CPU kernels (AMX/llamafile) for offloaded experts | Metal kernels (gather_qmm) for all experts |
| **Quantization** | INT4/INT8/FP8/BF16 on CPU, FP8/FP16 on GPU | 4-bit/8-bit quantized on unified memory |
| **Parallelism** | NUMA-aware thread pool + CUDA streams | Metal GPU compute + lazy evaluation |

### 8.2 What MLX Can Learn

1. **Expert frequency profiling for caching**: KTransformers' `frequency` strategy and `dynamic-expert-update` show that knowing which experts are "hot" provides 20-30% throughput improvement. MLX could profile expert activation patterns and use this to optimize memory access patterns (e.g., ensuring hot expert weights are in GPU caches).

2. **Deferred expert execution**: The concept of computing high-weight experts immediately and deferring low-weight experts could be adapted for MLX. Even on unified memory, deferring low-weight experts to overlap with the next layer's attention could reduce latency. This is essentially a quality-latency tradeoff that could be user-configurable.

3. **Chunked prefill for MoE**: KTransformers uses chunked prefill (default 4096-32768 tokens) to bound memory usage during prefill. MLX's MoE prefill could similarly chunk to control peak memory.

4. **Gate+Up projection interleaving**: KTransformers interleaves gate and up projections (2x parallelism) rather than computing them sequentially. MLX's Metal kernels could potentially fuse or parallelize these operations similarly.

5. **Expert-grouped batching for prefill**: During prefill, KTransformers groups all tokens assigned to the same expert and processes them as a batch. This is more efficient than processing token-by-token. MLX's `gather_qmm` already does something similar via grouped GEMM, which is actually a more elegant solution -- it handles the grouping implicitly in the kernel.

### 8.3 What KTransformers Could Learn from MLX

1. **Unified memory eliminates the transfer bottleneck**: KTransformers spends significant engineering on pinned memory, async transfers, and double buffering. MLX's unified memory approach avoids this entirely.

2. **Grouped GEMM via gather_qmm**: MLX's approach of using a single Metal kernel that gathers the correct expert weights via index tensors is more elegant than KTransformers' scatter-gather-per-expert approach. This is possible because of unified memory's random access capability.

3. **Lazy evaluation**: MLX's lazy evaluation graph can potentially optimize the full MoE layer as a unit, rather than requiring explicit pipelining.

### 8.4 Key Takeaways for mlx-lm Optimization

1. **Expert activation profiling is valuable** even on unified memory. If MLX could track which experts are frequently activated and ensure their weights stay in GPU L2 cache (or are pre-fetched), it could significantly improve decode latency.

2. **The deferred/pipelined execution idea translates well**: Even without CPU offloading, computing the top-k experts by weight and overlapping lower-weight experts with the next layer could reduce effective latency.

3. **NUMA/memory bandwidth is the bottleneck for MoE on CPU**: KTransformers' elaborate NUMA-aware TP splitting shows that memory bandwidth is the primary bottleneck for expert computation. On Apple Silicon, the unified memory bandwidth (~400-800 GB/s on M-series) is the analogous bottleneck. MLX's gather_qmm kernels should be optimized for memory bandwidth utilization.

4. **Expert count per token matters for kernel design**: DeepSeek V3 uses top-8 out of 256 experts per token. This means 97% of expert weights are "cold" per token. Any approach that avoids loading cold weights (like KTransformers' sparse execution) wins. MLX's gather_qmm inherently handles this by only gathering the selected expert weights.

## 9. Code References

Key files analyzed:
- `kt-kernel/python/experts.py` -- Factory interface for MoE CPU backends
- `kt-kernel/python/experts_base.py` -- Base class with GPU mask, deferred experts, buffer management
- `kt-kernel/python/utils/amx.py` -- AMX backend wrapper
- `kt-kernel/python/utils/llamafile.py` -- Llamafile backend wrapper
- `kt-kernel/operators/amx/moe_base.hpp` -- Core AMX MoE kernel (prefill+decode paths)
- `kt-kernel/operators/moe-tp.hpp` -- NUMA-aware tensor parallelism
- `kt-kernel/operators/common.hpp` -- GeneralMOEConfig with gpu_experts_mask
- `kt-kernel/operators/llamafile/moe.hpp` -- Llamafile MoE kernel
- `kt-sft/ktransformers/operators/experts.py` -- Original experts (KExpertsCPU, KDeepseekV3MoE, KTransformersExperts)
- `kt-sft/ktransformers/operators/gate.py` -- Gate/router operators
- `kt-sft/ktransformers/optimize/optimize_rules/*.yaml` -- Model-specific optimization rules
- `doc/en/kt-kernel/experts-sched-Tutorial.md` -- Expert scheduling tutorial with benchmarks
