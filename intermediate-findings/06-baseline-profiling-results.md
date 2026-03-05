# Baseline Profiling Results: Qwen3.5-35B-A3B-4bit MoE Prefill

## Model Configuration
- **Model**: mlx-community/Qwen3.5-35B-A3B-4bit
- **MoE layers**: 40 (out of 64 total layers)
- **num_experts**: 256
- **top_k**: 8 (each token routes to 8 experts)
- **hidden_size**: 2048
- **moe_intermediate_size**: 1024
- **Expert MLP**: Quantized 4-bit, group_size=64
- **Gate**: Quantized 8-bit

## MoE Block Throughput (single layer)

| seq_len | Time (ms) | Tokens/s | ms/token |
|---------|-----------|----------|----------|
| 1       | 1.16      | 863      | 1.160    |
| 16      | 2.39      | 6,693    | 0.149    |
| 64      | 4.22      | 15,153   | 0.066    |
| 128     | 10.35     | 12,372   | 0.081    |
| 256     | 9.24      | 27,696   | 0.036    |
| 512     | 10.33     | 49,586   | 0.020    |
| 1024    | 14.17     | 72,247   | 0.014    |
| 2048    | 23.71     | 86,365   | 0.012    |

### Key Observations:
1. **Sub-linear scaling**: 2048 tokens takes ~20x longer than 1 token, not 2048x
2. **Throughput increases with seq_len**: From 863 tok/s (1 token) to 86K tok/s (2048 tokens)
3. **Anomaly at 128**: 128 tokens is slower than 256 (10.35ms vs 9.24ms) - gather_sort threshold effect

## Time Breakdown

| seq_len | Routing (%) | Expert MLP (%) | Shared Expert (%) |
|---------|-------------|----------------|-------------------|
| 128     | 5.7%        | 87.5%          | 6.7%              |
| 512     | 3.5%        | 88.8%          | 7.8%              |
| 1024    | 3.2%        | 88.9%          | 8.0%              |
| 2048    | 2.6%        | 89.1%          | 8.3%              |

**Expert MLP dominates at ~89% of total time.**

## gather_qmm Performance Comparison (CORRECTED - fair eval)

| seq_len | gather_qmm sorted | gather_qmm unsorted | Explicit grouped | # groups |
|---------|-------------------|---------------------|------------------|----------|
| 128     | **3.33 ms**       | 3.67 ms             | 11.92 ms         | 250      |
| 256     | **3.88 ms**       | 7.22 ms             | 13.78 ms         | 256      |
| 512     | **5.01 ms**       | 14.34 ms            | 15.53 ms         | 256      |
| 1024    | **7.50 ms**       | 69.37 ms            | 16.87 ms         | 256      |
| 2048    | **24.38 ms**      | 117.56 ms           | 25.46 ms         | 256      |
| 4096    | **39.34 ms**      | 229.92 ms           | 43.85 ms         | 256      |

### Critical Conclusion:
**MLX's gather_qmm with sorted_indices=True is already the fastest approach.** Explicit Python-level
grouped matmuls have too much overhead from launching 256 separate Metal kernels. The gather_qmm
batched kernel handles all 256 experts in a single dispatch.

The sorting (gather_sort) provides massive speedups:
- seq_len=1024: 9.2x faster sorted vs unsorted
- seq_len=4096: 5.8x faster sorted vs unsorted

## Token Distribution Across Experts

### seq_len=1024 (representative case)
- Token-expert pairs: 8192 (1024 tokens × 8 experts each)
- Mean tokens/expert: 32.0, Std: 15.4
- Min/Max: 3 / 105
- p50/p90/p99: 31 / 52 / 68
- Experts with 0 tokens: 0

### seq_len=2048
- Token-expert pairs: 16384
- Mean tokens/expert: 64.0, Std: 31.4
- Min/Max: 4 / 234
- p50/p90/p99: 64 / 100 / 155

## Where Optimization Opportunity Lies

Since gather_qmm is already well-optimized with sorted indices, the remaining opportunities are:

1. **Fusing operations**: gate_proj + up_proj could potentially be fused into a single
   gather_qmm call (concatenated weights) to reduce kernel launches from 3 to 2
2. **Better sort thresholds**: The 64-token threshold seems to cause an anomaly at seq_len=128
3. **Overlapping compute**: Can shared_expert and routed_expert computation overlap?
4. **The sort itself**: _gather_sort uses mx.argsort which may not be optimal for this pattern
5. **Padding**: Aligning expert group sizes to tile boundaries (BM=16 for standard, BM=64 for NAX)
