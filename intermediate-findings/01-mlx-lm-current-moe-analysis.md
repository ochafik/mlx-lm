# mlx-lm Current MoE Implementation Analysis

## Architecture Overview (Qwen3.5-35B-A3B)

The Qwen3.5-35B-A3B model uses the `qwen3_5_moe` → `qwen3_5` → `qwen3_next` model chain:
- `qwen3_5_moe.py`: Weight sanitization wrapper
- `qwen3_5.py`: Model definition using `Qwen3NextSparseMoeBlock` from `qwen3_next.py`
- `qwen3_next.py`: Core MoE block using `SwitchGLU` from `switch_layers.py`

## MoE Layer Structure

### Qwen3NextSparseMoeBlock (qwen3_next.py:298-334)
```python
class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, args):
        self.gate = nn.Linear(dim, num_experts, bias=False)  # Router
        self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)  # Expert MLP
        self.shared_expert = Qwen3NextMLP(dim, shared_expert_intermediate_size)  # Always-on expert
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)  # Shared expert gate

    def __call__(self, x):
        gates = softmax(self.gate(x))          # [B, S, num_experts]
        inds = argpartition(gates, top_k)       # [B, S, top_k]
        scores = take_along_axis(gates, inds)   # [B, S, top_k]

        y = self.switch_mlp(x, inds)            # Expert computation
        y = (y * scores[..., None]).sum(-2)      # Weighted sum

        shared_y = sigmoid(shared_gate(x)) * shared_expert(x)  # Shared expert
        return y + shared_y
```

## Core Expert Computation: SwitchGLU (switch_layers.py:160-199)

```python
class SwitchGLU(nn.Module):
    def __call__(self, x, indices):
        x = mx.expand_dims(x, (-2, -3))         # Add dims for gather_mm

        do_sort = indices.size >= 64              # ← THRESHOLD: only sort for >=64 tokens
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)  # Sort tokens by expert ID

        x_up = self.up_proj(x, idx, sorted_indices=do_sort)     # gather_mm/gather_qmm
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)  # gather_mm/gather_qmm
        x = self.down_proj(activation(x_up, x_gate), idx, sorted_indices=do_sort)

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)
```

## _gather_sort Implementation (switch_layers.py:12-17)

```python
def _gather_sort(x, indices):
    *_, M = indices.shape        # M = top_k (e.g., 2)
    indices = indices.flatten()  # [B*S*top_k]
    order = mx.argsort(indices)  # Sort by expert ID
    inv_order = mx.argsort(order)
    return x.flatten(0, -3)[order // M], indices[order], inv_order
```

## What gather_mm / gather_qmm Do

These are MLX primitives that perform batched matrix multiplications where each "row" uses a different weight matrix selected by an index:

```python
# For each token i: output[i] = x[i] @ weight[indices[i]].T
mx.gather_mm(x, weight, rhs_indices=indices, sorted_indices=True)
```

The `sorted_indices=True` flag tells MLX that the indices are sorted, which allows it to:
1. Identify contiguous runs of tokens going to the same expert
2. Batch those tokens into a single larger matmul per expert
3. Avoid individual per-token matmuls

## Current Optimization Status

### What IS already optimized:
1. **Token sorting by expert ID** when >=64 tokens (gather_sort)
2. **sorted_indices flag** passed to gather_mm/gather_qmm enabling internal batching
3. **Quantized expert computation** via gather_qmm (4-bit, 8-bit)
4. **Shared expert** computation is a single dense matmul (always efficient)

### Key Question: How effective is MLX's gather_mm batching?

The current implementation delegates the actual batching to MLX's `gather_mm` primitive with
`sorted_indices=True`. The effectiveness depends on how MLX implements this internally:

- **Best case**: MLX identifies contiguous expert groups and dispatches one large matmul per expert.
  This would already achieve what vLLM/SGLang do.
- **Worst case**: MLX still loops over individual tokens or uses a gather-based approach that
  doesn't fully exploit the grouped structure.

### What might NOT be optimized:
1. **No explicit expert-level parallelism** - the code doesn't explicitly batch tokens per expert
   into separate matmul calls. It relies entirely on gather_mm to figure this out.
2. **The 64-token threshold** may be too conservative for large prefill sequences (1k+ tokens).
3. **No prefill/decode distinction** - the same code path is used regardless.
4. **No padding/alignment** - tokens per expert may not be padded to GPU-friendly sizes.
5. **Sequential gate → up → down projections** - these could potentially be parallelized.

## Qwen3.5-35B-A3B Specific Parameters
- `num_experts`: 128 (very large expert count)
- `num_experts_per_tok`: 8 (top-8 routing)
- `hidden_size`: 2560 (relatively small)
- `moe_intermediate_size`: 1024 (small per-expert)
- `shared_expert_intermediate_size`: 10240
- `decoder_sparse_step`: 1 (every layer has MoE)
- `num_hidden_layers`: 64

This means during prefill with 1024 tokens:
- Each token activates 8 out of 128 experts
- Total token-expert pairs: 1024 * 8 = 8192
- Average tokens per expert: 8192 / 128 = 64
- But distribution is non-uniform → some experts get many more, some fewer

## Next Steps
1. Understand how MLX's gather_mm actually works internally (is it truly batching by expert?)
2. Study vLLM's expert-parallel strategy for comparison
3. Study SGLang's approach
4. Determine if there's a meaningful optimization opportunity beyond what MLX already does
