# Helion Examples – NKI Validation Status

Status of porting `helion_nki/examples/` to the NKI backend.

## Validated (NKI version exists, passes)

| Example | Ops used | Notes |
|---------|----------|-------|
| **add_nki** | tensor_tensor (add) | |
| **sum_nki** | tensor_reduce (sum) | |
| **exp_nki** | activation (exp) | |
| **matmul_nki** | matmul (addmm) | |
| **batch_softmax_nki** | reduce, exp, subtract, divide (reciprocal+multiply) | |
| **softmax_decomposed_nki** | amax, exp, sum, divide | Same pattern as batch_softmax |
| **swiglu_nki** | silu, multiply | Fixed by keeping `aten.silu` undecomposed for NKI + direct NKI silu lowering |
| **broadcast_matmul_nki** | reshape + addmm loop | Fixed in backend by normalizing NKI tensor arg shapes at kernel entry |

## Partial / needs fix

| Example | Status | Blocker |
|---------|--------|---------|

## Likely to work (same ops as validated)

| Example | Ops | Blocker |
|---------|-----|---------|
| **swiglu_fwd** | sigmoid, multiply | sigmoid in activation list |
| **geglu** | silu, multiply | silu supported |
| **softmax** (simple) | F.softmax per tile | May use different decomposition |

## Needs backend implementation

| Example | Missing | Notes |
|---------|--------|-------|
| **bmm** | `aten.baddbmm` | Batch matmul accumulator |
| **rms_norm** | `mean` reduction | mean = sum/n; add to _NKI_REDUCTION_OPS or emit sum + scalar mul |
| **layer_norm** | `mean`, `var` | mean + variance; var = mean(x²) - mean² or sum-of-squares |
| **concatenate** | `torch.where`, `extra_mask` | Masking / conditional load |
| **embedding** | Indexed load (gather) | `weight[indices]` |
| **gather_gemv** | Indexed load | |
| **segment_reduction** | Scatter/gather, indices | Jagged structure |
| **jagged_*** | Dynamic shapes, offsets | |

## Needs NKI ISA / infra

| Example | Blocker |
|---------|---------|
| **split_k_barrier** | `hl.barrier` |
| **low_mem_dropout** | RNG |
| **distributed/** | NCCL, all_reduce, etc. |
| **fp8_gemm**, **int4_gemm**, **nvfp4_gemm** | Quantized matmul |
| **flex_attention** | Custom attention op |
| **mamba2_*** | Scan, complex control flow |

## Not yet tried

| Example | Complexity |
|---------|------------|
| matmul_layernorm | matmul + layer_norm (needs mean) |
| matmul_split_k | Split-K matmul |
| squeeze_and_excitation_net | matmul + reductions |
| welford | Welford variance |
| cross_entropy | log_softmax, indexing |
| kl_div, jsd | Log-space ops |
| grpo_loss | Complex loss |
| long_sum | Large reduction |
| blackwell_attention | Attention variant |
| aot_example | AOT flow |

## Run validated examples

```bash
cd /home/ubuntu/kernel_test && source aws_neuron_venv_pytorch/bin/activate
PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/run_nki_examples.py
```

## Implementation priorities

1. **mean reduction** – Implement as `sum / n` (sum + tensor_scalar multiply by 1/n) to unblock rms_norm, layer_norm.
2. **baddbmm** – Map to addmm in a loop (same pattern as matmul with batch dim).
3. **torch.where** – Add `where_expr` support in NKI backend (see NKI_STATEMENT_BASED_CODEGEN.md).
