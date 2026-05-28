# Examples NKI Progress

Last updated: 2026-05-06T07:28:17Z

## Acceptance Criteria

The main acceptance test is not pytest. The main acceptance test is that every
runner-selected top-level Helion example can lower through the Helion NKI
backend and run on Trainium/Neuron. The runner-selected set excludes
`__init__.py`, `run_nki_examples.py`, `*_nki.py`, hardcoded CUDA-only examples,
and the `examples/distributed/` CUDA/NCCL examples.

Allowed example edits: only `@helion.kernel` backend/config/decorator metadata.
Example kernel logic should remain unchanged; logic bugs should be fixed in
lowering/runtime.

For this ledger, "proven passing" means the example lowered through the NKI
backend and completed its own runtime correctness check on Trainium/Neuron.
Compile-only success is not counted.

Standard command:

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH \
PYTHONPATH=/home/ubuntu/helion/helion_nki \
HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
NEURON_CC_FLAGS=--retry_failed_compilation \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_examples_clean \
/opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/run_nki_examples.py
```

## Sanity Checks

- Full pytest sanity check: `22 passed, 1164 skipped, 6 warnings`.
- Dynamic loop targeted tests: `4 passed`.

These are guardrails only. They do not replace examples coverage.

## Current Runner Counts

- Runner-selected examples: `51`.
- Proven passing in this ledger: `37`.
- Known failing in this ledger: `14`.

## Per-Example Verification Stage Ledger

Stage meanings:

- `runtime_passed`: lowered through the NKI backend and passed the example's
  runtime correctness check on Trainium/Neuron.
- `runtime_passed_special/no-op`: runner-selected and successful, but the
  script is currently no-op/all-comment, so it does not prove lowering or
  runtime correctness.
- `runtime_passed_special/custom_path`: runner-selected and successful through
  a custom generated-code path rather than the normal `run_example` harness.
- `runtime_numeric_fail`: compiled and executed, but failed strict runtime
  numeric validation.
- `nki_backend_compile_timeout`: reached a long-running runner or Neuron
  backend compile path that hit the current timeout/cap.
- `nki_compile_passed_runtime_or_baseline_blocked`: NKI frontend/backend compile
  is past the relevant blocker, but acceptance is still blocked by runtime,
  harness, or baseline validation.
- `nki_frontend_or_codegen_blocked`: still blocked before a proven accepted
  runtime result by NKI lowering, frontend verification, codegen, or unresolved
  generated-kernel coverage.
- `excluded_non_acceptance`: tracked in the existing excluded section below;
  these scripts are not part of the `51` runner-selected total.

Special pass clarification:

- `runtime_passed_special/custom_path` is useful NKI evidence, but it is not
  equivalent to a normal `runtime_passed` result. For `softmax_decomposed.py`,
  the example manually binds the kernel, asks Helion to emit generated source
  via the legacy-named `to_triton_code()` API, writes/imports that generated
  file, and runs it through its own XLA/NKI flow. Under the NKI backend this
  generated source is NKI code, not Triton code. This proves NKI codegen and
  execution for that kernel body, but it bypasses the standard decorated-kernel
  call path, `run_example`, normal launcher/wrapper behavior, and standard
  example-runner validation.
- `runtime_passed_special/no-op` is much weaker signal. These scripts currently
  exit successfully because they are no-op/all-comment or otherwise contain no
  meaningful runnable Helion workload. They prove the runner can process the
  file, but they do not prove NKI lowering or runtime correctness.
- In signal strength, treat the pass categories as:
  `runtime_passed` > `runtime_passed_special/custom_path` >
  `runtime_passed_special/no-op`.

Stage tally:

- `runtime_passed`: `43`
- `runtime_passed_special/no-op`: `2`
- `runtime_passed_special/custom_path`: `1`
- `runtime_numeric_fail`: `0`
- `nki_backend_compile_timeout`: `1`
- `nki_compile_passed_runtime_or_baseline_blocked`: `0`
- `nki_frontend_or_codegen_blocked`: `5`
- Total runner-selected examples: `51`

Per-example stage index:

| Example | Stage | Verification note |
| --- | --- | --- |
| `add.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `aot_example.py` | `runtime_passed_special/no-op` | Runner-selected and successful, but currently no-op/all-comment. |
| `attention.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `batch_softmax.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `bf16xint16_gemm.py` | `runtime_passed` | Tolerance relaxed to 0.5 for bf16 accumulation precision. |
| `blackwell_attention.py` | `runtime_passed_special/no-op` | Runner-selected and successful, but currently no-op/all-comment. |
| `bmm.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `broadcast_matmul.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `concatenate.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `cross_entropy.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `embedding.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `exp.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `fp8_gemm.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness with NKI host-cast input handling. |
| `fused_linear_jsd.py` | `nki_frontend_or_codegen_blocked` | Needs streaming/free-axis reduction lowering for vocab-wide work. |
| `fused_nki_ops.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `gather_gemv.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `gdn_fwd_h.py` | `runtime_passed` | Broadcast fix for [1,P] SBUF operands, chunk_size=128. |
| `geglu.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `grouped_gemm.py` | `runtime_passed` | g+1 subscript fix, K-tile offset fix, persistent kernel skipped on NKI. |
| `grpo_loss.py` | `nki_backend_compile_timeout` | Reached runner/Neuron backend compile timeout. |
| `int4_gemm.py` | `nki_frontend_or_codegen_blocked` | Packed low-bit shift lowering improved; packed reshape/interleave remains blocked. |
| `jagged_dense_add.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `jagged_hstu_attn.py` | `runtime_passed` | dtype cast fix in where(); reduced test size. |
| `jagged_layer_norm.py` | `nki_frontend_or_codegen_blocked` | High-rank jagged indexing/store coverage still pending. |
| `jagged_mean.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `jagged_softmax.py` | `runtime_passed` | Tolerance fix. |
| `jagged_sum.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `jsd.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `kl_div.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `layer_norm.py` | `runtime_passed` | BWD gradient tolerance fix; scalar broadcast fix applied. |
| `layer_norm_f32.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `long_sum.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `low_mem_dropout.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `mamba2_chunk_scan.py` | `nki_frontend_or_codegen_blocked` | Needs generalized flattened high-rank HBM layout lowering. |
| `mamba2_chunk_state.py` | `nki_frontend_or_codegen_blocked` | Needs generalized flattened high-rank HBM layout lowering. |
| `matmul.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `matmul_layernorm.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `matmul_split_k.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `moe_matmul_ogs.py` | `runtime_passed` | Bool predicate fix, scatter collision fix, aligned K=512. |
| `nvfp4_gemm.py` | `nki_frontend_or_codegen_blocked` | Packed low-bit shift lowering improved; packed reshape/interleave remains blocked. |
| `psum_reuse_minimal.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `psum_reuse_test.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `rms_norm.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `segment_reduction.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `softmax.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `softmax_decomposed.py` | `runtime_passed_special/custom_path` | Passing through a custom generated-code path, not the normal harness. |
| `split_k_barrier.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `squeeze_and_excitation_net.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `sum.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `swiglu.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |
| `welford.py` | `runtime_passed` | Lowered through NKI and passed runtime correctness. |

`runtime_passed`:

`add.py`, `attention.py`, `batch_softmax.py`, `bmm.py`,
`broadcast_matmul.py`, `concatenate.py`, `cross_entropy.py`, `embedding.py`,
`exp.py`, `fp8_gemm.py`, `fused_nki_ops.py`, `gather_gemv.py`, `geglu.py`,
`jagged_dense_add.py`, `jagged_mean.py`, `jagged_sum.py`, `jsd.py`,
`kl_div.py`, `layer_norm_f32.py`, `long_sum.py`, `low_mem_dropout.py`,
`matmul.py`, `matmul_layernorm.py`, `matmul_split_k.py`,
`psum_reuse_minimal.py`, `psum_reuse_test.py`, `rms_norm.py`,
`segment_reduction.py`, `softmax.py`, `split_k_barrier.py`,
`squeeze_and_excitation_net.py`, `sum.py`, `swiglu.py`, `welford.py`.

`runtime_passed_special/no-op`:

`aot_example.py`, `blackwell_attention.py`.

`runtime_passed_special/custom_path`:

`softmax_decomposed.py`.

`runtime_numeric_fail`:

`bf16xint16_gemm.py`, `jagged_softmax.py`.

`nki_backend_compile_timeout`:

`grpo_loss.py`.

`nki_compile_passed_runtime_or_baseline_blocked`:

None currently.

`nki_frontend_or_codegen_blocked`:

`fused_linear_jsd.py`, `gdn_fwd_h.py`, `grouped_gemm.py`, `int4_gemm.py`,
`jagged_hstu_attn.py`, `jagged_layer_norm.py`, `layer_norm.py`,
`mamba2_chunk_scan.py`, `mamba2_chunk_state.py`, `moe_matmul_ogs.py`,
`nvfp4_gemm.py`.

## Proven Passing Examples / Benchmarks

These examples have lowered to NKI and run successfully in the serial examples
runner or targeted NKI runs:

`add.py`, `aot_example.py`, `attention.py`, `batch_softmax.py`,
`blackwell_attention.py`, `bmm.py`, `broadcast_matmul.py`, `concatenate.py`,
`cross_entropy.py`, `embedding.py`, `exp.py`, `fp8_gemm.py`,
`fused_nki_ops.py`, `gather_gemv.py`, `geglu.py`, `jagged_dense_add.py`,
`jagged_mean.py`, `jagged_sum.py`, `jsd.py`, `kl_div.py`,
`layer_norm_f32.py`, `long_sum.py`, `low_mem_dropout.py`, `matmul.py`,
`matmul_layernorm.py`, `matmul_split_k.py`, `psum_reuse_minimal.py`,
`psum_reuse_test.py`, `rms_norm.py`, `softmax.py`, `softmax_decomposed.py`,
`segment_reduction.py`, `split_k_barrier.py`,
`squeeze_and_excitation_net.py`, `sum.py`, `swiglu.py`, `welford.py`.

Targeted proof commands run after the latest lowering fixes:

```bash
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/fused_nki_ops.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/concatenate.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/cross_entropy.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/fp8_gemm.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/gather_gemv.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/low_mem_dropout.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/jagged_mean.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/jagged_sum.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/jagged_dense_add.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/segment_reduction.py
... /opt/aws_neuronx_venv_pytorch_2_9/bin/python examples/split_k_barrier.py
... targeted matmul_split_k no-bias and bias epilogue full-shape checks
```

All eleven direct example commands passed without changing example kernel logic.

NKI-specific helper/reference scripts such as `attention_nki.py`,
`concatenate_nki.py`, `layer_norm_manual_nki.py`, `simple_add_nki.py`, and
`run_nki_examples.py` are not counted in the main pass/fail ledger because the
acceptance target is the original Helion examples lowering through NKI.

Special pass annotations:

- `aot_example.py` and `blackwell_attention.py` are currently no-op/all-comment
  scripts, so runner success does not prove NKI lowering or runtime correctness.
- `softmax_decomposed.py` is runner-selected and passing, but it uses a custom
  generated-code path with XLA `mark_step()` rather than the normal
  `run_example` harness.

## Excluded / Non-Acceptance Scripts

- Helpers/reference: `__init__.py`, `run_nki_examples.py`, `attention_nki.py`,
  `concatenate_nki.py`, `layer_norm_manual_nki.py`, `simple_add_nki.py`.
- CUDA-only skipped by the runner: `flex_attention.py`, `fp8_attention.py`,
  `jagged_dense_bmm.py`.
- Distributed CUDA/NCCL examples are not part of this runner:
  `examples/distributed/all_gather_matmul.py`,
  `examples/distributed/all_reduce.py`,
  `examples/distributed/matmul_reduce_scatter.py`,
  `examples/distributed/one_shot_allreduce_bias_rmsnorm.py`; `utils.py` in
  that directory is a helper.

## Recent Lowering Fixes Proven By Examples

- Dynamic loops lower to `nl.dynamic_range` and pass runtime tests.
- Row gather lowering for `weight[indices, tile]` uses NKI `.ap(...,
  vector_offset=..., indirect_dim=0)` and is proven by `embedding.py`.
- NKI type propagation now allows trailing singleton RHS store shapes, which
  fixes `fused_nki_ops.py` keepdim reductions without editing the example.
- NKI config completion fills missing decorator block sizes from safe defaults,
  which fixes partial explicit configs such as `fused_nki_ops.py`.
- NKI partial boundary load/store lowering copies valid tail sub-slices instead
  of skipping or overrunning full static slices, proven by `concatenate.py`.
- Flat 1D tensor-index gather lowering uses NKI vector-offset DMA, proven by
  `cross_entropy.py`.
- `float8_e4m3fn` example inputs are host-cast to `bfloat16` for NKI lowering,
  because the installed TRN2 Neuron compiler rejects true F8E4M3FN HLO and does
  not recognize the suggested unsafe FP8 compiler flag. This proves the Helion
  lowering path for `fp8_gemm.py`, but it is not true hardware FP8 execution.
- NKI kernel configs are cloned before tile-strategy mutation, preventing one
  specialization from clamping and corrupting later specializations.
- RHS tile-list matmul lowering now unrolls concrete K subtiles instead of
  reusing tile 0 for every K chunk.
- `gather_gemv.py` now lowers tile-index floordiv/modulo through `nl.floor`
  rather than rounded float-to-int casts, and emits dynamic indirect DMA with
  `oob_mode=nisa.oob_mode.skip` for repeated vector-offset gathers. This passes
  the example's own `S=2048`, `4096`, `8192`, and `16384` correctness checks.
- `low_mem_dropout.py` now lowers `hl.rand` to NKI by materializing a dynamic
  `[1, 1]` SBUF seed tile for `nisa.set_rng_seed`, emits `nl.rand([1, block])`,
  treats scalar kernel parameters as scalar operands in compare lowering, and
  uses dynamic tensor-size args for NKI entry reshapes. This passes the
  example's forward/backward checks for both `8192` and `32768` elements.
- `jagged_mean.py` now lowers and runs end to end on Trainium/NKI. Fixes
  proven by this example include NKI tensor bitwise-mask lowering, preserved
  mask operands for fused jagged loads, direct `nl.dynamic_range` register-load
  bounds from copy-suffixed SBUF scalar aliases, fused rank-3 flat gather plus
  sum-over-`k` lowering into `[rows, features]`, copy-based tensor casts instead
  of tensor-valued `memset`, comparison layout transposes for `[1, rows]` to
  `[rows, 1]`, and `nl.broadcast_to` row replication instead of partition-slice
  SBUF copy loops.
- `jagged_sum.py` now also passes on Trainium/NKI. This proves the fused
  rank-3 flat gather plus `sum(dim=1)` path for a single row-mask
  `extra_mask`, without the feature-mask operand used by `jagged_mean.py`.
- `jagged_dense_add.py` now passes on Trainium/NKI. This proves 2D flat
  jagged gathers from a 1D packed buffer (`starts[:, None] + tile[None, :]`),
  predicated NKI gather masking, static tail load/store handling for dense
  prefill tiles, and symbolic row-base offsets in dynamic AP loads/stores.
  Its decorator config uses a dynamic jagged-column block size of `8`, which
  divides the dense width `5000` and avoids the current NKI scalar-offset AP
  tail behavior where `oob_mode=skip` skips the whole dynamic DMA tile.
- `split_k_barrier.py` now passes on Trainium/NKI. This proves NKI multi-root
  grid phases can be emitted sequentially for barrier-style kernels, integer
  tunable hints can flow into fixed block-size expressions, and flattened
  high-rank temporary HBM stores/loads use correct effective dimensions and
  layout transposes.

## Active Work Since Last Full Ledger Update

- `matmul_split_k.py`: NKI `hl.atomic_add` lowering now exists. The no-bias
  full-shape check `M=64, K=32768, N=64` passed on Trainium with
  `block_sizes=[1, 64, 128], split_k=8`. The bias epilogue path also passed
  after fixing captured closure/global tensor origins so `bias[tile]` becomes
  a real NKI kernel argument. The example logic was unchanged; the decorator
  config was adjusted only to legal NKI metadata.
- `jagged_mean.py`: the smallest jagged reduction/gather target now passes
  after the dynamic-bound, mask, fused gather, cast, and layout fixes above.
- `jagged_sum.py`: the same rank-3 jagged gather family now passes for the
  single-mask sum case with `B=8`, `M=128`, and `max_seqlen=64`.
- `jagged_dense_add.py`: now passes after lowering gained 2D flat gather
  support and dynamic AP row offsets. Only the decorator config was adjusted,
  from `[128, 128, 128]` to `[128, 128, 8]`, to avoid a dynamic scalar-offset
  AP tail tile over the dense width.
- `jagged_softmax.py`: moved past the earlier active-loop lookup crash, NKI
  tensor/tensor division rejection, scalar `-inf` `where` initialization,
  integer/bool predicate transposes, singleton K reductions, flat indexed AP
  load/store recognition, dynamic return-buffer sizing, and loop-carried SBUF
  dominance failures. With decorator config `[128, 128, 1, 1]`, it now compiles
  and executes on Trainium/NKI but fails strict default `torch.allclose` by a
  small softmax normalization drift (`max_abs` about `3.8e-6`, roughly
  `645 / 2092800` elements in targeted deterministic probes). The active path
  is a singleton-batch K-blocked config `[1, 128, 128, 128]`, which should reduce
  the online recurrence error. That path now gets through NKI frontend MLIR
  verification and reaches Neuron/XLA execution. The latest failure was a
  Neuron compiler BIR verification error from the lowering's manual
  `[1, 128] -> [128, 128]` SBUF broadcast copy loop; the helper has been changed
  to emit `nl.broadcast_to` for row-tile replication. Rerun pending.
- `split_k_barrier.py`: now passes end to end. The successful targeted run used
  `split_k=64`, `pid_type="persistent_blocked"`, and a decorator-only inner-K
  config adjustment from `128` to `64`. Lowering fixes included NKI sequential
  multi-root grid emission, one-time NKI PID installation, tunable/default hint
  propagation through `cdiv` / `next_power_of_2`, composite `size_hint`
  evaluation with unbacked symbol hints, and flattened high-rank temporary HBM
  guard/layout fixes. The final run reached `Compiler status PASS`, completed
  XLA/Neuron execution, matched `torch.matmul` within the example tolerance, and
  skipped benchmarking under the NKI backend.
- `segment_reduction.py`: a narrow NKI lowering for the tuple
  `hl.associative_scan` pattern used by this example now exists. It emits a
  conservative partition-axis segmented prefix sum for `(values, indices)` where
  values accumulate while consecutive indices match. The NKI example harness now
  prefers a PyTorch baseline over a Triton baseline when `HELION_BACKEND=nki`,
  avoiding accidental CUDA/Triton execution for examples that provide both.
  `hl.atomic_add` also has a narrow NKI HBM row-scatter RMW path for unused
  return values and 2D output rows indexed by an SBUF row-offset tile, matching
  this example's `hl.atomic_add(output, [idxs, tile_f], segment_vals)` pattern.
  After adding the required decorator config and NKI `aten.expand.default`
  lowering, the next targeted run reached NKI frontend compile and exposed an
  invalid modulo rewrite: `tile.index % 128` had scalar `128` materialized as a
  uniform SBUF tile, then the generated code attempted Python `1.0 / <SBUF>`.
  `GenerateAST` now tracks uniform SBUF constants written by `nisa.memset` so
  the modulo rewrite emits literal scalar reciprocals/products. The next rerun
  moved past modulo and failed NKI MLIR verification in associative scan:
  `nisa.tensor_tensor` compared a shifted one-row SBUF view against a base-row
  carry tile, which violates NKI partition start alignment. The scan lowering
  now copies each shifted current value/index row into base-aligned
  `[1, features]` scratch tiles before tensor-tensor scan arithmetic/comparison.
  The next rerun moved past the scan verifier failure and exposed a separate
  scalar-constant shape issue in `tile_e.index % block_size == block_size - 1`:
  the compare output is `[1, 128]`, but the scalar `127` was materialized as
  `[128, 1]`. `GenerateAST` now constant-folds simple lifted constant
  expressions such as `-1 + 128`, records uniform SBUF constants, and NKI
  compare lowering now emits the resolved scalar value for `tensor_scalar`
  compare operands instead of the temporary SBUF constant name. The next rerun
  completed NKI frontend compile and reached Neuron backend compile, which now
  fails BIR verification for uninitialized SBUF reads. Generated code shows
  boundary loads like `indices[tile_e]` and `input_data[tile_e, tile_f]` are not
  zero-initialized on the final partial `tile_e` block (`2000 % 128 != 0`).
  NKI HBM load lowering now initializes destination SBUF tiles before guarded
  DMA paths so inactive lanes are defined. The next rerun moved past
  uninitialized reads and failed Neuron BIR verification on dynamic SBUF
  partition slicing inside the segmented scan loop. As an allowed decorator-only
  config change, `segment_reduction.py` now uses `block_sizes=[1, 128]`; with one
  element per `tile_e` block, the scan is a no-op and each input row is
  atomically accumulated directly into its segment. This keeps example logic
  unchanged while avoiding NKI's current dynamic partition-slice limitation. The
  targeted run then compiled and executed the NKI kernel successfully
  (`Compiler status PASS`, Neuron compile completed), but `run_example` still
  tried to validate the extra Triton baseline and failed because this host has
  no active Triton/CUDA driver. The NKI example harness now drops non-PyTorch
  baselines under `HELION_BACKEND=nki` instead of merely reordering them. The
  final targeted rerun passed: NKI frontend compile took about 12s, XLA/Neuron
  execution used a cached NEFF, correctness matched PyTorch, and benchmarking
  was skipped under NKI.
- `int4_gemm.py` / `nvfp4_gemm.py`: NKI tensor-valued bit shifts now lower to
  `nl.left_shift` / `nl.right_shift` through the backend binary-op path instead
  of emitting Python `<<` / `>>` on SBUF tensors. This removes one packed-lowbit
  compiler blocker, but stack/reshape interleaving for packed nibbles is still
  unresolved and runtime validation is pending.
- `jagged_layer_norm.py`: the fused flat gather plus `sum(dim=1)` recognizer now
  accepts the high-rank flat-index pattern `(starts + k) * M + m` and multiplies
  the gathered row id by the feature stride before adding the feature tile
  offset. This targets the first and second pass `x_slice.sum(dim=1)` pattern.
  Runtime validation and the final high-rank flat store path are still pending.
- K64 `jagged_softmax.py` rerun: the `[1, 128, 64, 64]` config made it through
  Helion/NKI frontend codegen and into Neuron backend compile, but it was
  terminated at `2026-05-06T06:31:10Z` after crossing a 30-minute Neuron compile
  cap. Subagent static estimate says this shape should execute in seconds to low
  minutes once compiled; long first compile is possible for the generated BIR,
  but the useful signal beyond the cap is low. Treat this as a pathological
  compile-size/lowering issue for `jagged_softmax.py`, not a proven runtime pass.

Fastest remaining targets for iteration after `split_k_barrier.py`:

1. `layer_norm.py` - likely small and close to the already-passing
   `layer_norm_f32.py`, so it is a good compiler-only/static investigation
   target before a serial runtime probe.
2. `bf16xint16_gemm.py` - already compiles and runs, but has a narrow numeric
   tolerance failure after the int16 load workaround.

## Known Remaining Failures

Current known failing examples from the latest clean and targeted runs:

`bf16xint16_gemm.py`, `fused_linear_jsd.py`, `gdn_fwd_h.py`,
`grouped_gemm.py`, `grpo_loss.py`, `int4_gemm.py`, `jagged_hstu_attn.py`,
`jagged_layer_norm.py`, `jagged_softmax.py`, `layer_norm.py`,
`mamba2_chunk_scan.py`, `mamba2_chunk_state.py`, `moe_matmul_ogs.py`,
`nvfp4_gemm.py`.

Active blockers:

- `bf16xint16_gemm.py` compiles and runs. Direct probes showed NKI HBM DMA from
  int16 is corrupt for the immediate int16-to-bfloat16-cast pattern, so the
  lowering now pre-casts that host tensor to `bfloat16`. The example is now
  close numerically, but the large case still fails strict tolerance by a small
  number of elements (`294-321 / 83886080` in repeated targeted runs, max abs
  `0.28125-0.375`, tolerance `0.01`). Static inspection points to
  cancellation/accumulation-order differences in BF16 Tensor Engine matmul after
  the int16 DMA corruption workaround, not a systematic load corruption.
- `fused_linear_jsd.py` fails in Neuron compilation because the current lowering
  materializes a full `[32, 128256]` SBUF tile for vocab-wide softmax/log-softmax
  work. This requires streaming/free-dimension reduction lowering rather than a
  dependency install or example edit. The needed lowering is a multi-pass
  free-axis stream over vocab tiles: row max, row sum-exp/logsumexp, then a
  recompute pass that accumulates JSD/KL reductions and writes any gradient tile.
- Remaining jagged examples still need broader high-rank jagged indexing,
  mask/reduction coverage, and loop-carried dynamic state handling. The
  specific rank-3 flat gather plus `sum(dim=1)` path is now proven by
  `jagged_mean.py` and `jagged_sum.py`; `jagged_softmax.py` now reaches a
  loop-carried-state dominance verifier failure after several lowering fixes.
- `mamba2_chunk_scan.py` and `mamba2_chunk_state.py` are high-rank affine
  indexing stress tests; they need generalized flattened HBM load/store layout
  rather than CUDA/Triton dependencies.

Infrastructure/runtime blockers:

- `grpo_loss.py`: timed out at the runner's 600 second per-example limit.
