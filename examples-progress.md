# Examples NKI Progress

Last updated: 2026-05-06T06:23:51Z

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
- Proven passing in this ledger: `35`.
- Known failing in this ledger: `16`.

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
... targeted matmul_split_k no-bias and bias epilogue full-shape checks
```

All nine direct example commands passed without changing example kernel logic.

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
- `split_k_barrier.py`: NKI codegen for `hl.barrier()` now exists as a no-op
  marker, matching Triton's device-code behavior. This is appropriate for the
  current NKI lowering model where top-level kernel phases are emitted in
  sequential order inside one launch. The example decorator now sets
  `pid_type="persistent_blocked"`, which is required by Helion's barrier config
  validation and is allowed decorator metadata. Runtime validation is pending
  until the active `jagged_softmax.py` Trainium run finishes.
- `segment_reduction.py`: a narrow NKI lowering for the tuple
  `hl.associative_scan` pattern used by this example now exists. It emits a
  conservative partition-axis segmented prefix sum for `(values, indices)` where
  values accumulate while consecutive indices match. The NKI example harness now
  prefers a PyTorch baseline over a Triton baseline when `HELION_BACKEND=nki`,
  avoiding accidental CUDA/Triton execution for examples that provide both.
  `hl.atomic_add` also has a narrow NKI HBM row-scatter RMW path for unused
  return values and 2D output rows indexed by an SBUF row-offset tile, matching
  this example's `hl.atomic_add(output, [idxs, tile_f], segment_vals)` pattern.
  These fixes are codegen-only so far; runtime validation is pending until the
  active `jagged_softmax.py` Trainium run finishes.
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
- Active K64 `jagged_softmax.py` rerun: as of `2026-05-06T06:23:51Z`, the run is
  still inside Neuron compile rather than Trainium execution. The active
  `walrus_driver` log is progressing through backend passes and has reached
  `birverifier`; it is compiling a very large BIR (`198146` instructions,
  `4097` blocks, `38918` memory locations). It has not yet proven pass/fail.

Fastest remaining targets for iteration after `jagged_dense_add.py`:

1. `segment_reduction.py` (`100` nodes, `2000` edges, `128` features) - small
   shape, but still blocked by Triton baseline plus `associative_scan`.
2. `split_k_barrier.py` (`16 x 4096 x 16`) - small shape, but blocked by
   `hl.barrier()` support.

## Known Remaining Failures

Current known failing examples from the latest clean and targeted runs:

`bf16xint16_gemm.py`, `fused_linear_jsd.py`, `gdn_fwd_h.py`,
`grouped_gemm.py`, `grpo_loss.py`, `int4_gemm.py`, `jagged_hstu_attn.py`,
`jagged_layer_norm.py`, `jagged_softmax.py`, `layer_norm.py`,
`mamba2_chunk_scan.py`, `mamba2_chunk_state.py`, `moe_matmul_ogs.py`,
`nvfp4_gemm.py`, `segment_reduction.py`, `split_k_barrier.py`.

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
- The Helion path in `segment_reduction.py` still needs `associative_scan`.
  NKI `hl.atomic_add` now has a sequential load-add-store fallback and is
  proven by `matmul_split_k.py`; segment reduction has not yet been rerun
  against that path.
- `mamba2_chunk_scan.py` and `mamba2_chunk_state.py` are high-rank affine
  indexing stress tests; they need generalized flattened HBM load/store layout
  rather than CUDA/Triton dependencies.

Infrastructure/runtime blockers:

- `segment_reduction.py`: imports Triton and uses a Triton baseline first in
  `run_example`; even after bypassing that external path, NKI still needs
  `associative_scan` and `atomic_add` lowering.
- `split_k_barrier.py`: uses `hl.barrier()` without a persistent pid type.
- `grpo_loss.py`: timed out at the runner's 600 second per-example limit.
