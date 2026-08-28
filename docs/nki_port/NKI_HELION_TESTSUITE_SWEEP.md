# Helion native test-suite sweep against the NKI backend

**Goal:** run Helion's own test suite (the general functionality tests, not the
CUDA/Triton/CuTe/Pallas/TileIR-specific ones) against `HELION_BACKEND=nki`, at
the upstream test sizes, to gauge prod-readiness for upstreaming.

**Branch:** `nki-port-v2`. **Date:** 2026-06-10.

## How it was run

- 53 general-functionality test files (excluded: other-backend files cute*/
  pallas/tileir/metal/amd, distributed, triton-internals like inline_triton/
  tensor_descriptor/ptxas, and autotuner-MACHINERY files — see below).
- Test FILES ARE UNMODIFIED (upstream sizes). To make the Triton-gated test
  bodies execute against NKI, `test/conftest.py` gained an env-gated hook
  (`HELION_NKI_TEST_SWEEP=1`) that makes `onlyBackends` admit `nki` wherever it
  admits `triton`. Inert unless the flag is set.
- `HELION_BACKEND=nki HELION_NKI_SIMULATE=1` (CPU simulator — runs without
  Trainium contention and validates correctness against the PyTorch reference).
- `HELION_AUTOTUNE_EFFORT=none`: config-less kernels use a default config
  instead of a full LFBOTreeSearch (which, under sim with no timeout, runs
  effectively unbounded — a single test could dominate the whole sweep). The
  prod signal we want is "does the kernel compile+run correctly", not autotuner
  benchmarking. The 10 autotuner-machinery files (test_autotuner*,
  test_llm_autotuner, test_pretuned_kernels, test_external_autotune,
  test_aot_autotuning, test_matmul_heuristics, test_benchmarking,
  test_benchmark_worker, test_compile_time) are EXCLUDED — meaningless under
  effort=none. They need a separate hardware autotune run.
- Per-file subprocess, serial (parallel Neuron procs cause NRT failures), with
  a 600s per-test hang-guard (`--timeout=600 --timeout-method=signal`). This is
  NOT a short timeout — real NKI compiles finish well inside it; it only bounds
  genuine HANGS. Some hangs are in non-interruptible native code (signal can't
  preempt them) — those files were manually killed and recorded `rc=-9`.
- Harness: `/home/ubuntu/nki_test_sweep.py`; logs `/tmp/nki_test_sweep_logs/`.

## Headline result

**642 passed / 526 failed / 217 skipped** across 53 files (4937s). **16 files
fully green.** This is NOT prod-ready as-is — NKI passes a solid functional core
but has substantial op-coverage gaps, several numeric mismatches, a few real
backend bugs, and some kernels that HANG the NKI compiler.

**16 ALL-GREEN files:** test_autodiff (all skip — bwd gated), backend_registry,
barrier, codegen_comments, codegen_dict, debug_utils, dot_requirements,
dot_scaled, logging, loop_dependencies, memory_op_facts, print_ref_eager_mode,
quantized_ops, ref_eager, tensor_numel_constraints.

## Failure categories (triaged from --tb=line samples)

The 526 failures fall into these buckets (NOT all are NKI defects — some are
Triton-specific test assertions that can never pass on any non-Triton backend):

1. **Missing NKI op codegen (`BackendImplementationMissing`)** — real coverage
   gaps. Seen: `_reduce` (test_reduce: 14 fails), `device_print` (test_print),
   `_gelu_tanh_approx` / `_gelu_erf` (test_misc). These ops have no NKI codegen
   handler yet.
2. **Unsupported feature/dtype (`BackendUnsupported`)** — explicit, intentional
   gaps: `associative_scan` only supports fwd tuple scans over dim=0
   (test_associative_scan: ~70 fails, test_misc), dtype `uint64`
   (test_stack_tensor), dtype `float64` (test_associative_scan). These are
   honest "not implemented" rejections, not crashes.
3. **Real NKIBackend bugs** — `NKIBackend.scalar_load_expr() takes 2 positional
   args but 3 were given` (test_rng: 14 fails — the RNG seed path passes an
   extra arg the NKI override doesn't accept). This is a genuine signature bug
   worth fixing.
4. **Numeric mismatch (`Tensor-likes are not close`)** — kernel compiles+runs
   but output differs from PyTorch ref: test_control_flow (6), test_grid (2+),
   test_masking, test_broadcasting. Need per-case investigation (precision vs
   logic).
5. **Edge-case asserts** — zero-size tensors (`shape must be positive integers,
   got (0,32)` — test_zero_size, all 3), out-of-bound masking
   (`index range [96,127] exceed dim 100` — test_masking; NKI pads to 128 and
   the sim bounds-checks the unmasked tail).
6. **Compiler HANGS** — some kernels wedge the NKI compiler in
   non-interruptible native code (test_loops, test_unroll_tuples killed at
   rc=-9; ~19 timeouts inside test_examples, 7 inside test_indexing). These are
   the most serious — they don't error, they hang.
7. **Test-harness Triton-isms (NOT NKI bugs)** — assertions that grep the
   generated code for Triton-specific strings: `'virtual_pid' not found`
   (test_persistent_kernels — it's NKI code, has no virtual_pid),
   `triton_interpreter not defined` (test_print), and CUDA-only cases
   (`Found no NVIDIA driver`). These fail by construction on any non-Triton
   backend and should be re-gated, not "fixed".

## Per-file tally

ALL-GREEN (16): autodiff backend_registry barrier breakpoint codegen_comments
codegen_dict debug_utils dot_requirements dot_scaled logging loop_dependencies
memory_op_facts print_ref_eager_mode quantized_ops ref_eager
tensor_numel_constraints

WITH FAILURES (37):
associative_scan 7/38, atomic_ops 6/18, best_available 43/2, broadcasting 9/2,
cache 16/12, closures 3/3, config_api 39/1, constexpr 7/1, control_flow 9/5,
custom_op 3/4, dot 83/8, errors 32/3, examples 48/47, generate_ast 18/6,
graph_module 2/3, grid 1/10, indexing 13/58, int64_indexing 1/7, jagged_tile
4/7, loops 0/0(HANG rc=-9), masking 0/9, matmul 9/4, misc 23/18,
persistent_kernels 8/22, print 0/12, random 2/34, reduce 0/14, reductions 16/12,
register_tunable 4/1, rng 4/23, specialize 11/8, stack_tensor 0/7, torch_compile
125/114, type_propagation 10/2, unroll_tuples 0/0(HANG rc=-9), views 8/8,
zero_size 0/3   (format: passed/failed)

## Prod-readiness assessment

**Not ready to merge to upstream yet.** What this sweep establishes:
- A real functional core works on NKI (642 tests pass, 16 files fully green —
  basic loads/stores, reductions-via-sum, dot/matmul core, type propagation,
  config, registry).
- But there are: (a) genuine op-coverage gaps to implement (`_reduce`,
  `device_print`, gelu variants, broader `associative_scan`); (b) at least one
  real backend signature bug (`scalar_load_expr` in the RNG path); (c) numeric
  mismatches needing investigation; (d) **compiler hangs** on some kernels
  (loops, unroll_tuples, several examples) — the highest-priority issue since
  they don't fail gracefully; (e) zero-size / edge-case handling.
- A chunk of the "failures" are Triton-specific test assertions that should be
  re-gated rather than fixed — a real upstreaming task is deciding, per test,
  whether NKI should be admitted and adding `.expected_nki` goldens or
  NKI-appropriate assertions.

**Recommended next steps (priority order):** (1) the compiler hangs — they're
the worst failure mode; (2) the `scalar_load_expr` RNG signature bug; (3) the
missing-codegen ops with broad impact (`_reduce`); (4) the numeric mismatches;
(5) re-gate the Triton-specific tests + add NKI goldens. Items 1-4 are NKI
backend work; item 5 is test-suite hygiene for the PR.

NOTE: this was a SIM sweep. Hardware behavior can differ (the hangs especially
may manifest differently under neuronx-cc). The autotuner-machinery files and
the bwd/autodiff paths were not exercised here.
