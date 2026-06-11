# NKI backend — Helion test-suite failure analysis (in depth)

Companion to `NKI_HELION_TESTSUITE_SWEEP.md`. That file is the executive summary;
this file explains **every error category**, **which files it appears in**, **how
many times**, and **the exact mechanism that produces it**, with a per-file
appendix at the end.

## Provenance

- **What ran:** Helion's own unit-test suite (`test/test_*.py`), 53 general
  functionality files, at upstream test sizes. These are NOT the `examples/`
  kernels — each test defines its own small `@helion.kernel` to probe one
  feature. (`examples/` were covered by the separate hardware sweep, 49/51.)
- **How:** `HELION_BACKEND=nki`, CPU simulator (`HELION_NKI_SIMULATE=1`),
  `HELION_AUTOTUNE_EFFORT=none`, serial, 600s/test hang-guard. Triton-gated test
  classes were admitted to NKI via an env-gated `conftest.py` hook
  (`HELION_NKI_TEST_SWEEP=1`). Harness: `/home/ubuntu/nki_test_sweep.py`; raw
  logs: `/tmp/nki_test_sweep_logs/`.
- **Headline:** **642 passed / 526 failed / 217 skipped**; 16 files fully green.
- **Counting note:** the authoritative per-file failure counts are the pytest
  `N failed` summary lines (quoted below). The category counts come from parsing
  every failure traceback; where one root cause spans many parametrized cases
  (e.g. torch_compile), the category count and the summary count differ and both
  are given.

---

## How a backend op gets "missing" or "unsupported" (mechanism shared by 2 categories)

Helion lowers each device operation through a per-backend registry. For an op
like `_reduce`, codegen calls `codegen_fn = lookup(op, backend.name)`; if no NKI
entry exists it raises `BackendImplementationMissing` (inductor_lowering.py:1019).
A backend can also explicitly reject something it sees but can't handle by
raising `BackendUnsupported` from inside its codegen (e.g. an unmapped dtype or
reduction op). So:
- **`BackendImplementationMissing`** = "no NKI handler registered for this op."
- **`BackendUnsupported`** = "NKI saw it but deliberately refuses this
  variant/dtype." Both are *graceful* (clean exceptions), not crashes.

---

# Error categories

## 1. Missing NKI op codegen — `BackendImplementationMissing` (≈40 failures)

The op has no NKI codegen handler. Most never existed for NKI in the reference
fork either, so these are unimplemented features, not port regressions.

| Op | Count | Files | What it is |
|----|-------|-------|------------|
| `_reduce` | 14 | **test_reduce** | `hl.reduce` — the generic custom-combine reduction (fold with an arbitrary binary fn). No NKI lowering exists. |
| `device_print` | 8 | **test_print** | on-device `hl.device_print`. NKI has no print-from-kernel codegen. |
| atomics (`atomic_max/min/and/or/xor/cas/xchg`) | 9 | **test_atomic_ops** | atomic RMW ops; only a subset are wired for NKI. |
| `_gelu_erf` | 3 | **test_examples** (2), **test_misc** (1) | exact-erf GELU activation; not mapped to an NKI activation. |
| `split` | 3 | **test_views** | `torch.split` lowering. |
| `_gelu_tanh_approx` | 1 | **test_misc** | tanh-approx GELU. |
| `join` | 1 | **test_views** | tensor join/concat-style op. |
| `inline_triton` | 1 | **test_torch_compile** | inline-Triton escape hatch — inherently Triton-only; should be skipped for NKI. |

**How we got it:** the test builds a kernel using one of these ops; lowering
reaches the op, finds no `(op, "nki")` codegen entry, raises
`BackendImplementationMissing`. Fix = implement the handler (or, for
`inline_triton`, skip on NKI).

## 2. Explicitly unsupported features/dtypes — `BackendUnsupported` (≈39 failures)

NKI sees the construct and refuses it on purpose.

| Message | Count | Files |
|---------|-------|-------|
| `associative_scan only supports forward tuple scans over dim=0` | 21 | **test_associative_scan** (20), **test_misc** (1) |
| `dtype torch.uint64` | 7 | **test_stack_tensor** |
| `reduction 'argmax' not mapped to NKI op` | 5 | **test_reductions** |
| `activation op 'cos' is not mapped for NKI codegen` | 4 | **test_reductions** (1), **test_generate_ast** (1), **test_graph_module** (2) |
| `reduction 'argmin' not mapped to NKI op` | 1 | **test_reductions** |
| `expand from [4,16] to [16,16] is not broadcastable` | 1 | **test_indexing** |
| `dtype torch.float64` | (within associative_scan) | **test_associative_scan** |

**How we got it:** NKI's scan codegen only implements the forward-dim-0 tuple
form; its reduction map lacks argmax/argmin; its activation map lacks `cos`; and
it rejects 64-bit dtypes. Each raises `BackendUnsupported` with the message
above. These are honest capability gaps; fixing = extending the respective maps.

## 3. The `scalar_load_expr` TypeError — **FIXED this session** (was 20 failures in test_rng)

| Message | Count | File |
|---------|-------|------|
| `NKIBackend.scalar_load_expr() takes 2 positional arguments but 3 were given` | 14 shown / 20 total | **test_rng** |

**Mechanism:** base `Backend.scalar_load_expr(self, tensor_name, index_expr=None)`
takes an optional index; `NKIBackend` overrode it as `(self, tensor_name)`,
dropping `index_expr`. The RNG-seed path (`rng_utils.py:153`) calls it with two
args → Python raises `TypeError` (a hard crash) before the method body runs.
**Fix:** restored the base signature so the call reaches the body and raises the
*intended* `BackendUnsupported("scalar tensor loads ... use dma_copy path")`.
**Confidence: high** — verified the TypeError is gone (becomes BackendUnsupported)
and add/matmul/gdn_fwd_h codegen is byte-identical (error-path-only change). NOTE:
this turns a crash into a graceful rejection; it does **not** make RNG work on
NKI — `test_rng` still fails (now as BackendUnsupported), because scalar tensor
loads genuinely aren't implemented. Committed in `e4382d64`.

## 4. Numeric mismatches — `Tensor-likes are not close` (34 failures)

Kernel compiles and runs, but output differs from the PyTorch reference. These
are the highest-value *correctness* bugs (logic or precision).

| File | Count |
|------|-------|
| **test_grid** | 8 |
| **test_indexing** | 5 |
| **test_atomic_ops** | 5 |
| **test_examples** | 4 |
| **test_generate_ast** | 3 |
| **test_control_flow** | 3 |
| **test_jagged_tile** | 2 |
| test_views, test_masking, test_cache, test_broadcasting | 1 each |

**How we got it:** `torch.testing.assert_close(result, reference)` failed inside
the test. Each needs individual triage — could be a real codegen logic error or a
bf16/fp32 accumulation-precision difference. Not yet root-caused per-case.

## 5. DMA element-count mismatch — `dma_copy requires src and dst to have the same number of elements` (58 failures — the single most common failure signature)

This is an NKI **codegen shape bug**, surfaced by the simulator's dma_copy
validator. Examples: `got src=4096, dst=64`, `src=64, dst=8`, `src=16000,
dst=16384`. The generated `nisa.dma_copy` has a source slice whose element count
doesn't match the destination SBUF tile — i.e. our load/store codegen computed
mismatched shapes for that indexing pattern.

| File | count |
|------|-------|
| **test_indexing** | 28 |
| **test_specialize** | 8 |
| **test_examples** | 8 |
| **test_views** | 2 |
| **test_register_tunable** | 2 |
| **test_random** | 2 |
| **test_matmul** | 2 |
| **test_int64_indexing** | 2 |
| **test_broadcasting** | 2 |
| **test_atomic_ops** | 2 |

(58 total — note this is by far the largest single root-cause family in the
whole sweep, almost all in indexing-heavy tests. Fixing the indexing→DMA-shape
codegen would clear the most failures of any single fix.)

**How we got it:** a tile shape / slice-width the load/store codegen produced
doesn't match between HBM source and SBUF dest. Same family as the gdn_fwd_h /
jagged_hstu_attn bugs fixed earlier — but on indexing patterns those fixes didn't
cover. Real NKI codegen bugs.

## 6. Out-of-bound simulator access — `Out-of-bound access ... index range [...] exceed dimension size of N` (28 failures)

| File | Count | Example |
|------|-------|---------|
| **test_indexing** | 10 | `dim1 range [192,207] > size 200` |
| **test_masking** | 8 | `dim0 range [96,127] > size 100` |
| **test_examples** | 3 | `dim0 range [0,63] > size 32` |
| test_graph_module, test_grid, test_int64_indexing, test_misc, test_specialize | 1–2 each |

**Mechanism:** NKI tiles to a fixed partition width (e.g. 128) and the masking is
supposed to keep the over-hang lanes inert. Here the generated DMA reads the
**unmasked tail** of a non-power-of-2 dimension (e.g. reading [96,127] of a
size-100 dim), and the simulator's bounds-checker rejects it. This is the
masking/padding interaction — NKI codegen isn't clamping or masking the tail
slice for these shapes. Real codegen bug, subtler than the dma element-count one.

## 7. Compiler/simulator HANGS (8 timeout events; 2 whole files killed)

**This is the NKI *simulator*, not the Helion compiler and not the on-device NKI
compiler.** Confirmed from the captured traceback (test_examples / bf16xint16):
the hang is in
`nki/backends/simulator/tensor_view.py:_compute_indices_vectorized`, called from
`nki.simulator.run_kernel` while executing a `nisa.dma_copy`.

- Why a "compiler" hang under CPU sim: `HELION_NKI_SIMULATE=1` runs the generated
  kernel through the **`nki.simulate` interpreter**, which executes each `nisa`
  instruction in Python/NumPy. The interpreter spins computing the DMA
  access-pattern indices for certain patterns our codegen emits (likely a huge or
  pathological strided pattern). So: our codegen emits a legal-but-nasty access
  pattern → the simulator's index expansion blows up.

| File | Outcome |
|------|---------|
| **test_examples** | 4 timeout banners → 2 marked `Failed: Timeout` |
| **test_indexing** | 1 timeout |
| **test_generate_ast** | 1 timeout |
| **test_loops** | whole file wedged in non-interruptible native code → killed (`rc=-9`); 54 tests' results lost |
| **test_unroll_tuples** | same — killed (`rc=-9`); 29 tests' results lost |

The 2 killed files are the worst: the spin is in native NumPy that `--timeout`
(even signal-method) can't interrupt, so only a process kill stops it. Highest
priority because they don't fail gracefully.

## 8. Zero-size tensors — `shape must be a tuple of positive integers, got (0,32)` (5 failures)

| File | Count |
|------|-------|
| **test_zero_size** | 3 |
| **test_indexing** | 1 |
| **test_torch_compile** | 1 |

**Mechanism:** the test creates a tensor with a 0-length dim; NKI's `nl.ndarray`
rejects non-positive shapes. Helion would need to special-case zero-extent
tiles (no-op the kernel). Likely a moderate fix.

## 9. torch.compile graph break — `torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped` (108 of test_torch_compile's 114 failures)

**One root cause × 108 parametrized cases.** `torch.compile` tries to trace
through the Helion kernel call but hits a graph break at
`helion/runtime/kernel.py:418` (the NKI launcher path calls into code Dynamo
"marks as skipped" / can't trace). So nearly all of test_torch_compile's 114
failures are this single integration gap (NKI launcher isn't Dynamo-traceable),
not 114 distinct bugs.

## 10. Test-harness Triton-isms (NOT NKI defects) — re-gate, don't "fix"

These assert on Triton-specific generated-code strings or need CUDA. They fail by
construction on any non-Triton backend.

| Signature | Count | Files |
|-----------|-------|-------|
| `Found no NVIDIA driver` (CUDA-only test) | 19 | test_rng(9), test_misc(3), test_examples(2), test_type_propagation(2), test_config_api(1), test_indexing(1), test_print(1) |
| `'tl.xxx' not found in <generated code>` (greps for Triton ops) | ~9 | test_errors(`tl.load`), test_int64_indexing(`tl.int64` x5), test_matmul(`tl.atomic_add`), test_constexpr(`_BLOCK_SIZE_0`) |
| `'virtual_pid'/'pid_shared'/'(_NUM_SM,)' not found` (persistent-kernel Triton codegen strings) | 12 | **test_persistent_kernels** (its entire failure set) |
| `name 'triton_interpreter' is not defined` | 3 | test_print |
| `Philox round constants not found` | 2 | test_random |

**How we got it:** these tests `assertIn("tl.something", generated_code)` or
import Triton-only helpers. The real upstreaming task is per-test triage: decide
whether NKI should be admitted and add `.expected_nki` goldens / NKI-appropriate
assertions, or skip on NKI.

## 11. Other / mixed (real bugs worth noting)

- `Tensor engine transpose requires shape <= [128,128], got [1,256]` — **test_generate_ast** (1). Same >128-transpose class as the gather-index fix; another site not yet tiled.
- `Unknown reduction operator: <function max>` — **test_jagged_tile** (4), **test_examples** (1). A reduction-lowering gap for `max` via a particular path.
- `nc_matmul stationary dtype int8 not supported` — **test_dot** (4). int8 matmul operands unsupported on the NKI tensor engine path.
- `name '_nki_mm_result'/'store'/'create' is not defined` — **test_dot** (4), **test_examples** (10), **test_indexing**, **test_specialize**. NameErrors in generated code → a codegen path emits a variable/symbol it never defined (real codegen bugs).
- `Cannot convert symbols to int` (TypeError) — **test_random** (7), **test_examples**, **test_control_flow**, **test_misc**. A symbolic value reached a path that needs a concrete int.
- `tensor_copy dst partition dimension 1024 exceeds maximum 128` — **test_random** (3). A >128 partition not tiled.
- `0 active drivers` (RuntimeError) — **test_random** (3). The transient Neuron-runtime hiccup seen before (self-recovers in the autotuner, but here surfaces as a failure).
- `a leaf Variable that requires grad ... in-place operation` (RuntimeError) — **test_examples** (5+). The nki.simulate `copy_`-into-leaf-grad harness limitation (orthogonal to codegen).
- `argument of type 'DynamicAP' is not iterable` — **test_jagged_tile** (1). A codegen path mishandles the DynamicAP sentinel.

---

# Per-file appendix (authoritative summary counts + dominant cause)

Format: `file: passed/failed — dominant failure cause(s)`

- **ALL-GREEN (16):** test_autodiff (all-skip, bwd gated), test_backend_registry, test_barrier, test_breakpoint, test_codegen_comments, test_codegen_dict, test_debug_utils, test_dot_requirements, test_dot_scaled, test_logging, test_loop_dependencies, test_memory_op_facts, test_print_ref_eager_mode, test_quantized_ops, test_ref_eager, test_tensor_numel_constraints
- **test_associative_scan**: 7/38 — BackendUnsupported (fwd-dim0-only scan) ×many + float64.
- **test_atomic_ops**: 6/18 — missing atomic codegen (9) + numeric mismatch (5) + dma/OOB.
- **test_best_available**: 43/2 — "unexpectedly None" (backend-availability assertions).
- **test_broadcasting**: 9/2 — dma element-count mismatch (1), numeric (1).
- **test_cache**: 16/12 — Triton/env cache-key assertions ("unexpectedly None", `TRITON_CACHE_DIR`).
- **test_closures**: 3/3 — "no origin found for sNN" (symbol-origin gap in codegen).
- **test_config_api**: 39/1 — CUDA-only.
- **test_constexpr**: 7/1 — Triton-string assert (`_BLOCK_SIZE_0`).
- **test_control_flow**: 9/5 — numeric (3), reshape ValueError (1), symbol→int TypeError (1).
- **test_custom_op**: 3/4 — (custom-op integration; mixed).
- **test_dot**: 83/8 — int8 matmul unsupported (4) + `_nki_mm_result` NameError (4).
- **test_errors**: 32/3 — negative tests expecting Triton behavior (`tl.load` assert, "X not raised").
- **test_examples**: 48/47 — broadest mix: grad-harness RuntimeErrors (9), dma/shape asserts (7), NameErrors (6), broadcast ValueErrors (5), numeric (4), TypeErrors (3), OOB (3), gelu missing (2), + 4 sim-hangs.
- **test_generate_ast**: 18/6 — numeric (3), >128 transpose (1), `cos` unsupported (1), 1 hang.
- **test_graph_module**: 2/3 — `cos` unsupported (2), OOB (1).
- **test_grid**: 1/10 — numeric mismatch (8) dominant + OOB + unpack ValueError.
- **test_indexing**: 13/58 — largest: dma element-count (18) + OOB (10) + numeric (5) + 1 hang + misc.
- **test_int64_indexing**: 1/7 — `tl.int64` Triton-string asserts (5) + dma + OOB.
- **test_jagged_tile**: 4/7 — `Unknown reduction max` (4) + numeric (2) + DynamicAP TypeError.
- **test_loops**: 0/0 (**HANG, killed rc=-9**) — wedged sim; 54 tests unresolved.
- **test_masking**: 0/9 — OOB tail-access (8) + numeric (1). Masking/padding bug.
- **test_matmul**: 9/4 — dma mismatch (1), `tl.atomic_add` assert (1), bf16-on-CUDA RuntimeError (1).
- **test_misc**: 23/18 — CUDA (3), symbol→int + empty_like TypeErrors (2), gelu missing (2), scan unsupported (1), OOB (1).
- **test_persistent_kernels**: 8/22 — ALL Triton-string asserts (`virtual_pid`, `pid_shared`, `_NUM_SM`). Re-gate.
- **test_print**: 0/12 — `device_print` missing codegen (8) + `triton_interpreter` NameError (3).
- **test_random**: 2/34 — symbol→int TypeError (7), >128 partition (3), Philox/determinism asserts, `0 active drivers` (3).
- **test_reduce**: 0/14 — ALL `_reduce` missing codegen. Single root cause.
- **test_reductions**: 16/12 — argmax/argmin unsupported (6), `cos` (1), broadcast ValueError.
- **test_register_tunable**: 4/1 — dma element-count mismatch (1).
- **test_rng**: 4/23 — `scalar_load_expr` TypeError (14→now BackendUnsupported, FIXED) + CUDA (5).
- **test_specialize**: 8/8 — dma element-count (4) + OOB (2) + `create` NameError.
- **test_stack_tensor**: 0/7 — ALL `dtype uint64` unsupported. Single root cause.
- **test_torch_compile**: 125/114 — ~108 are the single Dynamo graph-break (NKI launcher not traceable) + zero-size + inline_triton.
- **test_type_propagation**: 10/2 — CUDA-only.
- **test_unroll_tuples**: 0/0 (**HANG, killed rc=-9**) — wedged sim; 29 tests unresolved.
- **test_views**: 8/8 — `split`/`join` missing codegen (4) + broadcast/dma + numeric.
- **test_zero_size**: 0/3 — ALL zero-extent shape rejection. Single root cause.

---

# Triage priorities (what's a real NKI defect vs noise)

**Real NKI backend defects (fix these):**
1. **Simulator hangs** (test_loops, test_unroll_tuples, 4 in examples/indexing/generate_ast) — worst failure mode; our codegen emits an access pattern the sim can't expand.
2. **DMA element-count mismatches** (~20, mostly test_indexing/specialize) — codegen shape bugs, same family as the gdn/hstu fixes.
3. **OOB tail access / masking** (28, test_masking/indexing) — non-power-of-2 tail not masked/clamped.
4. **Numeric mismatches** (34) — correctness, need per-case triage.
5. **NameErrors in generated code** (~15: `store`, `create`, `_nki_mm_result`) — codegen emits undefined symbols.
6. **Missing-op codegen** (40) — implement `_reduce`, atomics, gelu, split/join, device_print as needed.

**Intentional/known gaps (extend if required):** BackendUnsupported set (scan,
argmax/argmin, cos, uint64/float64) — 39.

**Not NKI defects (re-gate / skip / harness):** Triton-string asserts (~26 incl.
all of test_persistent_kernels), CUDA-only tests (19), torch.compile graph-break
(~108), grad-harness RuntimeErrors in examples (~7). Total ≈160 of the 526.

So of 526 failures, roughly **160 are test-harness/integration artifacts**, **~80
are intentional unsupported features**, and the remaining **~286 are genuine NKI
backend work** — dominated by the **58 DMA element-count mismatches** (indexing
codegen), plus OOB/masking (28), numeric mismatches (34), missing-op codegen
(40), NameErrors (~15), and the simulator hangs.
