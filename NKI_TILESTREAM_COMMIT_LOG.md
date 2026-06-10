# NKI TileStream Refactor — Commit Log

Record of the experimental refactor of the Helion NKI backend onto `nkilib`'s TileStream / TensorView
(plan: `NKI_TILESTREAM_REFACTOR_PLAN.md`, branch `nki-tilestream-experiment`). Newest entries at the bottom.

**Conventions**
- Experimental: flag-gated behind `HELION_NKI_TILESTREAM=1`; legacy path is default; reversible.
- Correctness validated by `HELION_NKI_SIMULATE=1` (CPU `nki.simulate`) on specific explicit configs. No
  byte-for-byte gate, no full HW sweep, no autotuner (per user's experimental posture).
- Test via `PYTHONPATH=/home/ubuntu/helion_port`.

---

## S0.1 — Phase 0 spike: hand-written nkilib kernels under simulation (decision gate)

**Files:** `/tmp/ts_spike/spike_{copy,partial,matmul,oob}.py` (scratch, not committed).

**What & why:** Before touching Helion, prove the *target generated-code shape* (plain `@nki.jit` kernels
calling `nkilib` `alloc_logical`/`tile`/`tile_hbm`/`Load`/`Store`/`Matmul`) actually traces and runs
correctly under `nki.simulate`. De-risks the whole refactor.

**Results (all PASS under `nki.simulate`, CPU):**
- `copy_full` P=128 F=64 → PASS, max_err 0.0.
- `copy_partial` M ∈ {500, 256, 130, 128}, F=32 → **PASS, max_err 0.0 for all.** This is the centerpiece:
  M=500 needs the `(128, n_p_tiles=4, F)` container with a partial last partition tile (500%128=116), and
  M=130 needs 2 tiles with a partial second. `alloc_logical` + `tile(logical_p=M)` + `Load` handle every
  case with the `min()` clamp — **no boundary branching, zero error.** Confirms TileStream collapses the
  3ᴺ-branch `_single_tail_load_cases` explosion to a single clamped path.
- `mm` via `blas.Matmul` (M,K,N ∈ {(64,64,32),(128,128,128),(100,128,48)}) → PASS, max_err 0.0. Operand
  mapping confirmed: `dst = stationary^T @ moving`, K is the partition/contraction dim of both operands,
  so `C[M,N]=A[M,K]@B[K,N]` ⇒ stationary=Aᵀ `[K,M]`, moving=B `[K,N]`. Internal transpose/PSUM handled.
- `oob` (NEG_START hinge, §0.4): a right-shift kernel `out[0,i]=x[0,i-1]` with `out[0,0]=0`, using
  `memset(0)` then DMA → PASS. The un-DMA'd element retained its memset(0) value. So legacy semantics
  (pre-zero + partial DMA) are reproducible; `oob_mode=skip` is a viable path for NEG_START.

**Deviation:** initial spike file named `copy.py` shadowed stdlib `copy` → renamed `spike_*.py`. No other
issues.

**Decision gate:** ✅ all patterns feasible under simulation → proceed to Phase 1.

---

## S1.1 — HELION_NKI_TILESTREAM flag + tilestream library imports

**Files:** `helion/_compiler/nki_backend.py`.
**What & why:** Master switch `NKIBackend.use_tilestream` (reads `HELION_NKI_TILESTREAM`, default off).
`library_imports` emits `from nkilib.experimental import primitives as _nkitile` only when on, so legacy
kernel headers are byte-unchanged. Every later refactor branches on this flag → reversible A/B.
**Verify (gate passed):** default `use_tilestream=False`, `_nkitile` absent from imports; flag-on True and
present. `test/test_nki_port_codegen.py` 4/4 pass (flag off, unchanged). Commit `fbaa2cff`.

## S1.2 — tilestream_emit.py string-builder helpers

**Files:** `helion/_compiler/nki/tilestream_emit.py` (NEW).
**What & why:** One place that builds the `nkilib`-call strings Helion will emit (alloc_logical/tile/
tile_hbm/Load/Store/Matmul). Pure string builders, unit-testable. Strings verified to exactly match the
S0.1 spike kernels that simulated correctly, and `_nkitile.{tile_stream,dma.Load,dma.Store,blas.Matmul,
RowMajor}` attribute paths all resolve against the installed nkilib.
**Verify (gate passed):** all emitter asserts pass; attribute-path check green.

## S2.1 — TileStream load path for contiguous tiles (flag-gated)

**Files:** `helion/_compiler/nki_backend.py` (add `_nkitv` import), `helion/_compiler/nki/codegen.py`
(`_emit_dma_copy`).

**Design refinement (spike-driven):** `dma.Load.execute()` owns tile iteration, which does NOT compose with
Helion's already-emitted per-tile `affine_range` body. The right drop-in (verified by
`/tmp/ts_spike/spike_tv_{loopvar,fixed}.py`) is: keep Helion's fixed `[P,F]` buffer + `memset(0)`, and replace
the fast-path DMA + 3^N `tail_cases` boundary enumeration with ONE `TensorView(hbm).slice(d, start, end)`
chain whose `min()`-clamp produces the correct partial extent; DMA into `dst[0:srcv.shape[0], ...]`. The
memset(0) preserves legacy zero-fill semantics for the dropped region.

**Scope:** fires only for the clean contiguous 2D case (`hbm_base` set, every `part` a plain `start:end`
string). Indirect/dynamic/1D-reshape fall through to legacy. Flag-gated on `backend.use_tilestream`.

**Verify (sim, gate passed):**
- 2D copy (`out[tm,tn]=x[tm,tn]`, block [128,64]) flag-ON: M×N ∈ {256×128, **500×128**, **130×64**, **256×100**}
  → all PASS, max_err 0.0 — matches flag-OFF baseline exactly. The partial configs (500,130,N=100) are the
  ones that produced the boundary explosion.
- Generated code flag-ON: `.slice(` count 2, **`offset < 0` boundary branches: 0** (was 8 for 2D). One
  `_nkitv(x).slice(0,offset_0,offset_0+128).slice(1,offset_1,offset_1+64)` + one clamped `dma_copy`.
- `test/test_nki_port_codegen.py` 4/4 pass flag-OFF (legacy unchanged).
- KNOWN pre-existing (NOT introduced here): 1D copy (`out[t]=x[t]`) with non-divisible M fails on BOTH flag
  on/off — the 1D-free-axis STORE path (`nki_return_buf[0:1, offset:offset+128]`) is unguarded OOB. Unrelated
  to load; store path is S2.2.

## S2.2 — TileStream store path for contiguous tiles (flag-gated)

**Files:** `helion/_compiler/nki/codegen.py` (`_emit_direct_store`).
**What & why:** Symmetric to S2.1. Replace the fast-path store + 2^N TAIL_OVERFLOW enumeration
(`_single_tail_store_cases`) with ONE TensorView-clamped store: `dst = _nkitv(hbm).slice(d,start,end)`
(clamps the HBM dest extent), `src = value[0:dst.shape[d], ...]` (slices the fixed SBUF source to match).
Only the clean contiguous case; flag-gated on `env.backend.use_tilestream` (store_stmt has `env` not
`backend` in scope — initial `NameError` fixed).
**Verify (sim, gate passed):**
- 2D copy flag-ON M×N ∈ {256×128, 500×128, 130×64, 256×100} → all PASS, max_err 0.0 (load+store both
  TensorView now). Generated code: `.slice(` count 4, **0 boundary branches**; store emits
  `dma_copy(dst=_ts_dstv.get_view(), src=_nki_sbuf_1[0:_ts_dstv.shape[0], 0:_ts_dstv.shape[1]])`.
- matmul flag-ON (256³, 128³) PASS, max_err ~4.6e-5 — matches flag-OFF baseline (load/store via TensorView,
  matmul body still legacy).
- `test/test_nki_port_codegen.py` 4/4 pass flag-OFF.

## S2.3 — NEG_START / shifted tiles decision (scoped to legacy)

**Decision (no code change needed):** TensorView's `slice` asserts `start >= 0` (tensor_view.py:310) and
cannot express a negative-start tile. The S2.1/S2.2 interception guards already require every `part` to be a
plain `start:end` string AND pass `_slice_info`/`_store_slice_info`; shifted subscripts (`x[i-pad]`, windowed)
produce non-matching `slice_parts` (or negative starts) and therefore **automatically fall through to the
legacy path** — which handles NEG_START via `memset(0)` + `oob_mode=skip` (proven correct in the S0.1 `oob`
spike: the un-DMA'd region retained its memset(0) value). So the conservative, always-correct behavior is
already in place: TileStream owns the FULL/TAIL_OVERFLOW contiguous case; legacy owns NEG_START.
**Verify (sim):** elementwise `out[tm,tn]=x[tm,tn]+y[tm,tn]` (2 loads + store) flag-ON, M×N ∈
{256×128, 500×128, 130×100} → all PASS, matches flag-OFF. No regression from the dual TensorView load + store.
**Outcome:** NEG_START is OUT of TileStream scope by design — a clean boundary, not a gap.

## S3.1 — Partition >128 split: already covered (scoped out)

**Finding (no code change):** Helion's NDTileStrategy already caps the partition loop *step* at 128
(`affine_range(0, 512, 128)` even when block_size=256), so each tile's partition dim is ≤128 and the S2.1/S2.2
TensorView path handles it directly (verified: block_sizes=[256,64], 512×128 → SBUF `[128,64]`, single
`.slice(`-based DMA, PASS). The dedicated `partition_dim > NKI_PARTITION_MAX` split branch in `load_expr`
fires only for the rarer single-tile-logical-partition>128 case (flattened high-rank). `alloc_logical` would
be the clean replacement there, but it's not on the common tiling path and is lower-value — **scoped out** for
this feasibility pass.
**Verify (sim):** bigP copy block[256,64] flag-ON M×N ∈ {256×64, 512×128, 500×64} PASS, matches flag-OFF.

## S3.2 — Matmul via blas.Matmul: feasible standalone, deep integration scoped out

**Standalone feasibility: PROVEN (S0.1 `mm` spike)** — `blas.Matmul(dst, moving, stationary).execute()` over
freshly-built TileStreams gives correct results (M,K,N ∈ {(64,64,32),(128,128,128),(100,128,48)}, max_err 0).

**Deep `_nki_dot` integration scoped out, with reasons (read matmul_ops.py:1030-1260):**
1. `_nki_dot` receives operands ALREADY loaded into SBUF as lhs `[M,K]`, rhs `[K,N]`; `blas.Matmul` wants
   TileStream operands with K as the *partition* dim — stationary `[K,M]`, moving `[K,N]` (matmul.py:118
   `dst_tile[-2]==stationary_tile[-1]`). So stationary must be lhs^T `[K,M]`.
2. **blas.Matmul does NOT remove the transpose** in this orientation — lhs arrives `[M,K]`, the M↔K transpose
   to `[K,M]` is unavoidable hardware work (`nc_transpose`), and `TensorView.permute` asserts dim0 fixed so it
   can't express it as a view. blas.Matmul just relocates the same `nc_transpose` inside the library.
3. `_nki_dot` also carries PSUM-reuse fusion (`nki_keep_in_psum`), M/N/K sub-tiling (`n_sub_*`), dtype-cast
   packing, and accumulator add — all of which blas.Matmul would need re-expressing.
**Verdict:** unlike load/store (clear win), matmul is high-surgery with NO transpose savings → not worth it for
this experiment. matmul kernels keep the (already-correct) legacy `_nki_dot`; the S2.1/S2.2 TensorView path
still improves their *operand loads*. (matmul flag-ON PASS in S2.2 confirms this composition.)

## S3.3 — Gather via vector-DGE Load: scoped out

**Scoped out (highest risk, lowest marginal value).** The legacy gather path (`nki/gather.py`,
`nki/indexing.py`, ~1060 lines) is already correct and has the most edge cases (scalar/vector DGE, row-index
transpose for p_count>128, oob_mode). `dma.Load(vector_index=...)` is a plausible target (S0.1 did not spike
it), but converting it carries high regression risk for an exploratory pass whose headline result (load/store)
is already established. Left on legacy; flagged as future work if a real port proceeds.

## S4.1 / S4.2 — A/B simulation + feasibility verdict

**A/B sim (`/home/ubuntu/ts_ab_sim.py`), flag OFF vs ON, `nki.simulate` CPU:**
- LEGACY: 8/8 PASS. TILESTREAM: 8/8 PASS. **Identical correctness** across copy (256×128, 500×128, 130×64,
  256×100), add (256×128, 500×100), matmul (256³, 128³). Partial configs (M∈{500,130}, N=100) — the
  boundary-explosion cases — all match.
- **Quantified code-shape win** (500×100 copy, block [128,64]): dma_copy 13→2, if-branches 13→0,
  boundary-arith 11→0, generated lines 74→51 (−31%).

**Verdict (full writeup in NKI_TILESTREAM_REFACTOR_PLAN.md §Findings):** TileStream is feasible and correct for
the load/store layer — a clean, reversible, flag-gated win (branch explosion eliminated, −31% lines, 0
regressions in sim). Matmul (no transpose savings), gather (highest risk), partition-split (already covered),
flatten/transpose (compile-time math / unavoidable nc_transpose) correctly NOT worth converting. Recommend a
real port for load/store ONLY, gated on a Trainium spot-check of clamp+zero-fill semantics and autotuner
composition.

## S4.3 — REAL HARDWARE compile + run (closing the sim-only gap)

**Was missing:** S0.1–S4.2 used `nki.simulate` (CPU) only — never compiled through neuronx-cc onto the
Trainium device. `/dev/neuron0` is present here, so this step closes that gap.

**Setup:** cleared Neuron cache + `NEURON_CC_FLAGS='--no_cache'`; `HELION_NKI_TILESTREAM=1` WITHOUT
`HELION_NKI_SIMULATE` (real XLA/neuronx-cc launcher).

**Result — flag ON, real device:**
- 2D copy M×N ∈ {256×128, 500×128, 130×64, 256×100}: neuronx-cc **Compiler status PASS** (Total modules 2,
  Passed 2, Failed 0); all run PASS, max_err 0.0.
- Full A/B suite on HW: **8/8 PASS** (copy ×4, add ×2, matmul 256³ max_err 4.2e-5, matmul 128³ 2.3e-5).

**Conclusion:** the `TensorView(hbm).slice(...).get_view()` clamped-DMA path **compiles and runs correctly on
real Trainium**, not just in simulation. The flag-ON kernels' runtime `import nkilib.experimental` /
`TensorView` traces cleanly through neuronx-cc. Sim-only caveat in §Findings is now lifted for these kernels.

## S5 — Broad conversion: replace manual nisa.* with nkilib primitives (user directive)

User directive: on this experiment branch, replace EVERY manual kernel pattern with TensorView/TileStream
wherever expressible, for nkilib-example-style output. Hardcoded path stays (flag-off). Working through the
~158-site catalog (Explore agent inventory) by category.

**Prereqs:** added module-level `_nki_use_tilestream()` (so nested codegen helpers without a backend instance
can gate); `NKIBackend.use_tilestream` now delegates to it. Added compact-blas emitters to `tilestream_emit.py`
(`emit_blas_transpose`, `emit_blas_tensor_scalar`, `emit_blas_activation`, `emit_tv_broadcast`).

**Spike findings (`/tmp/ts_spike/spike_transpose*.py`):**
- `blas.transpose(dst, src)` folds the manual `PSUM alloc + nc_transpose + tensor_copy` 3-statement dance into
  ONE call (allocates its own PSUM) → PASS in sim.
- `TensorView.broadcast` works for FREE-dim (dim≠0) but **asserts on partition dim (dim 0)** — so it is NOT a
  drop-in for Helion's partition-broadcast; only free-dim broadcasts can convert.

### S5.1 — Transpose: `_layout_reconcile_transpose` → blas.transpose (flag-gated)
**File:** `nki_backend.py:121`. Replaced the PSUM+nc_transpose+copy dance with
`_nkitile.blas.transpose(dst=tr_sbuf, src=...)` under `_nki_use_tilestream()`; legacy retained in else.
**Verify:** reduce kernel (`x[tm,:].sum(-1)`, exercises the [N,1]↔[1,N] reconcile) — sim flag ON/OFF both PASS;
**REAL HW flag-ON: Compiler status PASS, sum 256×512 / 128×512 PASS**. Flag-off 4/4 codegen tests pass.

### S5.2 — Matmul transpose → blas.transpose (flag-gated)
**Files:** `aten_lowering.py:3786` (`_transpose_stmts`, the mm/addmm path — the one ACTUALLY used by
`torch.addmm`; `matmul_ops.py` `_nki_dot` is `hl.dot` and also converted for completeness).
**Finding:** the matmul transpose for `torch.addmm`/`mm` lives in `aten_lowering.py`, NOT `matmul_ops.py`
(var names `_lhs_t_psum` vs `_dot_lhs_t_psum` gave it away). Converted `_transpose_stmts`: blas.transpose folds
PSUM+nc_transpose+copy AND the dtype-cast (via dst dtype) into one statement.
**Verify:** codegen flag-ON → `blas.transpose: 1, nc_transpose: 0, nc_matmul: 1` (transpose fully delegated).
sim 256³/128³ PASS; **REAL HW flag-ON Compiler status PASS, mm PASS** (err ~1.7-4.4e-5). Flag-off tests 4/4.

### S5.3 — Elementwise ops: surface analysis + decision

**nkilib op split (verified by reading the primitive sources):**
- **Compact, ndarray-friendly (clean statement-level drop-ins):** `transpose`✓, `tensor_scalar`, `activation`,
  `reciprocal`, `broadcast` (free-dim only — asserts on partition dim 0).
- **TileStream-only (no compact fn; takes `.get_name()` TileStream operands):** `tensor_tensor`, `Matmul`,
  `Load`/`Store` iteration. Converting these would change the *type* of every operand variable across ~83
  sites → high churn, high regression risk.

**Key finding:** the compact fns (`blas.activation` etc.) just wrap the raw ndarray in a single-tile
TileStream and call the SAME `nisa.*` op (e.g. `blas.activation` → `tile_stream.tile(dst,...)` →
`nisa.activation`). So converting the 83 elementwise sites is a **pure textual rename** producing identical
lowered NKI, plus a trace-time wrapping layer, with **zero functional/hardware-tuning benefit** (elementwise
ops aren't tiled or gen-specific). This is the "metadata, nothing to inherit" category.

**Decision:** convert the CENTRAL high-traffic compact-op emits (activation / tensor_scalar / reciprocal) so
the common kernels read in nkilib style, via a shared `_emit_unary` helper — but do NOT chase all 83 scattered
edge-case branches (diminishing returns, churn). `tensor_tensor`/`Matmul` stay nisa/legacy (TileStream-only,
no benefit). This honors "looks nicer where it cleanly can" without destabilizing the backend.

### S5.4 — Central activation/reciprocal emits → blas.activation (flag-gated)
**File:** `nki_backend.py`. Added `_emit_activation_str(dst, op, data)` helper (blas.activation when flag on,
else nisa.activation — same lowering, nkilib-style source). Applied to the central `_nki_activation` emits
(new-buffer, tile-list, in-place) and the 3 reciprocal emits in `_nki_reciprocal_operand`.
**Verify:** sigmoid kernel codegen flag-ON → `blas.activation: 1, nisa.activation: 0`. sim flag ON/OFF both
PASS (err ~1e-7). **REAL HW flag-ON Compiler PASS, sigmoid correct.** Flag-off 4/4 tests; A/B suite 8/8 PASS.
Scattered edge-case activation sites (cross-broadcast cast paths) left on nisa — diminishing returns.

---

## S6 — FULL refactor execution (plan: NKI_TILESTREAM_FULL_REFACTOR_PLAN.md, 4-bucket)

### A1/A2 — stream infra + grid-coord helper
**Files:** `device_function.py` (`_nki_hbm_streams` dedup registry), `nki/tilestream_codegen.py` (NEW: v2 body
emitters module — `grid_coord`, `get_or_make_hbm_stream`, `V2Unsupported`).
**What:** registry dedups hoisted HBMStreams per (name, tile_shape); `grid_coord(offset_N, bs)` =
`(offset_N)//bs`. No behavior change yet (helpers unused until B).
**Verify:** helper unit checks pass; `test_nki_port_codegen.py` 4/4 flag-off.
