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
