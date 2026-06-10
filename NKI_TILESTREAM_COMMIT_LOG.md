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
