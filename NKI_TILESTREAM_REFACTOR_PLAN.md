# NKI Backend Refactor onto `nkilib` TileStream / TensorView — Feasibility Plan

> Generated 2026-06-10. **This is an EXPLORATORY feasibility plan, not a production port.**
> The goal is to learn whether re-targeting the Helion NKI backend's *tiling / load / store /
> matmul* codegen onto `nkilib`'s `TileStream` + `TensorView` abstractions is feasible and
> whether the resulting kernels still work — NOT to ship a byte-identical replacement.
>
> Working tree: `/home/ubuntu/helion_port`. Branch: **`nki-tilestream-experiment`** (already checked out,
> forked from `nki-port-v2` @ `07eb8b4a`). Test via `PYTHONPATH=/home/ubuntu/helion_port` (the editable
> install resolves to `helion_nki`; NEVER reinstall — see the editable-install note).
>
> **Posture (per user):** "this is more experimental so no need to be super careful, I just want to see
> if it's feasible and if the kernels work. Don't do big kernel runs and just simulate specific configs."
> So: **no byte-for-byte gate, no full HW sweep, no autotuner sweeps.** Correctness is validated by
> `HELION_NKI_SIMULATE=1` (CPU `nki.simulate`) on a handful of explicit `helion.Config`s per pattern.

---

## 0. Orientation

### 0.1 What exists today (the thing we're refactoring)

The NKI backend is already a working, structured codebase on `nki-port-v2` (Phase 1/2/3 of the upstream
port are done; 38 examples pass on Trainium). The tiling/load/store/matmul codegen we care about lives in:

| Concern | File / symbol | ~Lines |
|---|---|---|
| Tile-loop structure (`for offset_N in nl.affine_range`, `iota` index setup) | `tile_strategy.py` `_codegen_grid_nki` (L3558), `_codegen_device_loop_nki` (L3969) | ~270 + ~200 |
| **Partial-tile boundary DMA explosion** (3ᴺ−1 branches) | `_compiler/nki/codegen.py` `_single_tail_load_cases` (L2993), `_slice_bounds_guard` (L2854), `_emit_dma_copy` (L3068), `_slice_info` | ~330 |
| Load/store entry points (shims → subpackage) | `language/memory_ops.py` `@codegen(load/store,"nki")` (L6680/6691) → `nki/codegen.py` `load_expr` / `store_stmt` | thin |
| Partition >128 split into `(128, n_p_tiles, *F)` | `nki_backend.py` `cast_ast` (L5802), `_NKI_PARTITION_MAX=128` (L6048), `_nki_alloc_sbuf` (L3858) | ~100 |
| Matmul transpose→PSUM→accumulate | `language/matmul_ops.py` `_nki_dot` (L1114), `_nki_copy_psum_to_sbuf` | ~250 |
| Gather / indirect (DGE) | `nki/gather.py`, `nki/indexing.py` | ~1060 |

Generated code today (for `x[tile_m, tile_k]` in a 256×256 matmul tiled at 128) emits one fast-path
`dma_copy` + up to 8 guarded boundary `dma_copy`s, plus a manual `nc_transpose`→PSUM→`tensor_copy` dance.

### 0.2 What `nkilib` gives us (the thing we're refactoring *onto*)

`nkilib.experimental.primitives` (verified importable in this venv):

- **`TensorView`** (`nkilib.core.utils.tensor_view`) — zero-copy view algebra. Every op (`slice`,
  `select`, `permute`, `broadcast`, `reshape_dim`, `flatten_dims`, …) is pure `(shape, strides, offset)`
  arithmetic returning a new view; `get_view()` collapses the whole chain into ONE `ap()` access pattern.
  **Crucially:** `slice` clamps `end = min(end, shape[dim])` (partial tiles for free), but asserts
  `start >= 0` (negative-start / shifted tiles are NOT handled by `slice` — see §0.4).
- **`TileStream` / `HBMStream`** + `tile()`, `tile_hbm()`, `alloc_logical()` (`primitives.tile_stream`) —
  declarative tiling. `alloc_logical((P,*F), pdim_size)` makes the `(pdim, n_p_tiles, *F)` container that
  Helion builds by hand. `tile(buf, tile_shape, iter_order=...)` + `get_tile()` walks the grid; partial
  tiles handled by `min()` clamp in `_get_tile_impl`. `RowMajor`/`ColMajor`/`DimOrder`/`ViewOrder` encode
  loop order.
- **`dma.Load` / `dma.load`** (`primitives.dma.load`) — zips an HBMStream→TileStream, emits `nisa.dma_copy`
  per tile; has built-in scalar/vector DGE (gather) and `dma_transpose`; uses `oob_mode=skip` for OOB.
- **`blas.Matmul`** (`primitives.blas.matmul`) — `dst/moving/stationary` TileStreams; does the
  transpose→PSUM→accumulate internally.

### 0.3 The key architectural fact (decides everything below)

`TensorView`/`TileStream` are **traced runtime objects** that run inside `@nki.jit` (i.e. at *neuronx-cc
trace time*), NOT Helion-compile-time helpers. So this refactor is **NOT** "call TensorView from
`nki/codegen.py`." It is: **make Helion's codegen EMIT TEXT that calls `nkilib`** — e.g. generate

```python
src_ts = tile_stream.tile_hbm(x, tile_shape=(128, 128))
dst_ts = tile_stream.tile(_sbuf, tile_shape=(128, 128))
dma.Load(dst=dst_ts, src=src_ts).execute()
```

into the kernel body instead of the hand-rolled `ndarray`+`memset`+9×`dma_copy`. The view algebra then
resolves during neuronx-cc's trace, one layer below Helion. Consequence: **Helion gains an
`import nkilib...` runtime dependency in generated kernels**, and correctness now depends on neuronx-cc
tracing `nkilib.experimental` as we expect.

### 0.4 The one correctness hinge to watch (NEG_START)

`_single_tail_load_cases` enumerates THREE per-dim states: `FULL`, `NEG_START` (tile starts before 0 —
shifted/padded subscripts like `x[i-pad]`), `TAIL_OVERFLOW` (last tile past the end). TensorView's `slice`
clamps the tail (`TAIL_OVERFLOW` ✓) but **rejects negative start** (`kernel_assert(start >= 0)`).
`nkilib` handles OOB instead via DMA `oob_mode=skip` (seen in `dma/load.py`). Today Helion `memset(0)`s the
tile first, so it depends on dropped regions reading as zero. **Whether `oob_mode=skip` leaves the
un-DMA'd SBUF region at its pre-memset value is the single most important thing to verify** before trusting
any shifted/windowed kernel. We isolate this in Step 5.

### 0.5 Scope — IN and OUT

**IN (the parts TileStream genuinely simplifies — proven over the prior analysis):**
- Partial-tile load/store (`_single_tail_load_cases` & friends) → TileStream `tile`/`Load`.
- Partition >128 split → `alloc_logical`.
- Matmul transpose/PSUM dance → `blas.Matmul`.
- Gather → `dma.Load(vector_index=...)`.

**OUT (TileStream does not help; leave as-is — established earlier):**
- Elementwise op lowering (`NKIOpOverrides`: add/mul/where/exp/…) — TileStream hands you views, you still
  emit every op. ~5000 lines untouched.
- N-D→2D flatten (`_squeeze_shape_2d`) — compile-time list math is simpler and strictly better than
  emitting runtime `flatten_dims`. Do NOT convert.
- Partition↔free transpose reconciliation (`_layout_reconcile_transpose`) — `TensorView.permute` asserts
  dim 0 fixed and can't express a partition transpose; it still compiles to `nc_transpose`. No win.

**OUT (experimental posture):** byte-for-byte gate, full 47/52 HW sweep, autotuner integration, performance
benchmarking. We only check **functional correctness on explicit configs under simulation.**

### 0.6 Method conventions (inherited from the prior plan)

- **Reference for nkilib APIs:** read the installed source under
  `/opt/aws_neuronx_venv_pytorch_2_9/lib/python3.12/site-packages/nkilib/`. Do not guess signatures.
- **Test invocation (no hardware):** codegen is pure Python.
  ```bash
  PYTHONPATH=/home/ubuntu/helion_port HELION_BACKEND=nki HELION_NEURON_TARGET=trn2 python <gen_script>.py
  ```
- **Correctness invocation (CPU sim, no neuronx-cc):**
  ```bash
  PYTHONPATH=/home/ubuntu/helion_port HELION_BACKEND=nki HELION_NEURON_TARGET=trn2 \
      HELION_NKI_SIMULATE=1 python <run_script>.py
  ```
  `default_nki_launcher` (runtime/__init__.py L264) routes to `_nki_simulate_launcher` (L210) when
  `HELION_NKI_SIMULATE=1`. This is the user's "simulate specific configs" path — fast, CPU-only.
- **One logical step = one commit.** End commit messages with
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do not `git push`.
- **Feature-flagged, reversible:** every refactor goes behind `HELION_NKI_TILESTREAM=1` (env, read once into
  an `NKIBackend` property) so the legacy path stays the default and the experiment is A/B-comparable and
  trivially abandonable. This is the single most important design choice — it makes "is it feasible" answerable
  without destabilizing the working backend.

---

## Phase 0 — Spike: prove the generated-code shape works under simulation (NO Helion changes yet)

**Why first:** before touching the backend, confirm the *target generated code* — hand-written `nkilib`
calls — actually traces and runs correctly under `nki.simulate`. If a hand-written `tile()/Load()` copy
kernel doesn't simulate, the whole refactor is moot and we stop here. This de-risks everything for ~an hour.

### S0.1 — Hand-write the three target kernels as plain `@nki.jit` (no Helion)
**File:** `/tmp/ts_spike/{copy,matmul,partial}.py` (scratch, not committed).
**Action:** Write, by hand, three `@nki.jit` kernels using `nkilib` directly:
1. **copy** — `alloc_logical` + `tile_hbm` + `tile` + `dma.Load(...).execute()` + store, full tiles.
2. **matmul** — `blas.Matmul` over tiled `dst/moving/stationary`.
3. **partial** — a copy where the tensor size is NOT a multiple of the tile (e.g. M=500, tile 128) so the
   last partition tile is partial; AND a shifted variant (`x[i-1]`-style negative start) to exercise the
   NEG_START/`oob_mode` question from §0.4.
**Verify (sim):** run each under `nki.simulate` directly (no Helion) and `torch.allclose` vs a numpy/torch
reference. Record: does the partial tile work via `min()` clamp? Does the shifted variant need
`oob_mode=skip`, and does the dropped region read as zero or garbage?
**Gate:** all three simulate and match reference (within NKI tolerance). **If `partial`'s shifted variant
mismatches, document the `oob_mode` semantics precisely — this dictates Step 5's design.**
**Commit:** none (scratch). Record findings in `NKI_TILESTREAM_COMMIT_LOG.md` (created in S1.1).

> Decision point after S0.1: if the spike works → proceed. If `nkilib` can't express something we need
> (e.g. dynamic/jagged loops, a subscript pattern), note it and SCOPE IT OUT — the experiment can still
> succeed for the patterns that do work.

---

## Phase 1 — Scaffolding: feature flag + emit-helpers (Helion changes begin)

### S1.1 — Add the `HELION_NKI_TILESTREAM` feature flag + commit log
**Files:** `helion/_compiler/nki_backend.py` (NKIBackend), `NKI_TILESTREAM_COMMIT_LOG.md` (NEW).
**Action:**
1. Add a cached property `NKIBackend.use_tilestream` → reads `os.environ.get("HELION_NKI_TILESTREAM","0")`
   (truthy = on). Default OFF. This is the master switch every later step branches on.
2. Add `"nkilib.experimental.primitives": "from nkilib.experimental import primitives as _nkitile"` (and the
   `dma`/`blas`/`tile_stream` sub-imports as needed) to `NKIBackend.library_imports` (L5641) — but ONLY emit
   them when `use_tilestream` is on (guard at emit time so legacy kernels keep identical headers).
3. Create `NKI_TILESTREAM_COMMIT_LOG.md` mirroring `NKI_PORT_COMMIT_LOG.md` conventions (step, what/why,
   sim-verify result, deviations). Record the S0.1 spike findings as the first entry.
**Verify:**
```bash
PYTHONPATH=/home/ubuntu/helion_port python -c "from helion._compiler.nki_backend import NKIBackend; b=NKIBackend(); print(b.use_tilestream)"  # False by default
PYTHONPATH=/home/ubuntu/helion_port HELION_NKI_TILESTREAM=1 python -c "from helion._compiler.nki_backend import NKIBackend; print(NKIBackend().use_tilestream)"  # True
```
**Commit:** `experiment(nki): add HELION_NKI_TILESTREAM flag + tilestream library imports`

### S1.2 — Add a `nki/tilestream_emit.py` helper module (the text generators)
**File:** `helion/_compiler/nki/tilestream_emit.py` (NEW).
**Action:** Create one place that builds the `nkilib`-call STRINGS Helion will emit, so the per-pattern steps
stay thin. Functions (each returns statement strings / an emitter that calls `state.add_statement`):
- `emit_alloc_logical(var, logical_shape, pdim_size, dtype, buffer="nl.sbuf") -> str`
- `emit_tile(var, src_var, tile_shape, tile_dims=None, iter_order="RowMajor()") -> str`
- `emit_tile_hbm(var, src_var, tile_shape, tile_dims=None) -> str`
- `emit_load(dst_ts, src_ts, vector_index=None, index_dim=None, transpose=False) -> str`
- `emit_matmul(dst_ts, moving_ts, stationary_ts) -> str`
These are pure string builders (testable without a kernel). Mirror Pallas's "subpackage holds the logic,
shim stays thin" structure (per `NKI_SUBPACKAGE_REFACTOR_GUIDE.md`).
**Verify:** unit-import + a couple of `assert emit_tile("ts","buf",(128,128)) == "ts = _nkitile.tile_stream.tile(buf, tile_shape=(128, 128), iter_order=_nkitile.RowMajor())"`-style string checks.
**Commit:** `experiment(nki): add tilestream_emit string-builder helpers`

---

## Phase 2 — Refactor the load/store path (the biggest, clearest win)

> This is the centerpiece. `_single_tail_load_cases` + `_slice_bounds_guard` + `_emit_dma_copy` (~330 lines
> of 3ᴺ-branch boundary logic in `nki/codegen.py`) collapse into "build a clamped TileStream + one
> `Load().execute()`." Branch on `use_tilestream` so legacy stays default.

### S2.1 — Route the load shim through a TileStream emitter (full-tile case only)
**Files:** `helion/_compiler/nki/codegen.py` (`load_expr`).
**Action:** At the top of `load_expr`, add `if backend.use_tilestream and <pattern is a plain contiguous
tile load>: return _load_expr_tilestream(state, subscript, tensor)`. Implement `_load_expr_tilestream`
using the S1.2 emitters: allocate the SBUF dest, build `tile_hbm` over the HBM source and `tile` over the
dest with the configured block sizes, emit `dma.Load(...).execute()`. **Start with the easy case only**
(contiguous, full or tail-overflow tiles — the `min()` clamp handles tails). Fall through to the legacy path
for anything else (indirect, shifted/NEG_START, dynamic). This keeps the step small and always-correct.
**Verify (sim):** copy kernel (`out[t]=x[t]`) at configs `block_sizes=[128]` (divides) and `[128]` with
M=500 (tail-overflow partial), run under `HELION_NKI_SIMULATE=1`, `allclose` vs torch. Also generate code
with flag OFF and confirm it's unchanged (legacy path intact).
**Commit:** `experiment(nki): TileStream load path for contiguous tiles (flag-gated)`

### S2.2 — Route the store shim through a TileStream emitter
**Files:** `helion/_compiler/nki/codegen.py` (`store_stmt`).
**Action:** Symmetric to S2.1: `dma.Load` reversed direction (SBUF→HBM via `nisa.dma_copy`; `nkilib` store
is `dma.store` / the tile-zip). Full/tail-overflow tiles only; legacy fallback otherwise.
**Verify (sim):** same copy kernel, full round-trip; partial M=500.
**Commit:** `experiment(nki): TileStream store path for contiguous tiles (flag-gated)`

### S2.3 — Decide NEG_START / shifted tiles (the §0.4 hinge)
**Files:** `helion/_compiler/nki/codegen.py`.
**Action:** Using the S0.1 finding, EITHER (a) emit `dma.Load` with `oob_mode=skip` + an explicit
`nisa.memset(0)` of the dest first (so dropped regions are zero, matching legacy semantics), OR (b) if
`oob_mode` semantics don't match, keep shifted/NEG_START on the **legacy** path permanently and document it
as out-of-scope for the experiment. Either is an acceptable feasibility outcome — the point is to KNOW.
**Verify (sim):** a shifted/windowed kernel (e.g. `out[t] = x[t-1] + x[t]` with boundary) under sim.
**Commit:** `experiment(nki): handle (or scope out) NEG_START tiles in TileStream load`

---

## Phase 3 — Refactor partition split + matmul + gather

### S3.1 — Partition >128 split via `alloc_logical`
**Files:** `helion/_compiler/nki_backend.py` (`cast_ast` L5802 region, `_nki_alloc_sbuf` L3858).
**Action:** When `use_tilestream` and the logical partition dim > 128, emit `tile_stream.alloc_logical(
(P,*F), pdim_size=128, dtype=...)` to produce the `(128, n_p_tiles, *F)` container instead of the manual
`min(resolved[0],128)` per-tile loop. The downstream load (S2.1) then `tile`s it. Legacy fallback otherwise.
**Verify (sim):** copy/elementwise kernel with M=256, tile 128 (n_p_tiles=2) under sim.
**Commit:** `experiment(nki): alloc_logical partition split (flag-gated)`

### S3.2 — Matmul via `blas.Matmul`
**Files:** `helion/language/matmul_ops.py` (`_nki_dot` L1114).
**Action:** When `use_tilestream`, emit `blas.Matmul(dst=dst_ts, moving=rhs_ts, stationary=lhs_ts).execute()`
over TileStreams instead of the hand-rolled `nc_transpose`→PSUM→`nc_matmul`→`tensor_copy`→accumulate. Map
Helion's `tile_m/tile_n/tile_k` block sizes to the TileStream tile shapes and `tile_dims` (K = reduction).
Legacy fallback otherwise. **Watch:** `blas.Matmul` does its own transpose/packing — confirm operand
orientation (stationary is transposed internally) matches Helion's `addmm(acc, x, y)` semantics.
**Verify (sim):** the `test_matmul_codegen` kernel (256×256, tiles 128) under sim, `allclose` vs `x@y`.
**Commit:** `experiment(nki): matmul via blas.Matmul (flag-gated)`

### S3.3 — Gather / indirect via `dma.Load(vector_index=...)`
**Files:** `helion/_compiler/nki/gather.py`, `nki/indexing.py`.
**Action:** When `use_tilestream`, route the row-index gather (`table[idx[t]]`) through
`dma.Load(dst, src, vector_index=idx_sbuf, index_dim=...)` (vector DGE). This replaces the hand-built
`.ap(vector_offset=...)` path. **Higher risk** (DGE has the most edge cases); if it doesn't cleanly map,
scope it out and keep gather on legacy. Legacy fallback otherwise.
**Verify (sim):** the `test_gather_codegen` kernel under sim.
**Commit:** `experiment(nki): gather via vector-DGE Load (flag-gated, or scoped out)`

---

## Phase 4 — Evaluate feasibility (the actual deliverable)

### S4.1 — A/B simulate the representative kernels, flag OFF vs ON
**File:** `/home/ubuntu/ts_ab_sim.py` (NEW scratch driver, not committed under helion/).
**Action:** For each of {copy, copy-partial-M500, shifted-window, elementwise-M256, matmul-256, gather}, and
for 2–3 explicit `helion.Config` block sizes each (NO autotuning, NO sweep — the user's "simulate specific
configs"), run under `HELION_NKI_SIMULATE=1`:
- flag OFF (legacy) → `allclose` vs torch reference (baseline sanity).
- flag ON (TileStream) → `allclose` vs the SAME reference.
Record pass/fail per (kernel, config, flag) in a table.
**Verify:** table is filled; note any (kernel,config) where ON fails but OFF passes (a real feasibility gap).
**Commit:** `experiment(nki): A/B simulation results for TileStream refactor`

### S4.2 — Write the feasibility verdict
**File:** `NKI_TILESTREAM_REFACTOR_PLAN.md` (append a "## Findings" section) + commit log.
**Action:** Summarize, grounded in S4.1:
1. **Which patterns work** under TileStream (copy/partial/matmul/gather/…), with sim evidence.
2. **Which were scoped out** and why (NEG_START semantics, dynamic/jagged loops, DGE edge cases).
3. **Code delta:** approximate lines removed (the ~330-line boundary explosion, ~100 partition-split,
   ~250 matmul) vs lines added (emitters + nkilib calls). Net simplification estimate.
4. **The dependency/altitude trade** restated concretely (runtime `import nkilib.experimental` in generated
   kernels; correctness now via neuronx-cc tracing a library we don't control).
5. **Recommendation:** is a real (non-flagged, byte-reviewed, HW-swept) port worth doing, and if so for which
   subset (the load/store layer is the strongest candidate; flatten/permute/transpose explicitly are not).
**Commit:** `experiment(nki): TileStream refactor feasibility findings + recommendation`

---

## Risk register (experiment-scoped)

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `nkilib.experimental` API is unstable / version-specific. | Pin to the installed `nki-0.4.0+...`; read source for signatures; flag-gate so legacy is unaffected. |
| R2 | **NEG_START / shifted tiles**: `oob_mode=skip` ≠ legacy `memset(0)`+partial-DMA semantics → silent numeric error. | S0.1 spike isolates it; S2.3 either matches via explicit memset or scopes it out. The §0.4 hinge. |
| R3 | TileStream resolution happens at neuronx-cc *trace* time, not Helion compile — a pattern that "compiles" in Helion may fail in the traced library. | Phase 0 spike proves the generated shape traces+sims BEFORE any Helion change. |
| R4 | `blas.Matmul` operand orientation (internal transpose) mismatches Helion's `addmm` → wrong results. | S3.2 verifies `allclose` vs `x@y` under sim; documented operand mapping. |
| R5 | Dynamic-range / jagged tile loops have no clean TileStream equivalent (`ViewOrder` covers static splits only). | Explicitly OUT of scope; keep `_codegen_device_loop_nki` dynamic path on legacy; note in findings. |
| R6 | Simulation (`nki.simulate`) diverges from real Trainium DMA/`oob_mode` behavior → "works in sim, not on HW". | Accept as experiment limitation; flag findings as sim-validated only; recommend one HW spot-check before any real port. |
| R7 | Flag-gated branches rot / double-maintain two paths. | Experiment branch only; not merged. If findings say "do the real port," the flag is removed in that follow-up, not kept. |
| R8 | Editable install shadows the port tree (`import helion` → helion_nki). | Always `PYTHONPATH=/home/ubuntu/helion_port`; every script prepends it (per editable-install note). |
| R9 | `HELION_NKI_SIMULATE` casts/limitations hide a real codegen bug. | Cross-check at least one config's sim output against the legacy-path sim output (S4.1 A/B), not just torch. |

---

## Do-NOT-do list (experiment scope discipline)

- **Do NOT** convert `_squeeze_shape_2d` (N-D→2D flatten) to `TensorView.flatten_dims` — compile-time list
  math is simpler and faster; runtime views here are pure regression.
- **Do NOT** route `_layout_reconcile_transpose` (partition↔free transpose) through `TensorView.permute` —
  it asserts dim 0 fixed and still compiles to `nc_transpose`; zero benefit.
- **Do NOT** touch the ~5000-line `NKIOpOverrides` elementwise lowering — TileStream is orthogonal to it.
- **Do NOT** add a byte-for-byte gate, full HW sweep, or autotuner integration — out of experimental scope.
- **Do NOT** remove the legacy path or the flag in this branch — reversibility is the point.
- **Do NOT** `git push` or merge to `nki-port-v2`.

---

## Flat execution checklist

**Phase 0 — spike (no Helion changes)**
1. S0.1 — hand-write copy/matmul/partial `@nki.jit` kernels with `nkilib`; simulate; record `oob_mode` finding. **Decision gate.**

**Phase 1 — scaffolding**
2. S1.1 — `HELION_NKI_TILESTREAM` flag + tilestream library imports + commit log.
3. S1.2 — `nki/tilestream_emit.py` string-builder helpers.

**Phase 2 — load/store (the win)**
4. S2.1 — TileStream load path, contiguous/tail tiles (flag-gated).
5. S2.2 — TileStream store path (flag-gated).
6. S2.3 — NEG_START decision (match via memset, or scope out).

**Phase 3 — partition / matmul / gather**
7. S3.1 — `alloc_logical` partition split.
8. S3.2 — `blas.Matmul`.
9. S3.3 — vector-DGE gather (or scope out).

**Phase 4 — verdict**
10. S4.1 — A/B simulate representative kernels × specific configs.
11. S4.2 — feasibility findings + recommendation.

Relevant paths (all under `/home/ubuntu/helion_port/`):
`helion/_compiler/nki/{codegen.py, gather.py, indexing.py, tilestream_emit.py(new)}`,
`helion/_compiler/{nki_backend.py, tile_strategy.py}`, `helion/language/{memory_ops.py, matmul_ops.py}`,
`helion/runtime/__init__.py` (sim launcher, unchanged). nkilib reference source:
`/opt/aws_neuronx_venv_pytorch_2_9/lib/python3.12/site-packages/nkilib/`.
