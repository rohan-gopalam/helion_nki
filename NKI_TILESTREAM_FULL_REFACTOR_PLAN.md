# NKI Backend — FULL TileStream Refactor Plan (loop-owns-iteration → TileStream-owns-iteration)

> Generated 2026-06-10. Branch: **`nki-tilestream-experiment`**. This is the LARGE codebase change:
> replace Helion's "backend emits the loop + per-tile body" model with nkilib's TileStream model, so
> generated kernels read like nkilib examples (`alloc_logical` / `tile` / `tile_hbm` / `dma.Load().execute()`
> / `blas.Matmul().execute()`) — the library owns tile extraction and DMA, Helion owns the schedule.
>
> Builds on the completed S1–S5 work (flag-gated `.slice`/`blas.transpose`/`blas.activation`). This plan
> takes it to the structural level. Still flag-gated behind `HELION_NKI_TILESTREAM=1` (now the "v2 codegen"),
> legacy default, reversible. Validated by `HELION_NKI_SIMULATE=1` + **real `/dev/neuron0` compile/run**.
> Per the standing assumption: **nkilib is treated as fully supported / future-proofed** — adopting its idiom
> is the goal, not a liability.

---

## 0. The decisive architectural finding (shapes the whole plan)

A full investigation of `tile_strategy.py` + `generate_ast.py` + `device_ir.py` + `inductor_lowering.py`
established two facts that **constrain every design choice below**:

**Finding A — `offset_N` / `indices_N` are a load-bearing contract referenced by 15+ downstream sites.**
The scalar `offset_var(block_idx)` (e.g. `offset_0`) and the SBUF index vector `index_var(block_idx)`
(e.g. `indices_0`) are consumed by: load/store codegen (`nki/codegen.py`, `nki/indexing.py`), masking
(`_setup_mask`), reductions (`reduction_strategy.py`), matmul (`aten_lowering.py`), cute paths, and
`variable_origin.py`. They are resolved on demand via `state.codegen.active_device_loops[block_idx][-1]
.strategy.offset_var(...)`. **Any rewrite that removes `offset_N` breaks all of them.**

**Finding B — the loop body is populated by scope-redirection, not insertion.**
`_codegen_device_loop_nki` returns a `DeviceLoopState` whose `for_node.body` is an *empty list reference*.
The caller (`device_ir.py` `ForLoopGraphInfo.codegen`) opens `with state.codegen.add_device_loop(...)`,
which (a) pushes the loop onto `active_device_loops` and (b) redirects `add_statement()` into
`device_loop.inner_statements` (the same list as `for_node.body`). Then `codegen_call_with_graph` runs the
FX graph nodes, and every load/store/compute appends into that body. **The body is filled by graph
execution, not by the loop emitter** — so we cannot "replace the loop" without preserving this mechanism.

**Finding C — real nkilib kernels STILL use `for ... in affine_range(...)`.**
Inspected `experimental/output_projection/output_projection_tkg_primitives.py`: the nkilib idiom keeps an
outer Python `for ... in affine_range(...)` / `TiledRange` loop and puts **`tile()` + `Load().execute()`
INSIDE** it. The library makes *tile extraction + DMA* declarative; it does NOT eliminate the schedule loop.

**Consequence — the refactor is NOT "delete the for-loop."** It is: **keep Helion's loop scaffold and the
`offset_N`/`active_device_loops` contract, but replace the BODY-level mechanics** — buffer alloc, index
materialization, tile slicing, and DMA — with TileStream/`dma.Load`/`dma.Store` objects constructed from the
loop's `offset_N`. This is the same altitude as the real nkilib examples, far lower risk than rewriting the
loop driver, and it composes with the autotuner (which only knows about block sizes + loop order, both of
which the scaffold still owns).

This reframes "the big refactor" into something tractable and correct: **a complete migration of the
body-level codegen to TileStream objects, with the loop scaffold and downstream `offset_N` contract intact.**

---

## 1. Target generated-kernel shape (the contract we're building toward)

Today (flag-on, post-S5), a 2-D copy emits Helion-style scaffold + `.slice` DMA. The FULL-refactor target
makes the body read like nkilib — allocate via `alloc_logical`, build streams, `Load`/`Store`/`Matmul`:

```python
@nki.jit
def _helion_k(x):
    x = x.reshape([500, 128])
    nki_return_buf = nl.ndarray([500, 128], dtype=nl.float32, buffer=nl.shared_hbm)
    # v2: stream views over the whole tensors, built once
    _x_hbm   = _nkitile.tile_stream.tile_hbm(x,             tile_shape=(128, 64))
    _out_hbm = _nkitile.tile_stream.tile_hbm(nki_return_buf, tile_shape=(128, 64))
    for offset_0 in nl.affine_range(0, 500, 128):     # <-- scaffold KEPT (Finding C)
        for offset_1 in nl.affine_range(0, 128, 64):
            _buf    = _nkitile.tile_stream.alloc_logical((128, 64), 128, nl.float32)   # alloc_logical
            _dst_ts = _nkitile.tile_stream.tile(_buf, tile_shape=(128, 64))
            _grid   = (offset_0 // 128, offset_1 // 64)
            _nkitile.dma.Load(dst=_dst_ts, src=_x_hbm.tile_at(_grid)).execute()        # declarative load
            _nkitile.dma.Store(dst=_out_hbm.tile_at(_grid), src=_dst_ts).execute()     # declarative store
```

Key idea: the loop variable `offset_N` becomes a **grid coordinate** (`offset_N // block_size`) handed to
`HBMStream.tile_at(...)`, so the library does the slice/clamp/partition math. `offset_N` still exists (Finding
A satisfied); the body is now nkilib-native (Finding C matched).

> Decision recorded: we do NOT pursue the "`Load().execute()` owns the whole iteration with no Python for"
> variant — Finding C shows even nkilib doesn't do that, and it would fight `active_device_loops` (Finding B).

---

## 1b. Coverage policy: convert what nkilib HAS a primitive for; legitimately keep nisa where it doesn't

> **Revised 2026-06-10 after auditing the full nkilib tree (the original "zero raw nisa / loud-error on
> anything uncovered" target was WRONG — see Hole 1 below).**

**Audit finding (decisive):** nkilib has **NO primitive** for reductions, scans, cumulative, predicated
(masking), or index materialization. And nkilib's OWN composed kernels (`cross_entropy.py`,
`ring_attention_bwd.py`, `foreach_norm.py`, `transformer_tkg.py`) **call `nisa.tensor_reduce` /
`nisa.tensor_tensor_scan` / `nisa.tensor_copy` directly.** So "express everything in nkilib, zero raw nisa"
is **not achievable and not idiomatic** — even AWS mixes. The real nkilib idiom is:
`tile_stream`+`dma`+`blas` for **tiling / data movement / matmul / elementwise / transpose**, and **raw `nisa`
for reductions / scans / predication / iota / memset** (no primitive exists).

**Four buckets (every emitted op is in exactly one):**

1. **CONVERT — has a nkilib primitive (the refactor target):** tiling, `alloc_logical`, load/store
   (`dma.Load`/`Store`), transpose (`blas.transpose`), matmul (`blas.Matmul`), elementwise unary/scalar
   (`blas.activation`/`tensor_scalar`), elementwise binary (`blas.TensorTensor`), gather/dynamic
   (`vector_select`/`select`+DGE `Load`).
2. **HARD-CASE — convert, but with a DIFFERENT primitive than the analogous view op:**
   - R-NEG (NEG_START / shifted `x[i-pad]`): `dma.Load(oob_mode=skip)` (TensorView.slice asserts start≥0).
   - R-BC (partition broadcast `[1,F]→[P,F]`): `blas.broadcast` (TensorView.broadcast asserts dim 0).
3. **STAYS nisa — LEGITIMATE, no nkilib primitive exists, idiomatic (nkilib does the same):**
   `nisa.tensor_reduce`, `nisa.tensor_partition_reduce`, `nisa.tensor_scalar_reduce`,
   `nisa.activation_reduce`, `nisa.tensor_tensor_scan` (scan), `nisa.tensor_scalar_cumulative` (cumsum),
   `nisa.scalar_tensor_tensor`, `nisa.tensor_copy_predicated` (where/masking), `nisa.reduce_cmd`,
   `nisa.register_load`, AND `nisa.iota`/`nisa.memset` (index/init materialization tied to the `indices_N`
   contract — Finding A). **These are NOT failures and NOT fallthrough — they are the boundary of the
   library.** ~38+ emission sites. Kernels exercising them (softmax, rms_norm, layernorm, cumsum, scan,
   any `where()`) keep these nisa ops and still work.
4. **LOUD ERROR — genuinely unexpected gap only:** an op that is neither convertible (bucket 1/2) nor a
   known-legitimate nisa op (bucket 3) raises `exc.BackendUnsupported(name, "v2: <case>")`. This now fires
   ONLY for true surprises (e.g. a new subscript shape we haven't classified), not for whole categories the
   library doesn't cover.

**Revised no-fallthrough meaning:** v2 still does NOT silently fall through to *legacy body-codegen* for
bucket-1/2 ops — those MUST convert or loud-error. But bucket-3 ops legitimately emit raw `nisa` in BOTH the
legacy and v2 paths (identical), because that's what nkilib itself does. Legacy body-codegen stays flag-OFF
as the A/B correctness reference. The S2.1/S2.2 `.slice` interception folds into the total v2 path.

**Goal restated honestly:** a flag-on kernel uses nkilib (`tile_stream`/`dma`/`blas`) for everything with a
primitive, and raw `nisa` ONLY for reduce/scan/predicate/iota/memset — i.e. it matches the structure of a
hand-written nkilib kernel, NOT "zero nisa."

## 2. Design: a parallel "v2 body codegen" behind the existing flag

`HELION_NKI_TILESTREAM=1` already exists. We extend its meaning: when on, the NKI backend uses **v2
body-codegen** for the structured operations. Implementation strategy — **a new module
`helion/_compiler/nki/tilestream_codegen.py`** that owns the v2 emitters, dispatched from the existing
shims/handlers. Legacy bodies remain literally unchanged (flag-off path untouched), so reversibility +
byte-identical legacy are preserved by construction.

The v2 module exposes, per structured op, an emitter that consumes the SAME inputs the legacy path gets
(`state`, `subscript`, `tensor`, the active `offset_N` via `active_device_loops`) and emits TileStream calls:

- `v2_load(state, subscript, tensor) -> ast` — build/reuse an `HBMStream` for `tensor`, an SBUF buffer via
  `alloc_logical`, a dest `TileStream`, and `dma.Load(...).execute()`; return the SBUF buffer name.
- `v2_store(state, ...)` — symmetric `dma.Store`.
- `v2_matmul(...)` — `blas.Matmul` over dst/moving/stationary streams (the deep integration scoped out in
  S3.2, now in scope because we own the body).
- `v2_alloc(shape, pdim, dtype) -> alloc_logical(...)` — used by `full`/`hl.zeros`/accumulators.
- Stream caching: an `HBMStream` per (tensor, tile_shape) is built ONCE outside the loop (Finding C / the
  nkilib example builds `wgt_hbm_grid` once) and `.tile_at(grid_pos)` inside — so we add a
  `state.device_function._nki_hbm_streams` registry + hoist-before-loop emission.

---

## 3. Phased implementation (each phase: flag-gated, sim + real-HW gate, one-commit-per-step)

### PHASE A — Stream infrastructure (no behavior change yet)
- **A1. `tilestream_codegen.py` skeleton + stream registry.** Create the module; add
  `_nki_hbm_streams: dict[(name, tile_shape) -> stream_var]` to `DeviceFunction`; add a "hoist a statement to
  before the current device loop" helper (uses `DeviceLoopState.outer_prefix`, which already exists).
  *Gate:* import + a unit test that the registry dedupes. Commit.
- **A2. grid-coordinate helper.** Given a block_idx and its `offset_var`, emit `offset_N // block_size` as the
  grid coord for `tile_at`. Handle the partition-tile-step cap (Finding: Helion caps partition step at 128).
  *Gate:* string-assert. Commit.

### PHASE B — Load path to TileStream objects (the core) — TOTAL, no fallthrough
- **B0. Install the v2 dispatch + bucket-aware guard.** In `load_expr`, when `use_tilestream`, route bucket-1/2
  cases to `v2_load` (convert); bucket-3 ops that appear in a load context (e.g. a predicated/masked load) stay
  nisa and are passed through unchanged; a bucket-4 (unclassified) case raises
  `BackendUnsupported("v2 load: <case>")`. Remove the S2.1 `.slice` as a *fallthrough to legacy* (fold it into
  v2 as the contiguous case). *Gate:* a contiguous kernel converts; a masked load still emits its
  `tensor_copy_predicated`; an unclassified case raises loudly (asserted in a test). Commit.
- **B1. Contiguous (full / tail-overflow).** Hoist `HBMStream` once; `alloc_logical` dst; `dma.Load(dst,
  src.tile_at(grid)).execute()`. *Gate:* copy+add, sim on==off, **real HW**; codegen shows
  `alloc_logical`/`tile_hbm`/`dma.Load`. Commit.
- **B2. 1-D reshape + partition>128 + scalar index.** `reshape_dim`/`tile_hbm` for 1-D; `alloc_logical` split
  for >128; `TensorView.select` for `x[5,:]`. Each sub-step sim+HW + commit.
- **B3. R-NEG (NEG_START / shifted `x[i-pad]`).** `dma.Load(..., oob_mode=skip)` over memset(0) buffer (NOT
  `slice`). *Gate:* a windowed/shifted kernel, sim on==off + HW. Commit.
- **B4. Gather (`table[idx[t]]`, IndirectAP) + scalar-dynamic (DynamicAP).** `TensorView.vector_select` +
  `dma.Load(vector_index=, index_dim=)`; dynamic via `select(runtime_idx)`. *Gate:* embedding/gather kernel
  sim+HW. Commit. (Highest-risk load case; this is what makes "no fallback" real.)

### PHASE C — Store path — TOTAL, no fallthrough
- **C1. Contiguous store** `dma.Store(dst=out_hbm.tile_at(grid), src=buf_ts)`. Loud guard for uncovered.
  *Gate:* copy round-trip + add, sim on==off + HW. Commit.
- **C2. Partial + 1-D + scatter.** scatter via `dma.Store`/DGE; loud guard otherwise. Gate + commit per
  sub-step until store has zero `BackendUnsupported` on the test set.

### PHASE D — Matmul to blas.Matmul (now tractable — we own the body)
- **D1.** In `aten_lowering.py` mm/addmm: when `use_tilestream`, build moving/stationary/dst TileStreams
  (stationary = lhs as `[K,M]` via the existing transpose, OR let `blas.Matmul` consume the `[K,M]` stream),
  emit `blas.Matmul(...).execute()` accumulating into the dst stream; map Helion's `tile_m/n/k` to tile shapes
  and the K-loop to the stream's K grid. Preserve PSUM-reuse + accumulator semantics (fall through to legacy
  `nc_matmul` when `_keep_in_psum` or sub-tiling shapes aren't expressible).
  *Gate:* matmul 256³/128³/partial, sim + **HW**, vs `x@y`. Commit. (This is the riskiest phase — keep the
  legacy fall-through generous.)

### PHASE E — Allocation + elementwise (make the whole body nkilib-native) — TOTAL
- **E1.** Route accumulator/`full`/`hl.zeros`/matmul-result buffer allocs through `alloc_logical`. *Gate:*
  reduction + matmul kernels sim+HW. Commit.
- **E2.** Elementwise binary → `blas.TensorTensor` over streams (now type-compatible — the S5.3 blocker is
  removed because v2 operands ARE streams). Under no-fallthrough this is REQUIRED, not optional: every
  `nisa.tensor_tensor` site reachable flag-on must convert or raise. *Gate:* add/mul/sub/where/maximum
  kernels, sim on==off + HW. Commit.
- **E3. R-BC (partition broadcast `[1,F]→[P,F]`).** `blas.broadcast(dst, src, src_partition)` (NOT
  `TensorView.broadcast`, which asserts dim 0). Replaces `_emit_partition_broadcast`. *Gate:* a
  partition-broadcast kernel (e.g. bias add over rows), sim on==off + HW. Commit.
- **E4. Classify remaining `nisa.*` body emits into bucket 3 vs bucket 4.** For each remaining raw `nisa.*`
  site reachable flag-on: if it's a bucket-3 op (reduce/scan/cumulative/predicated/scalar_tensor_tensor/
  iota/memset) → **leave it as nisa, annotate `# v2: stays nisa — no nkilib primitive (idiomatic)`**. If it's
  an unclassified op → raise `BackendUnsupported`. Goal: a flag-on kernel emits nkilib for all bucket-1/2 ops
  and raw nisa ONLY for bucket-3 ops — matching a hand-written nkilib kernel's structure. Add a test that
  asserts no UNCLASSIFIED nisa op survives flag-on for the example set.

### PHASE F — Dynamic/jagged loops — must be covered too (no fallthrough)
- **F1.** Because there is no fallthrough, dynamic_range/jagged MUST either convert or raise loudly. Spike
  whether `dma.Load(vector_index/scalar_offset)` + the `_nki_dyn_loops` counter compose. If they do → convert
  (B4 mechanism extended to the counter offset). If a specific jagged nesting genuinely can't map → raise
  `BackendUnsupported("v2: nested jagged")` and record it as the one explicit gap (the experiment's honest
  answer: "everything EXCEPT X"). *Gate:* jagged_tile example sim+HW or documented loud gap. Commit.

### PHASE F2 — Bucket-3 regression guard (reductions / scans / predication / masking)
- **F2.** These have NO nkilib primitive (audited) and STAY nisa. The risk is not that they need converting —
  it's that the Phase B/E changes (operands now arriving as v2 streams / alloc_logical buffers) must still feed
  the existing `nisa.tensor_reduce`/`tensor_tensor_scan`/`tensor_copy_predicated` codegen correctly (shapes,
  dtypes, `_nki_sbuf_shapes` registration). Verify the reduction/scan/where paths in `reduction_strategy.py`
  and the `_setup_mask` jagged-mask path still work when their inputs are v2-produced.
  *Gate:* softmax, rms_norm, layernorm, cumsum, a `where()`/masked kernel, and the jagged_tile example — sim
  flag-ON must equal flag-OFF, AND real-HW compile+run. These exercise bucket-3 ops sitting next to converted
  bucket-1 loads/stores; confirm the mix composes. Commit.

### PHASE G — Full validation + cleanup
- **G1.** Run the example suite under sim (flag on) for the kernels that exercise B–E; A/B vs flag-off.
- **G2.** Real-HW compile+run spot-checks across copy/add/matmul/reduce/sigmoid/gather + 2–3 fuller examples.
- **G3.** Update `NKI_TILESTREAM_REFACTOR_PLAN.md` §Findings with the v2 results + a generated-kernel
  before/after. Include the **definitive op coverage table**: which ops convert to nkilib (bucket 1/2) and the
  exhaustive list that legitimately stays `nisa` because nkilib has no primitive (bucket 3) — the honest
  answer to "can everything be done in nkilib?" is "yes for movement/tiling/matmul/elementwise; reductions/
  scans/predication stay nisa, as they do in nkilib's own kernels." Decide per-category production
  recommendation. Commit.

---

## 4. Risk register

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Removing `offset_N` breaks 15+ downstream sites (Finding A). | **We KEEP the loop scaffold + `offset_N`**; v2 only changes body mechanics. Grid coord = `offset_N // bs`. |
| R2 | Body-population scope-redirection (Finding B) breaks if we emit the loop differently. | We don't touch `add_device_loop`/`active_device_loops`; v2 emits INTO the same `inner_statements`. |
| R3 | `tile_at(grid)` semantics mismatch Helion's offset math (off-by-tile). | Per-step sim A/B vs legacy AND real HW; start with copy (trivially checkable) before matmul. |
| R4 | Matmul stream orientation / PSUM-reuse / sub-tiling not expressible in blas.Matmul. | Generous legacy fall-through (D1); convert only the shapes that map; keep nc_matmul otherwise. |
| R5 | HBMStream hoisting before the loop interacts badly with nested loops / multiple uses. | Dedup registry keyed on (name, tile_shape); hoist to `outer_prefix` of the OUTERMOST active loop; test nested (matmul has 3 loops). |
| R6 | Dynamic/jagged counter mechanism (Finding: `_nki_dyn_loops`) incompatible with tile_at. | Phase F assesses; default = keep dynamic/jagged on legacy. Correctness preserved. |
| R7 | Coverage gaps under NO-fallthrough policy = hard failures, not silent-correct. | Every uncovered shape raises a LOUD `BackendUnsupported("v2: <case>")` — never a wrong result. flag-OFF legacy is the A/B reference to prove each converted case matches. Gaps are visible work items, not hidden bugs. |
| R-NEG | `TensorView.slice` refuses negative start (NEG_START/shifted). | Phase B3: `dma.Load(oob_mode=skip)` over memset(0) buffer (S0.1-proven), not slice. |
| R-BC | `TensorView.broadcast` refuses partition dim 0. | Phase E3: `blas.broadcast` partition-replicate primitive. |
| R8 | Autotuner composition (block sizes / loop order). | Scaffold still owns block sizes + loop order; `tile_at` just consumes `offset_N`. Validate one autotuned config in G. |
| R9 | Generated-code bloat (alloc_logical + stream per tile inside loop). | Hoist streams outside loop (Finding C pattern); `alloc_logical` of the reused buffer can also hoist. Measure lines in G. |
| R10 | nkilib `tile_at`/`alloc_logical` semantics differ from our spikes on real HW. | Every phase gates on real `/dev/neuron0` compile+run, not just sim. |
| R-HOLE1 | nkilib has **no primitive** for reduce/scan/cumulative/predicated/iota/memset (~38 sites). | Bucket 3: these STAY nisa (nkilib's own kernels do the same). NOT loud-errors, NOT fallthrough — they're the library boundary. Phase F2 regression-guards them; E4 classifies them explicitly. |
| R-HOLE3a | bucket-3 ops (esp. masking/`tensor_copy_predicated`, reductions) break when fed v2-produced stream/alloc_logical inputs. | Phase F2: dedicated regression gate on softmax/rms_norm/layernorm/cumsum/where/jagged, sim==legacy + HW, with converted loads feeding them. |
| R-HOLE3b | `tile_at` grid-coord composition unverified for the 3-deep matmul loop (m×n×k + PSUM accumulate). | Phase D spike on a single matmul BEFORE broad conversion; generous legacy fall-through if K-axis/PSUM-reuse shapes don't map. |
| R-HOLE3c | `nisa.iota`/`memset` are tied to the `indices_N` contract (Finding A); converting them risks the masking/index machinery. | Bucket 3: leave `iota`/`memset` as nisa (composed nkilib kernels emit 0 of them only because their index vectors come from `tile`/`affine_range` structure — Helion's `indices_N` contract still needs them). Do NOT convert; document. |

## 5. Do-NOT list
- Do NOT remove the Python `for offset_N in affine_range` scaffold (Finding C: nkilib keeps it).
- Do NOT touch `active_device_loops` / `add_device_loop` / `set_statements` (Finding B).
- Do NOT let v2 silently fall through to legacy — uncovered shapes raise `BackendUnsupported` (§1b).
- Do NOT delete the legacy body-codegen — it stays flag-OFF as the A/B correctness reference.
- Do NOT break flag-off: legacy bodies stay byte-identical (gate: `test_nki_port_codegen.py` 4/4 every step).
- Do NOT `git push` / merge.

## 6. Execution checklist
A1 stream registry → A2 grid-coord helper → B0 v2 dispatch + bucket-aware guard (convert bucket-1/2,
pass-through bucket-3 nisa, loud-error bucket-4; remove .slice fallthrough) → B1 contiguous →
B2 1-D/part>128/scalar-index → B3 NEG_START (oob_mode) → B4 gather+dynamic → C1 store contiguous →
C2 store partial/1-D/scatter → D1 blas.Matmul → E1 alloc_logical → E2 blas.TensorTensor →
E3 partition broadcast (blas.broadcast) → E4 classify remaining nisa (bucket-3 keep+annotate, bucket-4 error) →
F1 dynamic/jagged convert-or-loud-gap → F2 bucket-3 regression guard (softmax/rms_norm/layernorm/cumsum/where/
jagged: v2==legacy + HW) → G1 sim suite (v2==legacy) → G2 HW spot-checks → G3 findings (incl. the definitive
list of which ops have NO nkilib primitive and legitimately stay nisa).

Each step: flag-gated, sim A/B (on==off correctness), real-HW compile+run for the touched pattern,
`test_nki_port_codegen.py` 4/4 flag-off, one commit. Log every step in `NKI_TILESTREAM_COMMIT_LOG.md`.

Files: `helion/_compiler/nki/{tilestream_codegen.py(NEW), codegen.py, tilestream_emit.py}`,
`helion/_compiler/{nki_backend.py, aten_lowering.py, tile_strategy.py, device_function.py, creation_ops.py}`,
`helion/language/matmul_ops.py`. nkilib ref: `/opt/aws_neuronx_venv_pytorch_2_9/.../nkilib/`.
