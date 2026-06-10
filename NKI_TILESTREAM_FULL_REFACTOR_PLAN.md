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

### PHASE B — Load path to TileStream objects (the core)
- **B1. `v2_load` for the clean contiguous 2-D case.** In `nki/codegen.py` `load_expr`, when
  `use_tilestream` AND contiguous, route to `v2_load`: hoist an `HBMStream` for the tensor (once),
  `alloc_logical` the SBUF dst, build dst `TileStream`, emit `dma.Load(dst, src.tile_at(grid)).execute()`,
  return the buf. This SUPERSEDES the S2.1 `.slice` interception for the contiguous case (keep `.slice` as the
  fallback for shapes `v2_load` doesn't cover yet).
  *Gate:* copy + add kernels, sim flag on/off identical; **real HW compile+run**; codegen shows
  `alloc_logical`/`tile_hbm`/`dma.Load`. Commit.
- **B2. Extend `v2_load` coverage:** tail-overflow (partial M/N — `tile_at` clamps natively), then 1-D
  reshaped tensors, then partition>128 (now `alloc_logical` does the split). Each sub-step its own sim+HW
  gate + commit. Anything still uncovered falls through to S2.1 `.slice` (correctness never regresses).

### PHASE C — Store path
- **C1. `v2_store`** symmetric to B1 (`dma.Store(dst=out_hbm.tile_at(grid), src=buf_ts)`). Supersedes S2.2.
  *Gate:* copy round-trip + add, sim + HW. Commit.
- **C2.** Extend coverage (partial, 1-D, scatter falls through to legacy). Gate + commit per sub-step.

### PHASE D — Matmul to blas.Matmul (now tractable — we own the body)
- **D1.** In `aten_lowering.py` mm/addmm: when `use_tilestream`, build moving/stationary/dst TileStreams
  (stationary = lhs as `[K,M]` via the existing transpose, OR let `blas.Matmul` consume the `[K,M]` stream),
  emit `blas.Matmul(...).execute()` accumulating into the dst stream; map Helion's `tile_m/n/k` to tile shapes
  and the K-loop to the stream's K grid. Preserve PSUM-reuse + accumulator semantics (fall through to legacy
  `nc_matmul` when `_keep_in_psum` or sub-tiling shapes aren't expressible).
  *Gate:* matmul 256³/128³/partial, sim + **HW**, vs `x@y`. Commit. (This is the riskiest phase — keep the
  legacy fall-through generous.)

### PHASE E — Allocation + elementwise polish (make the whole body nkilib-native)
- **E1.** Route accumulator/`full`/`hl.zeros` buffer allocs through `alloc_logical` (creation_ops.py,
  matmul result bufs). *Gate:* reduction + matmul kernels sim+HW. Commit.
- **E2.** Where operands are already TileStreams (post B/C), switch elementwise `nisa.tensor_tensor` →
  `blas.TensorTensor` over those streams (now type-compatible, unlike the S5.3 blocker). Only where both
  operands are v2 streams; else keep `nisa`. *Gate:* add/mul/where kernels. Commit.

### PHASE F — Dynamic/jagged loops (decision phase)
- **F1.** Assess whether `dma.Load` + `tile_at` compose with the `_nki_dyn_loops` counter mechanism and the
  jagged nested-demotion logic. The dynamic path shares the `offset_N` slot (Finding A) and prepends a counter
  increment. Likely outcome: **keep dynamic/jagged on legacy** (it's correct, isolated, and nkilib's static
  `tile_at` doesn't model data-dependent bounds). Document the boundary. Spike before deciding. Commit (or
  documented scope-out).

### PHASE G — Full validation + cleanup
- **G1.** Run the example suite under sim (flag on) for the kernels that exercise B–E; A/B vs flag-off.
- **G2.** Real-HW compile+run spot-checks across copy/add/matmul/reduce/sigmoid/gather + 2–3 fuller examples.
- **G3.** Update `NKI_TILESTREAM_REFACTOR_PLAN.md` §Findings with the v2 results + a generated-kernel
  before/after. Decide per-category what is production-recommendable. Commit.

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
| R7 | Coverage gaps in v2_load/store silently wrong. | Every v2 path has an explicit guard; uncovered shapes FALL THROUGH to the S2.1/S2.2 `.slice` path (already HW-verified), never to broken code. |
| R8 | Autotuner composition (block sizes / loop order). | Scaffold still owns block sizes + loop order; `tile_at` just consumes `offset_N`. Validate one autotuned config in G. |
| R9 | Generated-code bloat (alloc_logical + stream per tile inside loop). | Hoist streams outside loop (Finding C pattern); `alloc_logical` of the reused buffer can also hoist. Measure lines in G. |
| R10 | nkilib `tile_at`/`alloc_logical` semantics differ from our spikes on real HW. | Every phase gates on real `/dev/neuron0` compile+run, not just sim. |

## 5. Do-NOT list
- Do NOT remove the Python `for offset_N in affine_range` scaffold (Finding C: nkilib keeps it).
- Do NOT touch `active_device_loops` / `add_device_loop` / `set_statements` (Finding B).
- Do NOT convert dynamic/jagged loops without the Phase F spike.
- Do NOT remove the S2.1/S2.2 `.slice` paths — they are the safety fall-through for uncovered shapes.
- Do NOT break flag-off: legacy bodies stay byte-identical (gate: `test_nki_port_codegen.py` 4/4 every step).
- Do NOT `git push` / merge.

## 6. Execution checklist
A1 stream registry → A2 grid-coord helper → B1 v2_load contiguous → B2 v2_load coverage (partial/1-D/part>128)
→ C1 v2_store → C2 store coverage → D1 blas.Matmul → E1 alloc_logical allocs → E2 blas.TensorTensor on streams
→ F1 dynamic/jagged decision → G1 sim suite → G2 HW spot-checks → G3 findings + recommendation.

Each step: flag-gated, sim A/B (on==off correctness), real-HW compile+run for the touched pattern,
`test_nki_port_codegen.py` 4/4 flag-off, one commit. Log every step in `NKI_TILESTREAM_COMMIT_LOG.md`.

Files: `helion/_compiler/nki/{tilestream_codegen.py(NEW), codegen.py, tilestream_emit.py}`,
`helion/_compiler/{nki_backend.py, aten_lowering.py, tile_strategy.py, device_function.py, creation_ops.py}`,
`helion/language/matmul_ops.py`. nkilib ref: `/opt/aws_neuronx_venv_pytorch_2_9/.../nkilib/`.
