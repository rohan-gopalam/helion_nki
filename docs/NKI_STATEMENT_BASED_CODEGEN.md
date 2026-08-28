# NKI statement-based codegen: mismatch and options

## The mismatch

- **Triton / JAX (Pallas)** are expression-based: `tl.sum(x, 0)` returns a value you use inline.
- **NKI** is statement-based: `nisa.tensor_reduce(dst=result, src=x, op=nl.add, axis=0)` writes to an explicit destination; you must allocate `result` and emit a statement.

Helion’s codegen assumes expression-based backends: backend methods return **expression strings** that get inlined. They have no standard way to “emit this statement, then use this variable as the expression.” Load/store work because those codegen paths have access to `state.codegen.add_statement()`; most other backend methods do not receive `state` and cannot emit statements.

---

## Where backend expression methods are called

| Backend method           | Call site(s)                                                                 | Has `CodegenState`? |
|-------------------------|-------------------------------------------------------------------------------|----------------------|
| `reduction_expr`        | `reduction_strategy.py`: `call_reduction_function` → `backend.reduction_expr` | Caller has `state`   |
| `reduction_index_expr`   | `reduction_strategy.py`: `_index_init_expr`                                  | Caller has `state`   |
| `broadcast_to_expr`      | `reduction_strategy.py`: `broadcast_str`; `_tracing_ops.py`: _mask_to codegen | Both have `state`    |
| `where_expr`            | `_tracing_ops.py`: _mask_to codegen                                           | Yes                  |
| `minimum_expr`          | `tile_ops.py`: tile_end codegen                                              | Yes                  |
| `full_expr`             | `reduction_strategy.py`: codegen_reduction, LoopedReductionStrategy          | Yes                  |
| `scalar_load_expr`      | `tile_strategy.py`: `_to_ast` (tensor → AST for bounds)                       | No (only `CompileEnvironment`) |

So almost all call sites either (1) are inside a codegen handler that has `state`, or (2) are called from a method that receives `state` (e.g. `codegen_reduction(state, ...)`). The main exception is `scalar_load_expr` in `_to_ast`, which has no state (and NKI already raises there).

---

## Options

### Option A: Pass `state` into backend methods

- Add optional `state: CodegenState | None = None` to methods that may need to emit (e.g. `reduction_expr`, `minimum_expr`, `where_expr`, `broadcast_to_expr`, `full_expr`).
- Call sites that have `state` pass it (e.g. `reduction_strategy` would pass `state` into `call_reduction_function` and down to `backend.reduction_expr(...)`).
- NKI backend: when `state` is not `None`, allocate result (e.g. `state.device_function.new_var(...)`), emit `state.add_statement(...)` for the NKI ISA call, return the result variable name. When `state` is `None`, raise `BackendUnsupported`.

**Pros:** Explicit; no global/context state.  
**Cons:** Signature change; every such call site must be updated to pass `state` where available.

---

### Option B: “Current codegen state” on `CompileEnvironment`

- Add optional `_codegen_state: CodegenState | None` (or a small “statement sink” interface) on `CompileEnvironment`, set at codegen entry and cleared when done.
- At codegen entry (e.g. when entering the visitor that runs strategies, or at the start of `codegen_reduction` / device loop codegen), set `CompileEnvironment.current()._codegen_state = state`.
- Backend methods that need to emit (NKI):  
  `state = getattr(CompileEnvironment.current(), '_codegen_state', None)`  
  If `state` is not `None` and backend is NKI: allocate result, `state.add_statement(...)`, return result var. Otherwise current behavior (or raise for NKI).

**Pros:** No backend signature change; one central “context” for codegen.  
**Cons:** Implicit context; `CompileEnvironment` docstring currently says “No config or codegen specific state” (would be a deliberate exception).

---

### Option C: NKI-specific branches at each call site

- Like load/store: at each place that calls these backend methods, add an `if backend.name == 'nki': ... else: ...` and for NKI use `state.add_statement(...)` plus a result var, then use that var as the expression.
- No backend API change; NKI logic lives in the same modules that already have `state` (e.g. `reduction_strategy.py`, `_tracing_ops.py`, `tile_ops.py`).

**Pros:** No backend or environment API change; each op’s NKI path is explicit.  
**Cons:** NKI logic scattered; every such op needs a branch and correct allocation/ISA usage.

---

## Recommendation

- **Short term:** Option C is the fastest way to unblock: add NKI branches only where you need them (e.g. reductions, masked fill, tile_end), reusing the same pattern as load/store.
- **Medium term:** Option B is a clean way to keep backend APIs expression-only while giving NKI a way to emit statements: set “current codegen state” at the top of the codegen paths that already have `state`, and have NKI backend methods use it when present. Option A is equivalent in power but requires threading `state` through more call chains; use it if you prefer explicit parameters over context.

---

## Option B implementation (done)

Option B is implemented as follows:

- **CompileEnvironment** (`_compiler/compile_environment.py`): Added `_codegen_state: object | None` and `set_codegen_state(state)` context manager. Docstring updated to allow this single codegen-specific state.
- **Entry points** set/clear `_codegen_state` via `env.set_codegen_state(state)`:
  - `inductor_lowering.py`: around `strategy.codegen_reduction(state, ...)` in reduction codegen.
  - `generate_ast.py`: in `visit_For` around `codegen_call_with_graph(self, root, [])`.
  - `device_ir.py`: in `DeviceLoopGraphInfo.codegen`, `IfGraphInfo.codegen`, `WhileLoopGraphInfo.codegen` (condition and body), and `HelperFunctionGraphInfo.codegen`.
- **NKIBackend** (`_compiler/backend.py`): Added `_nki_codegen_state()`. When it returns non-None:
  - **`reduction_expr`** uses the documented **`nisa.tensor_reduce(dst=..., src=..., op=..., axis=...)`** (with `nl.add` / `nl.max` / `nl.min` / `nl.mul`); allocates `nl.zeros(())` for the destination then emits the call, returns the result var name.
  - **`minimum_expr`** uses **`nisa.tensor_tensor(dst=..., data1=..., data2=..., op=nl.minimum)`** (same pattern as the add comment in the codebase); allocates a result var, emits the call, returns it.
  - **`full_expr`**, **`where_expr`**, **`broadcast_to_expr`** have no NKI ISA documented in this repo (no `nisa.full`, `nisa.where`, `nisa.broadcast_to`), so they still raise `BackendUnsupported`; add statement-based paths when the NKI docs define the corresponding ISA.

---

## Files to touch (for any option)

- **Backend methods (NKI):** `reduction_expr`, `reduction_index_expr`, `reduction_index_zero_expr`, `broadcast_to_expr`, `minimum_expr`, `where_expr`, `full_expr`, `scalar_load_expr` (already raise or use dma path).
- **Call sites:**  
  - `_compiler/reduction_strategy.py` (reduction_expr, broadcast_to_expr, full_expr, reduction_index_*, reshape_expr).  
  - `language/_tracing_ops.py` (_mask_to: where_expr, broadcast_to_expr, full_expr).  
  - `language/tile_ops.py` (tile_end: minimum_expr).  
  - `_compiler/tile_strategy.py` (scalar_load_expr in _to_ast — no state; keep NKI as raise or special-case only if needed).
