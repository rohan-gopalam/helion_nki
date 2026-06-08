# NKI Port — Commit Log

In-depth record of every commit made while executing `NKI_PORT_IMPLEMENTATION_PLAN.md`,
porting the NKI backend from `fix-nki-kernel-compilation` (`3056e85c`) onto `upstream/main`
(`eb61d5b8`) on branch `nki-port-v2`.

**How to read this:** newest entries are appended at the bottom. Each entry records the plan
step, what changed and why, how it was verified (with the actual gate result), and any
deviation from the plan or open follow-up. Commits are made only after the step's verification
gate passes.

**Conventions**
- Reference (NKI source of truth): `fix-nki-kernel-compilation`. Verbatim copies use `git show`/`git checkout`, never manual paste.
- Port tree tested via `PYTHONPATH=/home/ubuntu/helion_port` (editable install points elsewhere; never reinstall).
- Baseline (before any NKI work): `helion` imports OK; backends = `['triton','pallas','cute','tileir','metal']`; `nki` absent.

---

## Baseline — `eb61d5b8` (upstream/main, untouched)

- `import helion` succeeds from the port tree.
- `list_backends()` → `['triton','pallas','cute','tileir','metal']` (no `nki`).
- Working tree clean apart from the two planning docs.

---

## P1.1 — Copy NKI-only support files

**Files:** `helion/_compiler/nki_fusion.py` (NEW), `helion/language/_nki_dim_access.py` (NEW)

**What & why:** Copied both verbatim from the reference via `git checkout fix-nki-kernel-compilation -- …`
(no manual edits). These have no upstream equivalent, so there is no conflict surface.
- `_nki_dim_access.py` defines the `DimAccess` hierarchy (Contiguous/Scalar/Indirect/Dynamic/FullSlice/StridedGather)
  plus the `IndirectAP`/`DynamicAP` dataclasses (the post-sentinel-refactor representation) used by
  `memory_ops.py` and `atomic_ops.py` load/store codegen.
- `nki_fusion.py` provides the FX-graph fusion passes (`annotate_psum_reuse`, `annotate_tensor_scalar_reduce`,
  `annotate_activation_reduce`, `annotate_fx_graph`) that mark matmul→activation patterns for PSUM reuse.
  Its `matmul_ops` import is lazy (inside a function body) so no circular import at load time.

**Verification (gate passed):**
- `git diff fix-nki-kernel-compilation -- <both files>` → empty (byte-identical to reference).
- `from helion._compiler.nki_fusion import annotate_fx_graph; from helion.language._nki_dim_access import DimAccess, IndirectAP, DynamicAP` → OK.
- Regression: `import helion` still OK; backends unchanged `['triton','pallas','cute','tileir','metal']` (nki not yet registered — correct, that's P1.5/P1.6).

**Deviations/follow-ups:** none.

## P1.2 — Add record_fx_node_ast / ast_for_fx_node to CodegenInterface

**File:** `helion/_compiler/helper_function.py`

**What & why:** The NKI load/store codegen in `memory_ops.py` calls `ast_for_fx_node`/`record_fx_node_ast`
on the codegen interface 32 times (all inside function bodies — import-safe). Upstream's `CodegenInterface`
does not define them. Added both methods verbatim from the reference, immediately after `lift()`.
Both are no-op defaults on the base (record = pass, ast_for = return None); the real recording is wired in
`generate_ast.py` (P1.10). Kept upstream's `statement_owner_node` (and its `contextlib`/`Iterator`/`Node`
imports) untouched — the reference branch predates that method, but all three coexist cleanly.

**Insertion point:** after `CodegenInterface.lift`, before the module-level `extract_helper_function`.

**Verification (gate passed):**
- Inserted text is byte-identical to `fix-nki-kernel-compilation:helper_function.py` for both methods.
- Gate: `CodegenInterface` has all of `record_fx_node_ast`, `ast_for_fx_node`, `statement_owner_node`.
- Regression: `import helion` OK.

**Deviations/follow-ups:** none. (Note: did NOT port the reference's removal of `statement_owner_node` —
that method is upstream-new and orthogonal; keeping it is correct.)

## P1.3 — Reconcile jagged_tile / loops / exc / tunable deltas

**Files:** `helion/language/tunable_ops.py` (edited). **No edit needed:** `loops.py`, `exc.py`, `language/__init__.py`.

**MAJOR FINDING (plan correction):** The plan estimated a "~109-line NKI delta in loops.py." That figure was
the *merge-base → reference* diff, which only shows the reference *adding* `jagged_tile`. In the 829
intervening commits, **upstream independently added its own jagged_tile scaffolding — a strict superset** of
the reference's:
- `loops.py`: upstream's `jagged_tile` type_propagation handler is **byte-identical** to the reference's
  (verified via extracted-body diff); upstream's docstring is richer (more examples). Neither branch's
  `loops.py` contains any `nki`/`backend`-specific code. → **No edit.**
- `exc.py`: `InvalidJaggedTileUsage` already present upstream (L345). → **No edit.**
- `language/__init__.py`: `jagged_tile` export already present upstream (L26). → **No edit.**

**The ONE real NKI delta** (isolated via `merge-base → reference`, not the misleading `upstream → reference`
diff which conflates upstream's own additions like `signal`/`wait`/`CuteBackendUnavailable`): in
`tunable_ops._register_tunable_type`, the reference returns a `SymIntType(origin, env.create_unbacked_symint(default))`
for `int`-typed tunables instead of `NumericType.subtype(int).new_unbacked(origin)`. This makes an int
`register_tunable` participate as a symbolic int (needed so NKI tile/loop bounds derived from tunables get a
size hint). Added the `if python_type is int:` branch + `SymIntType` import.

**Verification (gate passed):**
- Edited region matches the reference verbatim.
- `create_unbacked_symint(hint=8192)` signature accepts the positional `default()` hint (compile_environment L667).
- `SymIntType`/`NumericType` import cleanly from `type_info` (and are re-exported by `type_propagation`).
- Gate: `jagged_tile`, `InvalidJaggedTileUsage`, `JaggedTileIndexType`, `register_tunable` all import.
- No residual mis-targeted `type_propagation` imports in `helion/language/`. Regression: `import helion` OK.

**Deviation from plan:** The `int → SymIntType` change is **unconditional (not NKI-guarded)** in the
reference, so it affects all backends. Applied as-is to match the reference (and per AGENTS.md "don't add
unnecessary guards"). **Open follow-up:** confirm at P1.22/P2 that this doesn't shift Triton codegen for
int tunables; if it does and that matters, revisit guarding. loops.py/exc.py/__init__.py edits from the
plan were correctly dropped.
