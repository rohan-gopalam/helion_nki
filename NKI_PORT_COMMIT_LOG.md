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
