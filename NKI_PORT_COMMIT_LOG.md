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
