# NKI Upstream Merge Guide

This document is a hands-on runbook for porting the Helion NKI backend onto
`upstream/main`. Read it fully before writing a single line of code.

---

## Core philosophy

**Make one small change at a time. Verify the generated kernel code looks correct.
Then commit. Then move to the next change.**

Do not batch multiple logical steps into one edit. If something breaks, you want to
know exactly which change caused it. The generated NKI kernel source is your ground
truth at every step — readable Python code that starts with `@nki.jit`. If it looks
wrong, stop and fix it before proceeding.

Every meaningful step in this guide ends with a verification command. Do not skip
them.

---

## Why not `git merge upstream/main`?

A direct merge produces 81 conflict hunks across 34 files. The mechanical text
conflicts are survivable, but `tile_strategy.py` has 10 semantic conflicts where NKI
injected an alternative codegen path into the middle of a function that upstream also
heavily refactored. A resolved-but-wrong merge will compile silently and produce
incorrect kernel code. The port-forward approach below is safer and produces a
cleaner history.

---

## Background: repository layout

### Merge base
All NKI work diverged from commit `1bfe577d`. The gap:
- Local `fix-nki-kernel-compilation`: **80 commits** ahead of merge base
- `upstream/main`: **829 commits** ahead of merge base

### What the NKI branch added

**NKI-only files** (no upstream equivalent — copy verbatim):
- `helion/_compiler/nki_fusion.py` — FX-graph fusion: marks matmul→activation
  patterns for PSUM reuse. Contains `annotate_fx_graph()` called from `generate_ast.py`.
- `helion/language/_nki_dim_access.py` — `DimAccess` dataclass hierarchy for NKI
  dimension classification in load codegen.
- `helion/autotuner/nki_search.py` — `NKIFiniteSearch` (Phase 3 only).

**NKI code woven into shared files** (the hard part):

| File | NKI additions |
|---|---|
| `helion/_compiler/backend.py` | `NKIOpOverrides` class (lines 636–5808) + `NKIBackend` class (lines 5809–7321) |
| `helion/language/memory_ops.py` | NKI helpers (lines 510–1419), NKI load codegen (1420–4792), NKI store codegen (4793–6337) |
| `helion/_compiler/aten_lowering.py` | 15 `@register_codegen("nki")` handlers |
| `helion/_compiler/inductor_lowering.py` | 4 `backend.codegen_name == "nki"` guards |
| `helion/_compiler/generate_ast.py` | 2 NKI guards + NKI fusion call |
| `helion/_compiler/device_function.py` | 4 NKI guards |
| `helion/_compiler/device_ir.py` | 1 NKI decomp-table guard |
| `helion/_compiler/tile_strategy.py` | 6 NKI guards + full `_codegen_grid_nki` method |
| `helion/_compiler/compile_environment.py` | NKI backend registration + 2 NKI guards |
| `helion/_compiler/output_header.py` | `_default_nki_launcher` in `disallowed_names` |
| `helion/runtime/__init__.py` | `default_nki_launcher` function |
| `helion/runtime/config.py` | `platform_target` field |
| `helion/runtime/settings.py` | `"nki"` in `BackendLiteral`, `platform_target` doc |
| `helion/_testing.py` | XLA device detection, relaxed tolerances, benchmark skip |
| `helion/language/loops.py` | 1 NKI guard in `jagged_tile` |
| `helion/language/matmul_ops.py` | `@codegen(dot, "nki")` |
| `helion/language/view_ops.py` | `@codegen(subscript, "nki")` |
| `helion/language/scan_ops.py` | `@codegen(_associative_scan, "nki")` |
| `helion/language/_tracing_ops.py` | `@codegen(_mask_to, "nki")` |
| `helion/language/random_ops.py` | `@codegen(rand, "nki")` |
| `helion/language/barrier.py` | `@codegen(barrier, "nki")` |
| `helion/language/atomic_ops.py` | `@codegen(atomic_add, "nki")` + 2 direct NKI codegen calls |

### What upstream changed (things to not clobber)

- **New backends**: `helion/_compiler/cute/` (40 files), `helion/_compiler/metal/`
  (7 files), extended Pallas support.
- **`backend_registry.py`** (new file): All backends register through
  `register_compiler_backend()`. `compile_environment.py` uses
  `get_backend_class(name)` instead of a hardcoded dict. **NKI should plug into
  this registry.**
- **`Backend` base class** grew ~30 new virtual methods. `NKIBackend` must implement
  or inherit all abstract ones.
- **`runtime/__init__.py`** grew from 464 → 3,071 lines (Pallas fast-path, CuTe
  launcher, XLA callable refactor). Do not overwrite — add `default_nki_launcher`
  alongside the existing launchers.
- **`memory_ops.py`** grew from 602 → 6,666 lines (CuTe/Metal/Pallas load+store
  codegens). Do not overwrite — append NKI codegens to the end.
- **`settings.py`**: `BackendLiteral` is now driven by
  `backend_registry.list_backends()` dynamically rather than a hardcoded
  `Literal["triton", "pallas", ...]`. This means once `"nki"` is registered, it
  automatically appears in the type.

---

## Understanding the test framework

Before writing any code, understand how the upstream test framework works. All NKI
tests must plug into it.

### The key testing tools

**`to_triton_code(config)`** — generates the NKI kernel source as a Python string.
This is **pure Python, no hardware required**. It runs the entire compilation
pipeline through AST generation and unparsing, producing the `@nki.jit` Python source
that would be compiled by neuronx-cc. Use this for all codegen snapshot tests.

**`code_and_output(kernel_fn, args, **config_kwargs)`** — compiles the kernel _and_
executes it. Requires Trainium hardware. Returns `(code_str, result_tensor)`. Use
this for correctness tests that can only run on device.

**`assertExpectedJournal(code_str)`** — snapshots the generated code to
`test/test_X.expected` (or `test/test_X.expected_nki` for NKI-specific files). On
first run it creates the file. On subsequent runs it diffs against the snapshot.
Regenerate with `EXPECTTEST_ACCEPT=1 pytest test/test_X.py`.

**`@onlyBackends(["nki"])`** — class decorator that skips the entire test class
unless `HELION_BACKEND=nki`. Use this on every NKI test class.

**`@onlyBackends(["triton"])` / `@onlyBackends(["pallas"])`** — the reverse: skip
NKI. Some existing tests use this and will naturally continue to skip when NKI is
the active backend.

### Two tiers of tests

**Tier 1 — Codegen snapshot tests (no hardware)**
```python
@onlyBackends(["nki"])
class TestMyFeature(TestCase):
    def test_generates_dma_copy(self):
        @helion.kernel(config=helion.Config(block_sizes=[128, 64]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_k in hl.tile(x.shape):
                out[tile_m, tile_k] = x[tile_m, tile_k]
            return out

        bound = kernel.bind((torch.zeros(256, 128),))
        code = bound.to_triton_code(bound._config)

        # Assertions about code structure:
        self.assertIn("@nki.jit", code)
        self.assertIn("nisa.dma_copy", code)
        self.assertNotIn("tl.load", code)
        self.assertNotIn("triton.jit", code)

        # Snapshot the full code for regression detection:
        self.assertExpectedJournal(code)
```

**Tier 2 — Correctness tests (requires hardware)**
```python
@onlyBackends(["nki"])
class TestMyFeatureCorrectness(TestCase):
    def test_output_matches_torch(self):
        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def softmax(x: torch.Tensor) -> torch.Tensor:
            ...

        x = torch.randn(256, 512, device=DEVICE)
        code, result = code_and_output(softmax, (x,))
        torch.testing.assert_close(result.cpu().float(),
                                   torch.softmax(x.cpu().float(), dim=-1),
                                   atol=0.05, rtol=0.05)
```

### Expected file naming convention

The `assertExpectedJournal` system uses a naming convention based on the test file:
- `test/test_foo.py` → `test/test_foo.expected` (default backend)
- `test/test_foo.py` → `test/test_foo.expected_nki` (when `HELION_BACKEND=nki`)

The file suffix is determined automatically by the current backend. This means NKI
tests in `test/test_foo.py` will write their snapshots to `test/test_foo.expected_nki`,
keeping them separate from the Triton snapshots in `test/test_foo.expected`.

For a new NKI-specific test file like `test/test_nki_backend.py`, all snapshots go
to `test/test_nki_backend.expected_nki`.

### Running tests

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /home/ubuntu/helion_nki

# Run a single test file (codegen tests only — no hardware needed):
HELION_BACKEND=nki python -m pytest test/test_nki_backend.py -v

# Run with snapshot acceptance (first time, or after intentional changes):
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py -v

# Run a single test method:
HELION_BACKEND=nki python -m pytest test/test_nki_backend.py::TestNKIBackend::test_matmul_codegen -v

# Run existing NKI tests:
HELION_BACKEND=nki python -m pytest test/test_nki_dynamic_loops.py test/test_nki_load_refactor.py -v
```

---

## Phase 1: Port NKI onto upstream/main

### Step 0: Create the branch

```bash
cd /home/ubuntu/helion_nki
git fetch upstream
git checkout -b nki-port-v2 upstream/main
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
pip install -e '.[dev]'
```

Verify the baseline:
```bash
python -c "import helion; print(helion.__version__)"
python -m pytest test/test_matmul.py -v -k "test_matmul_basic" --timeout=30
```
Both should pass. If the second fails, stop and fix the baseline before touching
anything NKI-related.

---

### Step 1: Copy NKI-only files

No conflicts here — these files have no upstream equivalent.

```bash
git checkout fix-nki-kernel-compilation -- \
  helion/_compiler/nki_fusion.py \
  helion/language/_nki_dim_access.py
```

**Verify:** Both files exist and import cleanly:
```bash
python -c "from helion._compiler.nki_fusion import annotate_fx_graph; print('ok')"
python -c "from helion.language._nki_dim_access import DimAccess; print('ok')"
```

**Commit:**
```bash
git add helion/_compiler/nki_fusion.py helion/language/_nki_dim_access.py
git commit -m "port: add NKI-only files (nki_fusion, _nki_dim_access)"
```

---

### Step 2: Create `helion/_compiler/nki_backend.py`

This is the centerpiece of the port. Extract `NKIOpOverrides` and `NKIBackend` from
the old `backend.py` into a new standalone file.

**Read before editing:**
1. `fix-nki-kernel-compilation:helion/_compiler/backend.py` lines 636–7321. This is
   the full NKI implementation. The class boundaries are:
   - `NKIOpOverrides`: lines 636–5808
   - `NKIBackend`: lines 5809–7321
2. `upstream/main:helion/_compiler/backend.py` lines 66–946. The new `Backend` base
   class — read every method and its default implementation.
3. `upstream/main:helion/_compiler/backend.py` lines 4663–4978. The `MetalBackend`
   class — it is the simplest full `Backend` implementation and shows exactly what
   methods a new backend must provide.
4. `upstream/main:helion/_compiler/backend_registry.py`. The registration pattern.

**Create `helion/_compiler/nki_backend.py`:**

Structure of the new file:
```python
"""NKI (Neural Kernel Interface) backend for AWS Trainium."""
from __future__ import annotations

# --- all imports from the current backend.py that NKIOpOverrides/NKIBackend use ---

class NKIOpOverrides:
    # copy verbatim from current backend.py lines 636-5808

class NKIBackend(Backend):
    # copy verbatim from current backend.py lines 5809-7321

# Registration — plug into the upstream backend registry
from .backend_registry import register_compiler_backend
register_compiler_backend(NKIBackend)
```

**Adapt `NKIBackend` to the new `Backend` API:**

The upstream `Backend` base class added ~30 new methods since the merge base. Most
have reasonable defaults that NKI can inherit. Check each abstract method (marked
`@abc.abstractmethod`) and confirm `NKIBackend` already implements it:
- `name` → returns `"nki"` ✓
- `dtype_str` → already implemented ✓
- `acc_type` → already implemented ✓
- `function_decorator` → returns `"nki.jit"` ✓
- `constexpr_type` → already implemented ✓
- `default_launcher_name` → returns `"_default_nki_launcher"` ✓
- `library_imports` → already implemented ✓

For new non-abstract methods, check each one in `MetalBackend` as a reference:
- `experimental` → return `True`
- `supports_precompile` → return `False`
- `validate_environment` → check `compile_environment.py` in the current branch for
  any NKI startup checks; if any exist, move them here
- `max_tensor_numel` → return `None` (shape validation is done by
  `validate_nki_tensor_shapes` at codegen time)
- `process_fake_tensor_load` → check current `compile_environment.py` around line
  555 for NKI int64 casting logic; if it can move here, do it; otherwise leave it
  in `compile_environment.py` and return the default here
- `create_synthetic_reduction_lanes` → return `None` (NKI does not use threads)
- `adjust_block_size_constraints` → return constraints unchanged for now
- `get_do_bench`, `get_interleaved_bench`, `get_paired_device_micros_bench` → return
  `None` (Phase 3 handles autotuning timing)
- `setup_compile_cache_dir`, `make_ephemeral_cache`, `finalize_ephemeral_cache` →
  NKI has its own Neuron compile cache; check if current `NKIBackend` implements
  these and copy them; otherwise return defaults

**Verify — import only (no hardware needed):**
```bash
python -c "from helion._compiler.nki_backend import NKIBackend; print('ok')"
python -c "from helion._compiler.nki_backend import NKIOpOverrides; print('ok')"
```

**Commit:**
```bash
git add helion/_compiler/nki_backend.py
git commit -m "port: create nki_backend.py with NKIOpOverrides and NKIBackend"
```

---

### Step 3: Register NKI in `backend_registry.py`

Open `helion/_compiler/backend_registry.py`. At the bottom of the file, after the
existing `_BUILTIN_BACKENDS` loop, add:

```python
# NKI backend — optional plugin; absent when torch_xla is not installed
def _maybe_register_nki() -> None:
    try:
        from .nki_backend import NKIBackend  # noqa: F401  (side-effect: registers itself)
    except ImportError:
        pass

_maybe_register_nki()
```

The `nki_backend.py` module calls `register_compiler_backend(NKIBackend)` at import
time, so importing it is sufficient.

**Verify:**
```bash
python -c "
from helion._compiler.backend_registry import list_backends
print(list_backends())  # should include 'nki'
"
```

Also verify non-NKI backends still work:
```bash
python -m pytest test/test_matmul.py -v -k "test_matmul_basic" --timeout=30
```

**Commit:**
```bash
git add helion/_compiler/backend_registry.py
git commit -m "port: register NKIBackend in backend_registry"
```

---

### Step 4: Wire `compile_environment.py`

The current branch has:
```python
from .backend import NKIBackend
...
"nki": NKIBackend,
```

The upstream version uses `get_backend_class(settings.backend)()` from the registry —
so no manual registration is needed in this file. Remove the `NKIBackend` import if it
is present in the upstream version (it won't be, but check).

Then find the two NKI-specific guards in the current `compile_environment.py` and
verify they are present (or add them if not):

1. **int64 fake-tensor promotion** (around line 555 of current branch):
   ```python
   if self.backend_name == "nki" and fake_dtype == torch.int64:
       fake_dtype = torch.int32
   ```
   Find the fake-tensor construction path in upstream and add this guard in the same
   relative position.

2. **NKI reduction DMA shape check** (around line 877):
   ```python
   if env.backend_name == "nki": ...
   ```
   Read the full context and add to the equivalent location in upstream.

3. **`_codegen_state` mechanism** — NKI uses `env.set_codegen_state(state)` so that
   statement-based codegen can emit statements during lowering. Search for
   `set_codegen_state` in upstream's `compile_environment.py`. If present, no change
   needed. If absent, add the method from the current branch (it is about 15 lines).

**Verify:**
```bash
python -c "
import os; os.environ['HELION_BACKEND'] = 'nki'
from helion._compiler.compile_environment import CompileEnvironment
print('compile_environment ok')
"
```

**Commit:**
```bash
git add helion/_compiler/compile_environment.py
git commit -m "port: add NKI guards to compile_environment.py"
```

---

### Step 5: `settings.py` and `config.py`

**`settings.py`:** Open both the current branch and upstream versions side by side.

1. **Backend list** — In upstream, `_get_backend()` calls `list_backends()` from the
   registry dynamically. Once Step 3 is done, `"nki"` appears automatically. Verify:
   ```bash
   python -c "
   import os; os.environ['HELION_BACKEND'] = 'nki'
   from helion.runtime.settings import Settings
   s = Settings()
   print(s.backend)  # should be 'nki'
   "
   ```

2. **`platform_target` field** — The current branch adds this to `Settings` (line
   ~417) and its docstring (line ~657). Check upstream's `Settings` dataclass. If
   the field is absent, add:
   ```python
   platform_target: str | None = None
   ```
   and its docstring entry:
   ```python
   "platform_target": "The hardware platform to compile for when using the NKI backend (e.g. 'trn1', 'trn2').",
   ```

**`config.py`:** The current branch adds `platform_target: str | None` to `Config`
(around lines 21 and 83). Check upstream's `Config`. If absent, add the field.

**Commit after both files:**
```bash
git add helion/runtime/settings.py helion/runtime/config.py
git commit -m "port: add platform_target to Settings and Config for NKI"
```

---

### Step 6: Add `default_nki_launcher` to `runtime/__init__.py`

The upstream `runtime/__init__.py` grew 6× — do not overwrite it. Add the NKI
launcher alongside the existing ones.

**Read first:** The current branch `helion/runtime/__init__.py` lines 132–200 (the
full `default_nki_launcher` function). The upstream version: find `default_launcher`
(line 162) to understand where to insert.

**Add the function** after `default_launcher` and before the Pallas launcher section.
Copy the implementation verbatim from the current branch.

**Also add `default_nki_launcher` to `output_header.py`:** Open
`helion/_compiler/output_header.py` and add `"_default_nki_launcher"` to the
`disallowed_names` dict (the current branch has this at line 37). This prevents user
kernels from shadowing the launcher name.

**Verify — import only:**
```bash
python -c "from helion.runtime import default_nki_launcher; print('ok')"
```

**Commit:**
```bash
git add helion/runtime/__init__.py helion/_compiler/output_header.py
git commit -m "port: add default_nki_launcher and output_header disallowed name"
```

---

### Step 7: Add `_testing.py` changes

The current branch adds four NKI-specific behaviors to `helion/_testing.py`. Read
both versions side by side, then add each one to the upstream version:

1. **XLA device detection** — In the `DEVICE` detection block (upstream has `cuda`,
   `tpu`, `mps`, `xpu`, `mtia`, `cpu`), add:
   ```python
   if _get_backend() == "nki":
       DEVICE = None  # NKI launcher moves tensors to XLA device
   ```
   Place this before the `cuda` check so it takes priority when the env var is set.

2. **Relaxed tolerances** — In `check_example` or wherever `atol`/`rtol` are set,
   add NKI overrides (current branch around line 953):
   ```python
   rtol = 5e-2 if _get_backend() == "nki" else 1e-2
   atol = 1.5 if _get_backend() == "nki" else 6e-2
   ```

3. **Benchmark skip** — In the benchmarking section (current branch around line 970):
   ```python
   if _get_backend() == "nki":
       print("Skipping benchmark on NKI backend.", file=sys.stderr)
       return
   ```

4. **Baseline dict for NKI** — (current branch around line 871): When
   `_get_backend() == "nki"` and `baseline_fn` is a dict, select the NKI baseline.
   Read the full context and copy the guard.

**Verify:**
```bash
python -c "
import os; os.environ['HELION_BACKEND'] = 'nki'
from helion._testing import DEVICE
print(f'DEVICE={DEVICE}')  # should be None
"
```

**Commit:**
```bash
git add helion/_testing.py
git commit -m "port: add NKI device detection and tolerance overrides to _testing.py"
```

---

### Step 8: Create `test/test_nki_backend.py` — the incremental test file

**Create this file now, before touching any codegen.** It will grow alongside the
implementation. Starting empty is fine — each subsequent step adds tests to it.

```python
"""
Incremental tests for the NKI backend port.

Codegen tests (Tier 1): use to_triton_code, no hardware needed.
Correctness tests (Tier 2): use code_and_output, require Trainium.

Run codegen tests: HELION_BACKEND=nki python -m pytest test/test_nki_backend.py -v
Accept new snapshots: HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py -v
"""
from __future__ import annotations

import os
os.environ.setdefault("HELION_BACKEND", "nki")

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


def _nki_code(kernel_fn: helion.Kernel, args: tuple, **config_kwargs) -> str:
    """Compile to NKI source without executing. No hardware required."""
    bound = kernel_fn.bind(args)
    config = helion.Config(**config_kwargs) if config_kwargs else bound._config
    return bound.to_triton_code(config)


# Tests are added below as each port step is completed.
```

**Commit:**
```bash
git add test/test_nki_backend.py
git commit -m "port: create test/test_nki_backend.py skeleton"
```

---

### Step 9: Wire `device_ir.py`

One guard to add. Open both versions side by side.

**Current branch line 94:**
```python
if CompileEnvironment.current().backend_name == "nki":
    decomp_table.pop(torch.ops.aten.silu.default, None)
```
**Why:** NKI has a native `silu` path; the default `x * sigmoid(x)` decomposition
aliases operands and breaks statement-based in-place NKI codegen.

Find the function that builds the ATen decomposition table in upstream's `device_ir.py`
(search for `decomp_table`). Add the guard in the same relative position.

**Add a codegen test to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKIDeviceIR(TestCase):
    def test_silu_not_decomposed(self):
        """silu must not be decomposed to x*sigmoid(x) in NKI codegen."""
        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = torch.nn.functional.silu(x[tile])
            return out

        x = torch.zeros(512, device=DEVICE)
        code = _nki_code(kernel, (x,))
        # silu should appear as a single NKI activation call, not x*sigmoid(x)
        self.assertNotIn("sigmoid", code)
        self.assertIn("nki", code.lower())
```

**Run the test:**
```bash
HELION_BACKEND=nki python -m pytest test/test_nki_backend.py::TestNKIDeviceIR -v
```

**Commit:**
```bash
git add helion/_compiler/device_ir.py test/test_nki_backend.py
git commit -m "port: add NKI silu decomp guard to device_ir.py"
```

---

### Step 10: Wire `generate_ast.py`

Three additions. Read both versions in full before editing.

**Addition 1 (current line 587):**
```python
is_nki = env.backend.name == "nki"
if len(self.host_function.device_ir.root_ids) == 1 or is_nki:
    body = self.device_function.body
```
NKI kernels always use a single body block even with multiple root IDs.

**Addition 2 (current line 665):** NKI-specific pre-codegen passes:
```python
if env.backend.name == "nki":
    env.backend.validate_nki_tensor_shapes(root)
    from .nki_fusion import annotate_fx_graph
    annotate_fx_graph(root)
```
Add immediately before `env.set_codegen_state(state)`.

**Addition 3 (current lines 94 and 309):** `codegen_name != "nki"` guards that skip
Triton-specific paths. Find them in the current branch, read the context, add them
in the equivalent locations in upstream.

**Add to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKIGenerateAST(TestCase):
    def test_generates_nki_jit_decorator(self):
        """The generated code must have @nki.jit, not @triton.jit."""
        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def copy_kernel(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                out[tile] = x[tile]
            return out

        code = _nki_code(copy_kernel, (torch.zeros(512, device=DEVICE),))
        self.assertIn("@nki.jit", code)
        self.assertNotIn("@triton.jit", code)
        self.assertNotIn("tl.load", code)
        self.assertNotIn("tl.store", code)
        self.assertExpectedJournal(code)
```

**Run:**
```bash
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py::TestNKIGenerateAST -v
```
The first run with `EXPECTTEST_ACCEPT=1` captures the baseline. Subsequent runs
without it assert no regression.

**Commit:**
```bash
git add helion/_compiler/generate_ast.py test/test_nki_backend.py
git commit -m "port: add NKI guards to generate_ast.py"
```

---

### Step 11: Wire `device_function.py`

Four additions. Read both versions in full before editing.

**Addition 1 — `_register_nki_dynamic_tensor_size_args` method (current line 796):**
Adds tensor size arguments for NKI dynamic shapes. Check if upstream's
`DeviceFunction` has this method. If not, add the full method.

**Addition 2 — NKI tensor shape normalization (current line 969):**
```python
if backend.name == "nki":
    # Normalize tensor argument views at kernel entry to the 2D logical shape
```
Find the function argument codegen section and add this block.

**Addition 3 — NKI SBUF reassignment rewriting (current line 1041):**
```python
if backend.name == "nki":
    self._rewrite_nki_sbuf_reassignments(fn_def.body, {}, 0)
```
Add after `fn_def` is constructed.

**Addition 4 — NKI sympy printer (current line 1249):**
```python
if env.backend.name == "nki":
    return HelionNKIPrinter().doprint(expr)
```
Add as early-return in the expression-printing method.

**Add to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKIDeviceFunction(TestCase):
    def test_kernel_has_correct_signature(self):
        """NKI kernel function must accept tensor args, not Triton-style constexpr."""
        @helion.kernel(config=helion.Config(block_sizes=[128, 64]))
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            _, N = y.shape
            out = torch.zeros([M, N], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (torch.zeros(256, 128, device=DEVICE), torch.zeros(128, 128, device=DEVICE))
        code = _nki_code(matmul, args)
        self.assertIn("@nki.jit", code)
        self.assertIn("nisa.nc_matmul", code)
        self.assertExpectedJournal(code)
```

**Run:**
```bash
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py::TestNKIDeviceFunction -v
```
Read the generated code in `test/test_nki_backend.expected_nki`. Verify it looks like
correct NKI Python: `@nki.jit`, `nl.ndarray`, `nisa.nc_matmul`, no Triton imports.

**Commit:**
```bash
git add helion/_compiler/device_function.py test/test_nki_backend.py
git commit -m "port: add NKI guards to device_function.py"
```

---

### Step 12: Wire `tile_strategy.py`

This is the most complex file. Six additions plus the full `_codegen_grid_nki` method.
Read both versions in full before touching anything.

**Understand the structure first.** In the current branch, `tile_strategy.py` contains
`NDTileStrategy` (and related classes). Search for `class NDTileStrategy` and
`def codegen_grid` to find the method. `_codegen_grid_nki` is a method on the same
class.

**Addition 1 — block size constexpr literal (current line 510):**
```python
if env.backend.name == "nki" and block_idx is not None:
    literal = env.block_sizes[block_idx].from_config_assert(state.config)
    state.device_function.block_size_var_cache[(block_idx,)] = str(int(literal))
    return
```
Inside the method that sets up block size constexprs. Add before `state.device_function.constexpr_arg_with_host_def(...)`.

**Addition 2 — NKI grid dispatch (current line 1002):**
```python
if env.backend.name == "nki":
    return self._codegen_grid_nki(state, block_ids, block_sizes, begins, ends)
```
Inside `codegen_grid` (or equivalent), after `begins`/`ends` are resolved and before
the Triton `pids = self.select_pid_strategy()` path. Then add the full
`_codegen_grid_nki` method to the same class — copy it verbatim from the current
branch.

**Additions 3–6** — NKI loop bound and dynamic range handling (current lines 1413,
1656, 1678). Read the full context around each guard in the current branch, find the
equivalent locations in upstream, and add them. These are inside the loop body
emission loop.

**Add to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKITileStrategy(TestCase):
    def test_matmul_uses_affine_range(self):
        """NKI matmul tiling must use nl.affine_range, not tl.range."""
        @helion.kernel(config=helion.Config(block_sizes=[128, 128, 128]))
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            _, N = y.shape
            out = torch.zeros([M, N], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (torch.zeros(256, 256, device=DEVICE), torch.zeros(256, 256, device=DEVICE))
        code = _nki_code(matmul, args)
        self.assertIn("nl.affine_range", code)
        self.assertNotIn("tl.range", code)
        self.assertExpectedJournal(code)

    def test_softmax_tiling_structure(self):
        """NKI softmax must tile the row dimension with affine_range."""
        @helion.kernel(config=helion.Config(block_sizes=[128]))
        def softmax(x: torch.Tensor) -> torch.Tensor:
            M, N = x.shape
            out = torch.empty_like(x)
            for tile_m in hl.tile(M):
                row = x[tile_m, :]
                out[tile_m, :] = row - row.max()
            return out

        args = (torch.zeros(512, 128, device=DEVICE),)
        code = _nki_code(softmax, args)
        self.assertIn("nl.affine_range", code)
        self.assertIn("nisa.dma_copy", code)
        self.assertExpectedJournal(code)
```

**Run and inspect:**
```bash
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py::TestNKITileStrategy -v
```
Open `test/test_nki_backend.expected_nki` and read the generated code. Verify:
- The function starts with `@nki.jit`
- Tiling uses `nl.affine_range`
- Load/store uses `nisa.dma_copy`
- No Triton or Python `range()` calls for the tiled loops

**Commit:**
```bash
git add helion/_compiler/tile_strategy.py test/test_nki_backend.py
git commit -m "port: add NKI grid codegen to tile_strategy.py"
```

---

### Step 13: Add NKI load/store to `memory_ops.py`

This is the largest single addition (~5,800 lines) but also the most mechanical — it
is all new code appended to the end of an existing file.

**Read first:**
- Current branch `helion/language/memory_ops.py` lines 510–6337 (the NKI additions)
- Upstream `helion/language/memory_ops.py` lines 6189–6666 (the CuTe load codegen,
  which is the last section — NKI goes after it)

**Add in three blocks to the upstream file:**

**Block 1** — NKI helper functions (copy current branch lines 510–1419):
- `_nki_shifted_tile_subscript`
- `_nki_indirect_gather`
- `_nki_lookup_sbuf_shape_dtype`
- `_nki_as_uint32_p1_vector`
- `_nki_row_index_gather`
- `_nki_subscript_block_id`

Place these after the existing shared helper functions and before the first
backend-specific codegen section (the Triton `@_decorators.codegen(store, "triton")`).

**Block 2** — NKI load codegen (copy current branch lines 1420–4792):
```python
@_decorators.codegen(load, "nki")
def _(state: CodegenState) -> ast.AST:
    ...  # 3,373 lines
```
Place after the CuTe `@_decorators.codegen(load, "cute")` at the end of the file.

**Block 3** — NKI store codegen (copy current branch lines 4793–6337):
```python
@_decorators.codegen(store, "nki")
def _(state: CodegenState) -> ast.AST:
    ...  # 1,542 lines
```
Place after the NKI load codegen.

**After adding, verify the import structure:** The NKI load/store functions import
from `_nki_dim_access.py` and `nki_fusion.py`. Verify these imports work:
```bash
python -c "from helion.language import memory_ops; print('ok')"
```

**Add load codegen test to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKILoadStore(TestCase):
    def test_2d_contiguous_load(self):
        """Simple 2D load generates nisa.dma_copy without ap()."""
        @helion.kernel(config=helion.Config(block_sizes=[128, 64]))
        def copy2d(x: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(x)
            for tile_m, tile_k in hl.tile(x.shape):
                out[tile_m, tile_k] = x[tile_m, tile_k]
            return out

        args = (torch.zeros(256, 128, device=DEVICE),)
        code = _nki_code(copy2d, args)
        self.assertIn("nisa.dma_copy", code)
        self.assertNotIn(".ap(", code)
        self.assertExpectedJournal(code)

    def test_matmul_load_uses_nc_matmul(self):
        """Matmul loads must use nisa.nc_matmul, not nisa.dma_copy."""
        @helion.kernel(config=helion.Config(block_sizes=[128, 128, 128]))
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            _, N = y.shape
            out = torch.zeros([M, N], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (torch.zeros(256, 256, device=DEVICE), torch.zeros(256, 256, device=DEVICE))
        code = _nki_code(matmul, args)
        self.assertIn("nisa.nc_matmul", code)
        self.assertExpectedJournal(code)
```

**Run:**
```bash
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py::TestNKILoadStore -v
```

**Commit:**
```bash
git add helion/language/memory_ops.py test/test_nki_backend.py
git commit -m "port: add NKI load/store codegens to memory_ops.py"
```

---

### Step 14: Add NKI language codegens

Seven files, each a small targeted addition. For each one: read both versions,
find where the other backends' codegens are registered, add the NKI one in the
same style.

**`helion/language/matmul_ops.py`** — `@codegen(dot, "nki")` at line 277.
**`helion/language/view_ops.py`** — `@codegen(subscript, "nki")` at line 109.
**`helion/language/scan_ops.py`** — `@codegen(_associative_scan, "nki")` at line 351.
**`helion/language/_tracing_ops.py`** — `@codegen(_mask_to, "nki")` at line 451.
**`helion/language/random_ops.py`** — `@codegen(rand, "nki")` at line 144.
**`helion/language/barrier.py`** — `@codegen(barrier, "nki")` at line 54.
**`helion/language/atomic_ops.py`** — `@codegen(atomic_add, "nki")` at line 330,
plus two direct NKI codegen calls at lines 678 and 690.

After adding all seven, run:
```bash
python -c "import helion.language; print('language ok')"
```

**Add a language codegen test:**
```python
@onlyBackends(["nki"])
class TestNKILanguageCodgens(TestCase):
    def test_dot_uses_nc_matmul(self):
        """hl.dot must lower to nisa.nc_matmul on NKI."""
        @helion.kernel(config=helion.Config(block_sizes=[128, 128, 128]))
        def dot_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            M, K = x.shape
            _, N = y.shape
            out = torch.zeros([M, N], device=x.device, dtype=x.dtype)
            for tile_m, tile_n in hl.tile([M, N]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(K):
                    acc += hl.dot(x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        args = (torch.zeros(256, 256, device=DEVICE), torch.zeros(256, 256, device=DEVICE))
        code = _nki_code(dot_kernel, args)
        self.assertIn("nisa.nc_matmul", code)
        self.assertExpectedJournal(code)
```

**Commit:**
```bash
git add helion/language/matmul_ops.py helion/language/view_ops.py \
        helion/language/scan_ops.py helion/language/_tracing_ops.py \
        helion/language/random_ops.py helion/language/barrier.py \
        helion/language/atomic_ops.py test/test_nki_backend.py
git commit -m "port: add NKI codegens to language modules"
```

---

### Step 15: Wire `aten_lowering.py`

15 `@register_codegen("nki")` handlers. These are self-contained and can be copied
verbatim. For each: find where the same `X_lowering` object is defined in upstream,
add the NKI handler after the last existing handler for that op.

The 15 ops (file locations in current branch):
`full_lowering` (173), `unsqueeze_lowering` (403), `squeeze_lowering` (427),
`view_lowering` (435), `reshape_lowering` (436), `permute_lowering` (534),
`stack_lowering` (746), `expand_lowering` (919), `silu_lowering` (1099),
`mm_lowering` (1737), `addmm_lowering` (1742), `bmm_lowering` (1747),
`baddbmm_lowering` (1752), `cumsum_lowering` (1760), `iota_lowering` (1865).

**Check for `stack_lowering`:** This may have been added in the NKI branch. If
upstream does not have `stack_lowering`, add the registration object itself (a
`Lowering` instance) not just the NKI handler.

**Commit:**
```bash
git add helion/_compiler/aten_lowering.py
git commit -m "port: add 15 NKI codegen registrations to aten_lowering.py"
```

---

### Step 16: Wire `inductor_lowering.py`

Four guards. Read both versions in full. Add each guard in the equivalent location:

1. **Activation ops fail-fast** (current line 907): `_NKI_ACTIVATION_NAMES` set and
   `BackendUnsupported` raise.
2. **Int dtype cast** (current line 963): NKI-specific int cast in dtype conversion
   path.
3. **Three NKI codegen paths** (current lines 1022, 1027, 1032): In the element-wise
   op dispatch. Search for the function that contains them in the current branch and
   find the equivalent in upstream.

**Commit:**
```bash
git add helion/_compiler/inductor_lowering.py
git commit -m "port: add NKI guards to inductor_lowering.py"
```

---

### Step 17: Smoke test — first end-to-end kernel

At this point all the wiring should be in place. Run the smoke test:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_portv2_cache \
  python -c "
import torch
import helion
import helion.language as hl

@helion.kernel(config=helion.Config(block_sizes=[128]))
def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(M):
        row = x[tile_m, :]
        out[tile_m, :] = row / row.sum()
    return out

x = torch.randn(256, 512)
result = softmax(x)
print('result shape:', result.shape)
print('smoke test PASSED')
"
```

**If it fails, consult this triage table:**

| Error | Most likely cause | Where to look |
|---|---|---|
| `Unknown backend: 'nki'` | Registry not populated | `backend_registry.py` Step 3 |
| `AttributeError: NKIBackend has no method X` | New abstract method in `Backend` base | `nki_backend.py` Step 2 |
| `ImportError: cannot import name 'NKIOpOverrides'` | Broken import chain | `nki_backend.py` imports section |
| `TypeError: NKI only supports LNC 1 or 2` | Old `nki_kernel[grid]` call | `runtime/__init__.py` Step 6 |
| `ValueError: nki.jit 'platform_target' is deprecated` | `function_decorator` returns wrong string | `nki_backend.py` `function_decorator` — must return `"nki.jit"` with no args |
| Generated code has `@triton.jit` | `generate_ast.py` guard missing | Step 10 |
| Generated code has `tl.load` | `memory_ops.py` NKI handler not registered | Step 13 |
| Generated code has `tl.arange` instead of `nisa.iota` | Language codegen missing | Step 14 |
| `AttributeError: DeviceFunction has no attr _nki_sbuf_shapes` | `device_function.py` initialization guard missing | Step 11 |

**Run the existing NKI tests to check for regressions:**
```bash
HELION_BACKEND=nki python -m pytest test/test_nki_dynamic_loops.py test/test_nki_load_refactor.py -v
```

**Commit the smoke test result:**
```bash
git add test/test_nki_backend.py
git commit -m "port: Phase 1 complete — NKI backend wired onto upstream/main"
```

---

### Step 18: Add examples and update test coverage

Now that the backend runs, add tests for the key examples. These mirror the structure
of upstream's `test/test_examples.py` but for NKI:

**Add to `test/test_nki_backend.py`:**
```python
@onlyBackends(["nki"])
class TestNKIExamples(TestCase):
    """Codegen snapshot tests for NKI examples. No hardware required."""

    def test_softmax_codegen(self):
        """softmax example generates valid NKI code."""
        from helion._testing import check_example, EXAMPLES_DIR, import_path
        kernel_fn = import_path(EXAMPLES_DIR / "softmax.py").softmax
        args = (torch.zeros(256, 512, device=DEVICE),)
        bound = kernel_fn.bind(args)
        code = bound.to_triton_code(helion.Config(block_sizes=[128]))
        self.assertIn("@nki.jit", code)
        self.assertIn("nisa.dma_copy", code)
        self.assertExpectedJournal(code)

    def test_matmul_codegen(self):
        """matmul example generates nc_matmul calls."""
        from helion._testing import EXAMPLES_DIR, import_path
        kernel_fn = import_path(EXAMPLES_DIR / "matmul.py").matmul
        args = (torch.zeros(256, 256, device=DEVICE), torch.zeros(256, 256, device=DEVICE))
        bound = kernel_fn.bind(args)
        code = bound.to_triton_code(helion.Config(block_sizes=[128, 128, 128]))
        self.assertIn("nisa.nc_matmul", code)
        self.assertExpectedJournal(code)

    def test_layer_norm_codegen(self):
        from helion._testing import EXAMPLES_DIR, import_path
        kernel_fn = import_path(EXAMPLES_DIR / "layer_norm.py").layer_norm
        args = (torch.zeros(256, 512, device=DEVICE),
                torch.ones(512, device=DEVICE),
                torch.zeros(512, device=DEVICE))
        bound = kernel_fn.bind(args)
        code = bound.to_triton_code(helion.Config(block_sizes=[128]))
        self.assertIn("@nki.jit", code)
        self.assertExpectedJournal(code)
```

Accept snapshots:
```bash
HELION_BACKEND=nki EXPECTTEST_ACCEPT=1 python -m pytest test/test_nki_backend.py::TestNKIExamples -v
```

Read every generated snapshot in `test/test_nki_backend.expected_nki`. Verify the
code is valid NKI Python:
- Starts with `from helion.runtime import default_nki_launcher as _default_nki_launcher`
- `@nki.jit` decorator
- `import nki.language as nl`, `import nki.isa as nisa`
- `nl.ndarray` for SBUF allocation
- `nisa.dma_copy` for loads/stores
- `nisa.nc_matmul` for matmuls
- `nl.affine_range` for tiled loops
- Launcher calls `_default_nki_launcher(kernel, grid, *args)`

---

## Phase 2: Full example sweep

Once the smoke test and `TestNKIExamples` pass, run the full suite:

```bash
nohup bash -c '
  source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
  HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_portv2_sweep \
    python -u examples/run_nki_examples.py
' > /tmp/sweep_v2.txt 2>&1 &
tail -f /tmp/sweep_v2.txt
```

For each failure, add a regression test to `test/test_nki_backend.py` that captures
the codegen and asserts the expected structure, **before** fixing the bug. This
ensures the fix is verified and won't regress.

**Known pre-existing failures from the old branch** (may still apply):
- `gdn_fwd_h.py` — strided 4D tensor indexing
- `int4_gemm.py` — bitwise ops on int8
- `jagged_hstu_attn.py` — 5D scatter with SBUF row index
- `jagged_mean.py` — complex 3D gather semantics
- `nvfp4_gemm.py` — fp4 packing DMA mismatch

Do not spend time on these unless the sweep reveals new regressions — they were known
failures on the old branch and are not introduced by this port.

**Clear the Neuron cache between code changes:**
```bash
rm -rf /tmp/helion_nki_portv2_sweep/*
```

**Never run parallel test processes** while a Neuron compile is running — it causes
NRT failures.

---

## Phase 3: Autotuning

After Phase 2 is stable:

```bash
git checkout fix-nki-kernel-compilation -- helion/autotuner/nki_search.py
```

Wire `NKIFiniteSearch` into the upstream autotuner:
1. Read `helion/autotuner/__init__.py` in upstream — find where autotuners are
   registered and how `effort_level` maps to a search strategy.
2. Check if `NKIBackend.get_do_bench()` needs to return an NKI timing function
   (wrapping `xm.mark_step()`). Implement if needed.
3. Register `NKIFiniteSearch` so `autotune_effort="quick"` picks it when the backend
   is NKI.
4. Add tests to `test/test_nki_backend.py` verifying that autotuning generates
   multiple configs and selects the best one.

---

## Invariants to preserve throughout

1. **NKI code is a plugin.** Core files (`backend.py`, `settings.py`,
   `runtime/__init__.py`) must not have hard imports of NKI-specific classes at
   module load time. All NKI logic is in `nki_backend.py` and the NKI-only files.

2. **No `torch_xla` import at module load time** in any shared file. All `torch_xla`
   imports are lazy (inside `default_nki_launcher` and `NKIBackend` methods).

3. **One commit per logical step.** Do not squash steps. If a step produces wrong
   kernel output, revert to the previous commit and try again.

4. **Read the generated kernel before committing.** For every codegen change, run
   `to_triton_code` and inspect the output. Wrong NKI code compiles silently and
   produces wrong numerics.

5. **`NEURON_PLATFORM_TARGET_OVERRIDE=trn2`** must be set for all compile and run
   commands on this hardware.

6. **Clear the Neuron cache** whenever you change codegen logic:
   ```bash
   rm -rf /tmp/helion_nki_portv2_*
   ```

7. **`EXPECTTEST_ACCEPT=1` is only for capturing new baselines.** Never run it on a
   test that was previously passing without reading the diff — it would silently
   accept a regression.

---

## Quick file reference

**New files to create:**
- `helion/_compiler/nki_backend.py`
- `test/test_nki_backend.py`

**Files to copy verbatim from `fix-nki-kernel-compilation`:**
- `helion/_compiler/nki_fusion.py`
- `helion/language/_nki_dim_access.py`
- (Phase 3) `helion/autotuner/nki_search.py`

**Files to modify (targeted additions to the upstream version):**
- `helion/_compiler/backend_registry.py` — register NKIBackend
- `helion/_compiler/compile_environment.py` — 3 NKI additions
- `helion/_compiler/aten_lowering.py` — 15 NKI codegen registrations
- `helion/_compiler/inductor_lowering.py` — 4 NKI guards
- `helion/_compiler/generate_ast.py` — 3 NKI additions
- `helion/_compiler/device_function.py` — 4 NKI additions
- `helion/_compiler/device_ir.py` — 1 NKI guard
- `helion/_compiler/tile_strategy.py` — 6 NKI guards + `_codegen_grid_nki` method
- `helion/_compiler/output_header.py` — 1-line addition to `disallowed_names`
- `helion/runtime/__init__.py` — add `default_nki_launcher` function
- `helion/runtime/config.py` — add `platform_target` field
- `helion/runtime/settings.py` — add `platform_target` doc
- `helion/_testing.py` — 4 NKI additions
- `helion/language/memory_ops.py` — NKI helpers + load + store codegens (append)
- `helion/language/matmul_ops.py` — NKI dot codegen
- `helion/language/view_ops.py` — NKI subscript codegen
- `helion/language/scan_ops.py` — NKI scan codegen
- `helion/language/_tracing_ops.py` — NKI mask_to codegen
- `helion/language/random_ops.py` — NKI rand codegen
- `helion/language/barrier.py` — NKI barrier codegen
- `helion/language/atomic_ops.py` — NKI atomic_add codegen
