# Helion NKI Backend: Handoff Document

This document describes changes made to the Helion codebase to support the NKI (Neuron Kernel Interface) backend for AWS Trainium, plus how to test and debug NKI kernels.

---

## 1. Summary of Changes (Since Last Git State)

All edits target the NKI backend only. Triton and other backends are unchanged unless noted.

---

## 2026-05-05 Dynamic Loop Status

Dynamic `hl.tile(..., tensor_bound, ...)` loops now lower to NKI `nl.dynamic_range`
and have focused runtime coverage on Trainium2.

Verified:
- Tensor-valued loop stops are loaded into an NKI register with `nisa.register_load`.
- Dynamic loop offsets are tracked with SBUF counters for `.ap(scalar_offset=...)`.
- Tile indices inside dynamic loops are materialized with `nisa.iota(offset=0)` plus
  a float counter tile, avoiding NKI's static-offset restriction.
- Dynamic loads/stores work for both partition-dimension and free-dimension AP
  offsets.
- Loop-carried accumulators update the outer SBUF buffer in place, so reductions
  after `nl.dynamic_range` no longer reference a buffer allocated inside the
  dynamic loop body.
- `hl.load(..., extra_mask=...)` has an NKI masked-load path using
  `nisa.tensor_copy_predicated` for non tile-list loads.

Focused test:

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH \
PYTHONPATH=/home/ubuntu/helion/helion_nki \
HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_dynloops_cache8 \
/opt/aws_neuronx_venv_pytorch_2_9/bin/python -m pytest -q \
  test/test_nki_dynamic_loops.py -x -s
```

Result: `4 passed, 6 warnings` on 2026-05-05.

Full-suite check after installing the missing dev test dependencies into the
active Neuron venv (`expecttest`, `hypothesis`, `pytest-timeout`):

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH \
PYTHONPATH=/home/ubuntu/helion/helion_nki \
HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_fullsuite_after_deps_cache \
/opt/aws_neuronx_venv_pytorch_2_9/bin/python -m pytest -ra --tb=short \
  --junitxml=/tmp/helion_nki_full_pytest_after_deps.xml test
```

Result on 2026-05-05: `22 passed, 1164 skipped, 6 warnings`.
No failures, no collection errors, and no NVIDIA/CUDA-only failures were observed
in the NKI full-suite run.

Latest full-suite recheck after the example/lowering updates below:

```bash
PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:$PATH \
PYTHONPATH=/home/ubuntu/helion/helion_nki \
HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_full_pytest_after_examples \
/opt/aws_neuronx_venv_pytorch_2_9/bin/python -m pytest -ra --tb=short \
  --junitxml=/tmp/helion_nki_full_pytest_after_examples.xml test
```

Result on 2026-05-05: `22 passed, 1164 skipped, 6 warnings`.

Coverage caveat: only `test/test_nki_dynamic_loops.py` is explicitly NKI-enabled
(`4` tests). The other `18` passing tests are backend-independent utility checks
from `test/test_aot_autotuning.py` and `test/test_codegen_comments.py`; they do
not exercise NKI code generation or Trainium execution. Most of the historical
Helion suite is still intentionally skipped under `HELION_BACKEND=nki`.

Current direct jagged-lowering status:
- The jagged dense-add path reaches `nl.dynamic_range`, vector-offset gather, and
  predicated masked loads.
- `examples/jagged_dense_add.py` was adjusted to prefill `out` from `y` before the
  dynamic loop instead of using `hl.tile(max_nnz, out.size(1))`; NKI ranges do not
  accept a tensor-valued start.
- Remaining blocker: `max_nnz = nnz.amax()` over the per-row int32 length tile is
  still lowered through `nisa.tensor_partition_reduce`, which neuronx-cc rejects
  with `NCC_IXCG864 ISA check failed`. This is separate from dynamic-loop lowering;
  the reduction layout/dtype selection for logical 1D tiles needs another pass.

Tooling note: `ruff` was not available in this environment, so touched files were
checked with `python -m py_compile` and `git diff --check`.

Example-run status after row-gather/scatter and backend-forcing work:
- `examples/run_nki_examples.py` was run serially with `HELION_BACKEND=nki`,
  `NEURON_PLATFORM_TARGET_OVERRIDE=trn2`, and
  `TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_examples_after_gather`.
- Passing canonical examples include: `add.py`, `attention.py`,
  `batch_softmax.py`, `bmm.py`, `broadcast_matmul.py`, `concatenate.py`,
  `embedding.py`, `exp.py`, `geglu.py`, `jsd.py`, `kl_div.py`,
  `layer_norm_f32.py`, `long_sum.py`, `matmul.py`, `matmul_layernorm.py`,
  `psum_reuse_minimal.py`, `psum_reuse_test.py`, `rms_norm.py`, `softmax.py`,
  `softmax_decomposed.py`, `squeeze_and_excitation_net.py`, `sum.py`,
  `swiglu.py`, and `welford.py` (plus quiet zero-output examples
  `aot_example.py` and `blackwell_attention.py`).
- Remaining runner failures are tracked in
  `examples-progress.md`. The major buckets are indexed/free-dim DMA
  shape gaps, jagged reduction/broadcast layout, quantized `torch.stack` unpack
  paths, NKI RNG/barrier/Triton benchmark infrastructure, and a few neuronx-cc
  internal errors.

Additional NKI lowering changes made during this pass:
- Row gather syntax like `weight[indices, tile_e]` now lowers to
  `tensor.ap(pattern=[[F, P], [1, tile]], offset=tile_start,
  vector_offset=indices_p1, indirect_dim=0)`.
- Row scatter for `out[rows, tile_n] = value` uses the matching AP
  `vector_offset` destination pattern.
- NKI loop index SBUF variables are registered with shape/dtype metadata so
  later memory lowering can recognize `indices_*` and their copy variables.
- Scalar-like LHS tensor arithmetic (for example `start + tile.index`) uses the
  tensor-scalar path, and integer SBUF scalar operands are cast to float32 before
  `nisa.tensor_scalar` to satisfy NKI verifier rules.
- Non-list RHS matmul K-subtile loops now pass the loop variable into `_rhs_ref`
  instead of hardcoding K sub-tile 0.
- Mean reduction scaling over squeezed 3D+ SBUF shapes now uses the logical
  reduction dimension, not the post-squeeze NKI axis.

---

### 1.1 `helion/_compiler/backend.py`

**NKIBackend / NKIOpOverrides**

- **`range_str(begin, end, step)`**  
  NKI uses `nl.sequential_range(begin, end, step)` with literal step (no Triton-style `tl.range` or constexpr). The f-string must use `end` correctly so placeholders like `{end}` are not misinterpreted (e.g. avoid `f"..., {end}"` patterns that could miss the placeholder).

- **`_is_scalar_like_tensor(x)`**  
  Used to decide when to emit `nisa.tensor_scalar` instead of `nisa.tensor_tensor`. Returns true when the current binary node’s RHS is:
  - a 0-dim tensor,
  - a tensor with all dimensions equal to 1 (e.g. `[1]`, `[1, 1]`),
  - or a non-tensor (e.g. scalar parameter).  
  Previously only `len(shape)==1 and shape[0]==1` was checked; it was generalized to `all(d == 1 for d in shape)` so that subscript results like `sum_2[:, None]` (shape `[1, 1]`) are treated as scalar-like for NKI.

- **`_nki_binary_op(..., allow_tensor_tensor=True)`**  
  New branch: when both operands are non-scalar but the RHS is “scalar-like” (e.g. reduction result with shape `[1, 1]`), emit `_nki_tensor_scalar(a, b, op_tensor_scalar)` instead of `tensor_tensor`. This avoids neuronx-cc errors like “Expect AP same number of elements” when one operand is `[1, 4096]` and the other is `[1, 1]`.

- **`sub(a, b)`**  
  If both args are tensors and `_subtract_tensor_tensor_supported()` is true (same condition as `_truediv_tensor_tensor_supported`: partition dims match, RHS free dim is 1), emit `_nki_tensor_scalar(a, b, "nl.subtract")` instead of `tensor_tensor`.

- **`mul(a, b)`**  
  Same broadcast handling as `sub`: when `_truediv_tensor_tensor_supported()` holds, emit `_nki_tensor_scalar(a, b, "nl.multiply")` so `[M, N] * [M, 1]` uses `tensor_scalar`.

- **`_truediv_tensor_tensor_supported()`**  
  True when lhs/rhs are 2D, partition dims match, and RHS free dim is 1. Used for truediv, sub, and mul broadcast paths.

- **`_subtract_tensor_tensor_supported()`**  
  Delegates to `_truediv_tensor_tensor_supported()` for the same broadcast pattern.

- **`truediv`**  
  Extended to support scalar-like RHS (via `_is_scalar_like_tensor`) and the `[M,N]/[M,1]`-style broadcast path using `tensor_scalar`.

- **`maximum(a, b)` / `minimum(a, b)`**  
  New NKI overrides that call `_nki_binary_op` with `nl.maximum` / `nl.minimum` (for use in e.g. kl_div, grpo_loss).

- **`exp2(x)`**  
  New NKI override (e.g. for mamba2_chunk_scan).

- **Dim-0 reduction (`amax`, `sum`, etc.)**  
  When reduction is over `dim == 0`, NKI cannot use `nisa.tensor_reduce` (which requires a free axis). The code now uses **`nisa.tensor_partition_reduce`**:  
  - Supported reduction types: `sum` → `nl.add`, `max` → `nl.maximum`, `mean` → `nl.add` plus a subsequent scale.  
  - Result shape is `[1, free_size]`.  
  - See backend code around “tensor_partition_reduce” and “dim == 0”.

- **`create_loop_strategy(...)`**  
  NKI now returns a custom **`_NKINDTileStrategy`** (subclass of `NDTileStrategy`) that overrides **`supports_index_rank_expansion()`** to return **`False`**.  
  That prevents the compiler from injecting `[None, :]`-style index rank expansion for NKI, which would produce 3D indexing on 2D SBUF tiles and break NKI’s partition/free axis semantics.

---

### 1.2 `helion/_compiler/device_function.py`

- **1D tensor reshape at kernel entry (NKI)**  
  In `codegen_function_def`, for NKI, tensor arguments with `dim() == 1` are reshaped from `[N]` to **`[1, N]`** in the generated kernel preamble. So the “partition” dimension is 1 and the “free” dimension is N, which:
  - Avoids tile-list fragmentation for large 1D tensors (e.g. weight/bias),
  - Keeps a single SBUF buffer for the whole vector when `N > NKI_PARTITION_MAX` (see memory_ops),
  - Aligns with load/store codegen that assume 1D inputs are viewed as `[1, N]`.

- **Multi-output kernels (NKI)**  
  Kernels that write to multiple output tensors (e.g. layer_norm’s `out`, `mean`, `rstd`) now:
  - Use **`_nki_return_buffers`**: a dict keyed by `id(tensor)` storing `buf_name`, `host_var`, `host_reshape` per output.
  - **`_nki_return_statements()`** builds the return: single buffer → `return buf_name`, multiple → `return (buf0, buf1, ...)`.
  - **`codegen_function_call`**: when `_nki_return_buffers` has more than one entry, the call is generated as `_nki_result = _launcher(...)`, and **`_nki_post_call_stmts`** is set to a list of statements that assign `_nki_result[0]`, `_nki_result[1]`, … to the host variables (with optional `.reshape(host_reshape)` for 1D outputs).

- **Constexpr / block size args (NKI)**  
  **`block_size_var(block_id)`** ensures the block size is registered as a constexpr argument and that the host side has a definition for it. If the cache already has a var name but that name is not in `_constexpr_args`, it calls **`constexpr_arg_with_host_def(var_name, block_value)`** so that the NKI kernel receives the argument and tracing does not fail with “_BLOCK_SIZE_0 not found” or “missing required argument”.

---

### 1.3 `helion/language/memory_ops.py`

**Load (NKI)**

- **1D / [1, N] path**  
  When `partition_dim > NKI_PARTITION_MAX` and free dims are `[1]` or `[]`, the load no longer creates a tile list. It allocates a **single** `nl.ndarray([1, partition_dim])` and one `nisa.dma_copy` with source slice **`name[0:1, orig_slice]`** to match the kernel-entry reshape. This fixes “_nki_sbuf_4 not found” when downstream code expected one buffer name.

- **Reduction dimension slicing**  
  For a subscript dimension that corresponds to a reduction block (e.g. `slice(None)` over the reduced dim), the slice uses the reduction loop’s **offset variable** and **block size** (e.g. `roffset_1 : roffset_1 + 4096`) instead of `0 : size`. This is done by resolving `block_id` from the FX subscript or from `output_shape[output_idx]` / `env.get_block_id(out_dim)` so that load slices align with `nl.sequential_range(...)`.

**Store (NKI)**

- **Multiple return buffers**  
  Each distinct output tensor (by `id(tensor)`) gets an entry in **`device_fn._nki_return_buffers`**: its own `nki_return_buf_*` HBM buffer, plus `host_var` and optional `host_reshape`. Stores write to the appropriate buffer name. Single-output kernels still set `_nki_return_buffer_name` / `_nki_return_host_var` for backward compatibility.

- **Reduction dimension slicing**  
  Same idea as load: for a store subscript over a reduction dimension, the destination slice uses the reduction loop’s offset and block size (via `env.get_block_id(dim_size)` or, for constant sizes, looping over `env.block_sizes` and **`rdim.size_matches(dim_sympy)`**) so that DMA writes use `roffset_1 : roffset_1 + block_size` instead of a fixed range.

---

### 1.4 `helion/_compiler/generate_ast.py`

- **`visit_For`**  
  After generating the kernel call with `device_function.codegen_function_call()`, the visitor checks for **`device_function._nki_post_call_stmts`**. If present (multi-output NKI kernel), it emits the call statement and then **each of the post statements** (unpacking `_nki_result[i]` into host variables), and returns `None` so the original call is not duplicated.

---

### 1.5 `helion/_compiler/tile_strategy.py`

- **`_BaseNDTileStrategy.mask_var(block_idx)`**  
  Previously this could assume a single “flat” mask. It now uses **`self.mask_vars.get(block_idx)`** so that per-block mask variables are returned correctly. This fixes KeyError when reduction or other code asked for `mask_var(1)` and only `mask_var(0)` or `mask_var(-1)` was defined.

---

### 1.6 `helion/_compiler/compile_environment.py`

- **`get_block_id(size)`**  
  Used by NKI store codegen to decide if a dimension corresponds to a reduction block. When the size is not directly tied to a symbol, the fallback now checks **reduction** block sizes with **`rdim.size_matches(expr)`** (comparing `numel` to the dimension expression) so that constant reduction dimensions (e.g. 10240) are correctly mapped to the reduction block and the right offset/block_size are used in store slices.

---

### 1.7 `helion/language/matmul_ops.py`

- **`hl.dot`**  
  **`@_decorators.codegen(dot, "nki")`** was added so that the NKI backend has a concrete implementation for `dot` (e.g. for gdn_fwd_h). The NKI codegen emits the appropriate `nisa.nc_matmul` (or equivalent) and accumulator handling; see the `codegen(dot, "nki")` function in that file.

---

### 1.8 `examples/layer_norm.py`

- **Test dimensions**  
  **`dim`** was changed from **10240** to **8192** so that the reduction dimension is evenly divisible by the reduction block size (4096). NKI kernels do not implement masking for partial tiles; using a dimension that is a multiple of the block size avoids out-of-bounds DMA (e.g. `roffset_1 + 4096` exceeding 10240 on the last iteration). The comment in the handoff that “input is already padded correctly” and “masking later in PyTorch” refers to this design choice: keep kernel dimensions divisible and handle masking outside NKI if needed.

---

## 2. Testing Workflow

### 2.1 Environment

- Use the Neuron/PyTorch environment (e.g. `source aws_neuron_venv_pytorch/bin/activate`).
- Set:
  - `PYTHONPATH=/path/to/kernel_test/helion_nki` (or repo root that contains `helion_nki`)
  - `HELION_BACKEND=nki`
  - `NEURON_PLATFORM_TARGET_OVERRIDE=trn1` (for Trainium; omit or set to trn2 for Trainium2)
/
### 2.2 Single example

```bash
cd /path/to/kernel_test
source aws_neuron_venv_pytorch/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
PYTHONPATH=helion_nki:$PYTHONPATH HELION_BACKEND=nki python helion_nki/examples/layer_norm.py
```

Each example typically:
- Compiles the Helion kernel to NKI.
- Runs it (via the default NKI launcher and Neuron runtime).
- Compares against a reference (e.g. `torch.nn.functional.layer_norm`) and prints PASSED/FAILED or raises on mismatch.

### 2.3 Full example suite

```bash
cd /path/to/kernel_test
source aws_neuron_venv_pytorch/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/run_nki_examples.py
```

- Runs all canonical NKI examples under `helion_nki/examples/` (excluding `run_nki_examples.py`, `_*`, `*_nki.py`, and CUDA-only scripts).
- Each example is run in a subprocess with `HELION_BACKEND=nki` and the same env.
- Exit code 1 if any example fails; stdout lists which failed.

### 2.4 Clearing caches

- **TorchInductor cache** (generated Python kernels): remove `/tmp/torchinductor_ubuntu/` (or the dir configured for your run) so the next run regenerates code.
- **Neuron compile cache**: controlled by `NEURON_COMPILE_CACHE_URL` / local cache; clear or set empty if you need to force recompile.

---

## 3. Debugging NKI Kernels

### 3.1 Trace / compile errors (before neuronx-cc)

- **“Missing placeholders: ['end']”**  
  Usually a bug in `range_str` or wherever the range string is built; ensure f-strings use the intended variable names (e.g. `end`).

- **“_nki_sbuf_* not found” / “index out of range”**  
  Often due to:
  - 1D tensors: ensure kernel entry reshapes to `[1, N]` and load uses a single SBUF with `name[0:1, orig_slice]` when `partition_dim > NKI_PARTITION_MAX` and free_dims in `[1]` or `[]`.
  - Multiple outputs: ensure each output has its own return buffer in `_nki_return_buffers` and stores write to the correct buffer.
  - Reduction dimension: ensure load/store slices use the reduction loop’s offset and block size (e.g. `roffset_1 : roffset_1 + 4096`), not `0 : size`.

- **“Backend does not support: reduction over dim 0”**  
  Dim-0 reduction must go through `nisa.tensor_partition_reduce` in backend.py; add or fix that path.

- **“_BLOCK_SIZE_* not found” / “missing required argument”**  
  Block sizes must be registered as constexpr and passed from host. Fix `block_size_var()` in device_function.py so it calls `constexpr_arg_with_host_def` for the cached var when needed.

- **“Expect AP same number of elements” / TensorTensor shape mismatch**  
  NKI’s `tensor_tensor` requires matching shapes on the free axis. When one operand is effectively a scalar (e.g. `[1, 1]`), use `tensor_scalar` instead. Extend `_is_scalar_like_tensor` (e.g. all-ones shapes) and/or add a broadcast path for the op (like `sub`/`mul`) that calls `_nki_tensor_scalar`.

- **“Failed to resolve an argument … expecting tensor access, got 'float'”**  
  Binary ops with a scalar or 0-dim tensor on one side must be handled by the scalar branch of `_nki_binary_op` or by an explicit scalar path (e.g. `_is_scalar_operand` / `_is_scalar_like_tensor`).

### 3.2 Inspecting generated code

- Generated NKI Python lives under the TorchInductor cache, e.g.:
  - `/tmp/torchinductor_ubuntu/<subdir>/<hash>.py`
- After a run, search for the module that defines your kernel (e.g. `_helion_layer_norm_fwd`):
  - `find /tmp/torchinductor_ubuntu -name '*.py' -exec grep -l '_helion_layer_norm_fwd' {} \;`
- Open that file to see:
  - Reshapes at kernel entry,
  - SBUF/HBM allocations,
  - `nl.sequential_range` / `nl.affine_range` and loop variables,
  - `nisa.dma_copy` source/dest slices,
  - `nisa.tensor_tensor` vs `nisa.tensor_scalar` usage,
  - Return value (single buffer vs tuple).

Use this to verify reduction slices, 1D load/store, and multi-output return handling.

### 3.3 neuronx-cc (BIR / backend) errors

- **“DMA Copy in/out shape must match”**  
  Dest and src shapes in `nisa.dma_copy` must match. Check that:
  - Reduction dimensions use the same offset/block_size in load and store,
  - Multi-output buffers have the correct shapes (no single shared return buffer for outputs of different shapes).

- **“Expect AP same number of elements … TensorTensor”**  
  Fix in the compiler: route that binary op to `tensor_scalar` when one operand is scalar-like (see above).

- **“Unrecognized InstIO type”**  
  Usually a follow-on from a shape or op mismatch; fix the earlier error first.

### 3.4 Numerical mismatch (kernel runs but results wrong)

- **Layer norm / reductions**  
  The generated kernel may ignore `.to(torch.float32)` and keep data in float16 in SBUFs. NKI codegen does not yet lower dtype casts; accumulations and sensitive math should be in float32. This is a known limitation; either add dtype lowering for NKI or adjust the kernel/reference to match the current behavior.

- **Reduction block size**  
  Ensure reduction dimensions are multiples of the reduction block size (e.g. 4096). Otherwise the last tile can go out of bounds; change test dimensions (as in layer_norm `dim=8192`) or add host-side padding and slicing.

### 3.5 Useful references

- NKI ISA (e.g. tensor ops, DMA, partition reduce):  
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/nki.isa.html  
  https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.tensor_partition_reduce.html
- Helion compiler flow: `helion/_compiler/` (backend, device_function, generate_ast, memory_ops, tile_strategy, compile_environment).
- NKI runtime/launcher: `helion/runtime/__init__.py` (`default_nki_launcher`).

---

## 4. Quick reference: files touched

| Path | Purpose of changes |
|------|--------------------|
| `helion/_compiler/backend.py` | NKI range_str, binary op scalar/broadcast, dim-0 reduce, _NKINDTileStrategy, maximum/minimum/exp2 |
| `helion/_compiler/device_function.py` | 1D reshape [1,N], multi-output return, constexpr/block_size_var |
| `helion/language/memory_ops.py` | 1D load single buffer, multi-output store, reduction slice in load/store |
| `helion/_compiler/generate_ast.py` | visit_For: inject _nki_post_call_stmts for multi-output |
| `helion/_compiler/tile_strategy.py` | mask_var(block_idx) via .get(block_idx) |
| `helion/_compiler/compile_environment.py` | get_block_id fallback with rdim.size_matches for reduction dim |
| `helion/language/matmul_ops.py` | codegen(dot, "nki") for hl.dot |
| `examples/layer_norm.py` | dim 8192 for divisible reduction |
