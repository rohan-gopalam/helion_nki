# Helion NKI Backend: Comprehensive Handoff Document

**Date**: 2026-03-11
**Scope**: All work done to make the `layer_norm` example (forward + backward) work on the Helion NKI backend for AWS Trainium.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Project Structure](#2-project-structure)
3. [Current Status](#3-current-status)
4. [Architecture Overview: How Helion NKI Codegen Works](#4-architecture-overview)
5. [All Bugs Fixed (Chronological)](#5-all-bugs-fixed)
6. [File-by-File Change Inventory](#6-file-by-file-change-inventory)
7. [The Backward Pass Fix: Full Story](#7-the-backward-pass-fix)
8. [Generated Kernel Analysis](#8-generated-kernel-analysis)
9. [Debug Scripts](#9-debug-scripts)
10. [How to Debug NKI Issues](#10-how-to-debug-nki-issues)
11. [Remaining TODOs](#11-remaining-todos)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Key NKI Concepts Reference](#13-key-nki-concepts-reference)

---

## 1. Environment Setup

### Activation

```bash
cd /home/ubuntu/kernel_test
source aws_neuron_venv_pytorch/bin/activate
```

### Running a single example

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
rm -rf /tmp/torchinductor_ubuntu/   # ALWAYS clear cache between code changes
PYTHONPATH=helion_nki:$PYTHONPATH HELION_BACKEND=nki python helion_nki/examples/layer_norm.py
```

### Running the full NKI example suite

```bash
PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/run_nki_examples.py
```

### Critical: Cache Clearing

The TorchInductor cache at `/tmp/torchinductor_ubuntu/` caches generated Python kernels. After **any** code change to the Helion compiler, you **MUST** clear this cache:

```bash
rm -rf /tmp/torchinductor_ubuntu/
```

Failure to do this will result in running stale generated code, making debugging extremely confusing.

### Neuron Runtime Issues

If you get `NRT_FAILURE` errors or the kernel hangs, stale Python/neuron processes may be holding the device:

```bash
# Kill stale processes
pkill -f "python.*layer_norm" 2>/dev/null
pkill -f "neuron" 2>/dev/null

# Restart neuron runtime daemon (if needed)
sudo systemctl restart neuron-rtd 2>/dev/null
```

---

## 2. Project Structure

```
/home/ubuntu/kernel_test/
├── helion_nki/                          # Main Helion codebase (modified copy)
│   ├── helion/
│   │   ├── _compiler/
│   │   │   ├── backend.py               # *** HEAVILY MODIFIED — NKI backend, cast_ast, reduction_expr
│   │   │   ├── device_function.py       # *** MODIFIED — 1D reshape, multi-output, SBUF shape tracking
│   │   │   ├── reduction_strategy.py    # *** MODIFIED — NKI reduction codegen (LoopedReduction)
│   │   │   ├── inductor_lowering.py     # *** MODIFIED — to_dtype NKI path
│   │   │   ├── aten_lowering.py         # *** MODIFIED — codegen_full_nki for [1,N] accumulators
│   │   │   ├── generate_ast.py          # *** MODIFIED — multi-output post-call statements
│   │   │   ├── compile_environment.py   # *** MODIFIED — resolve_block_id with rdim fallback
│   │   │   └── tile_strategy.py         # *** MODIFIED — mask_var per block_idx
│   │   └── language/
│   │       ├── memory_ops.py            # *** HEAVILY MODIFIED — NKI load/store codegen
│   │       └── matmul_ops.py            # MODIFIED — codegen(dot, "nki")
│   ├── examples/
│   │   ├── layer_norm.py                # *** MODIFIED — dim=8192, inner block_size=1
│   │   ├── EXAMPLES_NKI_STATUS.md       # Status tracking for all NKI examples
│   │   └── run_nki_examples.py          # Suite runner
│   └── HANDOFF_NKI_CHANGES.md           # Previous handoff doc (less detailed)
├── debug_layer_norm.py                  # Manual NKI forward kernel test script
├── debug_layer_norm_bwd.py              # Manual NKI backward kernel test script
└── aws_neuron_venv_pytorch/             # Python venv with neuron SDK
```

---

## 3. Current Status

### Forward Pass: PASSES ✓

Both with-bias and without-bias variants pass numerical validation:
- `atol=1e-3, rtol=1e-3`
- 0 assertion failures
- Tested with `BATCH=4096, DIM=8192`

### Backward Pass: COMPILES BUT NUMERICALLY WRONG (grad_weight mismatch)

**Verified 2026-03-11**: The backward kernel compiles and runs to completion, but fails numerical validation:

```
AssertionError: BWD: Gradient mismatch for tensor 1 with shape torch.Size([8192]) in helion
```

Tensor 1 is **`grad_weight`** (shape `[8192]`). `grad_x` and `grad_bias` may be correct — only `grad_weight` is confirmed wrong.

The generated backward kernel shape issue is **fixed** (all `nki_reduce = [1, 1]`), but the gradient accumulation logic for `grad_weight` produces incorrect values. This is a **numerical correctness bug** remaining to be diagnosed.

Likely suspects:
1. The `grad_w_acc += torch.sum(dy_mb * x_hat, dim=0)` accumulation uses `tensor_partition_reduce` (dim=0 reduction on `[1, 8192]`), but `[1, 8192]` has partition_dim=1 so `tensor_partition_reduce` reduces along the 1-element partition axis — effectively a no-op. The `sum_1 = nki_part_reduce[0, :]` then just extracts a `[8192]` row, and `tensor_tensor(dst=_nki_full, data1=_nki_full, data2=sum_1)` adds it. This chain needs to be verified for correctness.
2. The `x_hat` computation (`(x_mb - mean_mb[:, None]) * rstd_mb[:, None]`) uses `tensor_scalar` with `[1,1]` operands — this should be correct.
3. The in-place mutation pattern (`_nki_full_copy = _nki_full` etc.) in the inner loop may cause aliasing issues in NKI.

### Other Examples

See `helion_nki/examples/EXAMPLES_NKI_STATUS.md` for full status. Key validated examples:
- add, sum, exp, matmul, batch_softmax, softmax_decomposed, swiglu, broadcast_matmul

---

## 4. Architecture Overview

### How Helion Generates NKI Code

1. **User writes Helion kernel** (`@helion.kernel` decorated function with `hl.tile`, tensor ops)
2. **FX tracing**: PyTorch captures the kernel as an FX graph with symbolic shapes (SymInt)
3. **Tile strategy selection**: Helion picks `PersistentReductionStrategy` or `LoopedReductionStrategy`
4. **Code generation**: Each FX node is lowered to NKI code via:
   - `backend.py` — arithmetic ops (`tensor_tensor`, `tensor_scalar`), reductions (`tensor_reduce`, `tensor_partition_reduce`), casts (`cast_ast`)
   - `memory_ops.py` — loads (`dma_copy` from HBM to SBUF) and stores (`dma_copy` from SBUF to HBM)
   - `reduction_strategy.py` — reduction loop structure and final reduction codegen
   - `inductor_lowering.py` — dtype conversions, op dispatch
   - `aten_lowering.py` — `torch.zeros`/`torch.full` → `nl.ndarray` + `nisa.memset`
5. **Output**: A Python file with `@nki.jit` decorated function + host wrapper

### NKI Partition/Free Axis Semantics

NKI tensors in SBUF are always conceptually 2D: `[partition, free]`

- **Partition dimension** (dim 0): Maps to neuron cores. Max 128 elements per tile.
- **Free dimension** (dim 1): Maps to PE SIMD lanes. Can be large (e.g., 8192).
- When partition_dim > 128, tensors must be split into "tile lists" of 128-element chunks.
- `nisa.tensor_tensor(dst, data1, data2, op)`: Both operands must have matching shapes.
- `nisa.tensor_scalar(dst, data, op0, operand0, op1)`: `operand0` is broadcast across the free dim.
- `nisa.tensor_reduce(dst, op, data, axis, keepdims)`: Reduces along specified axis.
- `nisa.tensor_partition_reduce(dst, op, data)`: Reduces along partition axis (dim 0).

### Key Data Flow in layer_norm_bwd

```
Helion source (layer_norm.py):
  for mb_cta in hl.tile(x.size(0), block_size=m_block):     # outer loop: 32-row CTAs
    grad_w_acc = weight.new_zeros(n, dtype=torch.float32)     # [8192] accumulator
    for mb in hl.tile(mb_cta.begin, mb_cta.end, block_size=1): # inner loop: 1 row at a time
      x_mb = x[mb, :].to(torch.float32)                       # load row → fp32
      dy_mb = grad_out[mb, :].to(torch.float32)                # load row → fp32
      mean_mb = mean[mb].to(torch.float32)                     # load scalar
      rstd_mb = rstd[mb].to(torch.float32)                     # load scalar
      x_hat = (x_mb - mean_mb[:, None]) * rstd_mb[:, None]    # normalize
      grad_w_acc += torch.sum(dy_mb * x_hat, dim=0)           # accumulate grad_weight
      c1 = torch.sum(x_hat * wdy, dim=-1) / n                 # scalar reduction → [1,1]
      c2 = torch.sum(wdy, dim=-1) / n                         # scalar reduction → [1,1]
      dx = (wdy - (x_hat * c1[:, None] + c2[:, None])) * rstd_mb[:, None]
      grad_x[mb, :] = dx.to(x.dtype)                          # store grad_x row
    grad_weight_blocks[mb_cta.id, :] = grad_w_acc              # store accumulated block
```

Generated NKI:
- `_nki_full` = `[1, 8192]` SBUF accumulator (grad_w_acc)
- Inner loop: `nl.sequential_range(offset_0, tile_end, 1)` — one row at a time
- x, dy loaded as `[1, 8192]` fp16 → cast to fp32 via `memset(0) + tensor_tensor(add)`
- mean, rstd loaded as `[1, 1]` fp32 scalars
- c1, c2 computed via `tensor_reduce(axis=1)` → `[1, 1]` result
- Final store: `dma_copy` to HBM `nki_return_buf_1[tile_id:tile_id+1, 0:8192]`

---

## 5. All Bugs Fixed (Chronological)

### Bug 1: `_nki_sbuf_2 not found` (Forward Pass)

**Symptom**: Generated code referenced `_nki_sbuf_2` but it wasn't defined.
**Root cause**: SymInt subscripts in 2nd/3rd load of `x` lost their `BlockSizeOrigin` (FX node `val=64` concrete instead of symbolic `u2`), so `compute_shape` eliminated the partition dim, making `output_shape=[4096]` instead of `[1, 4096]`.
**Fix** (`memory_ops.py` load codegen):
- When tensor is 2D, subscript has SymInt for dim 0, but `output_shape` has only 1 element → re-insert dim 0 using active tile loop's `block_size`
- When `get_block_id` fails for SymInt subscript, fall back to matching active non-reduction `block_id`

### Bug 2: Wrong slice offset for reduction dimension (Forward)

**Symptom**: Load slices used `offset_0` (batch dim offset) for both dimensions instead of using reduction offset for the free dimension.
**Root cause**: Fallback heuristic matching `block_size=1` (trivially divides everything) for the free/reduction dimension → used wrong offset variable.
**Fix**: Skip `block_size <= 1` in the fallback heuristic. Also extend the heuristic to match 1D tensors (`tensor_dim_idx == 0 and tensor.dim() == 1`).

### Bug 3: 1D tensor tile lists for weight/bias (Forward)

**Symptom**: `weight[0*128:1*128]` index out of range on 1D tensor.
**Root cause**: 1D tensors like `weight[8192]` got `partition_dim=4096` from reduction block resolution, creating tile lists.
**Fix**: Added `[1, N]` path — when `partition_dim > NKI_PARTITION_MAX` and `free_dims` is empty, allocate single `[1, partition_dim]` SBUF and one DMA copy.

### Bug 4: Single return buffer for multi-output kernel (Forward)

**Symptom**: `nki_return_buf[0:4096, 0:8192]` too many indexes — all outputs wrote to one buffer.
**Root cause**: Store codegen used single `_nki_return_buffer_name` for all outputs.
**Fix**: Use per-tensor `_nki_return_buffers` dict keyed by `id(tensor)`.

### Bug 5: 1D return buffer dimension mismatch (Forward)

**Symptom**: `DMA Copy actual in/out dimensions must match` — 1D `[4096]` HBM buffer vs 2D SBUF source.
**Root cause**: NKI DMA copy requires matching dimensions. 1D output buffers `[N]` don't match 2D SBUF sources.
**Fix**: Allocate 1D output return buffers as 2D `[1, N]`, reshape on host side.

### Bug 6: `tensor_reduce` partition size using hint instead of config (Forward)

**Symptom**: `_SHAPE_DIM = 64` in generated code, but should be 1.
**Root cause**: `int(fake_input.size(0))` used hint (64) instead of configured block size (1) for reduce buffer allocation.
**Fix**: Resolve partition dim through `_bs_subs` substitutions (same technique as LoopedReductionStrategy).

### Bug 7: dtype casts dropped — fp16→fp32 and fp32→fp16 (Forward)

**Symptom**: 40% numerical mismatch — output values 2-3x too large.
**Root cause**: NKI `cast_expr` was a no-op. `.to(torch.float32)` was silently dropped.
**Fix**: Implemented `cast_ast` for NKI that emits:
```python
cast_var = nl.ndarray([shape], target_dtype, buffer=nl.sbuf)
nisa.memset(cast_var, value=0)
nisa.tensor_tensor(dst=cast_var, data1=cast_var, data2=src, op=nl.add)
```
Added NKI path in `to_dtype` that passes `src_dtype` to `cast_ast`.

### Bug 8: Cast shape using FX tensor shape instead of SBUF tile shape (Forward)

**Symptom**: `_nki_cast_3 [4096, 1]` allocated memory out of bound.
**Root cause**: Cast shape fallback used FX node's full tensor shape instead of SBUF tile shape.
**Fix**: Look up `_nki_sbuf_shapes` registry first, fall back to reduction block size `[1, rblock]`.

### Bug 9: Cast shape 3D `[32, 8192, 1]` (Forward)

**Symptom**: Generated 3D array allocation.
**Root cause**: Fallback used ALL block sizes `[32, 4096]` + trailing 1.
**Fix**: Use only reduction block size `[1, rblock]` as fallback.

### Bug 10: Mean/rstd loads as 1D `[32]` instead of `[32, 1]` (Forward)

**Symptom**: `_nki_sbuf_4[:, None]` too many indexes for 1D shape.
**Root cause**: When `free_dims` is empty, load produced 1D `[partition_dim]` SBUF.
**Fix**: Always add trailing 1 for NKI (`[partition_dim, 1]` when `free_dims` is empty).

### Bug 11: SBUF overflow `[32, 8192]` (Backward)

**Symptom**: `_nki_cast_3 is too big for SB, requires 524288 bytes`
**Root cause**: Inner tile with `block_size=32` (default) created `[32, 8192]` fp32 tiles exceeding SBUF limit (~512KB).
**Fix**: Changed `layer_norm.py` backward kernel inner loop to `block_size=1`.

### Bug 12: Store slice heuristic wrong for NKI (Forward + Backward)

**Symptom**: DMA store used wrong offsets.
**Root cause**: Same issues as load: SymInt fallback, `block_size <= 1` filter, 1D tensor support.
**Fix**: Applied same fixes as load to store codegen.

### Bug 13: Tile list store distributing along wrong dimension (Backward)

**Symptom**: `_nki_full_*` tiles stored with partition offsets in dim 0 instead of dim 1.
**Root cause**: Free-dim tile lists (e.g., `grad_w_acc` as `[1, 8192]`) were being distributed as if they were partition-split.
**Fix**: Added `_nki_free_dim_tile_lists` set and `tile_along_free` detection in store codegen to distribute along last dim.

### Bug 14: `nki_reduce = [32, 1]` instead of `[1, 1]` (Backward — THE MAIN BUG)

**Symptom**: `BIR verification failed: Expect AP same number of elements [[8192,1],[1,8192]]`
**Root cause**: In `reduction_strategy.py` `codegen_reduction` (LoopedReductionStrategy NKI path), `fake_input.size(0)` returns concrete `int(32)` — the FX trace **hint** for the inner tile block size — rather than a `torch.SymInt`. The SymInt→config substitution path was never taken.
**Fix** (multi-layered):
1. **SBUF shape registry lookup**: Check `device_fn._nki_sbuf_shapes.get(input_name)` and `get(acc)` for pre-registered shapes.
2. **shape_dims[0] eval**: When `shape_dims[0]` is a variable name like `"_BLOCK_SIZE_1"`, evaluate it using a lookup dict mapping block size variable names to configured values.
3. **Hint-to-config override**: When `fake_input.size(0)` is a concrete int matching a non-reduction block size's hint, substitute the configured value instead.
4. **`backend.py` `reduction_expr`**: Same SymInt→config substitution logic for `PersistentReductionStrategy` path.

The generated kernel now shows `nki_reduce = nl.ndarray([1, 1], ...)` — correct.

---

## 6. File-by-File Change Inventory

### `helion/_compiler/backend.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `cast_ast()` | New NKI dtype cast: `nl.ndarray` + `memset(0)` + `tensor_tensor(add)`. Looks up `_nki_sbuf_shapes` registry, falls back to `[1, rblock]`. Handles tile list sources. Resolves SymInt dims via config substitutions. | 1460–1568 |
| `reduction_expr()` | dim≥1 path: Derives partition size from `fake_input.size(0)` with SymInt→config substitution. dim=0 path: `tensor_partition_reduce`. | 1628–1766 |
| `_is_scalar_like_tensor()` | Generalized to `all(d == 1 for d in shape)` | ~1200 |
| `_nki_binary_op()` | Scalar-like RHS detection → route to `tensor_scalar` | ~1250 |
| `sub()`, `mul()` | Broadcast path for `[M,N] op [M,1]` → `tensor_scalar` | ~1300 |
| `full_expr()` / `full_memset_stmt()` | Two-line pattern for NKI `full/zeros` | 1578–1591 |
| `_NKINDTileStrategy` | Subclass of NDTileStrategy with `supports_index_rank_expansion()=False` | ~1400 |

### `helion/_compiler/device_function.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `_nki_sbuf_shapes` | New dict: `var_name → [partition, free]` shape | 267 |
| `_nki_free_dim_tile_lists` | New set: tracks free-dim tile lists | 270 |
| `_nki_return_statements()` | Generate `return (buf0, buf1, ...)` for multi-output | 709–720 |
| `codegen_function_def()` | 1D tensor reshape `[N] → [1, N]` at kernel entry for NKI | 769–786 |
| `codegen_function_call()` | Multi-output: `_nki_result = _launcher(...)` + per-output unpacking | 841–860 |

### `helion/language/memory_ops.py`

| Area | What Changed |
|------|-------------|
| NKI load codegen | `[1, N]` path for 1D tensors. SymInt fallback for `get_block_id`. Reduction dim slicing with offset variable. Always 2D SBUF (`[p, f]`). Register shapes in `_nki_sbuf_shapes`. |
| NKI store codegen | Per-tensor `_nki_return_buffers`. 1D output as `[1, N]` HBM with host reshape. SymInt fallback. Reduction dim slicing. Free-dim tile list detection. |

### `helion/_compiler/reduction_strategy.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `LoopedReductionStrategy.codegen_reduction()` NKI path | SBUF shape lookup → `shape_dims` eval → hint-to-config override for partition size. Generates `tensor_reduce` in `outer_suffix`. | 436–579 |

### `helion/_compiler/inductor_lowering.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `to_dtype()` | NKI-specific branch: when `src_dtype != dtype`, calls `backend.cast_ast()` | 963–966 |

### `helion/_compiler/aten_lowering.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `codegen_full_nki()` | `[1, N]` path when `partition_dim > 128` and no free dims. Registers in `_nki_sbuf_shapes` and `_nki_free_dim_tile_lists`. | 205–222 |

### `helion/_compiler/generate_ast.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `visit_For` | After kernel call, checks for `_nki_post_call_stmts` and emits multi-output unpacking statements | 397–404 |

### `helion/_compiler/compile_environment.py`

| Area | What Changed | Lines (approx) |
|------|-------------|----------------|
| `resolve_block_id()` | New method: falls back to matching constant reduction dimensions via `rdim.size_matches(expr)` | 686–707 |

### `helion/_compiler/tile_strategy.py`

| Area | What Changed |
|------|-------------|
| `mask_var(block_idx)` | Uses `.get(block_idx)` instead of direct access |

### `examples/layer_norm.py`

| Area | What Changed |
|------|-------------|
| `dim` | Changed from 10240 to 8192 (must be multiple of reduction block 4096) |
| Inner loop | Changed to `block_size=1` to keep SBUF under 512KB |
| Tolerances | `rtol=2e-3, atol=1e-2` for backward pass |

---

## 7. The Backward Pass Fix: Full Story

### The Problem

The backward pass computes `c1 = torch.sum(x_hat * wdy, dim=-1) / n` and `c2 = torch.sum(wdy, dim=-1) / n`. These are dim-1 reductions on `[1, 8192]` tensors (since `block_size=1` for the inner loop), producing `[1, 1]` scalar results.

The generated code was producing `nki_reduce = nl.ndarray([32, 1], ...)` instead of `[1, 1]`.

### Root Cause Chain

1. The backward kernel has **two** block sizes:
   - Block 0: `m_block = 32` (outer CTA tile, registered via `hl.register_block_size`)
   - Block 1: inner tile `block_size=1` (registered via `hl.tile(..., block_size=1)`)

2. During FX tracing, the inner tile's SymInt has a **hint** of 32 (or 64 depending on trace order), not 1. The actual value 1 is only known from the config.

3. The `LoopedReductionStrategy.codegen_reduction()` NKI path needs to determine the partition size for the `nki_reduce` buffer. It calls `fake_input.size(0)`.

4. `fake_input.size(0)` returns a **concrete int** (the hint value, e.g., 32) rather than a `torch.SymInt`, because the FX tracing had already concretized it.

5. Since it's a concrete int (not SymInt), the `isinstance(part_size_sym, torch.SymInt)` branch was skipped, and the raw hint value 32 was used directly.

### The Fix (Three Layers of Defense)

**Layer 1** (`reduction_strategy.py:459-490`): SBUF shape registry lookup
```python
sbuf_shape = device_fn._nki_sbuf_shapes.get(input_name)
if sbuf_shape is None:
    sbuf_shape = device_fn._nki_sbuf_shapes.get(acc)
```
If the input or accumulator was registered during load/full codegen, use its known shape.

**Layer 2** (`reduction_strategy.py:470-490`): `shape_dims` eval
```python
_dim_str = shape_dims[0]  # might be "1" or "_BLOCK_SIZE_1"
try:
    sbuf_shape = [int(_dim_str)]
except (ValueError, TypeError):
    # Build lookup: variable name → config value
    _local_vars = {str(_bs.symbol()): int(_bs.from_config_assert(config)), ...}
    sbuf_shape = [int(eval(_dim_str, {}, _local_vars))]
```
The tile strategy's `shape_dims` already resolves to the correct variable name for the partition dimension. If it's a variable name like `"_BLOCK_SIZE_1"`, evaluate it using config values.

**Layer 3** (`reduction_strategy.py:516-523`): Hint-to-config override
```python
for _bs in env.block_sizes:
    if not _bs.reduction:
        _hint_val = env.size_hint(_bs.var._sympy_())
        _cfg_val = int(_bs.from_config_assert(state.config))
        if int(_hint_val) == part_size and _cfg_val != part_size:
            part_size = _cfg_val
            break
```
When the concrete int from `fake_input.size(0)` matches a non-reduction block size's hint, substitute the configured value.

**Also in `backend.py:1708-1723`**: Same SymInt→config substitution for the `PersistentReductionStrategy` path (which calls `backend.reduction_expr` directly).

### Debug Print (Still Present)

There's a debug print on `reduction_strategy.py:501`:
```python
print(f"[REDUCE_DBG] input={input_name} fake_input.shape=...", file=sys.stderr)
```
This should be removed once everything is verified working.

---

## 8. Generated Kernel Analysis

### Forward Kernel Location
```bash
find /tmp/torchinductor_ubuntu -name '*.py' -exec grep -l '_helion_layer_norm_fwd' {} \;
```

### Backward Kernel Location
```bash
find /tmp/torchinductor_ubuntu -name '*.py' -exec grep -l '_helion_layer_norm_bwd' {} \;
```

### Current Backward Kernel (Verified Correct Structure)

File: `/tmp/torchinductor_ubuntu/lf/clfv24b7wpudknvtmjwitba65k3ihkua7gzhjhijhunn6nsp4mwy.py`

Key correct patterns:
```python
# Accumulators: [1, 8192] — correct, fits in SBUF
_nki_full = nl.ndarray([1, 8192], nl.float32, buffer=nl.sbuf)

# Weight cast: [1, 8192] fp16 → fp32 — correct
_nki_cast = nl.ndarray([1, 8192], nl.float32, buffer=nl.sbuf)
nisa.memset(_nki_cast, value=0)
nisa.tensor_tensor(dst=_nki_cast, data1=_nki_cast, data2=_nki_sbuf_1, op=nl.add)

# Mean/rstd: [1, 1] — correct scalar loads
_nki_sbuf_4 = nl.ndarray([1, 1], nl.float32, buffer=nl.sbuf)
nisa.dma_copy(dst=_nki_sbuf_4, src=mean[0:1, offset_1:offset_1 + 1])

# c1/c2 reductions: [1, 1] — THIS WAS THE BUG, NOW CORRECT
nki_reduce = nl.ndarray([1, 1], nl.float32, buffer=nl.sbuf)
nisa.tensor_reduce(dst=nki_reduce, op=nl.add, data=nki_binary_2, axis=1, keepdims=True)

# Scalar broadcast for dx computation — correct
subscript_2 = sum_3[:, None]  # [1, 1] → used as scalar in tensor_scalar
nisa.tensor_scalar(dst=_nki_cast_1, data=_nki_cast_1, op0=nl.multiply, operand0=subscript_2, op1=None)

# Store grad_x: fp32→fp16 cast + DMA — correct
_nki_cast_3 = nl.ndarray([1, 8192], nl.float16, buffer=nl.sbuf)
nisa.memset(_nki_cast_3, value=0)
nisa.tensor_tensor(dst=_nki_cast_3, data1=_nki_cast_3, data2=nki_binary_1, op=nl.add)
nisa.dma_copy(dst=nki_return_buf[offset_1:offset_1 + 1, 0:0 + 8192], src=_nki_cast_3)

# Store grad_weight block — correct
nisa.dma_copy(dst=nki_return_buf_1[offset_0 // 32:offset_0 // 32 + 1, 0:0 + 8192], src=_nki_full)
```

### Host Wrapper (lines 133-183)

```python
def layer_norm_bwd(grad_out, x, mean, rstd, weight, compute_bias_grad=True, *, _launcher=...):
    m_block = 32
    n = 8192
    grad_x = torch.empty_like(x)
    num_blocks = (x.size(0) + m_block - 1) // m_block
    grad_weight_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)
    grad_bias_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)
    _nki_result = _launcher(_helion_layer_norm_bwd, (1,), weight, x, grad_out, mean, rstd, num_blocks)
    grad_x = _nki_result[0]
    grad_weight_blocks = _nki_result[1]
    grad_bias_blocks = _nki_result[2]
    grad_weight = grad_weight_blocks.sum(0).to(weight.dtype)
    grad_bias = grad_bias_blocks.sum(0).to(weight.dtype)
    return (grad_x, grad_weight, grad_bias)
```

---

## 9. Debug Scripts

### `debug_layer_norm.py` — Forward Pass Manual NKI Kernel

A hand-written NKI kernel that implements layer_norm forward pass step by step. Used to:
- Verify that the NKI ISA can correctly implement the algorithm
- Compare against PyTorch reference
- Dimensions: `BATCH=4096, DIM=8192, RBLOCK=4096`

Key patterns established here that informed the codegen fixes:
```python
# fp16→fp32 cast pattern
def load_fp32(src_slice, shape):
    tmp = nl.ndarray(shape, nl.float16, buffer=nl.sbuf)
    nisa.dma_copy(dst=tmp, src=src_slice)
    out = nl.ndarray(shape, nl.float32, buffer=nl.sbuf)
    nisa.memset(out, value=0)
    nisa.tensor_tensor(dst=out, data1=out, data2=tmp, op=nl.add)
    return out
```

**Status**: PASSES all checks (mean, rstd, output)

### `debug_layer_norm_bwd.py` — Backward Pass Manual NKI Kernel

A hand-written NKI backward kernel with smaller dimensions (`BATCH=128, DIM=256, M_BLOCK=32`). Used to verify the backward algorithm works in NKI.

**Status**: PASSES (grad_w and grad_b both close to reference)

---

## 10. How to Debug NKI Issues

### Step 1: Read the generated kernel

```bash
rm -rf /tmp/torchinductor_ubuntu/
# Run the example (it will fail, but generates the kernel)
PYTHONPATH=helion_nki:$PYTHONPATH HELION_BACKEND=nki python helion_nki/examples/layer_norm.py 2>&1 || true
# Find the generated kernel
find /tmp/torchinductor_ubuntu -name '*.py' -exec grep -l '_helion_layer_norm' {} \;
```

### Step 2: Look for shape mismatches

In the generated kernel, check:
- Every `nl.ndarray([...], ...)` — is the shape correct?
- Every `nisa.tensor_tensor(dst=A, data1=B, data2=C)` — do A, B, C have the same shape?
- Every `nisa.tensor_scalar(dst=A, data=B, ..., operand0=S)` — is S actually scalar-like?
- Every `nisa.dma_copy(dst=D, src=S)` — do D and S have the same shape?

### Step 3: Add debug prints to Helion codegen

Add `print(...)` statements to the relevant codegen functions:
- `memory_ops.py` load/store codegen — print slice expressions, shapes
- `backend.py` `reduction_expr` — print `fake_input.shape`, `part_size`
- `reduction_strategy.py` — print `shape_dims`, `sbuf_shape`
- `aten_lowering.py` `codegen_full_nki` — print partition_dim, free_dims

Remember to clear `/tmp/torchinductor_ubuntu/` after adding prints.

### Step 4: Compare with manual kernel

Run the debug scripts (`debug_layer_norm.py`, `debug_layer_norm_bwd.py`) to verify the algorithm is correct in isolation.

### Step 5: Common error messages

| Error | Likely Cause |
|-------|-------------|
| `_nki_sbuf_N not found` | Load codegen produced wrong var name or tile list issue |
| `Expect AP same number of elements [[A,B],[C,D]]` | Shape mismatch in `tensor_tensor` — one operand is wrong shape |
| `DMA Copy in/out shape must match` | Store/load slice shape doesn't match SBUF buffer shape |
| `too big for SB, requires N bytes` | Tile too large — reduce block_size or split into tile list |
| `BIR verification failed` | Usually a shape issue in tensor ops — check generated kernel |
| `NRT_FAILURE` | Neuron runtime crash — kill stale processes, restart |

---

## 11. Remaining TODOs

### Immediate (BLOCKING)

1. **Fix `grad_weight` numerical mismatch** — The backward pass compiles and runs but `grad_weight` (tensor 1, shape `[8192]`) is numerically wrong. Confirmed 2026-03-11. The kernel structure is correct (shapes match), but the accumulation logic produces wrong values.

   **How to investigate**:
   ```bash
   # Look at generated kernel
   find /tmp/torchinductor_ubuntu -name '*.py' -exec grep -l '_helion_layer_norm_bwd' {} \;
   # Known location: /tmp/torchinductor_ubuntu/lf/clfv24b7wpudknvtmjwitba65k3ihkua7gzhjhijhunn6nsp4mwy.py
   ```
   Check lines 83-91 in that file — the `tensor_partition_reduce` path for grad_w_acc:
   ```python
   nki_part_reduce = nl.ndarray([1, 8192], nl.float32, buffer=nl.sbuf)
   nisa.tensor_partition_reduce(dst=nki_part_reduce, op=nl.add, data=nki_binary)
   sum_1 = nki_part_reduce[0, :]
   nisa.tensor_tensor(dst=_nki_full, data1=_nki_full, data2=sum_1, op=nl.add)
   ```
   This reduces a `[1, 8192]` tensor along its partition axis (1 element) — which is a no-op. So `sum_1` equals `nki_binary[0, :]` unmodified, and then it adds to `_nki_full` correctly. This looks correct.

   The likely issue may be in the **variable copy aliasing** (`_nki_cast_copy`, `_nki_full_copy` etc.) in lines 51-56 of the generated kernel — NKI may not support Python-style variable aliasing inside `sequential_range` loops.

   **Alternative debug approach**: Use `debug_layer_norm_bwd.py` as a reference. Scale it up to full dimensions and compare results to the Helion-generated kernel output manually.

2. **Remove debug print**: Remove the `[REDUCE_DBG]` print at `reduction_strategy.py:501`.

3. **Test without-bias backward**: The backward pass has a `compute_bias_grad` parameter. Need to verify the `bias=None` path also works.

### Short-term

4. **Harden the partition size resolution**: The current 3-layer approach (SBUF registry → shape_dims eval → hint-to-config override) is fragile. A cleaner solution would be to ensure `fake_input.size(0)` always returns a SymInt (not a concrete int) during FX tracing, so the standard substitution path always works.

5. **Investigate loop variable aliasing**: The generated kernel has lines like `_nki_cast_copy = _nki_cast` and `_nki_full_copy = _nki_full` inside `sequential_range`. These are Python variable aliases — NKI likely treats them as the same SBUF pointer. If the loop is unrolled differently in the NKI compiler this could cause correctness issues.

6. **Port more examples**: See `EXAMPLES_NKI_STATUS.md` for priorities:
   - rms_norm (needs mean reduction — now supported)
   - layer_norm_f32 (same pattern, fp32 inputs)
   - cross_entropy, kl_div, jsd (need indexing, log-space ops)

### Long-term

7. **NKI masking**: Currently requires dimensions to be exact multiples of block sizes. Need `where_expr` support for partial tiles.

8. **Tile list optimization**: The `[1, N]` path avoids tile lists but limits partition parallelism. For large N, could benefit from partition-split tile lists with correct free-dim accumulation.

9. **SBUF memory management**: No explicit tracking of SBUF usage. Large kernels could overflow without warning.

---

## 12. Common Pitfalls

### 1. Forgetting to clear cache

After **any** change to Helion codegen, run:
```bash
rm -rf /tmp/torchinductor_ubuntu/
```
Otherwise you'll run stale generated code and wonder why your fix didn't work.

### 2. SymInt hints vs config values

FX tracing assigns "hints" to SymInts that may differ from the actual configured values. For example:
- `hl.register_block_size(x.size(0))` creates a SymInt with `hint=64` (default)
- But the config may set it to `32`
- `fake_input.size(0)` may return the hint as a concrete int

Always resolve through config substitutions, never trust raw `int(fake_input.size(0))`.

### 3. NKI 2D requirement

All NKI SBUF tensors must be at least 2D (`[partition, free]`). If your codegen produces a 1D buffer, NKI will crash. Always ensure:
- Loads produce `[p, f]` buffers (add trailing 1 when free_dims is empty)
- Stores source from 2D buffers
- Casts produce 2D output

### 4. tensor_tensor vs tensor_scalar

`tensor_tensor` requires **exact shape match** on both operands. If one operand is effectively a scalar (e.g., `[1, 1]`), you **must** use `tensor_scalar` instead. The `_is_scalar_like_tensor` check handles this, but new ops need to go through the same check.

### 5. Neuron runtime single-process

Only one Python process can use the Neuron device at a time. If a previous run crashed, kill it before starting a new one.

### 6. SBUF size limit

SBUF is ~512KB. A `[32, 8192]` fp32 buffer = 32 × 8192 × 4 = 1MB — too large. The solution is either:
- Use `block_size=1` for the inner loop (current approach for layer_norm_bwd)
- Split into tile lists of 128 elements each

---

## 13. Key NKI Concepts Reference

### Tensor Layout

```
SBUF tensor: [partition_dim, free_dim]
  - partition_dim ≤ 128 per tile (hardware neuron cores)
  - free_dim: any size (PE SIMD lanes)
```

### Key NKI ISA Operations

```python
# Memory
nisa.dma_copy(dst=sbuf, src=hbm_slice)    # HBM → SBUF
nisa.dma_copy(dst=hbm_slice, src=sbuf)    # SBUF → HBM

# Allocation
buf = nl.ndarray([p, f], dtype, buffer=nl.sbuf)  # SBUF buffer
buf = nl.ndarray([p, f], dtype, buffer=nl.shared_hbm)  # HBM buffer

# Initialize
nisa.memset(buf, value=0)

# Compute (element-wise, matching shapes)
nisa.tensor_tensor(dst=C, data1=A, data2=B, op=nl.add)      # C = A + B
nisa.tensor_tensor(dst=C, data1=A, data2=B, op=nl.multiply)  # C = A * B
nisa.tensor_tensor(dst=C, data1=A, data2=B, op=nl.subtract)  # C = A - B

# Compute (broadcast scalar along free dim)
nisa.tensor_scalar(dst=C, data=A, op0=nl.multiply, operand0=scalar, op1=None)
nisa.tensor_scalar(dst=C, data=A, op0=nl.subtract, operand0=scalar, op1=None)
nisa.tensor_scalar(dst=C, data=A, op0=nl.add, operand0=scalar, op1=None)

# Reduce along free axis (axis=1)
nisa.tensor_reduce(dst=result, op=nl.add, data=input, axis=1, keepdims=True)

# Reduce along partition axis (axis=0)
nisa.tensor_partition_reduce(dst=result, op=nl.add, data=input)

# Activation functions
nisa.activation(dst=result, op=nl.rsqrt, data=input)

# Loops
for i in nl.affine_range(start, stop, step):    # parallel iterations
for i in nl.sequential_range(start, stop, step): # sequential iterations
```

### dtype cast pattern (fp16 → fp32)

```python
fp32_buf = nl.ndarray([p, f], nl.float32, buffer=nl.sbuf)
nisa.memset(fp32_buf, value=0)
nisa.tensor_tensor(dst=fp32_buf, data1=fp32_buf, data2=fp16_src, op=nl.add)
# The add with mismatched dtypes performs the conversion on hardware
```

---

## Appendix: Diff Summary

To see all changes relative to the original codebase:
```bash
cd /home/ubuntu/kernel_test/helion_nki
# If git is set up:
git diff
# Or compare against the unmodified copy:
diff -r helion_nki/ heng_helion_nki/ --exclude='__pycache__' --exclude='*.pyc' | head -500
```

---

*End of handoff document.*
