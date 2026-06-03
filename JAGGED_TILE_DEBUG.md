# Jagged Tile NKI Backend Debug State

**Branch**: `nki-load-codegen-refactor`  
**Last updated**: 2026-06-03

---

## Test Results (current working tree, seed=42)

| Example | Size | Status | max_diff | Notes |
|---|---|---|---|---|
| `jagged_sum` | rows=32, maxcols=16 | ✅ PASS | 9.5e-7 | Small size passes |
| `jagged_sum` | rows=128, maxcols=64 | ❌ FAIL | ~164 | Fails at large size |
| `jagged_mean` | rows=32, M=128 | ✅ PASS | 2.1e-5 | Fixed this session |
| `jagged_layer_norm` | B=32, M=32 | ❌ FAIL | 0.46 | Down from 2.29, pre-existing baseline is also 0.46 |
| `jagged_softmax` | rows=32, M=32 | ❌ FAIL | ~1.0 | Mask not applied to max/exp reductions |
| `jagged_dense_add` | rows=128, cols=128 | ✅ PASS | 0 | Small size passes |
| `jagged_dense_add` | rows=256, cols=5000 | ❌ FAIL | 4.33 | Fails at large size — 39-iter dynamic_range bug |

---

## Committed Changes (commit `d23b1cc3`)

Fixed `jagged_mean` — three coordinated compiler changes:

1. **`helion/_compiler/tile_strategy.py`**: Detect when an outer jagged tile has a nested inner jagged tile by checking `active_device_loops` for a lower-block-id jagged tile. When detected, demote the outer tile from `dynamic_range` to `affine_range` (static max_size bound). neuronx-cc does not support nested `dynamic_range` loops.

2. **`helion/language/memory_ops.py`**: Gather predicate mask selection now iterates `active_device_loops` in **descending** block_id order so the innermost jagged tile's mask is picked first.

3. **`helion/_compiler/backend.py`**: In `where()` codegen, AND outer jagged tile masks (those converted to affine_range) into the predicate to zero out positions beyond each row's valid feature count.

---

## Uncommitted Changes (working tree)

### `helion/_compiler/tile_strategy.py` (+77 lines net)

**Root cause fixed**: The original commit used block_id ordering to detect outer jagged tiles (`any(other_bid > block_idx for other_bid in jagged_tile_parent_ids)`). This was correct for `jagged_mean` (truly nested) but too broad — it also fired for `jagged_layer_norm`'s three **sequential** tile_k passes, wrongly converting all three to `sequential_range(0, 8192, 64)` (static large bound) instead of `dynamic_range`.

**New detection logic** (retroactive demotion):
- Every jagged tile initially gets `dynamic_range` setup.
- When an INNER nested jagged tile is detected (a lower-block-id jagged tile is in `active_device_loops`), the OUTER tile's already-emitted `for_node.iter` is retroactively rewritten from `dynamic_range` to `affine_range(0, max_size, step)`.
- Sequential jagged tiles never see each other in `active_device_loops` (each is cleaned up when its `add_device_loop` context exits).

**Debug instrumentation**: Set `HELION_DEBUG_DYNRANGE=1` to trace decisions at each `codegen_device_loop` call.

### `helion/_compiler/generate_ast.py` (+15 lines)

After `add_device_loop`'s `yield` returns, removes the current loop's block_ids from `_nki_dyn_loops`. This ensures sequential sibling jagged loops don't see each other's `_nki_dyn_loops` entries when checking `_is_outer_jagged`. Nested loops still work because the outer loop's entry is present during the entire inner yield.

---

## Bug Analysis: Each Failing Test

### `jagged_sum` — size-dependent failure

- **Passes**: rows=32, maxcols=16 (≤32 iterations of `dynamic_range`)
- **Fails**: rows=128, maxcols=64 (up to 64-iter `dynamic_range`, max_diff~164)
- **Pattern**: Output values are ~17× larger than reference — looks like multiple passes accumulating into the same output location.
- **Threshold**: Bug appears between 32 and 39 iterations (established in previous debugging of `jagged_dense_add`, likely same root cause).
- **Root cause hypothesis**: NKI's neuronx-cc recycles some SBUF slot (likely `_dyn_counter` or related) after ~32 iterations of `dynamic_range`, causing the column offset counter to reset to 0. Later iterations then write to cols 0..N (overwriting correct earlier values) instead of their correct col offset.
- **Not a regression**: This failure exists in the committed baseline (`d23b1cc3`) for large sizes. Small sizes pass.

### `jagged_layer_norm` — max_diff 0.46

- **Root cause**: With the uncommitted fix, all three `tile_k` passes now use `dynamic_range`. Max_diff dropped from 2.29 → 0.46.
- **Remaining 0.46**: This is the pre-existing baseline failure (exists in the commit before this session). The output values are close but off — likely a floating-point accumulation error from the three-pass mean/var/normalize structure. Not caused by the dynamic_range fix.
- **Status**: The dynamic_range fix is correct. The 0.46 residual needs separate investigation.

### `jagged_softmax` — max_diff ~1.0

- **Root cause**: The first pass (compute max/L for online softmax) uses `dynamic_range` correctly now, but `mask_2` (the jagged validity mask: `tile_k_index < seqlens`) is **generated but never applied** to the gathered data before the max-reduction and exp-sum. OOB positions in the gather get 0.0 (from `oob_mode=skip`), which:
  - Corrupts `block_max`: if valid x values are negative, `max(x, 0.0)` picks 0 instead of `max(x)`
  - Inflates `block_L`: `exp(0 - block_max)` adds spurious probability mass for invalid k positions
- **Manual fix confirmed**: Applying `mask_2` to replace OOB positions with `-inf` before max-reduce, and zero them after exp, gives correct results (max_diff → 2.3e-6).
- **Compiler fix needed**: In `_try_emit_flat_gather_sum_dim1` or the NKI load path, when a gather is inside a jagged tile loop and uses `oob_mode=skip`, apply the jagged mask to set OOB positions to the reduction's identity element (`-inf` for max, `0` for add) rather than leaving them as 0.

### `jagged_dense_add` — size-dependent failure (rows=256, cols=5000)

- **Passes**: rows=128, cols=128 (≤32 iterations of `dynamic_range`)
- **Fails**: rows=256, cols=5000 (up to 40 iterations of `dynamic_range` for max_nnz≈4993, max_diff=4.33)
- **Pattern**: Exactly 14883 wrong cells in batch 0 (rows 0-127), cols 0..119. These are positions where `h == y` (x_data contribution missing). Batch 1 (rows 128-255) has 896 wrong cells from the second loop.
- **Threshold confirmed**: `dynamic_range` with bound ≤4096 (32 iterations) passes. Bound=5000 (39 iterations) fails. The breakpoint is between 32 and 39 iterations.
- **Ruled out**: SBUF aliasing, float counter init, HBM-based counter, register-based counter, reloading starts/nnz from HBM. All produce identical failure counts — the root cause is NOT in any of these variables.
- **Current hypothesis**: NKI's neuronx-cc recycles SBUF memory for the `_dyn_counter` SBUF after ~32 loop iterations. At iteration 33+, the counter reads 0 instead of the correct offset (e.g., 4096), causing the DMA AP `scalar_offset=_dyn_counter` to write to cols 0..127 instead of cols 4096..4223. This overwrites earlier correct (y+x) values with pure-y values.
- **Key evidence**: The threshold is exactly at 32 iterations. The SBUF recycling is a neuronx-cc compiler behavior for long `dynamic_range` loops — beyond some SBUF-usage threshold, the compiler reuses SBUF slots from the loop preamble for loop-body variables.
- **Suspected fix**: Either (a) use a NKI register (not SBUF) to hold the counter, or (b) find a way to prevent neuronx-cc from recycling the SBUF. Option (a) was attempted but the register approach doesn't expose the counter value as a DMA AP `scalar_offset` (which requires NkiTensor, not VirtualRegister). Alternative: store counter in `shared_hbm` and reload each iteration (tested, same result — probably because it hits the same cached kernel).

---

## Quick Repro

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /home/ubuntu/helion_nki

# All tests
rm -rf /tmp/jagged_check && HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/jagged_check python3 -c "
import sys; sys.path.insert(0, '.')
# ... (see test script above)
"

# Debug dynamic_range decisions
HELION_DEBUG_DYNRANGE=1 HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/dbg python3 -c "
import sys; sys.path.insert(0, '.')
from examples.jagged_layer_norm import jagged_layer_norm_kernel, create_test_jagged_tensor
from helion._testing import DEVICE; import torch
x, o = create_test_jagged_tensor(32, 32, 128, DEVICE, torch.float32)
try: jagged_layer_norm_kernel(x, o, 1e-6)
except: pass
"
```

---

## Files Changed (working tree vs HEAD)

| File | Change | Purpose |
|---|---|---|
| `helion/_compiler/tile_strategy.py` | +77 net | Retroactive outer-tile demotion; debug prints |
| `helion/_compiler/generate_ast.py` | +15 | `_nki_dyn_loops` cleanup after loop exits |

**5 example files** have cosmetic diffs (unrelated to jagged work, can be discarded).

---

## Next Steps (priority order)

1. **Remove debug prints from tile_strategy.py** (the `HELION_DEBUG_DYNRANGE` blocks), commit the fix.
2. **`jagged_softmax`**: In the NKI load/reduce path, apply the jagged mask to OOB positions before partition-reduce (max) and exp-sum. The mask `mask_2` is available in the generated kernel but unused in the reduce path.
3. **`jagged_sum` / `jagged_dense_add` size threshold**: Understand why neuronx-cc recycles SBUF at 33+ iterations. Try wrapping the counter SBUF in a way that prevents recycling (e.g., mark it as "live across iterations" if such NKI annotation exists).
4. **`jagged_layer_norm` residual 0.46**: Investigate whether this is a precision issue in the three-pass mean/var accumulation or something structural.
