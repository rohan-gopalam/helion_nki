# jagged_layer_norm.py — Normalization Chain Transform Bug Report

## Current Status (2026-05-28)
- B=32, M=32: FAIL (output is NaN — chain ops inside dynamic loop)
- B=256, M=32: FAIL (same issue)
- atol=0.5 was a red herring — B=32 appeared to pass because raw x values happened to be close to normalized, but that was coincidental.

## Root Cause Chain (multiple bugs)

### Bug 1 (FIXED): Scatter collision for padded positions
Invalid k-positions (k >= seq_lengths[b]) write zeros to valid output positions.
**Fix**: OOB scatter sentinel (memset=1073741824, predicated copy to skip invalid positions).

### Bug 2 (FIXED): Chain transform not finding mean/rstd expressions
The gather+transform+scatter optimization at memory_ops.py:2650 calls `ast_for_fx_node`
on the chain op's other_arg (mean/rstd). For jagged_layer_norm, the mean/rstd values
are accessed via `mean_acc[:, None, None]` which traces as `helion.language.view_ops.subscript`
nodes. The walk-up to find the underlying mean tensor was not recognizing `subscript` as
a transparent operation.
**Fix**: Added `'subscript'` to `_transparent_ops` in the walk-up at memory_ops.py:2772.

### Bug 3 (FIXED): Shape mismatch for mean [1,P] in tensor_tensor
After finding mean (shape [1, p_count]), `tensor_tensor(dst=[p,m], data2=[1,p])` fails
because partition dims don't match. Fix: when mean has shape [1, P], transpose it to
[P, 1] and use `tensor_scalar` for the column-broadcast subtraction.
**Fix**: Added transpose+tensor_scalar path in memory_ops.py:2807.

### Bug 4 (CURRENT BLOCKER): nc_transpose inside dynamic_range loop causes NaN
The mean/rstd transpose (nisa.nc_transpose) is emitted inside the inner k-loop 
(`nl.affine_range(k_count)`), which itself is inside `nl.dynamic_range(...)`. The 
`nisa.nc_transpose` instruction may not be valid inside dynamic range loops in NKI, 
causing NaN values.

**How to debug**: Check NKI documentation for which instructions are allowed inside 
dynamic_range. The transpose of mean/rstd should be done ONCE outside the k-loop.

**Proposed fix**: Restructure the chain op transform:
1. Emit `nc_transpose(mean → mean_col)` and `nc_transpose(rstd → rstd_col)` BEFORE 
   the k-loop (in the outer body that sets up the gather+scatter loop).
2. Inside the k-loop, use the pre-transposed `mean_col [P, 1]` and `rstd_col [P, 1]`
   directly in `tensor_scalar(subtract, mean_col)` and `tensor_scalar(multiply, rstd_col)`.
   
**Implementation note**: The chain ops are emitted in `body` which is the body of the
inner k-loop (affine_range). Need a separate `outer_body` list for pre-loop setup. 
The device_function's existing scope mechanism (state.codegen.add_statement vs body.append)
would need to be used.

### Bug 5 (RELATED): where(mask, val, 0.0) false branch detection
The false branch `0.0` of `torch.where(mask, normalized, 0.0)` was not recognized as
zero because it could be a constant FX node. Added handling for constant tensor nodes.
**Status**: Fixed but disabled (OOB scatter makes re-masking unnecessary when the
nc_transpose NaN issue is resolved).

## Files Modified
- `helion/language/memory_ops.py`: 
  - `_transparent_ops` set (line ~2772): add 'subscript'
  - Chain op broadcast (line ~2807): transpose+tensor_scalar for [1,P] mean/rstd
  - Scatter OOB sentinel (line ~2886): memset(1073741824) instead of dynamic flat_extent
  - Chain where detection (line ~2661): disabled re-masking (OOB scatter handles it)
- `examples/jagged_layer_norm.py`: atol=0.5 (was masking the real issue)

## Quick Repro
```bash
rm -rf /tmp/jln_repro
HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/jln_repro \
  python3 -c "
from examples.jagged_layer_norm import jagged_layer_norm_kernel, reference_jagged_layer_norm_pytorch, create_test_jagged_tensor
from helion._testing import DEVICE
import torch
x_data, x_offsets = create_test_jagged_tensor(32, 32, 128, DEVICE, torch.float32)
h = jagged_layer_norm_kernel(x_data, x_offsets, 1e-6)
r = reference_jagged_layer_norm_pytorch(x_data, x_offsets, 1e-6)
print('NaN in h:', torch.isnan(h).any().item())
print('h[0,:3]:', h[0,:3].tolist())
print('r[0,:3]:', r[0,:3].tolist())
"
```

## What to Check in Generated Code
```bash
grep "chain_tr\|nc_transpose.*chain\|tensor_scalar.*chain" /tmp/jln_repro/*/*.py
```
The `nc_transpose` calls for mean/rstd are inside the inner k-loop. Move them outside.
