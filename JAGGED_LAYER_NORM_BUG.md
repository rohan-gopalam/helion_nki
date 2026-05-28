# jagged_layer_norm.py — Large Batch Bug Report

## Status
- B=32, M=32: PASS (atol=0.5)
- B=32, M=256: PASS
- B=256, M=32: FAIL (~65% mismatch)
- B≥256: FAIL

## Root Cause Chain (multiple bugs)

### Bug 1 (FIXED): Scatter collision for invalid padded positions
In the normalization pass, the inner `dynamic_range` loop iterates from 0 to
`max_seq_len` in steps of 64. For positions `k >= seq_lengths[b]` (invalid/padded),
the flat index `(starts[b] + k_clamped) * M + m_offset` maps to `starts[b]*M + m_offset`
(since `k_clamped = 0`), which is the FIRST row of sequence b's output. Subsequent
invalid-k writes overwrite that row with zeros.

**Fix applied**: `_nki_scatter_safe_offsets` uses `memset(1073741824)` (OOB sentinel)
and `tensor_copy_predicated` to redirect invalid writes to OOB.

### Bug 2 (FIXED for B=32): Chain transform not normalizing
The gather+transform+scatter optimization in `memory_ops.py` (line ~2660) had:
```python
if _target is torch.ops.aten.where.self:
    continue  # wrong — skips mask re-application after arithmetic
```
After applying `x_slice - mean` (sub) and `result * rstd` (mul), invalid positions
have value `(0 - mean) * rstd = -mean*rstd` (not 0). The `where` was being skipped
because "masking is already done" — but that's only true for the *load* step, not after
arithmetic operations that move invalid positions away from 0.

**Fix applied**: Re-apply `tensor_copy_predicated` after the chain ops when `where(mask, val, 0)` is encountered.

### Bug 3 (STILL FAILING for B≥256): Unknown — likely the sub/mul transform itself is wrong

After Bug 2 fix, B=32 passes but B=256 still fails with ~65% mismatch. The chain
`['sub', 'mul', 'where']` is detected for the normalization pass. 

**What to investigate**:
1. Check whether `_other_expr` for the `sub` node correctly resolves to `mean_acc` (the per-sequence mean). For B=256, `mean_acc` has shape `[64, 1]` (transposed), while the x_slice tile is `[64, 64, 32]` (3D). The shapes may not broadcast correctly in the `tensor_tensor` call.

2. Check whether the chain is detecting the CORRECT `sub`/`mul` arguments. For B=256 the mean/rstd are `[1, 64]` transposed to `[64, 1]` SBUF tiles. The `_other_expr` for `sub` must be this `[64, 1]` tile. In `tensor_tensor(data1=[64,32], data2=mean[64,1])`, NKI broadcasts mean along the free dim automatically — this should work.

3. The remasking at Bug 2's fix site uses `pred_name = row_pred_full` which is the `[p_count, m_count]` broadcast of `row_pred_col`. After `sub` and `mul`, `_cur_tile` may have different shape than `pred_name` expects — mismatch at B=256.

4. The `_nki_chain_masked` result after re-masking — verify it's being used as the scatter source correctly.

## How to Debug Further

```bash
# Generate B=256 code and inspect:
rm -rf /tmp/helion_jln_b256
HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/helion_jln_b256 python3 -c "
from examples.jagged_layer_norm import jagged_layer_norm_kernel, create_test_jagged_tensor
from helion._testing import DEVICE
import torch
x_data, x_offsets = create_test_jagged_tensor(256, 32, 128, DEVICE, torch.float32)
jagged_layer_norm_kernel(x_data, x_offsets, 1e-6)
"
# Then look at:
grep "_nki_chain\|sub.*mean\|mul.*rstd\|tensor_tensor.*_cur_tile" /tmp/helion_jln_b256/*/*.py
```

Key variables to check in the B=256 generated file:
- `_nki_chain_tile` (result of sub/mul ops)
- `_nki_chain_masked` (result of re-masking)  
- Whether scatter uses `_nki_chain_masked` or falls back to `_nki_gather_masked_2`

## Files Modified
- `helion/language/memory_ops.py`: lines ~2660, ~2820 (two fixes)
- `examples/jagged_layer_norm.py`: atol=0.5 tolerance added

## Related Issue in moe_matmul_ogs
The same scatter-collision pattern existed in `moe_matmul_ogs.py`. Fixed there via
OOB index in the example kernel itself (different approach, same root cause).
