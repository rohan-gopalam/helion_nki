# Refactor: Replace AP Sentinel Strings with Dataclasses

## Background

NKI's DMA instruction (`nisa.dma_copy`) supports indirect memory access through the
`.ap()` modifier on tensor expressions. When Helion's load/store codegen encounters
a subscript that requires indirect addressing — a gather, a dynamic loop offset, or a
pre-built affine pattern — it cannot immediately emit the final NKI expression because
the full slice (free dimension size, total element count, strides) isn't assembled until
all dimensions of the subscript have been processed. It defers by encoding the relevant
information into a specially formatted string (a "sentinel") placed into `slice_parts`,
then decoding it later in `_build_hbm_src`.

There are three sentinels today, all in `helion/language/memory_ops.py`:

| Sentinel | Format | What it carries |
|---|---|---|
| `__AP_ROW_GATHER__` | `__AP_ROW_GATHER__{var}__` | Name of a `[P,1]` uint32 SBUF tile of row indices |
| `__AP_VEC_OFFSET__` | `__AP_VEC_OFFSET__{var}__{pattern}__` | Name of a vector-offset SBUF tile plus a pre-built pattern string |
| `__DYN_AP__` | `__DYN_AP__{counter}__{block_size}` | Name of a dynamic loop counter variable and the tile block size |

These are plain Python strings. Every consumption site does manual string operations:
`startswith`, `[len(prefix):]`, `.rstrip("_")`, `.rsplit("__", 1)`, string-in-string
checks. There are **~25 consumption sites** spread across `memory_ops.py` and one in
`atomic_ops.py`.

## Goal

Replace all three sentinel strings with two typed dataclasses. Eliminate all string
parsing at consumption sites. The `slice_parts` list becomes
`list[str | IndirectAP | DynamicAP]` where plain strings remain for ordinary contiguous
slices like `"offset_0:offset_0+128"`.

## New dataclasses — location: `helion/language/_nki_dim_access.py`

This file already exists and defines `DimAccess`, `Contiguous`, `Scalar`, `Indirect`,
`Dynamic`, `FullSlice`, and `StridedGather` for dimension classification. Add the two
new dataclasses here.

```python
@dataclasses.dataclass(frozen=True)
class IndirectAP:
    """Deferred indirect DMA: vector_offset= path in .ap().

    Covers both __AP_ROW_GATHER__ (pattern=None, needs_reshape=True) and
    __AP_VEC_OFFSET__ (pattern is pre-built string, needs_reshape=False).

    At consumption time in _build_hbm_src:
    - If pattern is None: compute flat_vec = vec_var * F + f_start, emit
      tensor.reshape([total_elems, 1]).ap(pattern=[[1,P],[1,F]], vector_offset=flat_vec, indirect_dim=0)
    - If pattern is a string: emit
      tensor.ap(pattern=<pattern>, vector_offset=vec_var, indirect_dim=0)
    """
    vec_var: str          # name of the [P,1] uint32 SBUF tile
    p_count: int          # number of partition rows (P)
    pattern: str | None   # None = compute from F at consumption; str = pre-built

@dataclasses.dataclass(frozen=True)
class DynamicAP:
    """Deferred dynamic-loop DMA: scalar_offset= path in .ap().

    Covers __DYN_AP__. The full .ap() pattern (strides, counts) is computed at
    consumption time in _build_hbm_src using the static dims in slice_parts and
    the tensor shape.
    """
    counter: str      # name of the dynamic loop counter SBUF variable
    block_size: int   # tile size for this dynamic dimension
```

Import these from `_nki_dim_access` at the top of `memory_ops.py` and `atomic_ops.py`.

## Creation sites — replace f-string with dataclass construction

All of these are in `memory_ops.py` unless noted.

### `__AP_ROW_GATHER__` creation (5 sites + 1 in atomic_ops)

Every `return f"__AP_ROW_GATHER__{vec_offset_var}__"` becomes
`return IndirectAP(vec_var=vec_offset_var, p_count=<P>, pattern=None)`.

The `p_count` is always known at creation time — it is the `partition_dim` or
`int(block_size)` argument passed to `_nki_row_index_gather`. Propagate it.

Specific sites:
- `memory_ops.py:1183` — inside `_nki_row_index_gather`, after tensor_scalar add
- `memory_ops.py:1290` — inside `_nki_row_index_gather`, mul+mod path
- `memory_ops.py:1309` — end of `_nki_row_index_gather`
- `memory_ops.py:1948` — 1D strided gather early-exit, `slice_parts = [...]`
- `memory_ops.py:2267` — 3D early-exit after flat_var computation
- `memory_ops.py:3990` — `_combine_leading_dims` case 1 output
- `memory_ops.py:4092` — store codegen strided gather
- `memory_ops.py:5199/5218` — store 3D early-exit
- `memory_ops.py:5430` — store strided gather
- `atomic_ops.py:585` — row-scatter RMW path (constructs sentinel string from
  `_nki_row_index_gather` result; replace the whole prefix+strip pattern with
  `isinstance(row_part, IndirectAP)` and `row_part.vec_var`)

### `__AP_VEC_OFFSET__` creation (1 site)

`memory_ops.py:953`: `return f"__AP_VEC_OFFSET__{vec_offset_var}__{pattern}__"`
becomes `return IndirectAP(vec_var=vec_offset_var, p_count=P, pattern=pattern)`.
`P` is already known (it's in the `pattern` string like `[[N,P],[1,F]]` — also pass
it as a field so consumption doesn't need to parse the pattern).

### `__DYN_AP__` creation (3 sites)

- `memory_ops.py:1605` — `_classify_load_dim`, dynamic loop path
- `memory_ops.py:1746` — `_classify_load_dim`, dynamic with SBUF offset
- `memory_ops.py:5046` — store codegen dynamic path

All become `DynamicAP(counter=_counter, block_size=int(block_size))`.
The return type of `_classify_load_dim` changes from `tuple[str, bool, ...]` to
`tuple[str | IndirectAP | DynamicAP, bool, ...]`.

## Consumption sites — replace string ops with attribute access

### `_build_hbm_src` (memory_ops.py ~line 4011) — main consumption site

This function currently has three separate `for _p in parts` loops, one per sentinel.
Replace with a single isinstance dispatch:

```python
for p in parts:
    if isinstance(p, IndirectAP):
        if p.pattern is not None:
            # VEC_OFFSET path — fully self-contained
            return f"{name_str}.ap(pattern={p.pattern}, vector_offset={p.vec_var}, indirect_dim=0)"
        else:
            # ROW_GATHER path — needs F from the free-dim slice_part
            free_part = parts[1] if len(parts) > 1 else f"0:{f_total}"
            # ... compute f_start, f_count from free_part (same logic as today) ...
            _flat_vec = device_fn.new_var("_ig_flat_vec", dce=True)
            state.codegen.add_statement(...)  # nl.ndarray + tensor_scalar * F + f_start
            pattern = f"[[1, {p.p_count}], [1, {f_count}]]"
            src = f"{name_str}.reshape([{_total_elems}, 1])"
            return f"{src}.ap(pattern={pattern}, vector_offset={_flat_vec}, indirect_dim=0)"

for p in parts:
    if isinstance(p, DynamicAP):
        # same stride/pattern computation as today, using p.counter and p.block_size
        ...
```

### `_combine_leading_dims` (memory_ops.py ~line 3800)

Replace:
```python
if sp.startswith("__AP_ROW_GATHER__"):
    vec_var_inner = sp[len("__AP_ROW_GATHER__"):].rstrip("_")
    count_inner = int(vec_shape_inner[0]) if vec_shape_inner else partition_dim
    leading_block_sizes.append(count_inner)
```
With:
```python
if isinstance(sp, IndirectAP):
    leading_block_sizes.append(sp.p_count)
```

Replace:
```python
if leading_offsets[0].startswith("__AP_ROW_GATHER__"):
    vec_var_0 = leading_offsets[0][len("__AP_ROW_GATHER__"):].rstrip("_")
```
With:
```python
if isinstance(leading_offsets[0], IndirectAP):
    vec_var_0 = leading_offsets[0].vec_var
```

The output of `_combine_leading_dims` also emits a new `IndirectAP` into `slice_parts`
(currently `f"__AP_ROW_GATHER__{flat_var_0}__"`) — replace with
`IndirectAP(vec_var=flat_var_0, p_count=p_count_0, pattern=None)`.

### Guard checks — any/not any (6 sites)

Every:
```python
not any(p.startswith(("__DYN_AP__", "__AP_ROW_GATHER__")) for p in slice_parts)
```
Becomes:
```python
not any(isinstance(p, (IndirectAP, DynamicAP)) for p in slice_parts)
```

Sites: `memory_ops.py:1865`, `4013`, `5277`, `5337`, `5445` (single check on `[0]`).

### Single-element checks on `slice_parts[0]`

- `memory_ops.py:1961`: `slice_parts[0].startswith("__AP_ROW_GATHER__")`
  → `isinstance(slice_parts[0], IndirectAP) and slice_parts[0].pattern is None`
- `memory_ops.py:1975`: same pattern
- `memory_ops.py:5445`: `slice_parts[0].startswith("__AP_ROW_GATHER__")`
  → `isinstance(slice_parts[0], IndirectAP)`

### Store codegen DMA builder (memory_ops.py ~line 6054)

Replace:
```python
_has_dyn_store = "__DYN_AP__" in slice_str
_has_row_store = "__AP_ROW_GATHER__" in slice_str
...
_vec_offset = _row_part[len("__AP_ROW_GATHER__"):].rstrip("_")
```
With direct isinstance checks on `slice_parts` (no need to search a string at all):
```python
_has_dyn_store = any(isinstance(p, DynamicAP) for p in slice_parts)
_has_row_store = any(isinstance(p, IndirectAP) for p in slice_parts)
...
row_part = next(p for p in slice_parts if isinstance(p, IndirectAP))
_vec_offset = row_part.vec_var
```

### `atomic_ops.py:585`

Replace the entire prefix+strip pattern:
```python
prefix = "__AP_ROW_GATHER__"
if row_part is None or not row_part.startswith(prefix) or not row_part.endswith("__"):
    return None
vec_offset = row_part[len(prefix):-2]
```
With:
```python
if not isinstance(row_part, IndirectAP):
    return None
vec_offset = row_part.vec_var
```

### Boundary/guard checks that exclude sentinels from plain-slice logic

Several places guard `":" in part` or similar plain-slice assumptions:
- `memory_ops.py:4387`, `4460`, `4602`, `4734`, `4823`, `5891`

These all do `part.startswith(("__DYN_AP__", ...))` to skip sentinel entries.
Replace each with `isinstance(part, (IndirectAP, DynamicAP))`.

## Type annotation changes

- `slice_parts: list[str]` → `list[str | IndirectAP | DynamicAP]` everywhere it appears
- `_classify_load_dim` return type: `tuple[str, bool, str | None, bool]`
  → `tuple[str | IndirectAP | DynamicAP, bool, str | None, bool]`
- `_nki_row_index_gather` return type: `str | None` → `IndirectAP | None`
- `_nki_indirect_gather` and `_nki_shifted_tile_subscript`: check return types —
  these return plain strings (contiguous slice expressions), unchanged

## What does NOT change

- The DMA instruction emitted to the NKI output file is identical. This is purely
  an internal codegen representation change.
- `_nki_dim_access.py`'s existing `Indirect`, `Dynamic`, `StridedGather` classes
  are separate — they classify *subscript dimensions* during load analysis, not the
  deferred AP expressions in `slice_parts`. The new `IndirectAP`/`DynamicAP` classes
  are about what gets placed into `slice_parts` at classification time and consumed
  at DMA-emission time. The names are distinct to avoid confusion.
- Triton/CUDA/other backends are completely unaffected — they never use `slice_parts`
  or any of these sentinels.

## Testing

After the refactor, all existing NKI tests should pass unchanged:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/helion_ap_refactor_test \
  python -m pytest test/test_nki_dynamic_loops.py -x -s
```

And the full example suite (or a targeted subset) to confirm no regression in the
generated NKI code:

```bash
HELION_BACKEND=nki NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
  TORCHINDUCTOR_CACHE_DIR=/tmp/helion_ap_refactor_examples \
  python examples/embedding.py    # exercises ROW_GATHER
python examples/attention.py      # exercises contiguous + DYN_AP
python examples/layer_norm.py     # exercises contiguous only (regression guard)
```
