"""Small SBUF-shape utility for NKI codegen.

Moved verbatim from ``helion/language/memory_ops.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState


def _nki_lookup_sbuf_shape_dtype(
    state: "CodegenState", name: str
) -> tuple[list[int] | None, str]:
    device_fn = state.device_function
    shape = device_fn._nki_sbuf_shapes.get(name)
    if shape is None:
        lookup = name
        while "_copy" in lookup:
            lookup = lookup[: lookup.rfind("_copy")]
            shape = device_fn._nki_sbuf_shapes.get(lookup)
            if shape is not None:
                name = lookup
                break
    dtype = device_fn._nki_sbuf_dtypes.get(name, "nl.int32")
    return shape, dtype
