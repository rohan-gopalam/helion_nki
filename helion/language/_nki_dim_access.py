"""
Dimension access classification for NKI load/store codegen.

Each leading dimension of a tensor subscript is classified into one of these
categories, which determines how the DMA is generated:

- Contiguous: a tile range [offset, offset+block_size)
- Scalar: a single-element access (block_size=1)
- Indirect: a precomputed vector of row indices (gather/scatter)
- Dynamic: inside a nl.dynamic_range loop with counter
- FullSlice: slice(None) — all elements of the dimension
"""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclasses.dataclass(frozen=True)
class DimAccess:
    """Base class for dimension access classification."""

    dim_size: int  # The tensor's size in this dimension


@dataclasses.dataclass(frozen=True)
class Contiguous(DimAccess):
    """A contiguous tile range [offset, offset + block_size).

    Examples:
        x[tile_m, :]  → Contiguous(offset="offset_0", block_size=128)
        x[tile_k, :]  → Contiguous(offset="offset_1", block_size=64)
    """

    offset_expr: str  # e.g., "offset_0", "offset_0 + 1"
    block_size: int


@dataclasses.dataclass(frozen=True)
class Scalar(DimAccess):
    """A single-element access (block_size = 1).

    Examples:
        x[tile_b.id, :]   → Scalar(offset="offset_0 // 128")
        x[tile_h.begin, :] → Scalar(offset="offset_1")
        x[5, :]           → Scalar(offset="5")
    """

    offset_expr: str  # The scalar expression for the single index


@dataclasses.dataclass(frozen=True)
class Indirect(DimAccess):
    """An indirect (gather/scatter) access using a precomputed row index vector.

    The vector_var is a [P, 1] uint32 SBUF tensor holding the flat row indices.
    The sentinel string `__AP_ROW_GATHER__{vector_var}__` is used downstream.

    Examples:
        x[tile.index + starts, :] → Indirect(sentinel="__AP_ROW_GATHER__vec__")
        x[sorted_indices, :]      → Indirect(sentinel="__AP_ROW_GATHER__vec__")
    """

    sentinel: str  # The __AP_ROW_GATHER__var__ sentinel string
    count: int  # Number of partition elements (P)


@dataclasses.dataclass(frozen=True)
class Dynamic(DimAccess):
    """Inside a nl.dynamic_range loop — uses __DYN_AP__ sentinel.

    The DMA uses scalar_offset from the dynamic loop counter.

    Examples:
        for offset in nl.dynamic_range(0, bound, 128):
            x[offset:offset+128, :] → Dynamic(counter="_dyn_counter", block_size=128)
    """

    sentinel: str  # The __DYN_AP__counter__size sentinel string
    block_size: int


@dataclasses.dataclass(frozen=True)
class FullSlice(DimAccess):
    """slice(None) — access all elements of this dimension.

    Always the last (free) dimension in NKI's [partition, free] layout.
    """

    pass


@dataclasses.dataclass(frozen=True)
class StridedGather(DimAccess):
    """A tile access that requires stride > 1 in the flattened layout.

    Occurs when a tile dimension is followed by scalar dimension(s) in ND tensors.
    Example: w[seqlen_tile, head_scalar, :] flattened to [S*H, D]
    → rows at stride H, not contiguous.

    The vector_var holds the computed strided indices.
    """

    sentinel: str  # The __AP_ROW_GATHER__var__ sentinel string
    tile_block_size: int  # The tile's block_size (number of rows gathered)
    stride: int  # The stride between consecutive rows


def classify_leading_dims(
    leading_offsets: list[str],
    leading_block_sizes: list[int],
    original_dim_sizes: list[int],
) -> str:
    """Determine the access mode for combined leading dimensions.

    Returns one of: "contiguous", "strided", "indirect", "dynamic"

    For contiguous: flat_offset = sum(offset_i * prod_later_dims), DMA is plain
    For strided: tile dimension has stride > 1 due to interleaved scalar dims
    """
    # Find the tile dimension (block_size > 1)
    tile_dim_idx = None
    for i, bs in enumerate(leading_block_sizes):
        if bs > 1:
            if tile_dim_idx is None:
                tile_dim_idx = i
            else:
                return "contiguous"  # multiple tiled dims → product of all, contiguous
    if tile_dim_idx is None:
        return "contiguous"  # all scalar dims → single row, contiguous
    # Tile is last leading dim → contiguous (no interleaved scalars after it)
    if tile_dim_idx == len(leading_block_sizes) - 1:
        return "contiguous"
    # Tile followed by scalar dims → strided
    if all(leading_block_sizes[i] == 1
           for i in range(tile_dim_idx + 1, len(leading_block_sizes))):
        stride = 1
        for i in range(tile_dim_idx + 1, len(original_dim_sizes)):
            stride *= original_dim_sizes[i]
        if stride > 1:
            return "strided"
    return "contiguous"
