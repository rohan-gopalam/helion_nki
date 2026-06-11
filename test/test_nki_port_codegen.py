"""Codegen snapshot/structure tests for the NKI backend port.

These run WITHOUT Trainium hardware: ``to_triton_code`` is pure Python. They
guard against codegen regressions in the ported NKI backend by asserting the
shape of the generated ``@nki.jit`` kernel for representative patterns.

Run: HELION_BACKEND=nki HELION_NEURON_TARGET=trn2 \
     python -m pytest test/test_nki_port_codegen.py -v

The byte-for-byte parity with the reference branch (fix-nki-kernel-compilation)
was verified during the port for copy/matmul/reduce (identical) and gather
(semantically identical; see docs/nki_port/NKI_PORT_COMMIT_LOG.md P1.22). These tests assert
the structural invariants so future changes can't silently regress them.
"""

from __future__ import annotations

import os

os.environ.setdefault("HELION_BACKEND", "nki")
os.environ.setdefault("HELION_NEURON_TARGET", "trn2")

import torch

import helion
import helion.language as hl


def _code(kernel_fn: object, args: tuple) -> str:
    bound = kernel_fn.bind(args)  # type: ignore[attr-defined]
    return bound.to_triton_code(bound._config)  # type: ignore[attr-defined]


def test_copy_codegen() -> None:
    @helion.kernel(config=helion.Config(block_sizes=[128]))
    def k(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for t in hl.tile(x.size(0)):
            out[t] = x[t]
        return out

    code = _code(k, (torch.zeros(256),))
    assert "@nki.jit" in code
    assert "nisa.dma_copy" in code
    assert "nl.affine_range" in code
    assert "tl." not in code and "triton" not in code
    # host wrapper must capture + reshape the launcher return into `out`
    assert "= _launcher(" in code


def test_matmul_codegen() -> None:
    @helion.kernel(config=helion.Config(block_sizes=[128, 128, 128]))
    def k(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, kk = x.shape
        _, n = y.shape
        out = torch.zeros([m, n], device=x.device, dtype=x.dtype)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(kk):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc
        return out

    code = _code(k, (torch.zeros(256, 256), torch.zeros(256, 256)))
    assert "@nki.jit" in code
    assert "nc_matmul" in code
    assert "tl." not in code


def test_reduce_codegen() -> None:
    @helion.kernel(config=helion.Config(block_sizes=[64]))
    def k(x: torch.Tensor) -> torch.Tensor:
        m, _ = x.shape
        out = torch.empty([m], dtype=x.dtype, device=x.device)
        for tile_m in hl.tile(m):
            out[tile_m] = x[tile_m, :].sum(-1)
        return out

    code = _code(k, (torch.zeros(256, 512),))
    assert "@nki.jit" in code
    assert "tensor_reduce" in code
    assert "tl." not in code


def test_gather_codegen() -> None:
    @helion.kernel(config=helion.Config(block_sizes=[64]))
    def k(idx: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        n = idx.size(0)
        d = table.size(1)
        out = torch.empty([n, d], dtype=table.dtype, device=table.device)
        for t in hl.tile(n):
            out[t, :] = table[idx[t], :]
        return out

    code = _code(k, (torch.zeros(128, dtype=torch.int32), torch.zeros(512, 256)))
    assert "@nki.jit" in code
    assert ".ap(" in code or "dma_copy" in code  # indirect gather path
    assert "tl." not in code
