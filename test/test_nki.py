"""End-to-end tests for the NKI (AWS Trainium) backend.

Mirrors ``test_pallas.py``: dedicated ``@helion.kernel(backend="nki")`` kernels
and a backend-gated test class. The class is gated by ``@onlyBackends(["nki"])``
+ ``@skipUnlessNKI`` so it only runs when ``HELION_BACKEND=nki`` AND a runtime is
available (Trainium via torch_xla, or CPU via the ``nki`` simulator with
``HELION_NKI_SIMULATE=1``); on every other configuration the whole class skips.

Each test asserts BOTH the structural shape of the generated ``@nki.jit`` source
(no Triton leakage; the expected NKI ISA ops are present) AND numerical
correctness against a PyTorch reference via ``code_and_output``.

Run on Trainium:
    HELION_BACKEND=nki HELION_NEURON_TARGET=trn2 \
        python -m pytest test/test_nki.py -v
Run on CPU (no hardware) via the simulator:
    HELION_BACKEND=nki HELION_NKI_SIMULATE=1 HELION_NEURON_TARGET=trn2 \
        python -m pytest test/test_nki.py -v

Pure-codegen structural tests that need no runtime live in
``test_nki_port_codegen.py``.
"""

from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessNKI
import helion.language as hl


@helion.kernel(backend="nki", static_shapes=True)
def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(backend="nki", static_shapes=True)
def matmul_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
        out[tile_m, tile_n] = acc
    return out


@helion.kernel(backend="nki", static_shapes=True)
def row_sum_kernel(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.shape
    out = torch.empty([n], dtype=torch.float32, device=x.device)
    for tile_n in hl.tile(n):
        out[tile_n] = torch.sum(x[tile_n, :], dim=-1)
    return out


@helion.kernel(backend="nki", static_shapes=True)
def softmax_kernel(x: torch.Tensor) -> torch.Tensor:
    n, _m = x.shape
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        row = x[tile_n, :]
        row = row - torch.amax(row, dim=-1, keepdim=True)
        e = torch.exp(row)
        out[tile_n, :] = e / torch.sum(e, dim=-1, keepdim=True)
    return out


@onlyBackends(["nki"])
@skipUnlessNKI("NKI backend requires Trainium (torch_xla) or HELION_NKI_SIMULATE=1 + nki")
class TestNKI(TestCase):
    def test_add(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(add_kernel, (x, y), block_sizes=[128, 128])
        # NKI codegen, no Triton leakage
        assert "@nki.jit" in code
        assert "tl." not in code and "triton" not in code
        torch.testing.assert_close(result, x + y, rtol=1e-4, atol=1e-4)

    def test_matmul(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(
            matmul_kernel, (x, y), block_sizes=[128, 128, 128]
        )
        assert "@nki.jit" in code
        assert "nisa.nc_matmul" in code
        torch.testing.assert_close(result, x @ y, rtol=1e-2, atol=1e-2)

    def test_row_sum(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(row_sum_kernel, (x,), block_sizes=[128])
        assert "@nki.jit" in code
        assert "tl." not in code and "triton" not in code
        torch.testing.assert_close(result, x.sum(dim=-1), rtol=1e-3, atol=1e-3)

    def test_softmax(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        code, result = code_and_output(softmax_kernel, (x,), block_sizes=[128])
        assert "@nki.jit" in code
        assert "tl." not in code and "triton" not in code
        torch.testing.assert_close(
            result, torch.softmax(x, dim=-1), rtol=1e-3, atol=1e-3
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
