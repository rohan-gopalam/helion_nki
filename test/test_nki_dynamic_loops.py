from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
import helion.language as hl


@onlyBackends(["nki"])
class TestNKIDynamicLoops(TestCase):
    def test_tensor_bound_tile_loop_codegen(self) -> None:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_sum(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(x.size(1))
            for tile_m in hl.tile(x.size(0)):
                acc = hl.zeros([tile_m, bs])
                for tile_n in hl.tile(end[0], block_size=bs):
                    acc += x[tile_m, tile_n]
                out[tile_m] = acc.sum(-1)
            return out

        x = torch.randn([4, 16], device=DEVICE, dtype=torch.float32)
        end = torch.tensor([8], device=DEVICE, dtype=torch.int32)
        bound = dyn_sum.bind((x, end))
        code = bound.to_triton_code(bound.config_spec.default_config())

        self.assertIn("@nki.jit", code)
        self.assertIn("nl.dynamic_range(", code)
        self.assertIn("nisa.register_alloc", code)
        self.assertIn("nisa.register_load", code)
        self.assertIn(".ap(", code)
        self.assertNotIn("triton.jit", code)
        self.assertNotIn("tl.arange", code)

    def test_tensor_bound_tile_loop_runs(self) -> None:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_copy(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            bs = hl.register_block_size(x.size(0))
            for tile_n in hl.tile(x.size(1)):
                for tile_m in hl.tile(end[0], block_size=bs):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1
            return out

        x = torch.randn([16, 4], device=DEVICE, dtype=torch.float32)
        end = torch.tensor([8], device=DEVICE, dtype=torch.int32)
        _, result = code_and_output(dyn_copy, (x, end))
        expected = torch.zeros_like(x)
        expected[: int(end[0]), :] = x[: int(end[0]), :] + 1
        torch.testing.assert_close(result, expected)

    def test_tensor_bound_free_dim_tile_loop_runs(self) -> None:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_copy_free(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(x)
            bs = hl.register_block_size(x.size(1))
            for tile_m in hl.tile(x.size(0)):
                for tile_n in hl.tile(end[0], block_size=bs):
                    out[tile_m, tile_n] = x[tile_m, tile_n] + 1
            return out

        x = torch.randn([4, 16], device=DEVICE, dtype=torch.float32)
        end = torch.tensor([8], device=DEVICE, dtype=torch.int32)
        _, result = code_and_output(dyn_copy_free, (x, end))
        expected = torch.zeros_like(x)
        expected[:, : int(end[0])] = x[:, : int(end[0])] + 1
        torch.testing.assert_close(result, expected)

    def test_tensor_bound_reduction_loop_runs(self) -> None:
        @helion.kernel(
            backend="nki",
            autotune_effort="none",
            config=helion.Config(block_sizes=[4, 4]),
        )
        def dyn_sum(x: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
            out = x.new_empty([x.size(0)])
            bs = hl.register_block_size(x.size(1))
            for tile_m in hl.tile(x.size(0)):
                acc = hl.zeros([tile_m, bs])
                for tile_n in hl.tile(end[0], block_size=bs):
                    acc += x[tile_m, tile_n]
                out[tile_m] = acc.sum(-1)
            return out

        x = torch.randn([4, 16], device=DEVICE, dtype=torch.float32)
        end = torch.tensor([8], device=DEVICE, dtype=torch.int32)
        _, result = code_and_output(dyn_sum, (x, end))
        expected = x[:, : int(end[0])].sum(dim=1)
        torch.testing.assert_close(result, expected)
