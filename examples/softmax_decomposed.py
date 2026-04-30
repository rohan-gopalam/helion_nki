"""
NKI-targeted softmax_decomposed (2D softmax over last dim).

Run: cd /home/ubuntu/kernel_test && PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/softmax_decomposed.py
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_HELION_NKI = os.path.join(_REPO, "helion_nki")
if _HELION_NKI not in sys.path:
    sys.path.insert(0, _HELION_NKI)

os.environ.setdefault("NEURON_COMPILE_CACHE_URL", "")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")

import torch
import helion
import helion.language as hl


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128]),
)
def softmax_decomposed(x: torch.Tensor) -> torch.Tensor:
    """Softmax over last dim. x: [n, m]."""
    n, _m = x.size()
    out = torch.empty_like(x)
    for tile_n in hl.tile(n):
        values = x[tile_n, :]
        amax = torch.amax(values, dim=1, keepdim=True)
        exp = torch.exp(values - amax)
        sum_exp = torch.sum(exp, dim=1, keepdim=True)
        out[tile_n, :] = exp / sum_exp
    return out


def main() -> None:
    N, M = 128, 128
    x = torch.randn(N, M, dtype=torch.float32)

    bound = softmax_decomposed.bind((x,))
    config_dict = {"block_sizes": [128]}
    print("Generating NKI code (softmax_decomposed)...")
    code = bound.to_triton_code(config_dict, output_origin_lines=False)
    out_path = os.path.join(_REPO, "softmax_decomposed_kernel.py")
    with open(out_path, "w") as f:
        f.write(code)
    print(f"Wrote {out_path}")

    ref = torch.softmax(x, dim=1)
    print("Running on XLA...")

    from torch_xla.core import xla_model as xm

    device = xm.xla_device()
    x_dev = x.to(device)
    xm.mark_step()

    import importlib.util

    spec = importlib.util.spec_from_file_location("softmax_decomposed_kernel", out_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw_kernel = getattr(mod, "_helion_softmax_decomposed")
    out_dev = raw_kernel[1](x_dev)
    xm.mark_step()

    out = out_dev.cpu()
    err = (out - ref).abs().max().item()
    print(f"max error: {err}")
    if err >= 1e-2:
        raise AssertionError(f"Error {err} too large")
    print("PASSED")


if __name__ == "__main__":
    main()
