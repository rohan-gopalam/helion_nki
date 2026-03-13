from __future__ import annotations

import torch
import nki
import nki.language as nl

from helion._testing import DEVICE
from helion.runtime import default_nki_launcher as _default_nki_launcher
import nkilib.core.subkernels.layernorm_tkg as layernorm_tkg_mod


# Baseline shape to mirror helion_nki/examples/layer_norm.py.
_M = 4096
_N = 8192
_BLOCK_M = 128
_H0 = 128
_LNC = 2

# Disable internal sharding in this baseline; we launch single-core blocks.
layernorm_tkg_mod.SHARDING_THRESHOLD = 10**9


@nki.jit(platform_target="trn1")
def _manual_layer_norm_fwd(x, weight, bias, eps):
    x = x.reshape([1, _BLOCK_M, _N])
    weight = weight.reshape([1, _N])
    bias = bias.reshape([1, _N])
    # Returns [H0, B*S, H1] = [128, _BLOCK_M, _N/128]
    return layernorm_tkg_mod.layernorm_tkg(x, weight, bias=bias, eps=eps, output_in_sbuf=False)


def layer_norm_manual_nki(
    x: torch.Tensor,
    normalized_shape: list[int],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    _launcher=_default_nki_launcher,
) -> torch.Tensor:
    m, n = x.size()
    assert m == _M and n == _N, f"manual kernel supports only [{_M}, {_N}], got [{m}, {n}]"
    assert m % _BLOCK_M == 0, f"m must be divisible by {_BLOCK_M}, got {m}"
    assert normalized_shape == [n], f"expected normalized_shape [{n}], got {normalized_shape}"
    assert weight.shape == (n,), f"expected weight shape ({n},), got {tuple(weight.shape)}"
    if bias is None:
        bias = torch.zeros_like(weight)
    else:
        assert bias.shape == (n,), f"expected bias shape ({n},), got {tuple(bias.shape)}"

    y = torch.empty_like(x)
    for m0 in range(0, m, _BLOCK_M):
        x_blk = x[m0 : m0 + _BLOCK_M, :]
        # layernorm_tkg output layout is [H0, B*S, H1] where H1 packs [LNC, H2].
        y_tkg = _launcher(_manual_layer_norm_fwd, (1,), x_blk, weight, bias, eps)
        h2 = n // (_H0 * _LNC)
        y_blk = y_tkg.reshape([_H0, _BLOCK_M, _LNC, h2]).permute(1, 2, 0, 3).reshape(
            [_BLOCK_M, n]
        )
        y[m0 : m0 + _BLOCK_M, :] = y_blk
    return y


def main() -> None:
    torch.manual_seed(0)
    device = DEVICE
    eps = 1e-4

    x = -2.3 + 0.5 * torch.randn([_M, _N], device=device, dtype=torch.float16)
    weight = torch.randn([_N], device=device, dtype=torch.float16)
    bias = torch.randn([_N], device=device, dtype=torch.float16)

    y_ref = torch.nn.functional.layer_norm(x, [_N], weight, bias, eps)
    y_nki = layer_norm_manual_nki(x, [_N], weight, bias, eps)
    torch.testing.assert_close(y_nki, y_ref, rtol=1e-3, atol=1e-3)
    print("manual layer_norm (with bias): PASS")

    y_ref_no_bias = torch.nn.functional.layer_norm(x, [_N], weight, None, eps)
    y_nki_no_bias = layer_norm_manual_nki(x, [_N], weight, None, eps)
    torch.testing.assert_close(y_nki_no_bias, y_ref_no_bias, rtol=1e-3, atol=1e-3)
    print("manual layer_norm (no bias): PASS")


if __name__ == "__main__":
    main()
