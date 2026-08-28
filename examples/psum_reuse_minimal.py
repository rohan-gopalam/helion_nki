"""
Minimal PSUM-reuse smoke tests for the FX-graph fusion pass.

Strategy:
- Run standalone mock NKI kernels first to prove the PSUM-reuse pattern
  compiles and runs correctly on Trn1 hardware (independent of Helion).
- Then run a Helion kernel that exercises the fusion pass, and inspect
  the GENERATED NKI source (not just run it) to confirm the pattern.

Does not depend on hardware availability for the static inspection path.
"""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
import sys

import torch
import helion
import helion.language as hl
from helion._testing import DEVICE

_CACHE_DIR = "/tmp/torchinductor_ubuntu"


@helion.kernel(
    backend="nki",
    autotune_effort="none",
    config=helion.Config(block_sizes=[128]),
    static_shapes=True,
)
def mm_relu_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """relu(hl.dot(x_tile, y_full)) — single tile dim, full y load.

    With a 1D hl.tile(128) only tile_m is active. y is loaded in full
    via y[:, :] which sidesteps the pre-existing `y[:, tile_n]` 2D-tile
    codegen bug.
    """
    out = torch.empty([128, 128], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(128):
        prod = hl.dot(x[tile_m, :], y[:, :])
        out[tile_m, :] = torch.relu(prod)
    return out


def _clear_caches() -> None:
    shutil.rmtree(_CACHE_DIR, ignore_errors=True)
    # Neuron compile workdir can also cache kernels
    for d in glob.glob("/tmp/_helion_*_python_ast.klir"):
        try: os.remove(d)
        except OSError: pass


def _find_generated_kernel(hint: str) -> str:
    """Return path of most recently generated NKI kernel source matching hint."""
    files = sorted(
        glob.glob(f"{_CACHE_DIR}/**/*.py", recursive=True),
        key=os.path.getmtime,
        reverse=True,
    )
    for fp in files:
        try:
            txt = open(fp).read()
        except OSError:
            continue
        if "nc_matmul" in txt and hint in txt:
            return fp
    return ""


def run_helion_matmul_relu() -> str:
    """Compile a matmul→relu Helion kernel and return the generated source path."""
    x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
    try:
        got = mm_relu_kernel(x, y)
    except Exception as e:
        print(f"[COMPILE FAIL] {e!r}"[:400])
        return ""

    ref = torch.relu(x @ y)
    if torch.allclose(got, ref, atol=1e-3, rtol=1e-3):
        print("[PASS] numerical")
    else:
        print(f"[FAIL] max err = {(got - ref).abs().max().item()}")
    return _find_generated_kernel("mm_relu_kernel")


def inspect_generated(path: str) -> None:
    if not path:
        print("[SKIP] no generated file found")
        return
    print(f"\n===== Generated kernel: {path} =====")
    with open(path) as f:
        src = f.read()
    print(src)

    # Analysis
    print("\n===== Analysis =====")
    has_mm_sbuf_tmp = "_mm_sbuf_tmp" in src or "_dot_sbuf_tmp" in src
    has_psum_as_data = (
        "data=_mm_psum" in src or "data=_dot_mm_psum" in src
    )
    print(f"  mm_sbuf_tmp present:          {has_mm_sbuf_tmp}  (False = PSUM reuse fired)")
    print(f"  nisa.activation reads PSUM:   {has_psum_as_data}  (True = PSUM reuse fired)")

    ncmm = src.count("nisa.nc_matmul")
    tcp = src.count("nisa.tensor_copy")
    act = src.count("nisa.activation")
    print(f"  nc_matmul count:  {ncmm}")
    print(f"  tensor_copy count: {tcp}")
    print(f"  activation count:  {act}")


if __name__ == "__main__":
    print("===== Part 1: Mock NKI kernels (standalone, Helion-independent) =====")
    # Run the mock side-by-side to confirm PSUM-reuse is hardware-valid.
    mock_script = "/home/ubuntu/kernel_test/mock_psum_reuse.py"
    if os.path.exists(mock_script):
        res = subprocess.run(
            [sys.executable, mock_script],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "NEURON_PLATFORM_TARGET_OVERRIDE": "trn1"},
        )
        for line in res.stdout.splitlines():
            if "PASS" in line or "FAIL" in line:
                print(f"  {line}")
        if res.returncode != 0:
            print(f"  [mock exit={res.returncode}]")
            print(res.stderr[-500:])

    print("\n===== Part 2: Helion fusion pipeline =====")
    _clear_caches()
    path = run_helion_matmul_relu()
    inspect_generated(path)
