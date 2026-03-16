#!/usr/bin/env python3
"""
Run canonical Helion examples targeting NKI sequentially.

Usage:
  cd /home/ubuntu/kernel_test && source aws_neuron_venv_pytorch/bin/activate
  PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/run_nki_examples.py

This runner executes example scripts one-by-one, forcing HELION_BACKEND=nki.
Each example is responsible for generating kernels and checking correctness
against its own reference implementation.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES_DIR = Path(_REPO) / "helion_nki" / "examples"


# Examples that require CUDA (NVIDIA GPU); skipped when running on Neuron/CPU.
_CUDA_ONLY_EXAMPLES = {"flex_attention.py", "fp8_attention.py", "jagged_dense_bmm.py"}


def _example_scripts() -> list[Path]:
    scripts: list[Path] = []
    for p in sorted(_EXAMPLES_DIR.glob("*.py")):
        name = p.name
        if name.startswith("_"):
            continue
        if name == "run_nki_examples.py":
            continue
        if name.endswith("_nki.py"):
            continue
        if name in _CUDA_ONLY_EXAMPLES:
            continue
        scripts.append(p)
    return scripts


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{_REPO}/helion_nki:{env.get('PYTHONPATH', '')}"
    env["HELION_BACKEND"] = "nki"
    env.setdefault("NEURON_COMPILE_CACHE_URL", "")
    env.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    scripts = _example_scripts()
    failed = []
    for path in scripts:
        rel = path.relative_to(_EXAMPLES_DIR).as_posix()
        print(f"\n=== {rel} ===")
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=_REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired as e:
            print("FAILED (timeout after 600s)")
            failed.append(rel)
            continue
        if result.returncode == 0:
            if "PASSED" in result.stdout:
                print("PASSED")
            else:
                print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            print("FAILED")
            print(result.stderr[-800:] if result.stderr else result.stdout[-800:])
            failed.append(rel)

    print("\n" + "=" * 40)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    print("All canonical examples passed on NKI")


if __name__ == "__main__":
    main()
