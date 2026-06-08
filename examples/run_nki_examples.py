#!/usr/bin/env python3
"""
Run canonical Helion examples targeting NKI sequentially.

Usage:
  cd /home/ubuntu/kernel_test && source aws_neuron_venv_pytorch/bin/activate
  PYTHONPATH=helion_nki:$PYTHONPATH python helion_nki/examples/run_nki_examples.py

This runner executes example scripts one-by-one, forcing HELION_BACKEND=nki.
Each example is responsible for generating kernels and checking correctness
against its own reference implementation.

Per-example logs are written to /tmp/nki_example_logs/<stem>.log.
Delete that directory to clean up: rm -rf /tmp/nki_example_logs
"""
from __future__ import annotations

import datetime
import os
from pathlib import Path
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXAMPLES_DIR = Path(_REPO) / "helion_nki" / "examples"
_LOG_DIR = Path("/tmp/nki_example_logs")


# Examples that require CUDA (NVIDIA GPU); skipped when running on Neuron/CPU.
_CUDA_ONLY_EXAMPLES = {"flex_attention.py", "fp8_attention.py", "jagged_dense_bmm.py"}


def _example_scripts() -> list[Path]:
    scripts: list[Path] = []
    for p in sorted(_EXAMPLES_DIR.glob("*.py")):
        name = p.name
        if name.startswith("_"):
            continue
        if name in ("run_nki_examples.py", "test_sizes.py"):
            continue
        if name.endswith("_nki.py"):
            continue
        if name in _CUDA_ONLY_EXAMPLES:
            continue
        scripts.append(p)
    return scripts


def main() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    # helion is now installed editable; PYTHONPATH override not required. Keep
    # the prefix append for users running from a checkout without install.
    env["PYTHONPATH"] = f"{_REPO}/helion_nki:{env.get('PYTHONPATH', '')}"
    env["HELION_BACKEND"] = "nki"
    env.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)

    scripts = _example_scripts()
    failed = []
    start_all = datetime.datetime.now()
    print(f"Run started at {start_all.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Per-example logs: {_LOG_DIR}/  (clean up with: rm -rf {_LOG_DIR})")

    for path in scripts:
        rel = path.relative_to(_EXAMPLES_DIR).as_posix()
        log_path = _LOG_DIR / f"{path.stem}.log"
        print(f"\n=== {rel} ===", flush=True)
        t0 = datetime.datetime.now()
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=_REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except subprocess.TimeoutExpired:
            elapsed = (datetime.datetime.now() - t0).seconds
            msg = f"FAILED (timeout after {elapsed}s)\n"
            print(msg, end="", flush=True)
            log_path.write_text(f"=== {rel} ===\n{msg}")
            failed.append(rel)
            continue

        elapsed = (datetime.datetime.now() - t0).seconds
        combined = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr.strip() else "")
        log_path.write_text(f"=== {rel} (exit={result.returncode}, {elapsed}s) ===\n{combined}")

        if result.returncode == 0:
            if "PASSED" in result.stdout:
                print(f"PASSED ({elapsed}s)", flush=True)
            else:
                print(f"OK ({elapsed}s) — no PASSED marker; see {log_path.name}", flush=True)
                print(result.stdout[-300:] if len(result.stdout) > 300 else result.stdout)
        else:
            print(f"FAILED ({elapsed}s) — full log: {log_path}", flush=True)
            # Print tail of stderr then tail of stdout so the most useful info shows
            tail = ""
            if result.stderr.strip():
                tail = result.stderr[-1200:]
            elif result.stdout.strip():
                tail = result.stdout[-1200:]
            print(tail)
            failed.append(rel)

    elapsed_all = (datetime.datetime.now() - start_all).seconds
    print("\n" + "=" * 40)
    print(f"Total time: {elapsed_all}s")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        sys.exit(1)
    print("All canonical examples passed on NKI")


if __name__ == "__main__":
    main()
