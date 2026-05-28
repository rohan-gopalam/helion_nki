#!/usr/bin/env bash
# Run all examples in examples/ with HELION_BACKEND=nki.
# Skips _nki.py variants and files that require CUDA.
# Usage: bash examples/run_all_nki_examples.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=/opt/aws_neuronx_venv_pytorch_2_9/bin/activate
PER_EXAMPLE_TIMEOUT=300   # seconds per example

source "$VENV"
export HELION_BACKEND=nki
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

# Examples that require CUDA or are not standalone
SKIP=(
  flex_attention.py
  fp8_attention.py
  jagged_dense_bmm.py
  blackwell_attention.py   # Blackwell GPU
  aot_example.py           # requires special AOT setup
  simple_add_nki.py        # uses DEVICE which resolves to cpu w/o GPU
  attention_nki.py         # same
  concatenate_nki.py       # same
  layer_norm_manual_nki.py # same
  fused_nki_ops.py         # same
  run_nki_examples.py
  run_all_nki_examples.sh
  __init__.py
  test_nki_autotune.py
  test_nki_autotune_error_recovery.py
  test_nki_autotune_matmul.py
)

PASSED=()
FAILED=()
SKIPPED=()

cd "$REPO"

for script in examples/*.py; do
  name="$(basename "$script")"
  # Check skip list
  skip=0
  for s in "${SKIP[@]}"; do
    [[ "$name" == "$s" ]] && { skip=1; break; }
  done
  if [[ $skip -eq 1 ]]; then
    SKIPPED+=("$name")
    continue
  fi

  echo -n "  $name ... "
  output=$(timeout "$PER_EXAMPLE_TIMEOUT" python "$script" 2>&1) && rc=0 || rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "PASSED"
    PASSED+=("$name")
  elif [[ $rc -eq 124 ]]; then
    echo "TIMEOUT"
    FAILED+=("$name (timeout ${PER_EXAMPLE_TIMEOUT}s)")
  else
    echo "FAILED (exit $rc)"
    echo "--- last 20 lines ---"
    echo "$output" | tail -20
    echo "---------------------"
    FAILED+=("$name")
  fi
done

echo ""
echo "=============================="
echo "Results: ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "FAILED:"
  for f in "${FAILED[@]}"; do echo "  $f"; done
  exit 1
fi
echo "All examples PASSED"
