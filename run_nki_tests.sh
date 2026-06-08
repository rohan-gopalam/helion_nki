#!/bin/bash
set -e

cd /home/ubuntu/helion_nki
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate

# Clear all caches for fresh kernels
echo "=== Clearing kernel caches ==="
rm -rf /tmp/torchinductor_cache /tmp/helion_cache ~/.cache/torch_inductor
export TORCHINDUCTOR_CACHE_DIR=/tmp/nki_sweep_$(date +%Y%m%d_%H%M%S)
mkdir -p $TORCHINDUCTOR_CACHE_DIR

export HELION_BACKEND=nki
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2

LOGFILE=/home/ubuntu/helion_nki/nki_test_sweep_$(date +%Y%m%d_%H%M%S).log
echo "=== NKI Test Sweep Started: $(date) ===" | tee $LOGFILE
echo "Cache dir: $TORCHINDUCTOR_CACHE_DIR" | tee -a $LOGFILE
echo "Log: $LOGFILE"

# Run NKI-specific tests first
echo "" | tee -a $LOGFILE
echo "=== NKI-specific tests ===" | tee -a $LOGFILE
python -m pytest test/test_nki_load_refactor.py test/test_nki_dynamic_loops.py \
    -v --tb=short --no-header 2>&1 | tee -a $LOGFILE

# Run full test suite with NKI backend
echo "" | tee -a $LOGFILE
echo "=== Full test suite (NKI backend) ===" | tee -a $LOGFILE

# Run test files one by one so a crash in one doesn't kill the rest
TEST_FILES=(
    test/test_examples.py
    test/test_loops.py
    test/test_masking.py
    test/test_reduce.py
    test/test_reductions.py
    test/test_matmul.py
    test/test_indexing.py
    test/test_misc.py
    test/test_views.py
    test/test_broadcasting.py
    test/test_dot.py
    test/test_closures.py
    test/test_control_flow.py
    test/test_generate_ast.py
    test/test_loop_dependencies.py
)

PASS=0; FAIL=0; ERROR=0

for tf in "${TEST_FILES[@]}"; do
    echo "" | tee -a $LOGFILE
    echo "--- $tf ---" | tee -a $LOGFILE
    if python -m pytest "$tf" -v --tb=short --no-header -x 2>&1 | tee -a $LOGFILE; then
        PASS=$((PASS+1))
        echo "  RESULT: PASSED" | tee -a $LOGFILE
    else
        CODE=$?
        if [ $CODE -eq 5 ]; then
            echo "  RESULT: NO TESTS (skipped)" | tee -a $LOGFILE
        else
            FAIL=$((FAIL+1))
            echo "  RESULT: FAILED (exit $CODE)" | tee -a $LOGFILE
        fi
    fi
done

echo "" | tee -a $LOGFILE
echo "=== Sweep Complete: $(date) ===" | tee -a $LOGFILE
echo "File results: $PASS passed, $FAIL failed, $ERROR errors" | tee -a $LOGFILE
echo "Full log: $LOGFILE"
