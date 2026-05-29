# Running the NKI Example Sweep

This document explains how to run the full suite of Helion NKI examples
("the sweep") on Trainium2, interpret the results, and run individual
examples for debugging.

---

## Prerequisites

- Trainium2 instance (or compatible Neuron hardware)
- The `helion_nki` repo checked out at `/home/ubuntu/helion_nki`
- Helion installed editable: `pip install -e '.[dev]'` (already done on this machine)

---

## Quick Start — Full Sweep

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /home/ubuntu/helion_nki

HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_nki_sweep \
python examples/run_nki_examples.py
```

The runner exits 0 if all 51 selected examples pass, non-zero otherwise and
prints which ones failed.

**Runtime:** Allow 2–4 hours for a cold run (no Neuron cache). With a warm
Neuron cache the same examples complete in 30–60 minutes.

---

## What the Runner Does

`examples/run_nki_examples.py` discovers every `*.py` in `examples/` that is:
- not `__init__.py`, `run_nki_examples.py`, or `*_nki.py`
- not a known CUDA-only example (`flex_attention.py`, `fp8_attention.py`,
  `jagged_dense_bmm.py`)

It runs each script as a subprocess with `HELION_BACKEND=nki` forced, a
600-second timeout per example, and collects pass/fail. Each example script
is responsible for its own correctness check (usually via `run_example()`).

**51 examples** are in scope as of the current commit.

---

## Running a Single Example

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
cd /home/ubuntu/helion_nki

HELION_BACKEND=nki \
NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
TORCHINDUCTOR_CACHE_DIR=/tmp/helion_<name>_cache \
python examples/<name>.py
```

Replace `<name>` with the example, e.g. `mamba2_chunk_scan`, `moe_matmul_ogs`.

Use a **fresh cache dir** per run when you have changed compiler code, or the
Neuron compile cache may serve a stale `.neff` and silently hide bugs:

```bash
rm -rf /tmp/helion_<name>_cache
rm -rf /var/tmp/neuron-compile-cache/neuronxcc-*/MODULE_<hash>*
```

---

## Expected Results (current branch `fix-nki-kernel-compilation`)

As of commit `d67b46d1`, all 51 runner-selected examples pass:

| Status | Count | Examples |
|--------|-------|---------|
| `runtime_passed` | 48 | Everything below minus two no-ops and one custom path |
| `runtime_passed_special/no-op` | 2 | `aot_example.py`, `blackwell_attention.py` |
| `runtime_passed_special/custom_path` | 1 | `softmax_decomposed.py` |
| `nki_backend_compile_timeout` | 0 | — |
| `nki_frontend_or_codegen_blocked` | 0 | — |

Previously blocked examples now fixed in this commit:
`mamba2_chunk_scan.py`, `mamba2_chunk_state.py`, `moe_matmul_ogs.py`,
`fused_linear_jsd.py`.

Known limitations that do **not** cause test failures (params were tuned):
- `fused_linear_jsd.py` runs with `vocab_size=16384` instead of 128256.
  Full vocab requires streaming two-pass softmax, not yet implemented.
- `jagged_layer_norm.py` runs with `B ≤ 256` only.

---

## Caches and Warm Runs

There are two independent caches:

| Cache | Location | Key | When to clear |
|-------|----------|-----|---------------|
| TorchInductor Python | `TORCHINDUCTOR_CACHE_DIR` | FX graph hash | When compiler Python code changes |
| Neuron neff | `/var/tmp/neuron-compile-cache/` | HLO hash | When NKI kernel code changes |

Because the Neuron neff cache is keyed on HLO (not on the Python code that
generated it), stale `.neff` files can silently produce correct-looking results
even after a compiler bug is introduced. **Always clear both caches before a
correctness regression run.**

```bash
# Clear everything
rm -rf /tmp/helion_nki_sweep
rm -rf /var/tmp/neuron-compile-cache/neuronxcc-*/
```

Or clear just one specific example's Neuron entry:

```bash
# Find and delete by module hash shown in the compile log
rm -rf /var/tmp/neuron-compile-cache/neuronxcc-*/MODULE_<hash>*
```

---

## Parallelism

The sweep runner is **serial** (one example at a time). Each example occupies
all Neuron devices on the instance. Running examples in parallel is not
supported and will cause Neuron device conflicts.

---

## Debugging a Failure

**1. Look at the generated Python kernel**

TorchInductor writes the generated NKI Python to the cache dir:

```
ls /tmp/helion_<name>_cache/*/c*.py
```

Read it to see the DMA calls, tile allocations, and matmul lowering.

**2. Add `HELION_DEBUG_3D_LOAD=1`** to trace 3D gather load decisions.

**3. Inspect the Neuron compiler log**

Set `NEURON_CC_FLAGS=--verbose=35` to get detailed Neuron compiler output.

**4. Forced recompile**

Delete the example's TorchInductor cache entry so the Python kernel is
regenerated from scratch:

```bash
rm -rf /tmp/helion_<name>_cache
```

**5. Check for Neuron cache serving stale results**

If results look suspiciously correct after a code change, delete the Neuron
neff and re-run to force fresh XLA/HLO compilation.

---

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `HELION_BACKEND` | `triton` | Set to `nki` for Trainium |
| `NEURON_PLATFORM_TARGET_OVERRIDE` | (none) | Set to `trn2` for Trainium2 |
| `TORCHINDUCTOR_CACHE_DIR` | system temp | Python kernel cache location |
| `NEURON_CC_FLAGS` | (none) | Pass flags to Neuron compiler, e.g. `--verbose=35` |
| `TORCH_LOGS` | (none) | Set to `+inductor` for TorchInductor debug output |
</content>
</invoke>