# NKI Performance Benchmarking & Meta-Autotuning

## Context

Helion generates NKI (Neuron Kernel Interface) code targeting AWS Trainium. Traditional
autotuning (exploring hundreds of configs with sub-second kernel launches) is infeasible
because neuronx-cc compilation takes 10-60s per kernel. Instead, we optimize the
**code generator itself** by measuring how codegen decisions affect runtime performance
across the full suite of passing kernels.

## Goal

Systematically improve the performance of Helion-generated NKI kernels by:

1. **Establishing baselines** — measure compile time and execution latency for all 30
   passing examples on this trn2.3xlarge (4 NeuronCores, 96GB HBM).

2. **Identifying codegen knobs** — choices made during code generation that affect
   performance without changing correctness:
   - `nl.affine_range` (parallel) vs `nl.sequential_range` (serial) for tile loops
   - Tile sizes along partition and free dimensions
   - DMA pattern: single large copy vs tiled copies
   - Cast placement: early (smaller tiles) vs late (fewer casts)
   - Accumulator layout: partition-first vs free-first
   - Reduction strategy: axis choice, keepdims, partition_reduce vs tensor_reduce

3. **Iterating** — for each codegen variant, re-benchmark all kernels and compare
   against baseline. This is "meta-autotuning": tuning the generator, not the kernel.

4. **Parallelizing** — run 4 kernels simultaneously on 4 NeuronCores via
   `NEURON_RT_VISIBLE_CORES` process isolation to keep iteration time ~10 minutes
   per sweep instead of ~40 minutes serial.

## Hardware

- **Instance**: trn2.3xlarge
- **Neuron Device**: 1 device, 4 cores (IDs 0-3), 96 GB HBM
- **CPU**: Intel Xeon 8488C, 12 cores
- **Logical NeuronCore config**: 2 (LNC=2 for dynamic_range kernels)

## Usage

```bash
# Baseline sweep (30 kernels, 4 cores, ~10 min)
python bench/coordinator.py --tag baseline --output bench/baseline.csv

# After a codegen change
python bench/coordinator.py --tag experiment_name --output bench/experiment.csv

# Compare
python bench/compare.py bench/baseline.csv bench/experiment.csv

# Quick subset test (no need to wait for full sweep)
python bench/coordinator.py --examples add.py,exp.py,sum.py --cores 3 --reps 5

# Dry run (verify setup without touching NeuronCores)
python bench/coordinator.py --dry-run
```

## Files

| File | Purpose |
|------|---------|
| `bench/worker.py` | Runs one kernel on a pinned core, outputs JSON timing |
| `bench/coordinator.py` | Dispatches workers across 4 cores in parallel |
| `bench/compare.py` | Diffs two result CSVs, reports speedups/regressions |

## Key Insight

neuronx-cc IS the low-level optimizer (ISA scheduling, register allocation). Helion's
job is to feed it high-quality NKI. The question is not "which config is fastest" but
"which codegen patterns produce NKI that neuronx-cc optimizes best." This is measurable
and iteratable with 4-way parallelism on this machine.
