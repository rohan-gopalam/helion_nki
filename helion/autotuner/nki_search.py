"""NKI-specific autotuner that handles Trainium/XLA device constraints.

NKI compilation is slow (~10-30s per config) so we use FiniteSearch over a
small hand-crafted config list rather than a gradient-based search.  Key
differences from the Triton path:

- torch.accelerator.synchronize() is unavailable on XLA; we use xm.mark_step()
- triton.testing.do_bench is unavailable; we use wall-clock timing
- Errors like SBUF overflow must be caught per-config (not fatal)
- Block sizes are constrained: partition dim <= 128 for tensor-engine ops
"""
from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from .base_search import BaseSearch
from .base_search import BenchmarkResult
from .base_search import _clone_args

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from ..runtime.config import Config
    from .base_search import _AutotunableKernel

log = logging.getLogger(__name__)

# Maximum partition-dim block size for NKI tensor-engine operations.
# Exceeding this causes hardware errors on Trainium.
NKI_MAX_PARTITION_DIM = 128

# Maximum free-dim block size (SBUF row size constraint).
NKI_MAX_FREE_DIM = 512

# Candidate block sizes to try, indexed by effort level.
# NKI compilation is very slow (~5-10 min per config for complex kernels).
_PARTITION_CANDIDATES_QUICK = [32, 128]         # 2 values: quick search
_FREE_CANDIDATES_QUICK = [128, 512]             # 2 values: quick search
_PARTITION_CANDIDATES_FULL = [16, 32, 64, 128]  # 4 values: full search
_FREE_CANDIDATES_FULL = [64, 128, 256, 512]     # 4 values: full search


def _nki_synchronize() -> None:
    """Synchronize the XLA/Neuron device."""
    try:
        from torch_xla.core import xla_model as xm
        xm.mark_step()
    except Exception:
        try:
            import torch
            torch.accelerator.synchronize()
        except Exception:
            pass


def _nki_bench(fn: Callable[..., object], args: Sequence[object], repeats: int = 3, warmup: int = 1) -> float:
    """Time a NKI kernel using wall-clock time.

    NKI kernels compile lazily on first call. The caller should have
    already done one warmup run (compilation + first execution). We do
    `warmup` additional runs to ensure XLA JIT is fully warm before timing.

    Note: default_nki_launcher already calls xm.mark_step() internally,
    so fn(*args) is a synchronous blocking call.

    Returns median time in milliseconds.
    """
    # Extra warmup runs to stabilize performance after initial compilation.
    for _ in range(warmup):
        fn(*args)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    if not times:
        return float("inf")
    times.sort()
    return times[len(times) // 2]


def _is_nki_error(exc: Exception) -> bool:
    """Return True if this is a recoverable NKI compilation/runtime error."""
    msg = str(exc).lower()
    recoverable_patterns = [
        "sbuf",
        "out of memory",
        "overflow",
        "invalid",
        "shape",
        "neuron",
        "hlo",
        "compilation",
        "too large",
        "exceeds",
        "illegal",
    ]
    return any(p in msg for p in recoverable_patterns)


def _generate_nki_configs(
    config_spec: object,
    effort: str = "quick",
) -> list[Config]:
    """Generate a set of valid NKI configs to try during autotuning.

    NKI compilation is very slow (~5-10 min per config for complex kernels).
    "quick" generates 5 representative configs covering the design space.
    "full" generates up to 17 configs for a more thorough search.

    Respects NKI hardware constraints:
    - Partition dim (block_sizes[0]) <= NKI_MAX_PARTITION_DIM
    - Free dim (block_sizes[1]) <= NKI_MAX_FREE_DIM
    - All block sizes are powers of 2
    """
    from ..autotuner.config_spec import ConfigSpec
    from ..runtime.config import Config

    assert isinstance(config_spec, ConfigSpec)

    num_block_dims = len(config_spec.block_sizes)
    if num_block_dims == 0:
        return [config_spec.default_config()]

    default = config_spec.default_config()
    default_block_sizes = list(default.config.get("block_sizes", []))

    configs: list[Config] = []
    seen: set[tuple[int, ...]] = set()

    def _add_config(bs: list[int]) -> None:
        key = tuple(bs)
        if key not in seen:
            seen.add(key)
            configs.append(Config(block_sizes=bs))

    # Always include the default config's block_sizes first (baseline reference).
    if default_block_sizes:
        _add_config(list(default_block_sizes))

    # Choose search breadth based on effort level
    part_cands = _PARTITION_CANDIDATES_FULL if effort == "full" else _PARTITION_CANDIDATES_QUICK
    free_cands = _FREE_CANDIDATES_FULL if effort == "full" else _FREE_CANDIDATES_QUICK

    if num_block_dims >= 2:
        spec0 = config_spec.block_sizes[0]
        spec1 = config_spec.block_sizes[1]

        hint0 = int(max(spec0.size_hint, 1))
        hint1 = int(max(spec1.size_hint, 1))

        # Cap to hardware limits
        max0 = min(hint0, NKI_MAX_PARTITION_DIM)
        max1 = min(hint1, NKI_MAX_FREE_DIM)

        def _candidate_sizes(max_val: int, candidates: list[int]) -> list[int]:
            valid = [s for s in candidates if s <= max_val]
            if not valid:
                return [max(1, max_val)]
            if effort != "full":
                # For quick search, keep only smallest and largest
                return sorted({valid[0], valid[-1]})
            return valid

        part_sizes = _candidate_sizes(max0, part_cands)
        free_sizes = _candidate_sizes(max1, free_cands)

        def _fill_remaining(p: int, f: int) -> list[int]:
            bs = [p, f]
            for i in range(2, num_block_dims):
                if i < len(default_block_sizes):
                    raw = int(default_block_sizes[i])
                    capped = min(raw, NKI_MAX_FREE_DIM if i % 2 == 1 else NKI_MAX_PARTITION_DIM)
                    bs.append(capped)
                else:
                    bs.append(1)
            return bs

        for p in part_sizes:
            for f in free_sizes:
                _add_config(_fill_remaining(p, f))

    elif num_block_dims == 1:
        spec0 = config_spec.block_sizes[0]
        hint0 = int(max(spec0.size_hint, 1))
        valid = [s for s in free_cands if s <= hint0]
        if not valid:
            valid = [min(64, hint0)]
        if effort != "full":
            valid = sorted({valid[0], valid[-1]})
        for s in valid:
            _add_config([s])

    return configs


class NKIFiniteSearch(BaseSearch):
    """Finite search over NKI-safe configs.

    Unlike the Triton autotuner, this:
    - Uses xm.mark_step() for synchronization
    - Uses wall-clock timing (not CUDA events)
    - Skips configs that error (SBUF overflow, etc.) without aborting
    - Only tries a small set of NKI-safe block-size configurations
    """

    def __init__(
        self,
        kernel: _AutotunableKernel,
        args: Sequence[object],
        configs: list[Config] | None = None,
        effort: str = "quick",
    ) -> None:
        # Bypass base __init__ since it calls torch.accelerator.synchronize().
        # We replicate the minimal initialization needed.
        import collections
        import random

        from .logger import AutotuningLogger

        self.kernel = kernel
        self.settings = kernel.settings
        self.config_spec = kernel.config_spec
        self.args: Sequence[object] = args
        self.counters: collections.Counter[str] = collections.Counter()
        self.log = AutotuningLogger(self.settings)
        self.best_perf_so_far = math.inf

        seed = self.settings.autotune_random_seed
        random.seed(seed)
        self.log(f"[NKI autotune] random seed: {seed}")

        self._original_args = _clone_args(self.args)
        self._mutated_arg_indices: Sequence[int] = []
        self._baseline_output: object = None
        self._baseline_post_args: Sequence[object] | None = None
        # Use relaxed tolerances for NKI since different block sizes may use
        # different floating-point accumulation orders.
        self._effective_atol: float = 0.1
        self._effective_rtol: float = 0.1
        self._jobs: int = 1
        self._current_generation: int = 0

        # Use provided configs or generate NKI-safe ones based on effort level.
        if configs is not None:
            self.configs = list(configs)
        else:
            self.configs = _generate_nki_configs(self.config_spec, effort=effort)

        self.log(f"[NKI autotune] will try {len(self.configs)} configs: {self.configs}")

        # Compute baseline using the first config (default).
        self._compute_nki_baseline()

    def _compute_nki_baseline(self) -> None:
        """Compute baseline using the default NKI config for accuracy checking."""
        if not self.configs:
            return
        new_args = _clone_args(self._original_args)
        baseline_config = self.configs[0]
        try:
            fn = self.kernel.compile_config(baseline_config, allow_print=False)
            self._baseline_output = fn(*new_args)
            # Materialize baseline on CPU to avoid re-compilation during accuracy checks.
            import torch
            from torch.utils._pytree import tree_map_only
            self._baseline_output = tree_map_only(
                torch.Tensor,
                lambda t: t.cpu(),
                self._baseline_output,
            )
            self.log(f"[NKI autotune] baseline computed with config {baseline_config}")
        except Exception as e:
            self.log.warning(f"[NKI autotune] baseline computation failed: {e}")
            self._baseline_output = None

    def benchmark_nki_config(self, config: Config) -> tuple[object, float]:
        """Benchmark a single NKI config; return (fn, time_ms).

        Returns (fn, inf) if the config errors out.
        """
        self.counters["benchmark"] += 1
        try:
            fn = self.kernel.compile_config(config, allow_print=False)
        except Exception as e:
            self.log(f"[NKI autotune] compile failed for {config}: {type(e).__name__}: {e}")
            return None, math.inf

        # Run once to check correctness and trigger compilation.
        # fn() calls default_nki_launcher which calls xm.mark_step() internally,
        # so this is already synchronous.
        try:
            bench_args = _clone_args(self._original_args) if self._mutated_arg_indices else self.args
            output = fn(*bench_args)
        except Exception as e:
            self.log(f"[NKI autotune] first run failed for {config}: {type(e).__name__}: {e}")
            return fn, math.inf

        # Accuracy check against baseline (which is already on CPU).
        if self._baseline_output is not None and self.settings.autotune_accuracy_check:
            try:
                import torch
                from torch.utils._pytree import tree_flatten
                actual_flat, _ = tree_flatten(output)
                expected_flat, _ = tree_flatten(self._baseline_output)
                for act, exp in zip(actual_flat, expected_flat, strict=False):
                    if isinstance(act, torch.Tensor) and isinstance(exp, torch.Tensor):
                        act_cpu = act.cpu()
                        torch.testing.assert_close(
                            act_cpu, exp,
                            atol=self._effective_atol,
                            rtol=self._effective_rtol,
                        )
            except Exception as e:
                self.log.warning(f"[NKI autotune] accuracy mismatch for {config}: {e}")
                self.counters["accuracy_mismatch"] += 1
                return fn, math.inf

        # Time the kernel (wall-clock).  The first run already happened above
        # (compilation warmup), so we only need 1 more warmup + 3 timed runs.
        try:
            perf = _nki_bench(fn, list(self.args), repeats=3, warmup=1)
            self.log(f"[NKI autotune] {config}: {perf:.1f}ms")
            if perf < self.best_perf_so_far:
                self.best_perf_so_far = perf
            return fn, perf
        except Exception as e:
            self.log(f"[NKI autotune] benchmark failed for {config}: {type(e).__name__}: {e}")
            return fn, math.inf

    def parallel_benchmark(
        self, configs: list[Config], *, desc: str = "Benchmarking"
    ) -> list[BenchmarkResult]:
        """Override to use NKI-specific benchmarking."""
        results: list[BenchmarkResult] = []
        for config in configs:
            fn, perf = self.benchmark_nki_config(config)
            status = "ok" if math.isfinite(perf) else "error"
            results.append(BenchmarkResult(
                config=config,
                fn=fn if fn is not None else (lambda *a: None),
                perf=perf,
                status=status,
                compile_time=None,
            ))
        return results

    def _autotune(self) -> Config:
        """Search over NKI configs, return the fastest valid one."""
        best_config = self.configs[0]
        best_perf = math.inf

        for config in self.configs:
            _, perf = self.benchmark_nki_config(config)
            if perf < best_perf:
                best_perf = perf
                best_config = config

        if math.isinf(best_perf):
            self.log.warning(
                "[NKI autotune] all configs failed, returning default config"
            )

        return best_config

    def autotune(self, *, skip_cache: bool = False) -> Config:
        """Run NKI autotuning and return best config."""
        import time as _time

        start = _time.perf_counter()
        best = self._autotune()
        end = _time.perf_counter()
        self.log(
            f"[NKI autotune] complete in {end - start:.1f}s after {self.counters['benchmark']} configs.\n"
            f"Best config: {best}",
            level=logging.INFO + 5,
        )
        return best
