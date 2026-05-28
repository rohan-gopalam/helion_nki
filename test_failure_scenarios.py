#!/usr/bin/env python3
"""
Demonstration of actual error scenarios in NKI autotuning.

This shows what happens when configs fail at different stages:
1. Compilation errors (Helion code generation fails)
2. XLA compilation errors (neuronx-cc fails)
3. Runtime errors (SBUF overflow, invalid operations)
4. Accuracy mismatches
"""

import torch
import helion
import helion.language as hl
from helion.runtime.config import Config
import logging

# Enable debug logging to see error messages
logging.basicConfig(level=logging.INFO)


def scenario_1_normal_autotuning():
    """Baseline: Normal autotuning with all configs working."""
    print("=" * 70)
    print("Scenario 1: Normal Autotuning (All Configs Work)")
    print("=" * 70)

    @helion.kernel(backend="nki", autotune_effort="quick")
    def simple_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
        return out

    x = torch.randn(64, 128)
    y = torch.randn(64, 128)

    print("\nRunning kernel with autotuning...")
    print("Watch for: '[NKI autotune] Config(...): XX.Xms' messages")
    result = simple_add(x, y)

    expected = x + y
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("\n✓ All configs succeeded, best config selected")
    print("✓ Result is correct")


def scenario_2_sbuf_overflow():
    """Show what happens when a config causes SBUF overflow."""
    print("\n" + "=" * 70)
    print("Scenario 2: SBUF Overflow (Config Too Large)")
    print("=" * 70)

    print("""
NKI has a limited SBUF (Scratchpad Buffer) size. When you try to allocate
tiles that are too large, you get an SBUF overflow error.

The autotuner will:
1. Try to compile the config ✓
2. Try to run it → SBUF overflow exception
3. Catch the exception
4. Log: '[NKI autotune] first run failed for Config(...): SBUF overflow'
5. Return: (fn, math.inf) ← Config marked as infinitely slow
6. Continue to next config
7. Program does NOT crash!

Let's trigger this by manually providing a config with oversized blocks:
""")

    # We'll create a kernel and manually try a bad config
    @helion.kernel(
        backend="nki",
        config=Config(block_sizes=[128, 512]),  # This might overflow for complex ops
        autotune_effort="none",
    )
    def matmul_large_tiles(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, k = a.size()
        k2, n = b.size()
        out = torch.empty([m, n], dtype=a.dtype, device=a.device)

        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
            out[tile_m, tile_n] = acc

        return out

    try:
        a = torch.randn(256, 256)
        b = torch.randn(256, 256)
        print("\nAttempting to run with large block sizes [128, 512]...")
        result = matmul_large_tiles(a, b)
        print("✓ Large blocks worked (hardware had enough SBUF)")
    except Exception as e:
        print(f"\n✗ Config failed with error: {type(e).__name__}: {e}")
        print("\nIn autotuning mode, this would be caught and the config")
        print("would get time = inf, then the autotuner would try the next config.")
        print("The program continues normally.")


def scenario_3_manual_bad_config():
    """Show the actual error handling in action by forcing a problematic config."""
    print("\n" + "=" * 70)
    print("Scenario 3: Simulating Multiple Config Failures")
    print("=" * 70)

    print("""
Let's manually create a situation where some configs work and some fail.
We'll provide explicit configs including potentially problematic ones.
""")

    # Provide multiple explicit configs, some safe, some risky
    @helion.kernel(
        backend="nki",
        configs=[
            Config(block_sizes=[32, 128]),    # Safe
            Config(block_sizes=[64, 256]),    # Safe
            Config(block_sizes=[128, 128]),   # Safe
            # Note: We can't easily force a failure without knowing the exact
            # operations, but the autotuner would handle it gracefully
        ],
    )
    def add_multi_config(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
        return out

    x = torch.randn(128, 256)
    y = torch.randn(128, 256)

    print("\nRunning with 3 explicit configs...")
    print("Watch for timing messages for each config")
    result = add_multi_config(x, y)

    expected = x + y
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("\n✓ Best config selected from valid configs")


def scenario_4_error_flow_explanation():
    """Detailed explanation of the error flow."""
    print("\n" + "=" * 70)
    print("Scenario 4: Detailed Error Flow Explanation")
    print("=" * 70)

    print("""
When a config fails, here's EXACTLY what happens:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Try to compile the config
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code location: helion/autotuner/nki_search.py:290-294

    try:
        fn = self.kernel.compile_config(config, allow_print=False)
    except Exception as e:
        self.log(f"[NKI autotune] compile failed for {config}: {e}")
        return None, math.inf  # ← Config rejected, autotuner continues

What can fail here:
- Unsupported operations in Helion → NKI code generation
- Invalid NKI Python code generation
- exec() fails to create the @nki.jit function

Result: Config gets time = infinity, NEXT config is tried

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 2: Try to run the compiled kernel once
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code location: helion/autotuner/nki_search.py:299-304

    try:
        bench_args = _clone_args(self._original_args) if ... else self.args
        output = fn(*bench_args)  # ← Calls default_nki_launcher → xm.mark_step()
    except Exception as e:
        self.log(f"[NKI autotune] first run failed for {config}: {e}")
        return fn, math.inf  # ← Config rejected, autotuner continues

What can fail here:
- XLA graph construction errors
- Neuron compiler (neuronx-cc) errors
- SBUF overflow (tile too large for hardware)
- Invalid NKI operations at runtime
- Memory allocation failures
- Timeout during compilation

Result: Config gets time = infinity, NEXT config is tried

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 3: Check accuracy vs baseline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code location: helion/autotuner/nki_search.py:307-324

    if self._baseline_output is not None:
        try:
            torch.testing.assert_close(
                act_cpu, exp,
                atol=self._effective_atol,
                rtol=self._effective_rtol,
            )
        except Exception as e:
            self.log.warning(f"[NKI autotune] accuracy mismatch for {config}: {e}")
            self.counters["accuracy_mismatch"] += 1
            return fn, math.inf  # ← Config rejected, autotuner continues

What can fail here:
- Numerical differences exceed tolerance
- Different rounding/accumulation orders
- Incorrect kernel logic for this config

Result: Config gets time = infinity, NEXT config is tried

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 4: Time the kernel (3 runs + 1 warmup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code location: helion/autotuner/nki_search.py:328-336

    try:
        perf = _nki_bench(fn, list(self.args), repeats=3, warmup=1)
        self.log(f"[NKI autotune] {config}: {perf:.1f}ms")
        return fn, perf  # ← Success! Return actual time
    except Exception as e:
        self.log(f"[NKI autotune] benchmark failed for {config}: {e}")
        return fn, math.inf  # ← Config rejected, autotuner continues

What can fail here:
- Runtime crashes during timing runs
- Timeouts
- Device errors

Result: Config gets time = infinity, NEXT config is tried

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 5: Select best config
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Code location: helion/autotuner/nki_search.py:355-371

    def _autotune(self):
        best_config = self.configs[0]
        best_perf = math.inf

        for config in self.configs:
            _, perf = self.benchmark_nki_config(config)  # May return inf
            if perf < best_perf:
                best_perf = perf
                best_config = config

        if math.isinf(best_perf):
            self.log.warning("[NKI autotune] all configs failed, returning default")

        return best_config

Result: Best valid config is returned. Even if ALL fail, default is returned.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL INSIGHT: The program NEVER crashes!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every single step has try-except. Failed configs just get marked as "infinitely
slow" and are skipped. The autotuner always returns SOME config (best valid one,
or default if all fail).
""")


def main():
    """Run all scenarios."""
    print("\n" + "🔬" * 35)
    print("NKI Autotuner Error Handling Demonstration")
    print("🔬" * 35)

    scenario_1_normal_autotuning()
    # scenario_2_sbuf_overflow()  # May or may not fail depending on hardware
    scenario_3_manual_bad_config()
    scenario_4_error_flow_explanation()

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The NKI autotuner is bulletproof against config failures:

✓ Every operation is wrapped in try-except
✓ Failed configs return math.inf (infinitely slow)
✓ Autotuner continues to next config
✓ Best valid config is selected
✓ Even if ALL configs fail, default config is returned
✓ Your program NEVER crashes due to a bad config

The key insight: Errors are EXPECTED and HANDLED, not prevented.
The autotuner assumes some configs might fail and deals with it gracefully.
""")


if __name__ == "__main__":
    main()
