# NKI backend port — documentation

Working documentation for porting the Helion **NKI backend** (AWS Trainium) onto
upstream `main`. These are development/process docs, not user-facing API docs.

Read in roughly this order:

| Doc | What it is |
|-----|-----------|
| [NKI_PORT_IMPLEMENTATION_PLAN.md](NKI_PORT_IMPLEMENTATION_PLAN.md) | The original step-by-step port plan (P1.1–P2.x). Diff-grounded checklist for moving the NKI backend from the `fix-nki-kernel-compilation` reference fork onto upstream `main`. |
| [NKI_PORT_COMMIT_LOG.md](NKI_PORT_COMMIT_LOG.md) | The authoritative running log — every commit/fix on `nki-port-v2`, with root-cause analysis (gdn_fwd_h, jagged_hstu_attn, int4_gemm, long_sum, the >128 transpose, get_num_sm xla, etc.) and the hardware-sweep results (49/51 examples). |
| [NKI_SUBPACKAGE_REFACTOR_GUIDE.md](NKI_SUBPACKAGE_REFACTOR_GUIDE.md) | How the inline NKI load/store codegen was refactored out of `memory_ops.py` into `helion/_compiler/nki/`, mirroring the Pallas subpackage structure. |
| [NKI_HELION_TESTSUITE_SWEEP.md](NKI_HELION_TESTSUITE_SWEEP.md) | Executive summary of running Helion's own unit-test suite against `HELION_BACKEND=nki` (642 pass / 526 fail / 217 skip; 16 files green). Prod-readiness assessment. |
| [NKI_TESTSUITE_FAILURE_ANALYSIS.md](NKI_TESTSUITE_FAILURE_ANALYSIS.md) | In-depth companion to the sweep: every failure category, exact counts, which files, the mechanism, and a per-file appendix + triage priorities. |

Branch: `nki-port-v2`. The NKI backend implementation lives in
`helion/_compiler/nki_backend.py` and the `helion/_compiler/nki/` subpackage.
