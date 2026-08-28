from __future__ import annotations

import os
import warnings


def _maybe_enable_nki_on_general_tests() -> None:
    """Opt-in: run the general (Triton-gated) test bodies against the NKI backend.

    Many functionality tests are gated ``@onlyBackends(["triton", ...])`` and
    therefore SKIP under ``HELION_BACKEND=nki``. For the NKI prod-readiness
    sweep we want those upstream test bodies (with their UPSTREAM sizes — the
    files are unmodified) to actually execute against NKI. When
    ``HELION_NKI_TEST_SWEEP=1`` is set, monkeypatch ``onlyBackends`` so any
    class that admits ``triton`` also admits ``nki``. This is env-gated and
    affects nothing outside the sweep; backend-specific classes (cute/pallas/
    tileir only) still skip correctly because they don't list triton.
    """
    if os.environ.get("HELION_NKI_TEST_SWEEP") != "1":
        return
    import unittest

    from helion import _testing

    _orig = _testing.onlyBackends

    def _patched(backends):  # type: ignore[no-untyped-def]
        from helion.runtime.settings import _get_backend

        def wrapper(cls):  # type: ignore[no-untyped-def]
            if _get_backend() == "nki" and "triton" in backends:
                return cls  # admit nki wherever triton is admitted
            return _orig(backends)(cls)

        return wrapper

    _testing.onlyBackends = _patched
    # tests import onlyBackends by name, so patch the module attr BEFORE collection
    unittest  # noqa: B018  (kept to make the intent explicit; harmless)


def pytest_configure() -> None:
    # TODO(tcombes): remove this once Pallas RNG generation avoids int64.
    # JAX x64 is disabled on TPU, so RNG-generated int64s are truncated and
    # spam Pallas test logs with one warning per generated statement.
    warnings.filterwarnings(
        "ignore",
        message=(
            "Explicitly requested dtype int64 requested in .* is not available, "
            "and will be truncated to dtype int32.*"
        ),
        category=UserWarning,
    )
    _maybe_enable_nki_on_general_tests()
