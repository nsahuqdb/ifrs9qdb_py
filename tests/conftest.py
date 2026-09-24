"""Where the tests find their reference data.

The golden fixtures under ``tests/fixtures/`` ship with the repository and are
small and synthetic, so the suite means something on a fresh clone with nothing
configured.

The reconciliation tests need a real run, and a real run is real portfolio
data: customer identifiers, exposures and provisions. That never goes in git.
Point the environment at a copy instead:

    export IFRS9_REF_RUN=/path/to/runs/run_00001    # holds Output/
    export IFRS9_REF_RUN_PREV=/path/to/runs/run_00000   # optional, the run before
    export IFRS9_SRC_INPUTS=/path/to/extracts       # the raw Oracle files
    pytest -q

``IFRS9_REF_RUN_PREV`` is what the movement analytics want: two real consecutive
runs. Without it those tests fall back to a perturbed copy of the one run, which
exercises the arithmetic but not the shape of a real quarter.

Without them those tests skip and say which variable was missing, rather than
failing or - worse - silently passing on absent data.

The config and the static reference are NOT in this category: they ship inside
the package and are byte-identical to the R package's ``inst/``, so anything
that needs only those runs everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def _from_env(name: str) -> Path | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.exists() else None


def ref_run() -> Path | None:
    """A run directory holding ``Output/`` with the reference CSVs."""
    p = _from_env("IFRS9_REF_RUN")
    if p is None:
        return None
    # Accept either the run folder or the Output folder itself.
    return p.parent if p.name == "Output" else p


def ref_output() -> Path | None:
    run = ref_run()
    if run is None:
        return None
    out = run / "Output"
    return out if out.is_dir() else (run if (run / "StPD.csv").is_file() else None)


def ref_file(name: str) -> Path | None:
    """One reference CSV, from the run if configured, else a bundled fixture."""
    out = ref_output()
    if out is not None and (out / name).is_file():
        return out / name
    local = FIXTURES / name
    return local if local.is_file() else None


def prev_run() -> Path | None:
    """The run BEFORE the reference one, for the movement analytics."""
    p = _from_env("IFRS9_REF_RUN_PREV")
    if p is None:
        return None
    return p.parent if p.name == "Output" else p


def prev_file(name: str) -> Path | None:
    """One CSV from the previous run, or None when no previous run is set."""
    run = prev_run()
    if run is None:
        return None
    for cand in (run / "Output" / name, run / name):
        if cand.is_file():
            return cand
    return None


def src_inputs() -> Path | None:
    """The raw extracts the ETL reads."""
    return _from_env("IFRS9_SRC_INPUTS")


def packaged_config() -> Path:
    """The config that ships inside the package."""
    import ifrs9qdb
    return Path(ifrs9qdb.__file__).parent / "config"


needs_ref_run = pytest.mark.skipif(
    ref_output() is None,
    reason="set IFRS9_REF_RUN to a run folder holding Output/")

needs_src_inputs = pytest.mark.skipif(
    src_inputs() is None,
    reason="set IFRS9_SRC_INPUTS to the raw extract directory")

needs_report = pytest.mark.skipif(
    ref_file("FinalEclReport.csv") is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")
