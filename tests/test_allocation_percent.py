"""ALLOCATIONPERCENTAGE arrives as a percentage (0-100); the LIC file carries
the fraction. Divided by 100 when any value is above 1; a file with no value
above 1 is taken as fractions already. Mirrors the R engine's
tests/testthat/test-allocation-percent.R."""
from __future__ import annotations

import pandas as pd
import pytest

from ifrs9qdb.etl.transform import transform_allocation, write_outputs
from ifrs9qdb.validation import INPUT_STAGE
from ifrs9qdb.validation.schema import canonicalise


def _aca(p):
    return pd.DataFrame({"EXTRACTDA": "6/9/2026",
                         "COLLATERALID": ["C1", "C1", "C2"],
                         "CONTRACTID": ["1", "2", "2"],
                         "ALLOCATIONPERCENTAGE": p})


def _run(id_, p):
    v = next(v for v in INPUT_STAGE if v.id == id_)
    return v.fn(inputs=canonicalise({"AccountCollateralAllocation": _aca(p)}))


def test_divided_by_100_when_any_value_is_above_1(tmp_path):
    share = lambda p: list(transform_allocation(_aca(p), "6/9/2026")
                           ["AllocationPercentage"])
    assert share([60, 40, 100]) == pytest.approx([0.6, 0.4, 1])
    assert share([60, 0.5, 1]) == pytest.approx([0.6, 0.005, 0.01])
    # no value above 1: already fractions, passed through
    assert share([0.6, 0.4, 1]) == pytest.approx([0.6, 0.4, 1])
    # four decimals in the file, as R writes it
    write_outputs(tmp_path, {"AccountCollateralAllocation.csv":
                             transform_allocation(_aca([0.6, 0.4, 1]), "6/9/2026")})
    lines = (tmp_path / "AccountCollateralAllocation.csv").read_text().splitlines()
    assert lines[1] == "6/9/2026,C1,1,0.6000"


def test_the_range_and_sum_checks_read_percentages():
    r = "INPUT_ACA_allocation_in_percent_range"
    assert _run(r, [60, 40, 100])["passed"]
    assert _run(r, [60, 40, 120])["detail"] == "1 outside [0,100], 0 non-numeric"
    # C1 is allocated 70 + 60 = 130%
    x = _run("XFILE_ACA_allocation_sum_per_collateral", [70, 60, 100])
    assert not x["passed"]
    assert "1 collateral id(s) allocated above 100% (max 130.0%)" in x["detail"]
    assert _run("XFILE_ACA_allocation_sum_per_collateral", [70, 30, 100])["passed"]
