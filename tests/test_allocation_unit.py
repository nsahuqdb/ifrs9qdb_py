"""ALLOCATIONPERCENTAGE's unit is configured (run.allocation_percentage_unit),
not guessed. Mirrors the R engine's tests/testthat/test-allocation-unit.R."""
from __future__ import annotations

import pandas as pd
import pytest

from ifrs9qdb.etl.transform import transform_allocation
from ifrs9qdb.runconfig import allocation_percentage_unit
from ifrs9qdb.validation import INPUT_STAGE
from ifrs9qdb.validation.schema import canonicalise


def _aca(p):
    return pd.DataFrame({"EXTRACTDA": "6/9/2026",
                         "COLLATERALID": ["C1", "C1", "C2"],
                         "CONTRACTID": ["1", "2", "2"],
                         "ALLOCATIONPERCENTAGE": p})


def _cfg(u):
    return {"run": {"extract_date": "2026-06-09", "allocation_percentage_unit": u}}


def _run(id_, p, u):
    v = next(v for v in INPUT_STAGE if v.id == id_)
    return v.fn(inputs=canonicalise({"AccountCollateralAllocation": _aca(p)}),
                run_config=_cfg(u))


def test_the_transform_divides_by_the_configured_unit():
    share = lambda p, u: list(transform_allocation(_aca(p), "6/9/2026", unit=u)
                              ["AllocationPercentage"])
    assert share([60, 40, 100], "percent") == pytest.approx([0.6, 0.4, 1])
    assert share([0.6, 0.4, 1], "fraction") == pytest.approx([0.6, 0.4, 1])
    # percentages all at most 1%: the old guess read them as fractions
    assert share([0.5, 1, 0.25], "percent") == pytest.approx([0.005, 0.01, 0.0025])
    assert share([0.5, 1, 0.25], "auto") == pytest.approx([0.5, 1, 0.25])
    assert share([50, 1, 0.25], "auto") == pytest.approx([0.5, 0.01, 0.0025])
    assert allocation_percentage_unit({"run": {}}) == "percent"
    assert allocation_percentage_unit(_cfg(" Fraction ")) == "fraction"


def test_a_file_that_contradicts_the_unit_fails_the_pre_run_check():
    i = "INPUT_ACA_allocation_unit_consistent"
    assert _run(i, [60, 40, 100], "percent")["passed"]
    r = _run(i, [0.6, 0.4, 1], "percent")
    assert not r["passed"]
    assert "every AllocationPercentage is at most 1 (3 value(s), max 1)" in r["detail"]
    assert _run(i, [0.6, 0.4, 1], "fraction")["passed"]
    r = _run(i, [60, 0.4, 1], "fraction")
    assert "1 AllocationPercentage value(s) above 1 (max 60)" in r["detail"]
    assert _run(i, [0.6, 0.4, 1], "auto")["passed"]
    r = _run(i, [60, 40, 100], "percentage")
    assert "run.allocation_percentage_unit is 'percentage'" in r["detail"]
    v = next(v for v in INPUT_STAGE if v.id == i)
    assert v.severity == "ERROR" and v.suppressible


def test_the_range_and_sum_checks_read_the_configured_unit():
    r = "INPUT_ACA_allocation_in_percent_range"
    assert _run(r, [0.6, 0.4, 1], "fraction")["passed"]
    assert _run(r, [0.6, 0.4, 1.2], "fraction")["detail"] == "1 outside [0,1], 0 non-numeric"
    assert _run(r, [60, 40, 120], "percent")["detail"] == "1 outside [0,100], 0 non-numeric"
    # contract 2 draws 0.4 + 0.9 = 130% on two items
    t = _run("INPUT_ACA_total_allocation_per_contract", [0.6, 0.4, 0.9], "fraction")
    assert "1 ContractIds have total allocation > 100%" in t["detail"]
    # C1 allocated 0.7 + 0.6 = 130%: invisible to a percent threshold before
    x = _run("XFILE_ACA_allocation_sum_per_collateral", [0.7, 0.6, 1], "fraction")
    assert not x["passed"]
    assert "1 collateral id(s) allocated above 100% (max 130.0%)" in x["detail"]
    assert _run("XFILE_ACA_allocation_sum_per_collateral", [70, 30, 100], "percent")["passed"]
