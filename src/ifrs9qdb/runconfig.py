"""Values read from config.yml's ``run:`` block, as R's load_config.R reads them."""
from __future__ import annotations

import pandas as pd

__all__ = ["allocation_percentage_unit", "allocation_divisor", "ALLOCATION_UNITS"]

ALLOCATION_UNITS = ("percent", "fraction", "auto")


def allocation_percentage_unit(run_config) -> str:
    """The unit of AccountCollateralAllocation's ALLOCATIONPERCENTAGE:
    run.allocation_percentage_unit, "percent" when unset (R's
    allocation_percentage_unit()).

    "percent" (57.25 is 57.25%), "fraction" (0.5725 is 57.25%) or "auto" (the
    old guess: percent when any value is above 1, else fraction -- which reads
    a file of percentages that are all at most 1% as fractions, 100 times too
    large). Any other value is returned as given, lower-cased, for
    INPUT_ACA_allocation_unit_consistent to report; the run reads it as "auto".
    """
    run = run_config.get("run") if isinstance(run_config, dict) else None
    u = run.get("allocation_percentage_unit") if isinstance(run, dict) else None
    if u is None or (isinstance(u, float) and pd.isna(u)) or not str(u).strip():
        return "percent"
    return str(u).strip().lower()


def allocation_divisor(pct, unit: str) -> float:
    """What the values are divided by to give a share of 1 (R's
    allocation_divisor()): 100 for percent, 1 for fraction; for "auto" and an
    unknown unit, 100 when any value is above 1, else 1."""
    if unit == "percent":
        return 100.0
    if unit == "fraction":
        return 1.0
    x = pd.to_numeric(pd.Series(pct), errors="coerce")
    return 100.0 if bool((x > 1).any()) else 1.0
