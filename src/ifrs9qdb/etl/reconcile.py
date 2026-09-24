"""
Comparing a Python run against an R run.

The port is only trustworthy if its eighteen files agree with the ones the R
pipeline produced from the same inputs. This does that comparison and reports
where they differ, rather than asserting equality and stopping at the first
mismatch -- the whole picture is more useful than the first problem.

Numeric columns are compared with a tolerance because the two languages format
floats differently; text is compared exactly after trimming, because a stray
space in a key breaks a join downstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["compare_file", "compare_outputs", "reconciliation_report"]

DEFAULT_TOL = 1e-6


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype(str).str.strip().replace({"nan": "", "None": ""})
    return out


def compare_file(actual: Path, reference: Path, key: list[str] | None = None,
                 tol: float = DEFAULT_TOL) -> dict:
    """Compare one output file against its reference.

    Returns a verdict rather than raising, so a run can report on all eighteen.
    """
    if not Path(actual).is_file():
        return {"file": Path(reference).name, "status": "missing",
                "note": "not produced by the Python pipeline"}
    if not Path(reference).is_file():
        return {"file": Path(actual).name, "status": "no reference",
                "note": "nothing to compare against"}

    # Read as text. Contract ids like 1123000471 are numeric-looking but are
    # KEYS, and letting pandas infer them turns them into floats -- after which
    # every id compares unequal and the diff is meaningless.
    a = _norm(pd.read_csv(actual, dtype=str, keep_default_na=False,
                          low_memory=False))
    b = _norm(pd.read_csv(reference, dtype=str, keep_default_na=False,
                          low_memory=False))

    # The R writer leaves a trailing comma on some files, which pandas reads
    # back as an unnamed empty column. It carries no data and LIC ignores it,
    # so it is not a difference worth reporting.
    b = b.loc[:, [c for c in b.columns if not str(c).startswith("Unnamed:")]]
    a = a.loc[:, [c for c in a.columns if not str(c).startswith("Unnamed:")]]
    only_a = [c for c in a.columns if c not in b.columns]
    only_b = [c for c in b.columns if c not in a.columns]
    shared = [c for c in b.columns if c in a.columns]


    if len(a) != len(b):
        return {"file": Path(reference).name, "status": "row count differs",
                "rows_python": len(a), "rows_r": len(b),
                "note": f"{len(a):,} vs {len(b):,}",
                "extra_columns": only_a, "missing_columns": only_b}

    # order-independent where a key is given, otherwise positional
    if key and all(k in a.columns and k in b.columns for k in key):
        a = a.sort_values(key).reset_index(drop=True)
        b = b.sort_values(key).reset_index(drop=True)

    mismatches, worst = [], 0.0
    for c in shared:
        # Booleans read back as True/False, and numpy will not subtract them.
        # Compare them as text, which is also how they are written.
        if a[c].dtype == bool or b[c].dtype == bool:
            bad = int((a[c].astype(str).str.lower()
                       != b[c].astype(str).str.lower()).sum())
            if bad:
                mismatches.append({"column": c, "differing_rows": bad})
            continue
        an = pd.to_numeric(a[c], errors="coerce")
        bn = pd.to_numeric(b[c], errors="coerce")
        numeric = (an.notna().sum() > 0.9 * len(a)
                   and bn.notna().sum() > 0.9 * len(b))
        if numeric:
            diff = (an.fillna(0).astype(float) - bn.fillna(0).astype(float)).abs()
            scale = bn.abs().clip(lower=1.0)
            rel = (diff / scale).max()
            bad = int((diff / scale > tol).sum())
            worst = max(worst, float(rel) if np.isfinite(rel) else 0.0)
        else:
            bad = int((a[c].fillna("") != b[c].fillna("")).sum())
        if bad:
            mismatches.append({"column": c, "differing_rows": bad})

    status = "match" if not mismatches and not only_a and not only_b else "differs"
    return {
        "file": Path(reference).name, "status": status,
        "rows_python": len(a), "rows_r": len(b),
        "columns_differing": len(mismatches),
        "worst_relative_diff": worst,
        "extra_columns": only_a, "missing_columns": only_b,
        "detail": mismatches[:10],
    }


# Natural keys, so a file written in a different order still compares.
KEYS = {
    "AccountMaster_1.csv": ["ContractId"],
    "AccountMaster_2.csv": ["ContractId"],
    "CustomerMaster_1.csv": ["CustomerId"],
    "CustomerMaster_2.csv": ["CustomerId"],
    "CustomerStagingFlag_1.csv": ["CustomerId"],
    "CustomerStagingFlag_2.csv": ["CustomerId"],
    "Origination_1.csv": ["ContractId"],
    "Origination_2.csv": ["ContractId"],
    "Collateral.csv": ["CollateralId"],
    "AccountCollateralAllocation.csv": ["ContractId", "CollateralId"],
    "LifeTimeParameterOther.csv": ["ContractId", "MonthLifetime"],
    "StPD.csv": ["PortfolioCode", "PDBucketDim1", "MonthLifetime"],
    "Ratings.csv": ["Rating"],
    "Portfolios.csv": ["PortfolioCode"],
}


def compare_outputs(python_dir, reference_dir, tol: float = DEFAULT_TOL) -> pd.DataFrame:
    """Compare every file the reference run produced."""
    python_dir, reference_dir = Path(python_dir), Path(reference_dir)
    rows = []
    for ref in sorted(reference_dir.glob("*.csv")):
        if ref.name.startswith("FinalEclReport"):
            continue          # the report is an outcome, not an ETL output
        rows.append(compare_file(python_dir / ref.name, ref,
                                 key=KEYS.get(ref.name), tol=tol))
    return pd.DataFrame(rows)


def reconciliation_report(python_dir, reference_dir) -> str:
    """A readable summary, for a console or a screen."""
    df = compare_outputs(python_dir, reference_dir)
    if len(df) == 0:
        return "No reference files to compare against."
    counts = df["status"].value_counts().to_dict()
    lines = [
        f"{len(df)} files compared against {Path(reference_dir)}",
        "  " + "  ".join(f"{k}: {v}" for k, v in counts.items()),
        "",
    ]
    for r in df.itertuples():
        mark = "ok  " if r.status == "match" else "--> "
        note = ""
        if r.status == "differs":
            note = f"{r.columns_differing} column(s) differ"
            if getattr(r, "missing_columns", None):
                note += f", missing {r.missing_columns}"
        elif r.status == "row count differs":
            note = f"{r.rows_python:,} vs {r.rows_r:,} rows"
        elif r.status == "missing":
            note = "not produced yet"
        lines.append(f"  {mark}{r.file:<36}{note}")
    return "\n".join(lines)
