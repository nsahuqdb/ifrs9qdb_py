"""
Reconciling one run against another, and packaging a run for handover.

Two jobs that look unrelated and are not: both exist so that a figure can be
defended. Reconciliation answers "why is this quarter different from last", and
the export answers "here is everything that produced it".

The reconciliation compares files by KEY rather than by row order, because a
transformation that reorders rows is not a difference anyone cares about, and
comparing positionally reports thousands of false ones.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["KEY_SPEC", "summarise_output_dir", "compare_runs",
           "reconcile_report", "build_export"]

# The natural key of each output file. Without these a file written in a
# different order compares as entirely different.
KEY_SPEC = {
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
    "Ratings.csv": ["Rating", "RatingType"],
    "Portfolios.csv": ["PortfolioCode"],
    "FinalEclReport.csv": ["Contract Id"],
}


def _output_dir(run) -> Path:
    run = Path(run)
    return run / "Output" if (run / "Output").is_dir() else run


def summarise_output_dir(run) -> pd.DataFrame:
    """Row and column counts per file, for a quick shape comparison."""
    d = _output_dir(run)
    rows = []
    for p in sorted(d.glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False, dtype=str,
                             keep_default_na=False)
            rows.append({"file": p.name, "rows": len(df),
                         "columns": len(df.columns),
                         "bytes": p.stat().st_size})
        except Exception as exc:
            rows.append({"file": p.name, "rows": None, "columns": None,
                         "bytes": p.stat().st_size,
                         "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def _compare_one(a: Path, b: Path, key: list[str] | None,
                 tol: float = 1e-6) -> dict:
    """Compare one file against its counterpart."""
    if not a.is_file():
        return {"file": b.name, "status": "only in reference"}
    if not b.is_file():
        return {"file": a.name, "status": "only in this run"}

    # Read as text: numeric-looking KEYS become floats otherwise, and every
    # row then compares unequal.
    da = pd.read_csv(a, dtype=str, keep_default_na=False, low_memory=False)
    db = pd.read_csv(b, dtype=str, keep_default_na=False, low_memory=False)
    da = da.loc[:, [c for c in da.columns if not str(c).startswith("Unnamed")]]
    db = db.loc[:, [c for c in db.columns if not str(c).startswith("Unnamed")]]

    only_a = [c for c in da.columns if c not in db.columns]
    only_b = [c for c in db.columns if c not in da.columns]
    shared = [c for c in db.columns if c in da.columns]

    out = {"file": a.name, "rows_this": len(da), "rows_reference": len(db),
           "columns_added": only_a, "columns_missing": only_b}

    usable_key = key and all(k in da.columns and k in db.columns for k in key)
    if usable_key:
        ka = da[key].agg("|".join, axis=1)
        kb = db[key].agg("|".join, axis=1)
        added = sorted(set(ka) - set(kb))
        removed = sorted(set(kb) - set(ka))
        out["rows_added"] = len(added)
        out["rows_removed"] = len(removed)
        out["examples_added"] = added[:10]
        out["examples_removed"] = removed[:10]
        common = list(set(ka) & set(kb))
        da = da.assign(_k=ka).set_index("_k").loc[common].sort_index()
        db = db.assign(_k=kb).set_index("_k").loc[common].sort_index()
    elif len(da) != len(db):
        out["status"] = "row count differs"
        return out

    diffs = []
    for c in shared:
        sa, sb = da[c], db[c]
        na = pd.to_numeric(sa, errors="coerce")
        nb = pd.to_numeric(sb, errors="coerce")
        numeric = (na.notna().sum() > 0.9 * len(sa)
                   and nb.notna().sum() > 0.9 * len(sb))
        if numeric:
            scale = nb.abs().clip(lower=1.0)
            rel = (na.fillna(0) - nb.fillna(0)).abs() / scale
            bad = int((rel > tol).sum())
            worst = float(rel.max()) if len(rel) else 0.0
        else:
            bad = int((sa != sb).sum())
            worst = 0.0
        if bad:
            diffs.append({"column": c, "rows": bad, "worst_relative": worst})

    out["columns_differing"] = len(diffs)
    out["detail"] = sorted(diffs, key=lambda d: -d["rows"])[:10]
    out["status"] = ("match" if not diffs and not only_a and not only_b
                     and not out.get("rows_added") and not out.get("rows_removed")
                     else "differs")
    return out


def compare_runs(run, reference, tol: float = 1e-6) -> pd.DataFrame:
    """Compare every file of one run against another."""
    a, b = _output_dir(run), _output_dir(reference)
    names = sorted({p.name for p in a.glob("*.csv")}
                   | {p.name for p in b.glob("*.csv")})
    return pd.DataFrame([_compare_one(a / n, b / n, KEY_SPEC.get(n), tol)
                         for n in names])


def reconcile_report(run, reference) -> dict:
    """The provision movement between two runs, with the file comparison.

    The headline is the ECL difference, because that is the question asked in a
    review; the file-level comparison is what explains it.
    """
    from .analytics import ecl_walk, normalise

    files = compare_runs(run, reference)
    a = _output_dir(run) / "FinalEclReport.csv"
    b = _output_dir(reference) / "FinalEclReport.csv"

    out = {
        "files": files.to_dict(orient="records"),
        "files_matching": int((files["status"] == "match").sum()),
        "files_compared": len(files),
    }
    if a.is_file() and b.is_file():
        cur = normalise(pd.read_csv(a, low_memory=False))
        prev = normalise(pd.read_csv(b, low_memory=False))
        w = ecl_walk(prev, cur)
        out["walk"] = {
            "opening": w["opening"], "closing": w["closing"],
            "residual": w["residual"],
            "steps": w["steps"].to_dict(orient="records"),
            "counts": w["counts"],
        }
    return out


# =============================================================== export =====
def build_export(run, dest_zip, include_inputs: bool = False) -> dict:
    """Package a run for handover or archive.

    Everything needed to defend the figure goes in: the LIC input files, the
    report, the frozen configuration, the validation result, the approvals and
    the audit trail. A README lists what is present and what it is for, because
    a zip of CSVs with no explanation is not evidence.

    The raw source extracts are OPTIONAL and off by default: they are large and
    contain customer data, so including them should be a decision rather than
    an accident.
    """
    run = Path(run)
    out_dir = _output_dir(run)
    dest = Path(dest_zip)
    dest.parent.mkdir(parents=True, exist_ok=True)

    contents, skipped = [], []

    def add(zf, path: Path, arc: str):
        if path.is_file():
            zf.write(path, arc)
            contents.append({"path": arc, "bytes": path.stat().st_size})
        else:
            skipped.append(arc)

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("*.csv")):
            add(zf, p, f"Output/{p.name}")

        cfg = run / "config_used"
        if cfg.is_dir():
            for p in sorted(cfg.rglob("*")):
                if p.is_file():
                    add(zf, p, f"config_used/{p.relative_to(cfg)}")

        for name in ("manifest.json", "validation.json", "approval.json",
                     "audit.jsonl"):
            add(zf, run / name, name)

        if include_inputs:
            inp = run / "input"
            if inp.is_dir():
                for p in sorted(inp.iterdir()):
                    if p.is_file():
                        add(zf, p, f"input/{p.name}")

        zf.writestr("README.txt", _readme(run.name, contents, skipped))

    return {"zip": str(dest), "bytes": dest.stat().st_size,
            "files": len(contents), "skipped": skipped}


def _readme(run_id: str, contents: list[dict], skipped: list[str]) -> str:
    total = sum(c["bytes"] for c in contents)
    lines = [
        f"IFRS 9 ECL — run {run_id}",
        f"Packaged {datetime.now().isoformat(timespec='seconds')}",
        "",
        "WHAT IS HERE",
        "  Output/           the eighteen LIC input files and the ECL report",
        "  config_used/      the configuration frozen when the run was built,",
        "                    with a hash per file so it can be shown unaltered",
        "  manifest.json     what produced the run: inputs, engine version, steps",
        "  validation.json   every check that was run, passes included",
        "  approval.json     who signed the run off, and when",
        "  audit.jsonl       an append-only record of what was done to it",
        "",
        f"  {len(contents)} files, {total / 1e6:.1f} MB",
    ]
    if skipped:
        lines += ["", "NOT PRESENT",
                  "  " + ", ".join(skipped),
                  "  A missing approval or validation file means that step was",
                  "  never run, not that it passed."]
    lines += [
        "",
        "READING THE NUMBERS",
        "  The report carries the model figure and, where overlays were applied,",
        "  the overlay amount and the final figure separately. Stage 3 is booked",
        "  at the full outstanding on the QDB basis, which DIVERGES from a LIC",
        "  extract on every Stage 3 row; that is intended, not a defect.",
    ]
    return "\n".join(lines) + "\n"
