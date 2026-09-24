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
    "RatingTypes.csv": ["RatingType"],
    "PortfolioRatingType.csv": ["PortfolioCode", "RatingType"],
    "FxRate.csv": ["CurrencyCode"],
    "CollateralType.csv": ["CollateralTypeId"],
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


# ===========================================================================
# Investigating a difference.
#
# compare_outputs says WHICH files and columns differ. That is enough to know
# there is a problem and never enough to fix one: the next question is always
# "which rows, and what do they hold?". These write those rows out so they can
# be opened side by side.
# ===========================================================================

def _read_text(p: Path) -> pd.DataFrame:
    """Read as text, for the same reason compare_file does.

    Contract ids are numeric-looking KEYS; inferring them turns them into
    floats and every id then compares unequal.
    """
    df = pd.read_csv(p, dtype=str, keep_default_na=False, low_memory=False)
    return df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed:")]]


def _numeric_enough(a: pd.Series, b: pd.Series) -> bool:
    an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    n = max(len(a), 1)
    return an.notna().sum() > 0.9 * n and bn.notna().sum() > 0.9 * n


def dump_mismatches(python_dir, reference_dir, out_dir,
                    tol: float = DEFAULT_TOL, files=None) -> pd.DataFrame:
    """Write the rows that differ, so a difference can be looked at.

    Up to three files per output that has anything to report:

        <name>_unmatched_actual.csv      keys produced here that the reference
                                         does not have
        <name>_unmatched_reference.csv   the other way round
        <name>_value_diffs.csv           keys in both, values differing, with
                                         the two values side by side

    Only files with a natural key are dumped. Without one the rows can only be
    lined up positionally, and a positional "difference" on a file written in
    a different order is noise that buries the real ones.
    """
    python_dir, reference_dir = Path(python_dir), Path(reference_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = []
    names = files or sorted(p.name for p in reference_dir.glob("*.csv"))
    for name in names:
        if name.startswith("FinalEclReport"):
            continue
        key = KEYS.get(name)
        a_path, b_path = python_dir / name, reference_dir / name
        if not key or not a_path.is_file() or not b_path.is_file():
            continue
        try:
            a, b = _read_text(a_path), _read_text(b_path)
        except Exception as exc:
            written.append({"file": name, "kind": "unreadable", "rows": 0,
                            "path": "", "note": f"{type(exc).__name__}: {exc}"})
            continue
        if any(k not in a.columns or k not in b.columns for k in key):
            continue

        stem = name[:-4] if name.endswith(".csv") else name
        sep = "\u0001"
        ka = a[key].astype(str).agg(sep.join, axis=1)
        kb = b[key].astype(str).agg(sep.join, axis=1)
        a = a.assign(_k=ka)
        b = b.assign(_k=kb)

        only_a = a[~a["_k"].isin(set(kb))]
        only_b = b[~b["_k"].isin(set(ka))]
        for frame, kind in ((only_a, "unmatched_actual"),
                            (only_b, "unmatched_reference")):
            if len(frame):
                p = out / f"{stem}_{kind}.csv"
                frame.drop(columns=["_k"]).to_csv(p, index=False, na_rep="")
                written.append({"file": name, "kind": kind, "rows": len(frame),
                                "path": str(p), "note": ""})

        am = a[a["_k"].isin(set(kb))].drop_duplicates("_k").set_index("_k")
        bm = b[b["_k"].isin(set(ka))].drop_duplicates("_k").set_index("_k")
        common = [c for c in bm.columns if c in am.columns and c not in key]
        if not len(am) or not common:
            continue
        bm = bm.reindex(am.index)

        diff_mask = pd.Series(False, index=am.index)
        differing_cols = []
        for c in common:
            x, y = am[c], bm[c]
            if _numeric_enough(x, y):
                xn = pd.to_numeric(x, errors="coerce").fillna(0.0)
                yn = pd.to_numeric(y, errors="coerce").fillna(0.0)
                scale = yn.abs().clip(lower=1.0)
                col_diff = ((xn - yn).abs() / scale) > tol
            else:
                col_diff = x.fillna("").astype(str) != y.fillna("").astype(str)
            if col_diff.any():
                differing_cols.append(c)
                diff_mask |= col_diff

        if not diff_mask.any():
            continue
        rows = am.index[diff_mask]
        side = pd.DataFrame(index=range(len(rows)))
        for i, k in enumerate(key):
            side[k] = [r.split(sep)[i] for r in rows]
        for c in differing_cols:
            side[f"{c} (python)"] = am.loc[rows, c].to_numpy()
            side[f"{c} (reference)"] = bm.loc[rows, c].to_numpy()
        p = out / f"{stem}_value_diffs.csv"
        side.to_csv(p, index=False, na_rep="")
        written.append({"file": name, "kind": "value_diffs", "rows": len(side),
                        "path": str(p),
                        "note": f"{len(differing_cols)} column(s): "
                                + ", ".join(differing_cols[:5])})

    cols = ["file", "kind", "rows", "path", "note"]
    if not written:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(written, columns=cols)


def write_reconciliation_markdown(python_dir, reference_dir, path,
                                  tol: float = DEFAULT_TOL) -> Path:
    """The comparison as a document, for a reviewer rather than a console."""
    import datetime as _dt

    df = compare_outputs(python_dir, reference_dir, tol=tol)
    matched = df[df["status"] == "match"] if len(df) else df
    lines = [
        "# Reconciliation",
        f"Generated: {_dt.datetime.now().astimezone():%Y-%m-%d %H:%M:%S %z}",
        "",
        f"Produced: `{python_dir}`",
        f"Reference: `{reference_dir}`",
        "",
        f"**{len(matched)} of {len(df)} files match.**",
        "",
        "| File | Status | Rows (produced) | Rows (reference) | Columns differing |",
        "|---|---|---:|---:|---:|",
    ]
    def _cell(v) -> str:
        if v is None or (isinstance(v, float) and v != v):
            return ""
        return f"{int(v):,}" if isinstance(v, (int, float)) else str(v)

    for _, r in df.iterrows():
        lines.append(
            f"| {r['file']} | {r['status']} | "
            f"{_cell(r.get('rows_python'))} | {_cell(r.get('rows_r'))} | "
            f"{_cell(r.get('columns_differing'))} |")

    def _listy(v):
        """A cell that is absent for THIS row reads back as NaN, not as [].

        pandas fills a ragged column with NaN, so a file that reported no
        detail gives a float here and iterating it raises. Everything that
        walks these cells goes through this.
        """
        if isinstance(v, (list, tuple)):
            return list(v)
        return []

    def _texty(v) -> str:
        return "" if v is None or (isinstance(v, float) and v != v) else str(v)

    bad = df[df["status"] != "match"] if len(df) else df
    if len(bad):
        lines += ["", "## What differs", ""]
        for _, r in bad.iterrows():
            lines.append(f"### {r['file']} — {r['status']}")
            note = _texty(r.get("note"))
            if note:
                lines.append(f"- {note}")
            for m in _listy(r.get("detail")):
                lines.append(f"- `{m['column']}`: {m['differing_rows']:,} rows")
            extra, missing = _listy(r.get("extra_columns")), _listy(
                r.get("missing_columns"))
            if extra:
                lines.append("- only in the produced file: " + ", ".join(extra))
            if missing:
                lines.append("- only in the reference: " + ", ".join(missing))
            lines.append("")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
