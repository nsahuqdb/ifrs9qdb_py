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
        # De-duplicated before indexing. A key is not always unique -- a LIC
        # report repeats a contract id for a security held twice -- and
        # .loc[common] on a repeated key fans the rows out, giving the two
        # sides different shapes and a "Can only compare identically-labeled
        # Series" a page away from where the cause is.
        common = sorted(set(ka) & set(kb))
        da = (da.assign(_k=ka).drop_duplicates("_k").set_index("_k")
              .loc[common].sort_index())
        db = (db.assign(_k=kb).drop_duplicates("_k").set_index("_k")
              .loc[common].sort_index())
        dup = int(ka.duplicated().sum() + kb.duplicated().sum())
        if dup:
            out["duplicate_keys"] = dup
    elif len(da) != len(db):
        out["status"] = "row count differs"
        return out

    if len(da) != len(db):
        out["status"] = "row count differs"
        return out

    diffs = []
    for c in shared:
        # Compared as arrays, not as Series: the two frames are aligned by
        # construction at this point, and comparing on labels re-introduces an
        # alignment that has already been done.
        sa, sb = da[c], db[c]
        na = pd.to_numeric(sa, errors="coerce").to_numpy()
        nb = pd.to_numeric(sb, errors="coerce").to_numpy()
        import numpy as _np
        numeric = (_np.isfinite(na).sum() > 0.9 * len(na)
                   and _np.isfinite(nb).sum() > 0.9 * len(nb))
        if numeric:
            a0 = _np.nan_to_num(na)
            b0 = _np.nan_to_num(nb)
            scale = _np.clip(_np.abs(b0), 1.0, None)
            rel = _np.abs(a0 - b0) / scale
            bad = int((rel > tol).sum())
            worst = float(rel.max()) if rel.size else 0.0
        else:
            bad = int((sa.astype("string").fillna("").to_numpy()
                       != sb.astype("string").fillna("").to_numpy()).sum())
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
    if not run.is_dir():
        # An empty zip named after a run that does not exist is a trap: it
        # looks like a handover and carries nothing.
        raise FileNotFoundError(f"No run at {run}")
    out_dir = _output_dir(run)
    dest = Path(dest_zip)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Everything sits under <run_id>/ so unzipping gives one self-contained
    # folder rather than scattering CSVs into whatever directory it was
    # opened in.
    root = run.name

    contents, skipped = [], []

    def add(zf, path: Path, arc: str):
        if path.is_file():
            zf.write(path, f"{root}/{arc}")
            contents.append({"path": arc, "bytes": path.stat().st_size})
        else:
            skipped.append(arc)

    def add_tree(zf, src: Path, arc: str):
        if not src.is_dir():
            skipped.append(f"{arc}/")
            return
        for p in sorted(src.rglob("*")):
            if p.is_file():
                add(zf, p, f"{arc}/{p.relative_to(src)}")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.glob("*.csv")):
            add(zf, p, f"Output/{p.name}")

        add_tree(zf, run / "config_used", "config_used")
        # The whole reports tree, not a named handful: a run writes its
        # reconciliation, its input source and its run status there, and a
        # handover that carries only the ones somebody thought to list is a
        # handover that loses the one that mattered.
        add_tree(zf, run / "reports", "reports")
        add_tree(zf, run / "overrides", "overrides")

        # These live at the run root in some layouts and under reports/ in
        # others. Naming one as NOT PRESENT while the reviewer is holding
        # reports/manifest.json would send them looking for a file they have.
        packaged = {c["path"] for c in contents}
        for name in ("manifest.json", "validation.json", "approval.json",
                     "audit.jsonl"):
            if f"reports/{name}" in packaged:
                continue
            add(zf, run / name, name)

        if include_inputs:
            inp = next((run / n for n in ("input", "inputs", "Inputs")
                        if (run / n).is_dir()), None)
            if inp is None:
                skipped.append("input/")
            else:
                add_tree(zf, inp, "input")

        # Two files that answer at a glance what would otherwise be a hunt
        # through three YAML files.
        for name, text in (("code_version.txt", _code_version_block(run)),
                           ("approval_summary.txt", _approval_block(run))):
            zf.writestr(f"{root}/{name}", text)
            contents.append({"path": name, "bytes": len(text.encode())})

        zf.writestr(f"{root}/README.txt",
                    _readme(run.name, contents, skipped, include_inputs))

    return {"zip": str(dest), "bytes": dest.stat().st_size,
            "files": len(contents), "skipped": skipped,
            "contents": [c["path"] for c in contents]}


def _code_version_block(run: Path) -> str:
    """What the code was when the run happened, and what it is now.

    Both, because they answer different questions: the first is what produced
    these numbers, the second is what a reader checking the branch out today
    would get. Where they differ, saying so beats implying the current code is
    what ran.
    """
    import json as _json

    from .code_version import code_status

    lines = [f"Packaged {datetime.now().isoformat(timespec='seconds')}"]
    manifest = {}
    for cand in (run / "reports" / "manifest.json", run / "manifest.json"):
        if cand.is_file():
            try:
                manifest = _json.loads(cand.read_text(encoding="utf-8"))
                break
            except Exception:
                pass
    at_run = manifest.get("code") or manifest.get("code_version")
    lines.append(f"At run time:    {at_run if at_run else 'not recorded in this run'}")
    try:
        lines.append(f"At export time: {code_status()}")
    except Exception as exc:
        lines.append(f"At export time: unavailable ({exc})")
    return "\n".join(lines) + "\n"


def _approval_block(run: Path) -> str:
    """Who made this run, who approved it, and on what configuration."""
    from .run_status import read_run_status

    lines = [f"Run: {run.name}", ""]
    try:
        status = read_run_status(run) or {}
    except Exception:
        status = {}
    if status:
        for k in ("status", "run_type", "kind", "created_by", "created_at",
                  "submitted_by", "checker", "decided_at", "reason", "comment"):
            if status.get(k) not in (None, ""):
                lines.append(f"{k.replace('_', ' ').capitalize()}: {status[k]}")
    else:
        lines.append("No run_status record: this run was never submitted for "
                     "approval, or predates the maker-checker trail.")

    frozen = (run / "config_used").is_dir()
    lines += ["", "Configuration: " + (
        "frozen in config_used/, which is what produced these numbers"
        if frozen else
        "NOT frozen. This run predates the config freeze, so the files on "
        "disk now may differ from what was active at run time.")]
    marker = run / "config_used" / "config_used.yml"
    if marker.is_file():
        try:
            lines.append(marker.read_text(encoding="utf-8").strip())
        except Exception:
            pass

    ov = run / "overrides"
    lines += ["", f"Overrides applied: "
                  f"{len([p for p in ov.glob('*') if p.is_file()]) if ov.is_dir() else 0}"]
    return "\n".join(lines) + "\n"


def _readme(run_id: str, contents: list[dict], skipped: list[str],
            include_inputs: bool = False) -> str:
    total = sum(c["bytes"] for c in contents)
    lines = [
        f"IFRS 9 ECL — run {run_id}",
        f"Packaged {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Everything is under a single folder named for the run, so unzipping",
        "gives one self-contained tree.",
        "",
        "WHAT IS HERE",
        "  Output/           the eighteen LIC input files and the ECL report",
        "  config_used/      the configuration frozen when the run was built,",
        "                    with a hash per file so it can be shown unaltered",
        "  reports/          validation, reconciliation, input source, status",
        "  overrides/        manual overrides applied to this run, if any",
        "  manifest.json     what produced the run: inputs, engine version, steps",
        "  validation.json   every check that was run, passes included",
        "  approval.json     who signed the run off, and when",
        "  audit.jsonl       an append-only record of what was done to it",
        "  code_version.txt  the code at run time and at packaging time",
        "  approval_summary.txt  who made it, who approved it, on what config",
    ]
    if include_inputs:
        lines.append("  input/            the source extracts the run read")
    lines += ["", f"  {len(contents)} files, {total / 1e6:.1f} MB"]
    if skipped:
        lines += ["", "NOT PRESENT",
                  "  " + ", ".join(skipped),
                  "  A missing approval or validation file means that step was",
                  "  never run, not that it passed. A missing config_used/",
                  "  means the run predates the config freeze, so the files on",
                  "  disk now may differ from what was active at run time."]
    lines += [
        "",
        "READING THE NUMBERS",
        "  The report carries the model figure and, where overlays were applied,",
        "  the overlay amount and the final figure separately. Stage 3 is booked",
        "  at the full outstanding on the QDB basis, which DIVERGES from a LIC",
        "  extract on every Stage 3 row; that is intended, not a defect.",
    ]
    return "\n".join(lines) + "\n"
