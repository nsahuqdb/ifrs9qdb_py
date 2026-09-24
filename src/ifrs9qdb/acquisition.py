"""Getting the input bundle into a directory the ETL can read.

Three ways a quarter's extracts arrive, and none of them is "already in the
configured folder":

  * someone uploads a zip;
  * the data team drops a dated folder into a shared location;
  * a path is given directly.

This module only PREPARES an input directory. The ETL still calls
``read_all_inputs()`` on whatever it produces, so acquisition and loading stay
separable and a bundle can be inspected before anything is run against it.

The structural check here is deliberately cheap -- are the twelve files there,
and does each one open. It is not a substitute for the INPUT validators, which
run inside the ETL and check content. It exists so a malformed bundle is
rejected before a run is attempted, rather than failing partway through.
"""
from __future__ import annotations

import datetime as _dt
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import yaml

from .etl.read_inputs import INPUT_SPECS

__all__ = ["expected_input_files", "acquire_inputs_from_zip", "list_data_drops",
           "validate_input_directory", "record_input_source"]

_DATA_SUFFIXES = {".xlsx", ".xls", ".csv"}


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def expected_input_files(include_alternatives: bool = False) -> list[str]:
    """The file names a complete bundle carries.

    Format is detected from the bytes rather than the extension -- four of
    these arrive as SQL*Plus HTML with an .xls name -- so the alternatives
    matter for FINDING a file, not for reading it.
    """
    out = []
    for spec in INPUT_SPECS:
        names = [getattr(spec, "file", f"{spec.name}.xlsx")]
        if include_alternatives:
            names += list(getattr(spec, "file_alternatives", []) or [])
        out.extend(names)
    return out


def acquire_inputs_from_zip(zip_path, dest_root=None) -> dict:
    """Extract an uploaded bundle into a fresh timestamped directory.

    A new directory per upload, never a reused one: two uploads in the same
    session must not be able to mix, and a half-overwritten bundle is the kind
    of thing that produces a run nobody can explain afterwards.

    A zip that wraps everything in one top-level folder is unwrapped, because
    people zip the folder as often as its contents and both should work.
    """
    zp = Path(zip_path)
    if not zp.is_file():
        raise FileNotFoundError(f"uploaded zip not found: {zp}")

    root = Path(dest_root) if dest_root else \
        Path(tempfile.gettempdir()) / "ifrs9_uploaded_inputs"
    dest = root / _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zp) as z:
        for member in z.infolist():
            # A zip can name a member "../../etc/passwd". Resolve and refuse
            # anything that lands outside the destination.
            target = (dest / member.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise ValueError(
                    f"refusing to extract {member.filename!r}: it would write "
                    "outside the destination directory")
        z.extractall(dest)

    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        if any(p.suffix.lower() in _DATA_SUFFIXES for p in inner.iterdir()):
            dest = inner

    return {"path": str(dest), "source_zip": zp.name, "extracted_at": _now()}


def list_data_drops(drop_root) -> pd.DataFrame:
    """The dated folders in a shared drop location, newest first.

    No naming convention is enforced: the data team's structure changes, and a
    tool that only recognises `YYYY-MM` stops seeing the drop the month they
    rename it.
    """
    cols = ["name", "path", "modified", "n_files", "looks_complete"]
    if not drop_root:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    root = Path(drop_root)
    if not root.is_dir():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})

    wanted = {Path(f).name.lower() for f in expected_input_files(True)}
    stems = {Path(f).stem.lower() for f in expected_input_files()}
    rows = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = [p for p in sub.rglob("*") if p.is_file()]
        names = {p.name.lower() for p in files}
        found_stems = {p.stem.lower() for p in files
                       if p.suffix.lower() in _DATA_SUFFIXES}
        rows.append({
            "name": sub.name,
            "path": str(sub),
            "modified": _dt.datetime.fromtimestamp(sub.stat().st_mtime),
            "n_files": len(files),
            # Complete by STEM, not by exact name: the same extract arrives as
            # .xlsx one quarter and .xls the next.
            "looks_complete": stems <= found_stems or wanted <= names,
        })
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return (pd.DataFrame(rows, columns=cols)
            .sort_values("modified", ascending=False).reset_index(drop=True))


def validate_input_directory(input_dir) -> pd.DataFrame:
    """A cheap structural check: are the files there, and do they open.

    Returns one row per check so it renders the same way a validation result
    does. Content checks belong to the INPUT validators, which run inside the
    ETL; this only decides whether attempting a run is worthwhile.
    """
    rows = []

    def add(check, ok, detail=""):
        rows.append({"check": check, "status": "PASS" if ok else "FAIL",
                     "detail": str(detail)})

    d = Path(input_dir) if input_dir else None
    if d is None or not d.is_dir():
        add("Input directory exists", False, f"no directory at {input_dir}")
        return pd.DataFrame(rows, columns=["check", "status", "detail"])
    add("Input directory exists", True, str(d))

    from .etl.read_inputs import resolve_input_path

    found = {}
    for spec in INPUT_SPECS:
        try:
            p = resolve_input_path(d, spec)
        except Exception:
            p = None
        if p is None:
            candidates = [getattr(spec, "file", "")] + \
                list(getattr(spec, "file_alternatives", []) or [])
            add(f"{spec.name} present", False,
                "tried " + ", ".join(c for c in candidates if c))
        else:
            found[spec.name] = Path(p)
            add(f"{spec.name} present", True, Path(p).name)

    for name, p in found.items():
        try:
            with open(p, "rb") as fh:
                fh.read(2048)
            add(f"{name} readable", True, "")
        except OSError as exc:
            add(f"{name} readable", False, str(exc))

    am = found.get("AccountMaster")
    if am is not None:
        try:
            from .etl.read_inputs import read_input
            df = read_input(am, next(s for s in INPUT_SPECS
                                     if s.name == "AccountMaster"))
            cols = {"".join(ch for ch in str(c).lower() if ch.isalnum())
                    for c in df.columns}
            # The canary: if this column is missing the bundle is not the
            # extract this pipeline reads, whatever the file is called.
            add("AccountMaster has a CONTRACTID column", "contractid" in cols,
                f"{len(df):,} rows, {len(df.columns)} columns")
        except Exception as exc:
            add("AccountMaster has a CONTRACTID column", False,
                f"{type(exc).__name__}: {exc}")

    return pd.DataFrame(rows, columns=["check", "status", "detail"])


def record_input_source(run_dir, kind: str, details: dict | None = None) -> Path:
    """Write where a run's inputs came from, into the run itself.

    Without this the run records what it produced and not what it read, and a
    quarter cannot be reproduced without asking somebody to remember.
    """
    reports = Path(run_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    meta = {"schema_version": "1.0", "kind": kind, "recorded_at": _now(),
            "details": {k: str(v) for k, v in (details or {}).items()}}
    p = reports / "input_source.yml"
    p.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
                 encoding="utf-8")
    return p
