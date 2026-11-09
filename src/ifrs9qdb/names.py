"""Customer names for a run, wherever they are recorded.

A run's report carries the customer name from engine 0.48, its contract
reference from 0.45 (with the name from 0.48), and the LIC customer master
leaves the name blank. So a run older than 0.48 has no names of its own. A
customer's name does not change from one quarter to the next, so a name
missing from a run is taken from the other runs in the same runs folder --
the most recent first.

    customer_names(run_dir)      # {customer id: name} for the run
    name_book(runs_dir)          # every run's names, newest run first
"""
from __future__ import annotations

import functools
from pathlib import Path

import pandas as pd

from .ids import as_id

__all__ = ["customer_names", "name_book", "run_names", "fill_names"]


def _str(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _squash(c) -> str:
    return "".join(ch for ch in str(c).lower() if ch.isalnum())


@functools.lru_cache(maxsize=256)
def _read(path: str, mtime_ns: int) -> dict:
    """{customer id: name} from one file that has a customer id and a
    customer name column (a contract reference, a report); {} otherwise."""
    try:
        head = pd.read_csv(path, nrows=0)
    except Exception:                                     # noqa: BLE001
        return {}
    cols = {_squash(c): c for c in head.columns}
    cid, cname = cols.get("customerid"), cols.get("customername")
    if cid is None or cname is None:
        return {}
    try:
        d = pd.read_csv(path, usecols=[cid, cname], dtype=str, keep_default_na=False,
                        low_memory=False)
    except Exception:                                     # noqa: BLE001
        return {}
    out: dict = {}
    for a, b in zip(as_id(d[cid]).astype(str), d[cname]):
        b = _str(b)
        if a and b and a.lower() not in ("nan", "<na>"):
            out.setdefault(a, b)
    return out


def _sources(run_dir: Path) -> list[Path]:
    out = run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir
    files = [run_dir / "reference" / "contracts.csv", out / "FinalEclReport.csv"]
    files += sorted(out.glob("FinalEclReport_overlay_*.csv"))
    return [f for f in files if f.is_file()]


def run_names(run_dir) -> dict:
    """A run's own names: its contract reference, then its report(s)."""
    names: dict = {}
    for f in _sources(Path(run_dir)):
        for a, b in _read(str(f), f.stat().st_mtime_ns).items():
            names.setdefault(a, b)
    return names


def name_book(runs_dir) -> dict:
    """Every run's names in a runs folder, the most recent run first."""
    root = Path(runs_dir)
    if not root.is_dir():
        return {}
    runs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    stamp = []
    for p in sorted(runs, key=lambda p: p.name, reverse=True):   # run_00018 first
        for f in _sources(p):
            stamp.append((str(f), f.stat().st_mtime_ns))
    return _book(tuple(stamp))


@functools.lru_cache(maxsize=8)
def _book(stamp: tuple) -> dict:
    names: dict = {}
    for path, mtime in stamp:
        for a, b in _read(path, mtime).items():
            names.setdefault(a, b)
    return names


def customer_names(run_dir) -> dict:
    """{customer id: name} for a run: its own, completed from the other runs
    of its runs folder."""
    run_dir = Path(run_dir)
    own = run_names(run_dir)
    book = name_book(run_dir.parent)
    if not book:
        return own
    return {**book, **own}


def fill_names(ids, names, book: dict) -> pd.Series:
    """``names`` with each blank filled from ``book`` by customer id."""
    ids = as_id(pd.Series(ids)).astype(str).reset_index(drop=True)
    if names is None:
        cur = pd.Series([""] * len(ids))
    else:
        cur = pd.Series(names).reset_index(drop=True).map(_str)
    got = ids.map(lambda a: book.get(a, ""))
    return cur.where(cur != "", got)
