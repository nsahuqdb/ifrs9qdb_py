"""One refresh of the macro store: every dataset, every source, one vintage.

``refresh`` walks ``datasets.csv`` (the groups and datasets asked for), fetches
each from the IMF's SDMX API for the countries in ``countries.csv``, falls back
to the DataMapper (WEO and Fiscal Monitor) or DBnomics when the IMF cannot be
reached, adds the FRED series when a key is set, seeds the package's own
history files, and writes one vintage with a log line per dataset: what was
fetched, from where, and what failed and why. A dataset that fails keeps its
copy from the previous vintage (``MacroStore.save``), and the log says so.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Callable

import pandas as pd

from .catalogue import countries as _countries
from .catalogue import datasets as _datasets
from .catalogue import read_table
from .sources import (FRED_DEFAULT, PAUSE, BisSdmx, DataMapper, DBnomics, Fred,
                      ImfSdmx, WorldBank, _sleep, empty_frames, read_file, static_seed)
from .store import MacroStore

__all__ = ["refresh", "import_file", "SOURCES"]

SOURCES = ("IMF", "BIS", "WB", "DM", "DBNOMICS", "FRED", "STATIC")
DEFAULT_SOURCES = ("IMF", "BIS", "WB", "FRED", "STATIC")


def _static_dir(project_root=None) -> Path:
    if project_root and (Path(project_root) / "data-raw" / "static").is_dir():
        return Path(project_root) / "data-raw" / "static"
    return Path(__file__).resolve().parent.parent / "static"


def refresh(store: MacroStore, project_root=None, groups=None, datasets=None,
            country_codes=None, sources=DEFAULT_SOURCES,
            fred_key: str | None = None, fred_ids=None, transport=None,
            label: str = "", progress: Callable[[str], None] | None = None,
            only_default: bool = True) -> dict:
    """Fetch and save one vintage. Returns ``{vintage, log, series,
    observations}``; ``log`` has one row per dataset or source."""
    say = progress or (lambda msg: None)
    rows = _datasets(project_root, groups=groups, only_default=only_default and not datasets)
    if datasets:
        rows = rows[rows["dataset"].isin(list(datasets))]
    ctry = _countries(project_root)
    codes3 = list(country_codes or ctry["code"])
    codes2 = list(ctry.set_index("code").reindex(codes3)["iso2"].fillna(""))
    sers, obss, log = [], [], []
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def keep(s, o, entry):
        sers.append(s)
        obss.append(o)
        entry.update({"series": int(len(s)), "observations": int(len(o)),
                      "fetched": stamp})
        log.append(entry)

    if "STATIC" in sources:
        s, o = static_seed(_static_dir(project_root))
        keep(s, o, {"dataset": "Package history files", "source": "STATIC",
                    "status": "ok"})

    apis = {"IMF": ImfSdmx(transport=transport) if "IMF" in sources else None,
            "BIS": BisSdmx(transport=transport) if "BIS" in sources else None}
    dm_done = False
    for n, (_, r) in enumerate(rows.iterrows()):
        r = r.to_dict()
        src = str(r.get("source") or "IMF").upper()
        if src not in apis:
            continue
        say(f"{r['dataset']}: {r['label']}")
        entry = {"dataset": r["dataset"], "label": r["label"], "group": r["group"]}
        err = None
        api = apis[src]
        if api is None:
            continue
        if n:
            _sleep(PAUSE)
        try:
            s, o, info = api.fetch(r, codes3)
            keep(s, o, {**entry, "source": src, "status": "ok" if len(s) else "empty", **info})
            continue
        except Exception as e:
            err = str(e)
        if src != "IMF":
            log.append({**entry, "source": src, "status": "failed", "error": err,
                        "series": 0, "observations": 0, "fetched": stamp,
                        "note": "the previous vintage's copy is kept"})
            continue
        if "DM" in sources and r["dataset"] in ("WEO", "FM") and not dm_done:
            try:
                s, o, info = DataMapper(transport=transport).fetch(codes3)
                keep(s, o, {**entry, "dataset": "WEO, FM", "source": "DM",
                            "status": "ok" if len(s) else "empty",
                            "note": (f"IMF API failed ({err}); " if err else "")
                            + "DataMapper has the latest release only"})
                dm_done = True
                continue
            except Exception as e:
                err = f"{err}; DataMapper: {e}" if err else f"DataMapper: {e}"
        if "DBNOMICS" in sources:
            code = "WEO:latest" if r["dataset"] == "WEO" else \
                str(r["candidates"] or r["dataset"]).split("|")[0]
            try:
                s, o, info = DBnomics(transport=transport).fetch(r["dataset"], code,
                                                                 codes3, codes2)
                keep(s, o, {**entry, "source": "DBNOMICS",
                            "status": "ok" if len(s) else "empty", **info})
                continue
            except Exception as e:
                err = f"{err}; DBnomics: {e}" if err else f"DBnomics: {e}"
        log.append({**entry, "source": "-", "status": "failed",
                    "error": err or "no source selected", "series": 0,
                    "observations": 0, "fetched": stamp,
                    "note": "the previous vintage's copy is kept"})

    if "WB" in sources:
        say("World Bank: development indicators")
        try:
            wb = read_table("wb.csv", project_root)
            s, o, info = WorldBank(transport=transport).fetch(wb)
            keep(s, o, {"dataset": "WDI", "label": "World Bank development indicators",
                        "group": "World Bank", "source": "WB",
                        "status": "ok" if len(s) else "failed",
                        "error": "; ".join(info.get("failed", []))})
        except Exception as e:
            log.append({"dataset": "WDI", "source": "WB", "status": "failed", "error": str(e),
                        "series": 0, "observations": 0, "fetched": stamp,
                        "note": "the previous vintage's copy is kept"})

    if "FRED" in sources:
        say("FRED")
        key = fred_key or os.environ.get("FRED_API_KEY", "")
        cat = read_table("fred.csv", project_root)
        if not key:
            try:
                s, o, info = Fred(transport=transport).fetch_csv(cat)
                keep(s, o, {"dataset": "FRED", "label": "FRED series (CSV download)",
                            "source": "FRED", "status": "ok" if len(s) else "failed",
                            "missing": ", ".join(info.get("missing", []))})
            except Exception as e:
                log.append({"dataset": "FRED", "source": "FRED", "status": "failed",
                            "error": str(e), "series": 0, "observations": 0,
                            "fetched": stamp})
        if key:
            fred_ids = fred_ids or list(cat["series"]) or FRED_DEFAULT
            try:
                s, o, info = Fred(key, transport=transport).fetch(list(fred_ids or FRED_DEFAULT))
                keep(s, o, {"dataset": "FRED", "label": "FRED series", "source": "FRED",
                            "status": "ok", "missing": ", ".join(info.get("missing", []))})
            except Exception as e:
                log.append({"dataset": "FRED", "source": "FRED", "status": "failed",
                            "error": str(e), "series": 0, "observations": 0,
                            "fetched": stamp})

    s = pd.concat(sers, ignore_index=True) if sers else empty_frames()[0]
    o = pd.concat(obss, ignore_index=True) if obss else empty_frames()[1]
    vid = store.save(s, o, label=label, log=log)
    return {"vintage": vid, "log": pd.DataFrame(log),
            "series": int(len(store.series(vid))),
            "observations": int(len(store.observations(vid)))}


def import_file(store: MacroStore, path, source: str, dataset: str,
                country: str = "", freq: str | None = None, unit: str = "",
                label: str = "") -> dict:
    """Add one file's series as a new vintage (everything else carried
    forward): a QCB bulletin table, a Planning Council download, the
    Ministry of Finance budget, an SDMX-CSV from the IMF portal."""
    s, o = read_file(path, source=source.upper(), dataset=dataset.upper(),
                     country=country.upper(), freq=freq, unit=unit)
    if s.empty:
        raise ValueError("No series could be read from that file.")
    log = [{"dataset": dataset.upper(), "source": source.upper(), "status": "ok",
            "file": Path(path).name, "series": int(len(s)),
            "observations": int(len(o)),
            "fetched": _dt.datetime.now().strftime("%Y-%m-%d %H:%M")}]
    vid = store.save(s, o, label=label or f"{source.upper()} {dataset.upper()} file",
                     log=log)
    return {"vintage": vid, "log": pd.DataFrame(log), "series": int(len(s)),
            "observations": int(len(o))}
