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
    observations}``; ``log`` has one row per dataset or source.

    The IMF, the BIS, the World Bank and FRED are different servers, so they
    are fetched side by side; each one's datasets one after another, with a
    pause, so no server sees a burst. A dataset's countries are the core
    ones (Qatar, the GCC, the US, China, the euro area, the world) or, for
    the outlooks and the BIS's series, every major economy as well
    (``countries.csv``, ``scope``; ``datasets.csv``, ``countries``).
    ``progress`` hears one line per dataset as it starts and as it ends."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    say = progress or (lambda msg: None)
    rows = _datasets(project_root, groups=groups, only_default=only_default and not datasets)
    if datasets:
        rows = rows[rows["dataset"].isin(list(datasets))]
    ctry = _countries(project_root)
    if "scope" not in ctry.columns:
        ctry["scope"] = "core"
    if country_codes:
        ctry = ctry[ctry["code"].isin(list(country_codes))]
    core = list(ctry.loc[ctry["scope"] == "core", "code"]) or list(ctry["code"])
    major = list(ctry["code"])
    iso2 = ctry.set_index("code")["iso2"].fillna("").to_dict()
    sers, obss, log = [], [], []
    lock = threading.Lock()
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def keep(s, o, entry):
        entry.update({"series": int(len(s)), "observations": int(len(o)), "fetched": stamp})
        with lock:
            sers.append(s)
            obss.append(o)
            log.append(entry)

    def failed(entry, err):
        with lock:
            log.append({**entry, "status": "failed", "error": err, "series": 0,
                        "observations": 0, "fetched": stamp,
                        "note": "the previous vintage's copy is kept"})

    def timed(name, fn):
        t0 = time.time()
        say(f"{name}: fetching")
        out = fn()
        say(f"{name}: {out} ({time.time() - t0:.0f} s)")

    if "STATIC" in sources:
        s, o = static_seed(_static_dir(project_root))
        keep(s, o, {"dataset": "Package history files", "source": "STATIC", "status": "ok"})

    def sdmx_rows(src: str):
        api = (ImfSdmx if src == "IMF" else BisSdmx)(transport=transport)
        sel = rows[rows.get("source", pd.Series("IMF", index=rows.index))
                   .fillna("IMF").str.upper().replace("", "IMF") == src]
        dm_done = False
        for n, (_, r) in enumerate(sel.iterrows()):
            r = r.to_dict()
            codes = major if str(r.get("countries") or "core") == "major" else core
            entry = {"dataset": r["dataset"], "label": r["label"], "group": r["group"],
                     "source": src}
            if n:
                _sleep(PAUSE)

            def one():
                nonlocal dm_done
                try:
                    s, o, info = api.fetch(r, codes)
                    keep(s, o, {**entry, "status": "ok" if len(s) else "empty", **info})
                    return f"{len(s):,} series"
                except Exception as e:
                    err = str(e)
                if src == "IMF" and "DM" in sources and r["dataset"] in ("WEO", "FM") \
                        and not dm_done:
                    try:
                        s, o, info = DataMapper(transport=transport).fetch(codes)
                        keep(s, o, {**entry, "dataset": "WEO, FM", "source": "DM",
                                    "status": "ok" if len(s) else "empty",
                                    "note": f"IMF API failed ({err}); DataMapper has the "
                                            "latest release only"})
                        dm_done = True
                        return f"{len(s):,} series from the DataMapper"
                    except Exception as e:
                        err = f"{err}; DataMapper: {e}"
                if src == "IMF" and "DBNOMICS" in sources:
                    code = "WEO:latest" if r["dataset"] == "WEO" else \
                        str(r["candidates"] or r["dataset"]).split("|")[0]
                    try:
                        s, o, info = DBnomics(transport=transport).fetch(
                            r["dataset"], code, codes, [iso2.get(c, "") for c in codes])
                        keep(s, o, {**entry, "source": "DBNOMICS",
                                    "status": "ok" if len(s) else "empty", **info})
                        return f"{len(s):,} series from DBnomics"
                    except Exception as e:
                        err = f"{err}; DBnomics: {e}"
                failed(entry, err)
                return "failed: " + err[:160]
            timed(f"{src} {r['dataset']}", one)

    def world_bank():
        def one():
            try:
                wb = read_table("wb.csv", project_root)
                codes = ";".join(c for c in ctry.get("wb", ctry["code"]) if c)
                s, o, info = WorldBank(transport=transport).fetch(wb, countries=codes)
                entry = {"dataset": "WDI", "label": "World Bank development indicators",
                         "group": "World Bank", "source": "WB"}
                if len(s):
                    keep(s, o, {**entry, "status": "ok",
                                "error": "; ".join(info.get("failed", []))})
                    return f"{len(s):,} series"
                failed(entry, "; ".join(info.get("failed", [])) or "nothing returned")
                return "failed"
            except Exception as e:
                failed({"dataset": "WDI", "source": "WB"}, str(e))
                return "failed: " + str(e)[:160]
        timed("World Bank WDI", one)

    def fred():
        def one():
            key = fred_key or os.environ.get("FRED_API_KEY", "")
            cat = read_table("fred.csv", project_root)
            entry = {"dataset": "FRED", "label": "FRED series", "group": "FRED",
                     "source": "FRED"}
            try:
                if key:
                    s, o, info = Fred(key, transport=transport).fetch(
                        list(fred_ids or cat["series"]) or FRED_DEFAULT)
                else:
                    s, o, info = Fred(transport=transport).fetch_csv(cat)
            except Exception as e:
                failed(entry, str(e))
                return "failed: " + str(e)[:160]
            if not len(s):
                failed(entry, "nothing returned: " + ", ".join(info.get("missing", [])))
                return "failed"
            keep(s, o, {**entry, "status": "ok", "missing": ", ".join(info.get("missing", []))})
            return f"{len(s):,} series"
        timed("FRED", one)

    tasks = []
    if "IMF" in sources:
        tasks.append(lambda: sdmx_rows("IMF"))
    if "BIS" in sources:
        tasks.append(lambda: sdmx_rows("BIS"))
    if "WB" in sources:
        tasks.append(world_bank)
    if "FRED" in sources:
        tasks.append(fred)
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
        for f in [ex.submit(t) for t in tasks]:
            f.result()

    say("Saving the vintage")
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
