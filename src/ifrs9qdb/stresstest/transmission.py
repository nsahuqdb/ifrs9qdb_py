"""How a scenario reaches the provision.

The scenario's main variables and their linked effects (``macro.system``)
give every variable's path; three channels carry them to the book, each
priced through the same ``etl.report.compute_ecl`` the run used:

  1 PD model      non-oil GDP into the internal PD model and GCC growth into
                  the external one: the run's frozen config with the paths
                  added, the PD chain rebuilt (``stress.pd_chain``)
  2 stage moves   each Stage 1 contract's chance of moving to Stage 2 within
                  the year rises with its 12-month PD through the one-factor
                  model QDB's own stage history supports, plus the share a
                  payment-deferral programme moves; each extra share moving
                  costs its lifetime ECL less its 12-month ECL (an expected
                  value: no contract is picked at random)
  3 collateral    property collateral revalued by the lowest point property
                  prices reach over the horizon, so LGD moves through the
                  engine's own formula

The waterfall adds the channels in that order. Nothing here changes the
engine or a run: the baseline is the run's own report repriced (or, for a
run priced by an earlier version of the calculation, its config rebuilt on
this one).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..macro.catalogue import concepts as _concepts
from ..stress import REPRICE_KEYS, _ingredients, frozen_config, pd_chain, reprice, summarise_repricing
from .config import read_stress_config
from .scenarios import Scenario

__all__ = ["RunContext", "prepare", "transmit", "LAYERS", "scenario_paths", "channel_paths",
           "collateral_shares", "customer_sectors", "stage_moves", "stressed_pd12",
           "property_trough", "driver_edits", "severity"]

LAYERS = [("pd", "PD model"), ("staging", "Stage moves"), ("collateral", "Collateral")]


def _dt_year() -> int:
    import datetime as _dt
    return _dt.date.today().year


def customer_sectors(run_path, static=None) -> tuple[dict, str]:
    """Customer id -> sector, and where it came from.

    A run written by this version keeps ``stress/customer_sectors.csv``
    (the pipeline's own industry lookup). For an earlier run the industry
    extract is read again from the input folder its audit log names, with
    the run's frozen industry mapping -- right as long as that folder has not
    been overwritten by a later quarter's files, which is said."""
    run_path = Path(run_path)
    p = run_path / "stress" / "customer_sectors.csv"
    if p.is_file():
        t = pd.read_csv(p, dtype=str, keep_default_na=False)
        if {"CustomerId", "Sector"} <= set(t.columns):
            return ({k.strip(): v.strip() for k, v in zip(t["CustomerId"], t["Sector"])
                     if v.strip()}, "run")
    try:
        import json
        from ..etl.lending import _customer_sector
        from ..etl.read_inputs import INPUT_SPECS, read_input, resolve_input_path
        line = (run_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
        idir = Path(json.loads(line).get("input_dir", ""))
        spec = next(s for s in INPUT_SPECS if s.name == "IndustryCode")
        f = resolve_input_path(idir, spec) if idir.is_dir() else None
        if f is None or static is None:
            return {}, ""
        ind = read_input(f)
        ind.columns = [str(c).strip().upper().replace(" ", "") for c in ind.columns]
        ids = ind.get("CUSTOMERID")
        if ids is None:
            return {}, ""
        ids = ids.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        ind["CUSTOMERID"] = ids
        sec = _customer_sector(ids.drop_duplicates().reset_index(drop=True), ind,
                               static.get("industry_sector_mapping"))
        return ({k: v for k, v in zip(ids.drop_duplicates(), sec) if isinstance(v, str)},
                "inputs")
    except Exception:
        return {}, ""


def collateral_shares(out_dir, patterns: dict) -> dict:
    """For each concept, each contract's share of its net collateral (value
    x allocation x (1 - haircut), as the engine nets it) whose type's
    description matches the concept's pattern."""
    from ..etl.report import _chr, _col, _num, read_output_csv
    alloc = read_output_csv(out_dir, "AccountCollateralAllocation.csv")
    coll = read_output_csv(out_dir, "Collateral.csv")
    ctype = read_output_csv(out_dir, "CollateralType.csv")
    if alloc is None or coll is None or ctype is None:
        return {}
    tid = _chr(_col(ctype, "CollateralTypeId")).to_numpy()
    hc = pd.Series(_num(_col(ctype, "HaircutGeneral")).to_numpy(), index=tid)
    hc = hc[~hc.index.duplicated(keep="first")]
    desc = pd.Series(_chr(_col(ctype, "Description")).fillna("").to_numpy(), index=tid)
    desc = desc[~desc.index.duplicated(keep="first")]
    cid = _chr(_col(coll, "CollateralId")).to_numpy()
    cval = pd.Series(_num(_col(coll, "CollateralValue")).to_numpy(), index=cid)
    cval = cval[~cval.index.duplicated(keep="first")]
    ctyp = pd.Series(_chr(_col(coll, "CollateralTypeId")).to_numpy(), index=cid)
    ctyp = ctyp[~ctyp.index.duplicated(keep="first")]
    a_cid = _chr(_col(alloc, "ContractId"))
    a_clid = _chr(_col(alloc, "CollateralId"))
    a_pct = _num(_col(alloc, "AllocationPercentage"))
    typ = a_clid.map(ctyp)
    v = a_clid.map(cval).astype(float)
    h = typ.map(hc).astype(float)
    contrib = (v * a_pct * (1 - h)).where(lambda x: np.isfinite(x), 0.0)
    d = typ.map(desc).fillna("")
    keep = a_cid.notna()
    tot = contrib[keep].groupby(a_cid[keep]).sum()
    out = {}
    for concept, pat in (patterns or {}).items():
        try:
            rx = re.compile(pat)
        except re.error:
            continue
        hit = d.map(lambda s: bool(rx.search(str(s)))) & keep
        part = contrib[hit].groupby(a_cid[hit]).sum()
        share = (part / tot.reindex(part.index)).where(lambda x: np.isfinite(x), 0.0)
        out[concept] = share.clip(0, 1).to_dict()
    return out


@dataclass
class RunContext:
    """Everything about a run a scenario needs, read once."""
    inputs: object
    report: pd.DataFrame
    run_path: Path
    cfg: dict
    system: object
    H: int
    first_year: int
    ing: pd.DataFrame
    segment: np.ndarray
    external: np.ndarray
    stage0: np.ndarray
    pd12: np.ndarray
    coll_share: dict
    drivers_internal: dict      # concept -> (column, multiplier, active)
    drivers_external: dict      # concept -> country
    base_curves: dict | None
    base: pd.DataFrame
    notes: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def years(self) -> list:
        return list(range(self.first_year, self.first_year + self.H))

    @property
    def base_total(self) -> float:
        return float(np.nansum(self.base["ecl"]))


def _rt(v) -> int:
    t = str(v)
    return 2 if (t.startswith("External") or t in ("2", "2.0")) else 1


def prepare(inputs, report: pd.DataFrame, run_path, system, cfg: dict | None = None,
            project_root=None) -> RunContext:
    """Read a run for stress scenarios: its contracts, sectors, curves,
    collateral and PD model drivers, with the fitted macro system."""
    run_path = Path(run_path)
    cfg = cfg or read_stress_config(project_root)
    H = int(cfg.get("horizon_years", 5))
    ing = _ingredients(inputs, report)
    notes = []

    fz = frozen_config(run_path)
    sect, how = customer_sectors(run_path, fz.get("static") if fz.get("ok") else None)
    ext_pf = set(inputs.external_portfolios())
    external = (ing["portfolio"].isin(ext_pf) | ing["rating_type"].map(_rt).eq(2)).to_numpy()
    if sect:
        seg = np.array([sect.get(str(c).strip(), "Others") for c in ing["customer"]], dtype=object)
        if how == "inputs":
            notes.append("Sectors were read from the input folder this run was built "
                         "from; they are this run's only if that folder still holds "
                         "its files.")
    else:
        seg = ing["portfolio"].astype(str).to_numpy().astype(object)
    seg[external] = "External book"

    base_curves = None
    ch = pd_chain(inputs, report, run_path, weight_mode="hold")
    if ch.get("ok") and ch.get("base_curves") is not None:
        base_curves = ch["base_curves"]
        notes.append("This run was priced by another version of the calculation, so "
                     "scenarios are measured from its config rebuilt on this one.")
    di, de = {}, {}
    cat = _concepts(project_root)
    if fz.get("ok"):
        model_cfg = fz["model"] or {}
        varcfg = model_cfg.get("variables") or {}
        active = set()
        for m in (model_cfg.get("models") or {}).values():
            if str((m or {}).get("rating_type", "")).lower().startswith("internal"):
                for comp in (m.get("mev_components") or []):
                    if float(comp.get("weight") or 0) != 0:
                        active.add(comp.get("variable"))
        # the external model's driver: the variable set one path per group
        grouped = {v for v, c in varcfg.items()
                   if ((c or {}).get("forecast_source") or {}).get("aggregation")}
        for _, r in cat.iterrows():
            mv = str(r["model_variable"] or "")
            base, _, grp = mv.partition(":")
            if grp and (base in grouped or base == "GCC"):
                de[r["concept"]] = grp
            elif mv in varcfg:
                fs = (varcfg[mv] or {}).get("forecast_source") or {}
                col = fs.get("column_index")
                mult = float((varcfg[mv] or {}).get("stress_unit_multiplier") or 1.0)
                if col:
                    di[r["concept"]] = (int(col), mult, mv in active)

    drv = di.get((cfg.get("pd_link") or {}).get("internal", "NONOIL_GDP_QAT"))
    if drv is None or not drv[2]:
        notes.append("This run's internal PD model does not use non-oil GDP, so scenarios "
                     "reach the internal book through stages and collateral only.")
    base = reprice(inputs, report, pd_curves=base_curves)
    mi = (fz.get("raw") or {}) if fz.get("ok") else {}
    first = int(((mi.get("macro_source") or {}).get("first_forecast_year")) or 0) or None
    if first is None:
        ed = pd.to_datetime(report["extract_date"], errors="coerce").dropna() \
            if "extract_date" in report.columns else []
        first = int(ed.iloc[0].year) if len(ed) else _dt_year()
    rc = RunContext(inputs=inputs, report=report, run_path=run_path, cfg=cfg, system=system,
                    H=H, first_year=first, ing=ing, segment=seg, external=external,
                    stage0=ing["stage"].fillna(2).astype(int).to_numpy(),
                    pd12=np.array([]), coll_share=collateral_shares(inputs.out_dir,
                                                                    cfg.get("collateral") or {}),
                    drivers_internal=di, drivers_external=de, base_curves=base_curves,
                    base=base, notes=notes)
    rc.pd12 = stressed_pd12(rc, base_curves)
    return rc


# ------------------------------------------------------------------ paths --
def scenario_paths(scn: Scenario, system, H: int = 5) -> dict:
    """Every variable's deviation under a scenario: the main variables as
    set, the rest by the linked effects (``MacroSystem.simulate``)."""
    return system.simulate(scn.given(H), scn.linked, H)


def channel_paths(rc: RunContext, paths: dict) -> dict:
    """The three paths the book reads: the internal PD driver, GCC growth,
    and property prices."""
    pl = rc.cfg.get("pd_link") or {}
    z = np.zeros(rc.H)

    def get(v):
        p = paths.get(v)
        return z.copy() if p is None else np.asarray(p, float)[:rc.H]
    out = {"internal": get(pl.get("internal", "NONOIL_GDP_QAT")),
           "external": get(pl.get("external", "GDP_GCC"))}
    for c in rc.coll_share:
        out[c] = get(c)
    return out


def severity(rc: RunContext, scn: Scenario) -> dict:
    """Each main variable's largest move, in standard deviations of its
    yearly history."""
    out = {}
    for v, p in scn.given(rc.H).items():
        vals = [x for x in p if x is not None]
        sd = rc.system.sd(v) if v in rc.system.order else float("nan")
        if vals and sd and np.isfinite(sd) and sd > 0:
            w = max(vals, key=abs)
            out[v] = float(w / sd)
    return out


# --------------------------------------------------------------- channels --
def driver_edits(rc: RunContext, ch: dict, slope_scale: float = 1.0) -> tuple[dict, dict]:
    """The PD chain's edits: the internal driver's path into each active
    model variable's forecast column, GCC growth into every country's."""
    mev, gcc = {}, {}
    p = np.asarray(ch["internal"], float) * float(slope_scale)
    drv = rc.drivers_internal.get((rc.cfg.get("pd_link") or {}).get("internal", "NONOIL_GDP_QAT"))
    if drv is not None and np.any(np.abs(p) > 1e-12):
        col, mult, _ = drv
        mev[col] = list(p / mult)
    g = np.asarray(ch["external"], float)
    if np.any(np.abs(g) > 1e-12):
        for c, country in rc.drivers_external.items():
            gcc[country] = list(g)
    return mev, gcc


def stressed_pd12(rc: RunContext, pd_curves=None) -> np.ndarray:
    """Each contract's 12-month PD on the curves given (the run's, or a
    rebuilt chain's), at its own rating."""
    ctx = rc.inputs.ctx.copy(stpd=pd_curves) if pd_curves is not None else rc.inputs.ctx
    keys = rc.ing[["portfolio", "rating", "rating_type"]].astype(str).agg("|".join, axis=1)
    out = np.full(len(rc.ing), np.nan)
    memo: dict = {}
    for i, (k, r) in enumerate(zip(keys, rc.ing[["portfolio", "rating", "rating_type"]]
                                   .itertuples(index=False))):
        if k not in memo:
            cm = ctx.pd_curve(r.portfolio, r.rating, r.rating_type)
            if cm is None or len(cm) == 0:
                memo[k] = np.nan
            else:
                c = np.asarray(cm, dtype=float)
                memo[k] = float(c[min(12, c.size - 1)])
        out[i] = memo[k]
    return out


def _k(cfg: dict) -> float:
    s = cfg.get("staging") or {}
    rs, rd = float(s.get("rho_stage", 0.033)), float(s.get("rho_default", 0.032))
    return float(np.sqrt(rs / (1 - rs)) / np.sqrt(rd / (1 - rd)))


def stage_moves(rc: RunContext, pd12_after, restructured: float = 0.0) -> dict:
    """Each contract's extra chance of moving from Stage 1 to Stage 2 in the
    scenario: its rating part (the one-factor model on its 12-month PD's
    probit move) and the deferral programme's share on top."""
    s = rc.cfg.get("staging") or {}
    p12 = float(s.get("p12", 0.063))
    k = _k(rc.cfg)
    pb, ps = rc.pd12, np.asarray(pd12_after, float)
    elig = (rc.stage0 == 1) & np.isfinite(pb) & np.isfinite(ps) & (pb > 0) & (pb < 1) \
        & (ps > 0) & (ps < 1)
    if s.get("internal_only", True):
        elig &= ~rc.external
    dz = np.zeros(len(pb))
    dz[elig] = norm.ppf(ps[elig]) - norm.ppf(pb[elig])
    rating = np.where(elig, np.clip(norm.cdf(norm.ppf(p12) + k * dz) - p12, 0.0, 1.0), 0.0)
    total = np.where(elig, np.clip(norm.cdf(norm.ppf(p12) + k * dz) - p12
                                   + float(restructured), 0.0, 1.0), 0.0)
    return {"eligible": elig, "dz": dz, "rating": rating, "total": total, "k": k, "p12": p12}


def property_trough(path) -> float:
    """The lowest point a %-change path's cumulative level reaches, in per
    cent (0 when it never falls below the start)."""
    p = np.asarray(path, float)
    if not p.size:
        return 0.0
    lvl = np.cumprod(1.0 + p / 100.0) - 1.0
    return float(min(0.0, 100.0 * lvl.min()))


def _share(rc: RunContext, concept: str) -> np.ndarray:
    key = f"share:{concept}"
    if key not in rc._cache:
        shares = rc.coll_share.get(concept) or {}
        rc._cache[key] = np.array([shares.get(str(x), 0.0) if isinstance(x, str) else 0.0
                                   for x in rc.ing["cid"]])
    return rc._cache[key]


def _price(rc, kw) -> pd.DataFrame:
    return reprice(rc.inputs, rc.report, **{k: v for k, v in kw.items() if k in REPRICE_KEYS})


def _curves(rc: RunContext, mev: dict, gcc: dict):
    if not mev and not gcc:
        return rc.base_curves
    key = ("chain", repr(sorted((k, tuple(np.round(v, 10))) for k, v in mev.items())),
           repr(sorted((k, tuple(np.round(v, 10))) for k, v in gcc.items())))
    if key not in rc._cache:
        ch = pd_chain(rc.inputs, rc.report, rc.run_path, mev_deltas=mev or None,
                      gcc_deltas=gcc or None, weight_mode="hold")
        if not ch.get("ok"):
            raise RuntimeError(ch.get("reason", "The PD chain could not be rebuilt."))
        if len(rc._cache) > 64:
            for k in [k for k in rc._cache if isinstance(k, tuple) and k[0] == "chain"][:16]:
                rc._cache.pop(k, None)
        rc._cache[key] = ch["curves"]
    return rc._cache[key]


def price_channels(rc: RunContext, paths: dict, restructured: float = 0.0,
                   slope_scale: float = 1.0) -> dict:
    """Every contract's ECL after each channel in turn, and what each
    channel did."""
    ch = channel_paths(rc, paths)
    mev, gcc = driver_edits(rc, ch, slope_scale)
    curves = _curves(rc, mev, gcc)
    kw = {"pd_curves": curves} if curves is not None else {}
    rA = _price(rc, kw)
    e_pd = rA["ecl"].to_numpy(float)

    pd12 = stressed_pd12(rc, curves) if (mev or gcc) else rc.pd12
    sm = stage_moves(rc, pd12, restructured)
    dp = sm["total"]
    stage2 = rc.stage0.copy()
    stage2[sm["eligible"]] = 2
    moved = bool(np.any(dp > 0))
    if moved:
        e_s2 = _price(rc, {**kw, "stage": stage2})["ecl"].to_numpy(float)
        gap = np.where(sm["eligible"], np.nan_to_num(e_s2 - e_pd), 0.0)
    else:
        gap = np.zeros(len(e_pd))
    e_st = e_pd + dp * gap

    trough = {c: property_trough(ch[c]) for c in rc.coll_share}
    hit = {c: t for c, t in trough.items() if t < 0}
    e_col, coll_info = e_st, {"trough": trough}
    if hit:
        cn0 = rc.ing["collateral_net"].fillna(0.0).to_numpy(float)
        mult = np.ones(len(cn0))
        for c, t in hit.items():
            mult += _share(rc, c) * t / 100.0
        cn = np.maximum(0.0, cn0 * mult)
        rC = _price(rc, {**kw, "collateral_net": cn})["ecl"].to_numpy(float)
        if moved:
            rC2 = _price(rc, {**kw, "collateral_net": cn, "stage": stage2})["ecl"].to_numpy(float)
            gapc = np.where(sm["eligible"], np.nan_to_num(rC2 - rC), 0.0)
        else:
            gapc = np.zeros(len(rC))
        e_col = rC + dp * gapc
        coll_info.update({"contracts": int(np.sum(mult < 1)),
                          "collateral_before": float(cn0.sum()),
                          "collateral_after": float(cn.sum())})
    restr_part = float(np.nansum((dp - sm["rating"]) * gap))
    return {"after_pd": e_pd, "after_staging": e_st, "after": e_col, "frame": rA,
            "pd12": pd12, "stage": sm, "gap": gap, "collateral": coll_info,
            "restructured_part": restr_part, "channels": ch,
            "edits": {"internal": {str(k): v for k, v in mev.items()}, "external": gcc}}


def transmit(rc: RunContext, scn, waterfall: bool = True, paths: dict | None = None,
             slope_scale: float = 1.0) -> dict:
    """Price a scenario. Returns the totals, the waterfall by channel, every
    variable's path with its baseline, and the split by book, segment,
    portfolio, stage and customer."""
    if isinstance(scn, dict):
        scn = Scenario.from_dict(scn)
    if paths is None:
        sim = scenario_paths(scn, rc.system, rc.H)
    else:
        sim = {"paths": paths, "role": {v: ("main" if v in scn.given(rc.H) else "linked")
                                        for v in paths}, "flags": {}, "recommended": []}
    try:
        pc = price_channels(rc, sim["paths"], scn.restructured, slope_scale)
    except RuntimeError as e:
        return {"ok": False, "reason": str(e)}
    before = rc.base
    after = pc["frame"].copy()
    after["ecl"] = pc["after"]
    ok = before["ecl"].notna() & after["ecl"].notna()
    s = summarise_repricing(before, after)
    okv = ok.to_numpy()
    tot = lambda e: float(np.sum(np.asarray(e)[okv]))
    b0 = s["before"]
    wf = []
    prev = b0
    for (key, label), e in zip(LAYERS, (pc["after_pd"], pc["after_staging"], pc["after"])):
        t = tot(e)
        wf.append({"layer": label, "key": key, "change": t - prev, "ecl": t})
        prev = t
    det = s["detail"].copy()
    seg = pd.Series(rc.segment, index=rc.ing.index)
    book = pd.Series(np.where(rc.external, "Externally rated", "Internally rated"),
                     index=rc.ing.index)
    det["segment"] = seg.reindex(det.index).to_numpy()
    det["book"] = book.reindex(det.index).to_numpy()

    def roll(col):
        g = det.groupby(col).agg(contracts=("contract", "size"), exposure=("exposure", "sum"),
                                 before=("ecl_before", "sum"), after=("ecl_after", "sum"))
        g["change"] = g["after"] - g["before"]
        g["change_pct"] = 100 * g["change"] / g["before"].where(g["before"] != 0)
        return g.reset_index().sort_values("change", key=abs, ascending=False)

    sm = pc["stage"]
    dp = sm["total"]
    expo = rc.ing["exposure"].to_numpy(float) if "exposure" in rc.ing else \
        before["exposure"].to_numpy(float)
    st_rows = []
    for stg in (1, 2, 3):
        m = (rc.stage0 == stg) & okv
        n_ = float(m.sum())
        ex = float(np.nansum(expo[m]))
        if stg == 1:
            n_ -= float(dp[m].sum())
            ex -= float(np.nansum(dp[m] * expo[m]))
        if stg == 2:
            m1 = (rc.stage0 == 1) & okv
            n_ += float(dp[m1].sum())
            ex += float(np.nansum(dp[m1] * expo[m1]))
        st_rows.append({"stage": stg, "contracts": n_, "exposure": ex})
    by_stage = pd.DataFrame(st_rows)
    years = rc.years
    rows = []
    flags = sim.get("flags") or {}
    for v in rc.system.order:
        p = np.asarray(sim["paths"].get(v, np.zeros(rc.H)), float)
        role = (sim.get("role") or {}).get(v, "off")
        base = rc.system.baseline(v, years)
        for j, y in enumerate(years):
            b = base[j]
            rows.append({"variable": v, "label": rc.system.label(v), "unit": rc.system.unit(v),
                         "role": role, "year": y, "move": float(p[j]),
                         "baseline": b, "scenario": None if b is None else b + float(p[j]),
                         "beyond_data": j in (flags.get(v) or [])})
    pdx = pc["pd12"]
    w = np.where(okv & np.isfinite(rc.pd12) & np.isfinite(pdx) & (rc.stage0 < 3), expo, 0.0)
    out_pd = {"before": float(np.nansum(w * rc.pd12) / w.sum()) if w.sum() else None,
              "after": float(np.nansum(w * pdx) / w.sum()) if w.sum() else None}
    return {
        "ok": True, "name": scn.name, "id": scn.id,
        "before": b0, "after": s["after"], "delta": s["after"] - b0,
        "pct": 100 * (s["after"] - b0) / b0 if b0 else float("nan"),
        "years": years, "severity": severity(rc, scn),
        "paths": pd.DataFrame(rows),
        "recommended": sim.get("recommended", []),
        "waterfall": pd.DataFrame(wf, columns=["layer", "key", "change", "ecl"]),
        "by_book": roll("book"), "by_segment": roll("segment"),
        "by_portfolio": roll("portfolio"), "by_stage": by_stage,
        "by_customer": s["by_customer"],
        "staging": {"eligible": int(sm["eligible"].sum()),
                    "expected_moves": float(dp.sum()),
                    "expected_exposure": float(np.nansum(dp * expo)),
                    "rating_moves": float(sm["rating"].sum()),
                    "restructured": float(scn.restructured),
                    "restructured_part": pc["restructured_part"],
                    "k": sm["k"], "p12": sm["p12"]},
        "collateral": pc["collateral"], "pd12": out_pd,
        "edits": pc["edits"], "notes": list(rc.notes), "detail": det,
    }
