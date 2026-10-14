"""
Stress testing and what-if.

Everything here reprices the book through the SAME engine the run used, by
varying its inputs. Nothing adjusts an answer afterwards, so the EAD waterfall,
the LGD formula and its floor, the Stage 1 horizon cap, the discounting and the
exposure cap all apply unchanged.

The levers, and what each actually changes:

    stage        the horizon (Stage 1 caps at 12 months; Stage 3 books per config)
    rating       which PD curve applies, moved along the contract's OWN scale
    collateral   LGD, through the real formula rather than an override
    exposure     the EAD curve, scaled
    maturity     the term, by re-profiling the curve onto it
    pd_multiplier every cumulative PD curve, capped at 1

A what-if names customers; a stress changes a policy or a parameter across a
portfolio. Both end in the same repricing.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .analytics.profile import classify_stage
from .engine import EclConfig
from .inputs import EngineInputs

__all__ = [
    "StressSpec", "SCOPE_FIELDS", "STANDARD_PACKAGES", "standard_packages",
    "packages_path", "read_packages", "write_packages", "stress_scope",
    "reprice", "apply_stress", "compare_packages",
    "reverse_stress", "reverse_stress_all", "roll_forward", "tornado",
    "TORNADO_LEVERS", "REVERSE_LEVERS", "stretch_curve", "advance_curve",
    "conditional_pd", "staging_threshold", "staging_threshold_sweep",
    "advance_curves", "reprofile_curves", "customer_rows",
    "mev_stress", "MEV_WEIGHT_MODES", "stpd_to_curves",
    "run_scenario_weights", "scenario_weights_sensitivity",
    "filter_rating_type", "order_by_rating", "scenario_ecl_single_run",
]


# ------------------------------------------------------------- curve maths --
def stretch_curve(curve, new_len: int) -> np.ndarray:
    """Re-profile an amortisation curve onto a different term.

    The balance follows the same shape expressed as a fraction of the term and
    still runs down to the same end point at the new maturity. This is what a
    term extension means; it keeps the curve's own structure -- steps, deferral
    periods, expected drawdowns -- rather than replacing it with a straight
    line.
    """
    c = np.asarray(curve, dtype=float)
    n = max(1, int(new_len))
    if c.size == 0:
        return c
    if c.size == 1 or n == c.size:
        return np.resize(c, n)
    pos = np.linspace(0, c.size - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, c.size - 1)
    w = pos - lo
    return c[lo] * (1 - w) + c[hi] * w


def advance_curve(curve, months: int) -> np.ndarray:
    """Drop elapsed months off the FRONT of a curve.

    Rolling forward is not the same as shortening maturity: shortening keeps
    today's balance and squeezes the remaining repayments into less time, which
    is the wrong answer to the question "what will this look like next year".
    """
    c = np.asarray(curve, dtype=float)
    k = int(months)
    if k <= 0 or c.size == 0:
        return c
    return c[-1:] if k >= c.size else c[k:]


def conditional_pd(cum, k: int) -> np.ndarray:
    """Cumulative PD conditional on surviving ``k`` months.

        cumPD(t | survived k) = (cumPD(k+t) - cumPD(k)) / (1 - cumPD(k))

    A roll-forward must use this. Reusing the original curve charges for a
    default already known not to have happened.
    """
    c = np.asarray(cum, dtype=float)
    if k <= 0 or k >= c.size:
        return c
    base = c[k]
    if not np.isfinite(base) or base >= 1:
        return np.zeros(c.size - k)
    return np.clip((c[k:] - base) / (1 - base), 0, 1)


# ------------------------------------------------------------------ specs --
@dataclass
class StressSpec:
    """One stress: who it reaches, and what changes for them.

    Packages, what-ifs on named customers and staging-policy changes are all
    this. The scope is every contract in ``portfolios`` AND ``stages`` AND
    ``customers`` -- an empty list does not narrow it -- and every lever
    reaches the scope only: the rest of the book keeps its own price. ``None``
    for the DPD threshold and the LGD parameters means the run's own, so a
    stress that changes nothing reprices to the run's report. As R's
    stress_default().
    """
    name: str = "Stress"
    # who
    portfolios: list[str] = field(default_factory=list)
    stages: list[int] = field(default_factory=list)
    customers: list[str] = field(default_factory=list)
    # staging
    dpd_threshold: float | None = None
    contagion: bool = True
    tasdeer_collective: bool = True
    watchlist_triggers: bool = True
    local_triggers: bool = True
    stage_to: int | None = None
    default_top_n: int = 0
    # PD and rating
    pd_multiplier: float = 1.0
    pd_scenario: str = ""
    rating_notches: int = 0
    # LGD and collateral
    lgd_base: float | None = None
    lgd_floor: float | None = None
    collateral_pct: float = 100.0
    # exposure and term
    exposure_pct: float = 100.0
    maturity_years: float = 0.0

    def is_empty(self) -> bool:
        d = StressSpec()
        return all(getattr(self, k) == getattr(d, k)
                   for k in asdict(d) if k not in ("name",) + SCOPE_FIELDS)

    @classmethod
    def from_dict(cls, d: dict) -> "StressSpec":
        """A spec from a saved package or a request: unknown keys ignored,
        blanks back to their defaults, numbers and lists coerced."""
        d = dict(d or {})
        out = {}
        for f in cls.__dataclass_fields__.values():
            if f.name not in d:
                continue
            v = d[f.name]
            if f.name in SCOPE_FIELDS:
                if v is None or v == "":
                    v = []
                elif isinstance(v, str):
                    v = [t for t in v.replace(";", ",").split(",") if t.strip()]
                elif not isinstance(v, (list, tuple)):
                    v = [v]     # one value: R's YAML writes a vector of one so
                v = [str(x).strip() for x in v] if f.name != "stages" \
                    else [int(float(x)) for x in v]
            elif f.name in ("dpd_threshold", "lgd_base", "lgd_floor", "stage_to"):
                v = None if v is None or v == "" or (isinstance(v, float) and np.isnan(v)) \
                    else (int(float(v)) if f.name == "stage_to" else float(v))
                if f.name == "stage_to" and v == 0:
                    v = None
            elif f.name in ("default_top_n", "rating_notches"):
                v = int(float(v or 0))
            elif f.name in ("pd_multiplier", "collateral_pct", "exposure_pct",
                            "maturity_years"):
                v = float(v)
            elif f.name in ("contagion", "tasdeer_collective",
                            "watchlist_triggers", "local_triggers"):
                v = bool(v)
            elif f.name in ("name", "pd_scenario"):
                v = "" if v is None else str(v)
            out[f.name] = v
        return cls(**out)

    def to_dict(self) -> dict:
        return asdict(self)


SCOPE_FIELDS = ("portfolios", "stages", "customers")

# The standard packages: each a single coherent story rather than an arbitrary
# bundle, and the same in both engines (R's standard_packages()).
STANDARD_PACKAGES = [
    {"name": "Mild downturn", "pd_multiplier": 1.25, "rating_notches": 1,
     "collateral_pct": 90.0},
    {"name": "Severe downturn", "pd_multiplier": 2.0, "rating_notches": 3,
     "collateral_pct": 70.0, "exposure_pct": 110.0},
    {"name": "Collateral revaluation", "collateral_pct": 50.0},
    {"name": "Staging tightened to 30 days", "dpd_threshold": 30.0},
    {"name": "Largest five default", "default_top_n": 5},
]


def standard_packages(portfolios=None) -> list[StressSpec]:
    """The standard packages, scoped to ``portfolios`` (all if None)."""
    return [StressSpec.from_dict({**p, "portfolios": list(portfolios or [])})
            for p in STANDARD_PACKAGES]


def packages_path(project_root) -> Path:
    """Where saved packages live: the project's config/stress_packages.yml,
    the file R's stress_read() and stress_write() use."""
    return Path(project_root) / "config" / "stress_packages.yml"


def read_packages(path) -> list[StressSpec]:
    """Saved packages, or none. R writes and reads the same file."""
    import yaml
    p = Path(path)
    if not p.is_file():
        return []
    try:
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return [StressSpec.from_dict(d) for d in (y.get("packages") or [])
            if isinstance(d, dict)]


def write_packages(specs, path) -> Path:
    """Save packages for next quarter, in R's format (packages: [...]); a
    None (the run's own) is left out, as R leaves a NULL out."""
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for sp in specs:
        d = (sp if isinstance(sp, StressSpec) else StressSpec.from_dict(sp)).to_dict()
        rows.append({k: v for k, v in d.items() if v is not None})
    p.write_text(yaml.safe_dump({"packages": rows}, sort_keys=False,
                                allow_unicode=True), encoding="utf-8")
    return p


def stress_scope(ing: pd.DataFrame, spec: StressSpec) -> pd.Series:
    """The contracts a stress reaches: in its portfolios AND stages AND
    customers, an empty list not narrowing it."""
    keep = pd.Series(True, index=ing.index)
    if spec.portfolios:
        keep &= ing["portfolio"].isin(list(spec.portfolios))
    if spec.stages:
        keep &= pd.to_numeric(ing["stage"], errors="coerce").isin(
            [int(x) for x in spec.stages])
    if spec.customers:
        keep &= ing["customer"].astype(str).str.strip().isin(
            [str(c).strip() for c in spec.customers])
    return keep.fillna(False)


# --------------------------------------------------------------- repricing --
# Every repricing here -- a stress package, a what-if, a roll-forward, a macro
# path, a scenario -- prices on the run's own pricing context and contract
# facts with etl.report.compute_ecl(), the function that priced the run's
# report. A lever changes compute_ecl()'s inputs or its context, never the
# answer, so the EAD waterfall, the LGD formula, the staging horizons, the
# discounting, the Stage 3 method and the exposure cap all still apply, and
# an unchanged repricing IS the run's report. As the R engine.

_REPORT_COLS = ("contract", "customer", "stage", "exposure", "ecl", "collcov",
                "dpd", "watchlist", "restructured", "local2", "local3",
                "local4", "local5", "local6", "default_flag", "insolvency")


def add_months(dates, k) -> pd.Series:
    """Calendar months added to each date, the day clamped to the month's
    last -- the 31st plus one month is the 30th or the 28th. As R's
    add_months(), so a maturity or extract date moves the same way in both."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    k = np.broadcast_to(np.asarray(k, dtype=float), (len(d),))
    out = []
    for v, m in zip(d, k):
        if pd.isna(v) or not np.isfinite(m) or m == 0:
            out.append(v)
            continue
        t = v.year * 12 + (v.month - 1) + int(round(m))
        y, mo = divmod(t, 12)
        last = pd.Timestamp(year=y, month=mo + 1, day=1).days_in_month
        out.append(v.replace(year=y, month=mo + 1, day=min(v.day, last)))
    return pd.Series(out, index=pd.Series(dates).index if hasattr(dates, "index") else None)


def _ingredients(inputs: EngineInputs, report: pd.DataFrame) -> pd.DataFrame:
    """The report's contracts with the facts the run priced them on.

    One row per report row, in the report's order: the report's own stage,
    customer, exposure and staging flags (what the run booked), joined to
    the pipeline's contract facts (``etl.report.ecl_frame``) position by
    position -- a contract id is not unique (an investment held in two
    positions), so the n-th row of a contract in the report is the n-th in
    the frame. Joining on the id alone squares a repeated one: two rows
    against two is four, and the security is priced twice.
    """
    rep = report.reset_index(drop=True)
    cid = rep["contract"].astype(str)
    key = cid + "#" + cid.groupby(cid).cumcount().astype(str)
    left = rep[[c for c in _REPORT_COLS if c in rep.columns]].copy()
    left["key"] = key.to_numpy()
    facts = inputs.frame.drop(columns=[c for c in ("contract", "customer", "stage",
                                                   "dpd") if c in inputs.frame.columns])
    facts = facts.drop_duplicates("key")
    m = left.merge(facts, on="key", how="left")
    m.index = rep.index
    m["payment_frequency"] = m["pay_freq"]
    m["months_to_mat"] = m["months"]
    m["on_balance"] = pd.to_numeric(m["on_balance"], errors="coerce")
    m["collateral_net"] = [inputs.ctx.collnet.get(c, 0.0) if isinstance(c, str) else 0.0
                           for c in m["cid"]]
    return m


def _ctx_curves(curves) -> dict:
    """PD curves in the context's form, from an StPD table or a
    ``portfolio|bucket`` dict."""
    from .etl.report import _KEYSEP, EclContext
    if isinstance(curves, pd.DataFrame):
        return EclContext._stpd(curves)
    return {(k.replace("|", _KEYSEP, 1) if _KEYSEP not in k else k):
            np.asarray(v, dtype=float) for k, v in curves.items()}


def reprice(
    inputs: EngineInputs,
    report: pd.DataFrame,
    *,
    stage: pd.Series | None = None,
    rating: pd.Series | None = None,
    on_balance: pd.Series | None = None,
    collateral_net: pd.Series | None = None,
    maturity_shift_months: pd.Series | None = None,
    pd_curves=None,
    pd_multiplier: float = 1.0,
    cfg: EclConfig | None = None,
) -> pd.DataFrame:
    """Price every contract with ``etl.report.compute_ecl``, with any input
    substituted -- one value per report row for each:

        stage, rating          the contract facts
        on_balance             the balance, and a supplied EAD schedule
                               scaled with it (exposure)
        collateral_net         the context's net collateral, so LGD moves
                               through compute_lgd()
        maturity_shift_months  the maturity date, and a supplied schedule
                               re-profiled onto the new term
        pd_curves              another set of PD curves (an StPD table or a
                               ``portfolio|bucket`` dict): a scenario's, a
                               rebuilt chain's
        pd_multiplier          every PD curve scaled, capped at 1
        cfg                    LGD base and unsecured floor

    The run's own Stage 3 method, exposure cap and EAD fallback rules come
    with its context. Returns one row per report row with the ECL and the
    LGD the engine used.
    """
    from .etl.report import compute_ecl
    ing = _ingredients(inputs, report)
    n = len(ing)
    ctx = inputs.ctx
    frame = ing.copy()
    cid = frame["cid"]

    def vec(v):
        return pd.Series(list(v), index=frame.index)

    if stage is not None:
        frame["stage"] = vec(stage)
    frame["stage"] = pd.to_numeric(frame["stage"], errors="coerce").fillna(2).astype(int)
    if rating is not None:
        frame["rating"] = vec(rating).astype(str)
    changes = {}
    if on_balance is not None:
        new = pd.to_numeric(vec(on_balance), errors="coerce")
        old = frame["on_balance"]
        f = new / old
        moved = old.notna() & (old != 0) & new.notna() & (new != old) & cid.notna()
        if moved.any():
            sch = dict(ctx.schedules)
            for c, k in zip(cid[moved], f[moved]):
                if c in sch:
                    sch[c] = np.asarray(sch[c], dtype=float) * float(k)
            changes["schedules"] = sch
        frame["on_balance"] = new
    if collateral_net is not None:
        cn = dict(ctx.collnet)
        for c, v in zip(cid, pd.to_numeric(vec(collateral_net), errors="coerce").fillna(0.0)):
            if isinstance(c, str):
                cn[c] = float(v)
        changes["collnet"] = cn
    if maturity_shift_months is not None:
        sh = pd.to_numeric(vec(maturity_shift_months), errors="coerce").fillna(0.0).round()
        if (sh != 0).any():
            frame["d_maturity"] = add_months(frame["d_maturity"], sh)
            sch = changes.get("schedules", dict(ctx.schedules))
            for k in sorted(set(sh[sh != 0])):
                sch = reprofile_curves(sch, [c for c in cid[sh == k] if isinstance(c, str)], k)
            changes["schedules"] = sch
    if pd_curves is not None:
        changes["stpd"] = _ctx_curves(pd_curves)
    if pd_multiplier != 1.0:
        changes["stpd"] = {k: np.clip(np.asarray(v, float) * pd_multiplier, 0, 1)
                           for k, v in changes.get("stpd", ctx.stpd).items()}
    if cfg is not None:
        changes["lgd_base"] = cfg.lgd_base
        changes["lgd_floor"] = cfg.lgd_unsecured_floor
    if changes:
        ctx = ctx.copy(**changes)

    ecl = np.full(n, np.nan)
    lgd = np.full(n, np.nan)
    have = cid.notna().to_numpy()
    if have.any():
        res = compute_ecl(ctx, frame[have])
        ecl[have] = res["ecl"].to_numpy()
        lgd[have] = res["lgd"].to_numpy()

    out = ing[["contract", "customer", "portfolio", "rating", "stage",
               "exposure"]].copy()
    out["stage_after"] = frame["stage"].to_numpy()
    out["rating_after"] = frame["rating"].to_numpy()
    out["ecl"] = ecl
    out["lgd"] = lgd
    return out


# ------------------------------------------------------------ apply stress --
def apply_stress(inputs: EngineInputs, report: pd.DataFrame,
                 spec, cfg: EclConfig | None = None) -> dict:
    """Apply one stress and reprice -- a package, a what-if on named
    customers or a staging-policy change; there is no other.

    Every lever reaches the scope only (``stress_scope``), in this order:

      staging     the policy re-stages the scope, when it differs from the
                  run's own -- a stress that leaves it alone keeps the
                  reported stages, manual overrides included; then
                  ``stage_to`` moves the scope to one stage, and the scope's
                  ``default_top_n`` largest customers by exposure go to
                  Stage 3
      rating      notched along each contract's own scale
      exposure    the balance, and a supplied EAD schedule with it
      collateral  the net collateral, so LGD moves through compute_lgd()
      maturity    the maturity date, in calendar months, and a supplied
                  schedule re-profiled onto the new term
      PD, LGD     the scope is priced on its scenario's curves when one is
                  named, times the multiplier, and on the LGD base and floor
                  given; the rest of the book on the run's own

    Both sides are priced with ``etl.report.compute_ecl`` (``reprice``), so
    an empty stress is the run's report. As R's stress_apply().
    """
    if not inputs.ok:
        return {"ok": False, "reason": f"Engine inputs missing: {', '.join(inputs.missing)}"}
    if not isinstance(spec, StressSpec):
        spec = StressSpec.from_dict(spec)
    ing = _ingredients(inputs, report)
    if len(ing) == 0:
        return {"ok": False, "reason": "The report could not be matched to the inputs."}

    scoped = stress_scope(ing, spec)
    sc = scoped.to_numpy()
    if not sc.any():
        return {"ok": False, "reason": "No contract is in this stress's scope."}
    before = reprice(inputs, report, cfg=cfg)
    notes: list[str] = []

    # staging policy: re-staged only when it is not the run's own
    run_thr = staging_threshold(inputs.out_dir)
    thr = run_thr if spec.dpd_threshold is None else float(spec.dpd_threshold)
    stage = ing["stage"].fillna(2).astype(int).to_numpy().copy()
    if (thr != run_thr or not spec.contagion or not spec.tasdeer_collective
            or not spec.watchlist_triggers or not spec.local_triggers):
        local_any = pd.Series(False, index=ing.index)
        if spec.local_triggers:
            for c in ("restructured", "local2", "local3", "local4", "local5", "local6"):
                if c in ing.columns:
                    local_any |= (ing[c] == 1).fillna(False)
        watch = ing.get("watchlist") if spec.watchlist_triggers else None
        new_stage = classify_stage(
            ing.get("dpd"), ing.get("default_flag"), watch, local_any,
            ing["portfolio"], ing["customer"], dpd_threshold=thr,
            contagion=spec.contagion, tasdeer_collective=spec.tasdeer_collective)
        stage[sc] = np.asarray(new_stage)[sc]
        notes.append(f"Re-staged on DPD > {thr:g}"
                     + ("" if spec.contagion else ", no contagion")
                     + ("" if spec.tasdeer_collective else ", Tasdeer not collective")
                     + ("" if spec.watchlist_triggers else ", watchlist not a trigger")
                     + ("" if spec.local_triggers else ", local flags not triggers"))
    if spec.stage_to:
        stage[sc] = int(spec.stage_to)
        notes.append(f"Moved to Stage {int(spec.stage_to)}")

    # largest N forced to default: rank by EXPOSURE, and keep every one picked.
    # Ranking by ECL puts customers already in Stage 3 at the top -- they carry
    # a 100% provision by construction -- so the lever quietly delivered fewer
    # defaults than asked for.
    defaulted: list[str] = []
    if spec.default_top_n and spec.default_top_n > 0:
        elig = ing[scoped & (ing["exposure"] > 0)]
        rank = elig.groupby("customer")["exposure"].sum().sort_values(ascending=False)
        defaulted = [str(c) for c in rank.head(spec.default_top_n).index]
        hit = scoped & ing["customer"].astype(str).isin(defaulted)
        stage[hit.to_numpy()] = 3
        notes.append(f"{len(defaulted)} largest customers by exposure forced to Stage 3")

    rating = ing["rating"].astype(str).to_numpy().copy()
    if spec.rating_notches:
        for i in np.where(sc)[0]:
            scale = inputs.scale_for(ing.iloc[i].get("rating_type"))
            if scale:
                rating[i] = scale.notch(rating[i], spec.rating_notches)

    on_balance = ing["on_balance"].to_numpy().copy()
    if spec.exposure_pct != 100:
        on_balance[sc] *= spec.exposure_pct / 100

    cnet = ing["collateral_net"].fillna(0.0).to_numpy().copy()
    if spec.collateral_pct != 100:
        cnet[sc] *= spec.collateral_pct / 100

    shift = np.zeros(len(ing))
    if spec.maturity_years:
        shift[sc] = spec.maturity_years * 12

    curves = None
    if spec.pd_scenario:
        from .analytics.model_view import read_scenario_stpd
        curves = read_scenario_stpd(inputs.out_dir, spec.pd_scenario)
        if not curves:
            return {"ok": False, "reason": "This run wrote no PD curves for the "
                                           f"scenario {spec.pd_scenario!r}."}
        notes.append(f"Priced on the {spec.pd_scenario} PD curves")
    if spec.pd_multiplier != 1.0:
        notes.append(f"PD curves scaled by {spec.pd_multiplier:g}x (capped at 1)")
    lgd_cfg = cfg
    if spec.lgd_base is not None or spec.lgd_floor is not None:
        lgd_cfg = EclConfig(
            lgd_base=inputs.ctx.lgd_base if spec.lgd_base is None else spec.lgd_base,
            lgd_unsecured_floor=(inputs.ctx.lgd_floor if spec.lgd_floor is None
                                 else spec.lgd_floor))

    after = reprice(
        inputs, report, stage=stage, rating=rating, on_balance=on_balance,
        collateral_net=cnet, maturity_shift_months=shift, pd_curves=curves,
        pd_multiplier=spec.pd_multiplier, cfg=lgd_cfg)
    # The PD curves and the LGD parameters are the context's, so they reach
    # every contract priced on it: outside the scope the run's own price
    # stands. Everything else outside the scope is unchanged already.
    after.loc[~scoped, ["ecl", "lgd"]] = before.loc[~scoped, ["ecl", "lgd"]].to_numpy()

    ok = before["ecl"].notna() & after["ecl"].notna()
    d = before[["contract", "customer", "portfolio", "rating", "stage"]].copy()
    d["ecl_before"] = before["ecl"]
    d["ecl_after"] = after["ecl"]
    d["stage_after"] = after["stage_after"]
    d["rating_after"] = after["rating_after"]
    d["exposure"] = before["exposure"]
    d["change"] = d["ecl_after"] - d["ecl_before"]
    priced = d[ok]

    by_cust = (priced.groupby("customer")
               .agg(facilities=("contract", "size"),
                    exposure=("exposure", "sum"),
                    ecl_before=("ecl_before", "sum"),
                    ecl_after=("ecl_after", "sum"),
                    stage=("stage", "max"), stage_after=("stage_after", "max"))
               .reset_index())
    by_cust["change"] = by_cust["ecl_after"] - by_cust["ecl_before"]
    by_cust = by_cust.reindex(by_cust["change"].abs()
                              .sort_values(ascending=False).index)

    by_pf = (priced.groupby("portfolio")
             .agg(contracts=("contract", "size"),
                  before=("ecl_before", "sum"), after=("ecl_after", "sum"))
             .reset_index())
    by_pf["change"] = by_pf["after"] - by_pf["before"]

    moved = priced["stage"] != priced["stage_after"]
    mv = priced[moved]
    migration = pd.DataFrame()
    if len(mv):
        migration = (mv.assign(move=[f"Stage {int(x)} \u2192 Stage {int(y)}"
                                     for x, y in zip(mv["stage"], mv["stage_after"])])
                     .groupby("move")
                     .agg(contracts=("contract", "size"),
                          customers=("customer", "nunique"),
                          exposure=("exposure", "sum"),
                          ecl_change=("change", "sum"))
                     .reset_index())
        migration = migration.reindex(migration["ecl_change"].abs()
                                      .sort_values(ascending=False).index)
    return {
        "ok": True,
        "name": spec.name,
        "before": float(priced["ecl_before"].sum()),
        "after": float(priced["ecl_after"].sum()),
        "delta": float(priced["change"].sum()),
        "priced": int(ok.sum()),
        "scoped": int(sc.sum()),
        "moved": int(moved.sum()),
        "customers_moved": int(priced.loc[moved, "customer"].nunique()),
        "defaulted": defaulted,
        "notes": notes,
        "by_customer": by_cust,
        "by_portfolio": by_pf.sort_values("change", key=abs, ascending=False),
        "migration": migration,
        "detail": priced.reindex(priced["change"].abs()
                                 .sort_values(ascending=False).index),
    }


def compare_packages(inputs, report, specs: Iterable[StressSpec]) -> pd.DataFrame:
    """Several stresses priced on identical inputs, worst first -- the
    question a committee puts is which of these hurts most."""
    rows = []
    for s in specs:
        r = apply_stress(inputs, report, s)
        if not r.get("ok"):
            continue
        rows.append({"name": r["name"], "before": r["before"], "after": r["after"],
                     "delta": r["delta"],
                     "pct": 100 * r["delta"] / max(r["before"], 1),
                     "moved": r["moved"]})
    return pd.DataFrame(rows).sort_values("delta", ascending=False)


# --------------------------------------------------------- reverse stress --
REVERSE_LEVERS = {
    "pd_multiplier": {"label": "PD multiplier", "lo": 1.0, "hi": 10.0, "unit": "x"},
    "rating_notches": {"label": "Rating downgrade", "lo": 0, "hi": 20,
                       "unit": " notches", "integer": True},
    "collateral_pct": {"label": "Collateral value", "lo": 100.0, "hi": 0.0, "unit": "%"},
    "exposure_pct": {"label": "Exposure", "lo": 100.0, "hi": 300.0, "unit": "%"},
    "lgd_base": {"label": "LGD base", "lo": 0.45, "hi": 1.0, "unit": ""},
    "default_top_n": {"label": "Largest customers defaulting", "lo": 0, "hi": 100,
                      "unit": " customers", "integer": True},
}


def reverse_stress(inputs, report, lever: str, target_pct: float,
                   base: StressSpec | None = None, tol: float = 0.25,
                   max_iter: int = 16) -> dict:
    """Solve for the level of one lever that reaches a target provision.

    Every lever is monotonic in the provision, so bisection settles it in a
    dozen or so repricings rather than a grid search. A lever that cannot reach
    the target says so and reports the most it achieves, instead of returning
    its boundary as if it were an answer.
    """
    L = REVERSE_LEVERS.get(lever)
    if L is None:
        return {"ok": False, "reason": f"Unknown lever {lever!r}"}
    spec0 = base or StressSpec(name="reverse",
                               portfolios=inputs.internal_portfolios())

    def at(value):
        s = StressSpec(**{**asdict(spec0), lever: value})
        return apply_stress(inputs, report, s)

    b = at(L["lo"])
    if not b.get("ok"):
        return {"ok": False, "reason": b.get("reason", "Could not price the base.")}
    base_prov = b["before"]
    if base_prov <= 0:
        return {"ok": False, "reason": "Base provision is zero."}

    hi = at(L["hi"])
    hi_pct = 100 * (hi["after"] - base_prov) / base_prov
    if hi_pct < target_pct:
        return {"ok": True, "found": False, "lever": L["label"],
                "value": L["hi"], "unit": L["unit"], "achieved_pct": hi_pct,
                "provision": hi["after"], "base": base_prov}

    lo, up = L["lo"], L["hi"]
    for _ in range(max_iter):
        mid = (lo + up) / 2
        if L.get("integer"):
            mid = round(mid)
        r = at(mid)
        p = 100 * (r["after"] - base_prov) / base_prov
        if p >= target_pct:
            up = mid
        else:
            lo = mid
        if abs(p - target_pct) <= tol:
            break
        if L.get("integer") and abs(up - lo) <= 1:
            break

    value = round(up) if L.get("integer") else up
    final = at(value)
    return {"ok": True, "found": True, "lever": L["label"], "value": value,
            "unit": L["unit"],
            "achieved_pct": 100 * (final["after"] - base_prov) / base_prov,
            "provision": final["after"], "base": base_prov}


def reverse_stress_all(inputs, report, target_pct: float,
                       base: StressSpec | None = None) -> pd.DataFrame:
    rows = []
    for k in REVERSE_LEVERS:
        r = reverse_stress(inputs, report, k, target_pct, base)
        if not r.get("ok"):
            continue
        rows.append({
            "lever": r["lever"],
            "required": (f"{r['value']:g}{r['unit']}" if r["found"]
                         else f"beyond {REVERSE_LEVERS[k]['hi']:g}{r['unit']}"),
            "found": r["found"], "achieved_pct": r["achieved_pct"],
            "provision": r["provision"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["found", "achieved_pct"], ascending=[False, True])


# ------------------------------------------------------------ roll forward --
def roll_forward(inputs, report, months: int = 12,
                 portfolios: list[str] | None = None) -> dict:
    """The provision in ``months`` months if nothing else changes.

    The book in scope is moved forward ``k`` months and priced with
    ``etl.report.compute_ecl`` on the run's own context, as R's
    roll_forward():

      EAD    a contract's balance is what its own EAD path reaches at month
             k -- its supplied schedule, or its fallback over its full term,
             never the twelve months a Stage 1 provision looks at -- and a
             supplied schedule is advanced (elapsed months dropped);
      PD     conditional on surviving those months, not the original curve,
             which would charge for a default known not to have happened;
      term   the extract date moves k months, so a fallback curve is rebuilt
             over what remains, and any deferral shortens by k; a contract
             whose path ends inside the window has run off.

    Staging, ratings and collateral are held. Contracts out of scope keep
    their provision.
    """
    from .etl.report import compute_ecl, ead_path
    if not inputs.ok:
        return {"ok": False, "reason": f"Engine inputs missing: {', '.join(inputs.missing)}"}
    k = int(round(months))
    if k <= 0:
        return {"ok": False, "reason": "Roll forward by at least one month."}
    before = reprice(inputs, report)
    ing = _ingredients(inputs, report)
    ctx = inputs.ctx
    scoped = (ing["portfolio"].isin(portfolios) if portfolios
              else pd.Series(True, index=ing.index)) & ing["cid"].notna()

    on_balance = ing["on_balance"].to_numpy(dtype=float).copy()
    matured = np.zeros(len(ing), dtype=bool)
    for i in np.where(scoped.to_numpy())[0]:
        path = ead_path(ctx, ing.iloc[i])
        if path is None or k >= len(path):
            matured[i] = True
            on_balance[i] = 0.0
        else:
            on_balance[i] = float(path[k])

    live = scoped.to_numpy() & ~matured
    a = before["ecl"].to_numpy(dtype=float).copy()
    a[matured] = 0.0
    if live.any():
        frame = ing[live].copy()
        frame["stage"] = pd.to_numeric(frame["stage"], errors="coerce").fillna(2).astype(int)
        frame["on_balance"] = on_balance[live]
        frame["d_extract"] = add_months(frame["d_extract"], k)
        frame["deferral"] = (pd.to_numeric(frame["deferral"], errors="coerce")
                             .fillna(0) - k).clip(lower=0)
        adv = ctx.copy(
            schedules=advance_curves(ctx.schedules, list(frame["cid"]), k),
            stpd={key: conditional_pd(v, k) for key, v in ctx.stpd.items()})
        a[live] = compute_ecl(adv, frame)["ecl"].to_numpy()

    b = before["ecl"].to_numpy(dtype=float)
    ok = np.isfinite(b) & np.isfinite(a)
    return {
        "ok": True, "months": k,
        "before": float(b[ok].sum()),
        "after": float(a[ok].sum()),
        "delta": float(a[ok].sum() - b[ok].sum()),
        "matured": int(matured.sum()),
        "matured_exposure": float(np.nansum(
            ing["on_balance"].to_numpy(dtype=float)[matured])),
        "exposure_before": float(ing["on_balance"].to_numpy()[ok].sum()),
        "exposure_after": float(on_balance[ok].sum()),
    }


# ----------------------------------------------------------------- tornado --
TORNADO_LEVERS = [
    {"key": "pd_multiplier", "label": "PD +25%", "value": 1.25},
    {"key": "rating_notches", "label": "Downgrade 1 notch", "value": 1},
    {"key": "rating_notches", "label": "Downgrade 2 notches", "value": 2},
    {"key": "collateral_pct", "label": "Collateral -25%", "value": 75},
    {"key": "collateral_pct", "label": "Collateral -50%", "value": 50},
    {"key": "exposure_pct", "label": "Exposure +20%", "value": 120},
    {"key": "lgd_base", "label": "LGD base 0.45 to 0.55", "value": 0.55},
    {"key": "lgd_floor", "label": "LGD floor 0.5 to 0.7", "value": 0.7},
    {"key": "dpd_threshold", "label": "DPD threshold 60 to 30", "value": 30},
    {"key": "default_top_n", "label": "Largest 5 default", "value": 5},
    {"key": "contagion", "label": "Contagion off", "value": False},
    {"key": "tasdeer_collective", "label": "Tasdeer not collective", "value": False},
]


def tornado(inputs, report, base: StressSpec | None = None,
            levers: list[dict] | None = None) -> pd.DataFrame:
    """Move each lever on its own and rank by effect.

    The bars are comparable but do NOT add up: combining levers has
    interactions, which is what a stress package is for.
    """
    spec0 = base or StressSpec(name="tornado",
                               portfolios=inputs.internal_portfolios())
    b = apply_stress(inputs, report, spec0)
    if not b.get("ok"):
        return pd.DataFrame()
    base_prov = b["before"]
    rows = []
    for L in (levers or TORNADO_LEVERS):
        s = StressSpec(**{**asdict(spec0), L["key"]: L["value"]})
        r = apply_stress(inputs, report, s)
        if not r.get("ok"):
            continue
        rows.append({"lever": L["label"], "provision": r["after"],
                     "change": r["after"] - base_prov,
                     "pct": 100 * (r["after"] - base_prov) / max(base_prov, 1),
                     "moved": r["moved"]})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.reindex(out["change"].abs().sort_values(ascending=False).index)
    out.attrs["base"] = base_prov
    return out


# ------------------------------------------------------ staging policy -----
def staging_threshold(out_dir, default: float = 60) -> float:
    """The DPD threshold the run actually used.

    A run freezes its staging policy with its config, in
    ``config_used/static/staging_thresholds.csv`` beside ``Output/``; a copy
    beside the outputs is read when there is no frozen config. Sweeping around
    a hardcoded 60 when the run used something else compares the book against
    a policy nobody applied.
    """
    if out_dir is None:
        return default
    from .analytics.model_view import config_used
    p = Path(out_dir)
    cu = config_used(p)
    cands = ([cu["static"] / "staging_thresholds.csv"] if cu else []) + [
        p / "staging_thresholds.csv", p / "Output" / "staging_thresholds.csv"]
    f = next((c for c in cands if c.is_file()), None)
    if f is None:
        return default
    try:
        d = pd.read_csv(f)
    except Exception:
        return default
    if not {"key", "value"} <= set(d.columns):
        return default
    hit = d.loc[d["key"] == "dpd_stage2_threshold_days", "value"]
    if len(hit) == 0:
        return default
    v = pd.to_numeric(hit.iloc[0], errors="coerce")
    return default if pd.isna(v) else float(v)


def staging_threshold_sweep(inputs, report,
                            thresholds=(0, 15, 30, 45, 60, 75, 90),
                            base: StressSpec | None = None,
                            reference: float | None = None,
                            cfg: EclConfig | None = None) -> pd.DataFrame:
    """Reprice the book at each candidate Stage 2 DPD threshold.

    The curve this draws is rarely straight: most of the book is staged by
    watchlist, restructuring and contagion rather than by days past due, so
    lowering the threshold moves far less than people expect. Showing that is
    the point -- it is the difference between a policy debate and an argument
    about a number nobody has tested.

    Each row is ``apply_stress`` with only the threshold changed from
    ``base`` (the internally rated portfolios by default), and the run's own
    threshold is the reference, always priced.
    """
    spec0 = base or StressSpec(portfolios=inputs.internal_portfolios())
    if reference is None:
        reference = staging_threshold(inputs.out_dir)
    rows = []
    for t in sorted({float(x) for x in thresholds} | {float(reference)}):
        spec = replace(spec0, dpd_threshold=float(t), name=f"DPD > {t:g}")
        r = apply_stress(inputs, report, spec, cfg)
        if not r.get("ok"):
            continue
        rows.append({"threshold": float(t), "ecl": r["after"],
                     "moved": r["moved"], "customers_moved": r["customers_moved"]})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    ref = out.loc[out["threshold"] == float(reference), "ecl"]
    if len(ref) == 1:
        b = float(ref.iloc[0])
        out[f"vs_{reference:g}"] = out["ecl"] - b
        out[f"vs_{reference:g}_pct"] = 100 * (out["ecl"] - b) / max(abs(b), 1.0)
    out.attrs["reference"] = float(reference)
    return out


# ---------------------------------------------------- curve bulk edits -----
def advance_curves(curves: dict, contracts, months: int) -> dict:
    """Roll a set of EAD curves forward by ``months``.

    A new dict: the caller's curves are the run's own and repricing must not
    mutate them. That bug is silent -- the first stress looks right and every
    one after it compounds.
    """
    if not curves or months is None or months <= 0:
        return curves
    out = dict(curves)
    for cid in contracts:
        cur = out.get(str(cid))
        if cur is None or len(cur) == 0:
            continue
        out[str(cid)] = advance_curve(cur, int(round(months)))
    return out


def reprofile_curves(curves: dict, contracts, shift_months: float) -> dict:
    """Stretch or compress a set of EAD curves by ``shift_months``.

    A term extension, expressed on the curve rather than on the horizon: the
    balance keeps its own shape -- deferral periods, steps, expected drawdowns
    -- and simply runs off over a different number of months.
    """
    if not curves or not shift_months:
        return curves
    out = dict(curves)
    for cid in contracts:
        cur = out.get(str(cid))
        if cur is None or len(cur) == 0:
            continue
        out[str(cid)] = stretch_curve(cur, max(1, len(cur) + int(round(shift_months))))
    return out


def customer_rows(rows: pd.DataFrame, n: int | None = 100) -> pd.DataFrame:
    """Roll a repricing detail up to one row per customer.

    Attributes that vary within a customer collapse to ``"multiple"`` rather
    than to the first value: a customer with facilities in two portfolios is
    not in the first one, and picking it would put the whole relationship under
    the wrong heading. Stage takes the WORST facility, which is how staging
    works.

    Largest absolute move first, capped at ``n`` -- pass None for all of them.
    """
    if rows is None or len(rows) == 0:
        return pd.DataFrame()

    def one(s: pd.Series) -> str:
        u = s.dropna().unique()
        return str(u[0]) if len(u) == 1 else "multiple"

    agg = {"facilities": ("contract", "size"), "exposure": ("exposure", "sum"),
           "ecl_before": ("ecl_before", "sum"), "ecl_after": ("ecl_after", "sum")}
    for col, how in (("portfolio", one), ("rating", one),
                     ("rating_after", one), ("stage", "max"),
                     ("stage_after", "max")):
        if col in rows.columns:
            agg[col] = (col, how)

    out = rows.groupby("customer").agg(**agg).reset_index()
    out["change"] = out["ecl_after"] - out["ecl_before"]
    e = out["exposure"].where(out["exposure"] > 0)
    out["coverage_before"] = 100 * out["ecl_before"] / e
    out["coverage_after"] = 100 * out["ecl_after"] / e
    out = out.reindex(out["change"].abs().sort_values(ascending=False).index)
    out = out.reset_index(drop=True)
    return out if n is None else out.head(int(n))


# ----------------------------------------------------------- macro path ----
MEV_WEIGHT_MODES = ("auto", "hold", "custom")


def _same_pd_curves(x: dict, y: dict, tol: float = 1e-9) -> bool:
    """Two sets of PD curves are the same calculation (see the ECL bridge)."""
    if x.keys() != y.keys():
        return False
    return all(x[k].shape == y[k].shape
               and np.allclose(x[k], y[k], rtol=0.0, atol=tol) for k in x)


def stpd_to_curves(stpd: pd.DataFrame) -> dict[str, np.ndarray]:
    """An StPD table as the zero-prepended curves the engine prices with."""
    key = ("PortfolioCode", "PDBucketDim1", "MonthLifetime", "PDLifetime")
    if stpd is None or not set(key) <= set(stpd.columns):
        return {}
    t = pd.DataFrame({
        "pf": stpd["PortfolioCode"].astype(str),
        "bk": pd.to_numeric(stpd["PDBucketDim1"], errors="coerce"),
        "m": pd.to_numeric(stpd["MonthLifetime"], errors="coerce"),
        "v": pd.to_numeric(stpd["PDLifetime"], errors="coerce"),
    }).dropna()
    return {f"{p}|{int(b)}": np.concatenate([[0.0], g.sort_values("m")["v"].to_numpy()])
            for (p, b), g in t.groupby(["pf", "bk"])}


def mev_stress(inputs: EngineInputs, report: pd.DataFrame, run_path,
               mev_new: pd.DataFrame | None = None,
               shock: dict | None = None,
               weight_mode: str = "auto",
               weights: dict | None = None,
               cfg: EclConfig | None = None,
               external_weights: dict | None = None) -> dict:
    """Reprice the book on a different macroeconomic path.

    The whole PD chain is rebuilt from the run's OWN frozen config with the
    forecast edited, so the shift factors, the scenario curves and the monthly
    StPD all follow from the new path exactly as they would in a real run.
    Nothing is adjusted afterwards.

    ``mev_new`` sets individual cells -- a frame of ``year``, ``idx``, ``value``
    -- and ``shock`` adds a delta to one MEV across every year, keyed by its
    1-based index. Both may be given; the cells are set first.

    ``weight_mode`` decides what happens to the scenario weights, and it is a
    genuine choice rather than a detail. The internal weights run on
    ``auto_non_oil_gdp_cdf``, derived from the first forecast years of this
    very matrix, so editing non-oil GDP moves the weights AND the curves:

        auto    leave the config alone; the weights follow the new path
        hold    pin the internal weights to what the unedited config gives,
                isolating the PD effect
        custom  use the weights supplied: ``weights`` for the internal
                scale, normalised; the external scale keeps its own (set per
                year from GCC growth, which this screen does not edit)
                unless ``external_weights`` are given too

    ``before`` is the run rebuilt from its own config with no edit, so the
    change is the edit's alone. For a run priced by this version of the
    calculation that is the run's own provision; for one priced by an earlier
    version (scenario weights before 0.29, say) it is not, and ``provision``
    and ``note`` say by how much -- measured from the run's own curves, that
    change of method would have been counted as stress.

    Returns before, after and the split, or ``{"ok": False, "reason": ...}``
    naming what was missing -- a macro screen that goes blank explains nothing.
    """
    from .analytics.model_view import config_used
    from .etl.macro import build_stpd_from_static
    from .etl.static_ref import load_static_reference

    import yaml

    if weight_mode not in MEV_WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {MEV_WEIGHT_MODES}")
    if inputs is None or not inputs.ok:
        return {"ok": False,
                "reason": "The run's engine inputs could not be read."}
    cu = config_used(run_path)
    if cu is None:
        return {"ok": False,
                "reason": "This run has no frozen config (config_used), so "
                          "the PD chain cannot be rebuilt."}
    try:
        static = load_static_reference(cu["static"])
        model = yaml.safe_load((cu["config"] / "model.yml").read_text(encoding="utf-8"))
        raw = yaml.safe_load((cu["config"] / "model_inputs.yml")
                             .read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"The frozen config could not be read: {exc}"}

    import copy
    raw0 = copy.deepcopy(raw)          # the run's config, unedited
    fc = ((raw or {}).get("mev_forecasts") or {}).get("forecasts")
    if not fc:
        return {"ok": False,
                "reason": "mev_forecasts.forecasts is empty in this run's config."}
    fc = {y: [float(v) for v in vals] for y, vals in fc.items()}

    # The forecast years are YAML ints in this config and strings in others,
    # so an edit is matched on the year's VALUE rather than on its type. Keyed
    # on str() alone, every cell edit silently did nothing.
    year_key = {str(y): y for y in fc}
    if mev_new is not None and len(mev_new):
        for r in mev_new.itertuples(index=False):
            y = year_key.get(str(r.year))
            i = int(r.idx) - 1
            if y is not None and 0 <= i < len(fc[y]):
                fc[y][i] = float(r.value)
    for k, delta in (shock or {}).items():
        i = int(k) - 1
        delta = float(delta)
        if delta == 0:
            continue
        for y in fc:
            if 0 <= i < len(fc[y]):
                fc[y][i] += delta
    raw["mev_forecasts"]["forecasts"] = fc

    # "This run's" weights: what the unedited config gives on this engine --
    # the baseline's. The config's explicit_weights are a rounded snapshot,
    # and in a run from before 0.29 the earlier method's.
    from .etl.macro import resolve_internal_scenario_weights
    try:
        held = {k: float(v) for k, v in
                resolve_internal_scenario_weights(raw0, static).items()}
    except Exception:
        held = None
    base_internal = held or (raw.get("internal_scenario_weights") or {}).get("explicit_weights")
    if weight_mode == "hold":
        # Pin exactly those, so only the PD effect of the edit moves. The
        # external weights follow GCC growth, which this screen does not edit.
        if held is None:
            return {"ok": False,
                    "reason": "The run's scenario weights could not be resolved."}
        raw["internal_scenario_weights"] = {
            **(raw.get("internal_scenario_weights") or {}),
            "mode": "explicit", "explicit_weights": held}
    elif weight_mode == "custom" and weights:
        names = list(base_internal or weights)
        for node, given in (("internal_scenario_weights", weights),
                            ("external_scenario_weights", external_weights)):
            if not given:
                continue
            try:
                w = {k: float(v) for k, v in given.items()}
            except (TypeError, ValueError):
                w = {}
            unknown = [k for k in w if k not in names]
            if unknown:
                return {"ok": False, "reason": "Not a scenario of this run: "
                                               f"{', '.join(unknown)}."}
            miss = [k for k in names if k not in w]
            if miss:
                return {"ok": False,
                        "reason": f"Custom weights are missing: {', '.join(miss)}."}
            tot = sum(w.values())
            if not np.isfinite(tot) or tot <= 0 or \
                    any(not np.isfinite(v) or v < 0 for v in w.values()):
                return {"ok": False, "reason": "Custom weights must be numbers "
                                               "of zero or more that sum to "
                                               "more than zero."}
            raw[node] = {**(raw.get(node) or {}), "mode": "explicit",
                         "explicit_weights": {k: w[k] / tot for k in names}}
    try:                               # what the rebuilt chain weights by
        used_internal = {k: float(v) for k, v in
                         resolve_internal_scenario_weights(raw, static).items()}
    except Exception:
        used_internal = None

    extract = str(report["extract_date"].dropna().iloc[0]) \
        if "extract_date" in report.columns and report["extract_date"].notna().any() \
        else ""
    try:
        from .etl.model_registry import run_model_id
        stpd_new = build_stpd_from_static(static, model, raw, extract,
                                          model_id=run_model_id(run_path))
    except Exception as exc:
        return {"ok": False, "reason": f"Rebuilding the PD chain failed: {exc}"}
    curves = _ctx_curves(stpd_new)
    if not curves:
        return {"ok": False, "reason": "The rebuilt StPD had no usable curves."}

    # The baseline: the run rebuilt on THIS engine with no edit. When that is
    # not the run's own curves (it was priced by another version of the
    # calculation), the stress is measured from the rebuild and the gap to the
    # run's own provision is reported, not counted as stress.
    base_curves = None
    try:
        stpd_base = build_stpd_from_static(static, model, raw0, extract,
                                           model_id=run_model_id(run_path))
        bc = _ctx_curves(stpd_base)
        if bc and not _same_pd_curves(bc, inputs.ctx.stpd):
            base_curves = bc
    except Exception:
        base_curves = None
    own = reprice(inputs, report, cfg=cfg)
    before = own if base_curves is None else \
        reprice(inputs, report, pd_curves=base_curves, cfg=cfg)
    after = reprice(inputs, report, pd_curves=curves, cfg=cfg)

    d = before[["contract", "customer", "portfolio", "rating", "stage",
                "exposure"]].copy()
    d["ecl_before"] = before["ecl"].to_numpy()
    d["ecl_after"] = after["ecl"].to_numpy()
    ok = d["ecl_before"].notna() & d["ecl_after"].notna()
    priced = d[ok].copy()
    priced["change"] = priced["ecl_after"] - priced["ecl_before"]
    if len(priced) == 0:
        return {"ok": False, "reason": "Nothing could be priced on the new path."}

    by_pf = (priced.groupby("portfolio")
             .agg(contracts=("contract", "size"),
                  before=("ecl_before", "sum"), after=("ecl_after", "sum"))
             .reset_index())
    by_pf["change"] = by_pf["after"] - by_pf["before"]

    path = pd.DataFrame([{"year": int(y), **{f"mev_{i + 1}": v
                                             for i, v in enumerate(vals)}}
                         for y, vals in sorted(fc.items(), key=lambda kv: int(kv[0]))])

    # the run's own provision, as its report has it, over the contracts priced
    rep_ecl = pd.to_numeric(report["ecl"], errors="coerce") \
        if "ecl" in report.columns and len(report) == len(own) else own["ecl"]
    provision = float(pd.to_numeric(rep_ecl, errors="coerce").fillna(0.0)
                      .to_numpy()[ok.to_numpy()].sum())
    note = None
    if base_curves is not None:
        gap = float(priced["ecl_before"].sum()) - provision
        note = (f"This run was priced by another version of the calculation: "
                f"rebuilt on this engine with no edit it is {gap:+,.0f} from its "
                f"own provision ({provision:,.0f}). The change shown is "
                "measured from the rebuild, so it is the edit's alone.")

    return {
        "ok": True,
        "before": float(priced["ecl_before"].sum()),
        "after": float(priced["ecl_after"].sum()),
        "delta": float(priced["change"].sum()),
        "provision": provision,
        "rebuilt_baseline": base_curves is not None,
        "note": note,
        "priced": int(ok.sum()),
        "weight_mode": weight_mode,
        "weights_base": base_internal,
        "weights_used": used_internal,
        "by_portfolio": by_pf.reindex(
            by_pf["change"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True),
        "movers": customer_rows(priced, n=500),
        "path": path,
        "stpd": stpd_new,
    }


# ------------------------------------------------------- scenario weights --
def run_scenario_weights(run_path) -> dict:
    """The scenario weights a run is priced with, as this engine resolves
    them from its frozen config (``etl.macro.resolve_scenario_weights``):
    ``internal``, and ``external`` averaged over the forecast years it is set
    for, one vector a year from GCC growth.

    Not the config's ``explicit_weights``: on an ``auto`` mode those are a
    rounded snapshot, and in a config written before 0.29 the earlier
    method's.
    """
    import yaml

    from .analytics.model_view import config_used
    from .etl.macro import resolve_scenario_weights
    from .etl.static_ref import load_static_reference
    cu = config_used(run_path)
    if cu is None:
        return {"ok": False,
                "reason": "This run has no frozen config (config_used)."}
    try:
        static = load_static_reference(cu["static"])
        model = yaml.safe_load((cu["config"] / "model.yml").read_text(encoding="utf-8"))
        raw = yaml.safe_load((cu["config"] / "model_inputs.yml")
                             .read_text(encoding="utf-8"))
        internal, external = resolve_scenario_weights(static, model, raw)
    except Exception as exc:
        return {"ok": False,
                "reason": f"The run's scenario weights could not be resolved: {exc}"}
    names = list(static["scenario_severity"]["scenario"])
    ext = external["average"]
    return {"ok": True,
            "internal": {k: float(internal[k]) for k in names if k in internal},
            "external": {k: float(ext[k]) for k in names if k in ext.index}}


def scenario_weights_sensitivity(inputs, report, run_path,
                                 weights: dict | None = None,
                                 shift: float = 0.10,
                                 cfg: EclConfig | None = None) -> pd.DataFrame:
    """What moving ``shift`` of probability ONTO each scenario does.

    Each row takes it from the other scenarios -- pro rata, so their relative
    standing is unchanged -- gives it to one, and reprices the book on the
    result -- ``mev_stress`` with these as custom weights and the macro path
    unedited, so at the run's own weights nothing moves. From ``weights``,
    else the run's own. That is the question a committee asks: if we thought
    the downturn ten points more likely, what would we book?
    """
    if weights is None:
        rw = run_scenario_weights(run_path)
        if not rw.get("ok"):
            return pd.DataFrame()
        weights = rw["internal"]
    w = {k: float(v) for k, v in weights.items()}
    tot = sum(w.values())
    if not w or tot <= 0:
        return pd.DataFrame()
    w = {k: v / tot for k, v in w.items()}
    def reprice_on(weights):
        return mev_stress(inputs, report, run_path, weight_mode="custom",
                          weights=weights, cfg=cfg)

    base = reprice_on(w)
    if not base.get("ok"):
        return pd.DataFrame()
    from .analytics.scenarios import scenario_ecl_from_outputs
    own = scenario_ecl_from_outputs(run_path)
    own = own.get("ecl", {}) if own.get("ok") else {}

    rows = []
    for s in w:
        room = 1.0 - w[s]
        others = [o for o in w if o != s]
        denom = sum(w[o] for o in others)
        if room <= 0 or denom <= 0:
            continue
        take = min(shift, room)
        up = {o: w[o] * (1 - take / denom) for o in others}
        up[s] = w[s] + take
        r = reprice_on(up)
        if not r.get("ok"):
            continue
        rows.append({
            "scenario": s, "weight_before": w[s], "weight_after": up[s],
            "ecl_scenario": own.get(s, float("nan")),
            "ecl_base": base["after"], "ecl_shifted": r["after"],
            "change": r["after"] - base["after"],
            "pct": 100 * (r["after"] - base["after"]) / max(base["after"], 1.0),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.attrs["base_total"] = base["after"]
    out.attrs["shift"] = shift
    return (out.reindex(out["change"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


# -------------------------------------------------------- report helpers ---
def filter_rating_type(report: pd.DataFrame, inputs: EngineInputs,
                       rating_type: int) -> pd.DataFrame:
    """Restrict a report to the portfolios on one rating scale.

    The two scales are different models on different grades. Mixing them in a
    rating chart puts an agency Aa2 beside a QDB 5 as though they meant the
    same thing.
    """
    if report is None or inputs is None:
        return report
    keep = [p for p, t in inputs.rating_type_of_portfolio.items()
            if t == int(rating_type)]
    if not keep:
        return report
    return report[report["portfolio"].isin(keep)]


def order_by_rating(df: pd.DataFrame, col: str, levels) -> pd.DataFrame:
    """Order a frame by the rating scale's own hierarchy, best first.

    Rows carrying a rating not on the scale are dropped rather than sorted to
    one end: a grade the scale does not contain has no position on it, and
    putting it first or last invents one.
    """
    if df is None or len(df) == 0 or not levels:
        return df
    order = {r: i for i, r in enumerate(levels)}
    keep = df[df[col].astype(str).isin(order)]
    if len(keep) == 0:
        return keep
    return (keep.assign(_o=keep[col].astype(str).map(order))
            .sort_values("_o").drop(columns="_o").reset_index(drop=True))


def scenario_ecl_single_run(inputs: EngineInputs, report: pd.DataFrame,
                            run_path, scenarios=None,
                            cfg: EclConfig | None = None) -> dict:
    """Price the book under each scenario on its own, from ONE run.

    A run that wrote its five per-scenario reports needs none of this --
    ``analytics.scenario_ecl_from_outputs`` reads them and cannot disagree
    with the run. This is for the runs that did not: the PD chain is rebuilt
    from the run's frozen config with the whole probability mass on one
    scenario at a time, and the book is repriced on each resulting curve set.

    The weighted figure is deliberately absent. Weighting these back up would
    invite comparison with the booked number, and the two are not the same
    calculation -- the run weights its MARGINAL PDs before the curves are
    built, not its finished provisions.
    """
    from .analytics.model_view import config_used
    from .etl.macro import build_stpd_from_static
    from .etl.static_ref import load_static_reference

    import yaml

    if inputs is None or not inputs.ok:
        return {"ok": False,
                "reason": "The run's engine inputs could not be read."}
    cu = config_used(run_path)
    if cu is None:
        return {"ok": False,
                "reason": "This run has no frozen config (config_used), so "
                          "the PD chain cannot be rebuilt."}
    try:
        static = load_static_reference(cu["static"])
        model = yaml.safe_load((cu["config"] / "model.yml").read_text(encoding="utf-8"))
        model_inputs = yaml.safe_load(
            (cu["config"] / "model_inputs.yml").read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "reason": f"The frozen config could not be read: {exc}"}

    names = list(static["scenario_severity"]["scenario"])
    wanted = [s for s in (scenarios or names) if s in names]
    if not wanted:
        return {"ok": False, "reason": "None of those scenarios are configured."}

    extract = str(report["extract_date"].dropna().iloc[0]) \
        if "extract_date" in report.columns and report["extract_date"].notna().any() \
        else ""
    from .etl.model_registry import run_model_id
    out = {}
    for name in wanted:
        one_hot = {s: (1.0 if s == name else 0.0) for s in names}
        try:
            stpd = build_stpd_from_static(static, model, model_inputs, extract,
                                          scenario_weights=one_hot,
                                          model_id=run_model_id(run_path))
        except Exception as exc:
            return {"ok": False,
                    "reason": f"Rebuilding the PD chain for {name} failed: {exc}"}
        curves = _ctx_curves(stpd)
        if not curves:
            return {"ok": False,
                    "reason": f"The rebuilt StPD for {name} had no curves."}
        priced = reprice(inputs, report, pd_curves=curves, cfg=cfg)
        out[name] = float(pd.to_numeric(priced["ecl"], errors="coerce").sum())

    return {"ok": True, "ecl": out, "scenarios": wanted,
            "rebuilt": True, "weighted": None}
