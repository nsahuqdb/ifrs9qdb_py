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

from .analytics.profile import classify_stage, customer_view
from .engine import EclConfig, compute_lgd, sum_marginal_ecl
from .inputs import EngineInputs

__all__ = [
    "StressSpec", "Rule", "reprice", "apply_stress", "compare_packages",
    "reverse_stress", "reverse_stress_all", "roll_forward", "tornado",
    "TORNADO_LEVERS", "REVERSE_LEVERS", "stretch_curve", "advance_curve",
    "conditional_pd", "staging_threshold", "staging_threshold_sweep",
    "advance_curves", "reprofile_curves", "customer_rows",
    "mev_stress", "MEV_WEIGHT_MODES", "stpd_to_curves",
    "match_rules", "reprice_rules", "filter_rating_type",
    "order_by_rating", "scenario_ecl_single_run",
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
    """A named set of levers applied together."""
    name: str = "Stress"
    portfolios: list[str] = field(default_factory=list)
    dpd_threshold: float = 60
    contagion: bool = True
    tasdeer_collective: bool = True
    watchlist_triggers: bool = True
    local_triggers: bool = True
    pd_multiplier: float = 1.0
    lgd_base: float = 0.45
    lgd_floor: float = 0.5
    collateral_pct: float = 100.0
    exposure_pct: float = 100.0
    rating_notches: int = 0
    maturity_years: float = 0.0
    default_top_n: int = 0

    def is_empty(self) -> bool:
        d = StressSpec()
        return all(getattr(self, k) == getattr(d, k)
                   for k in asdict(d) if k not in ("name", "portfolios"))


@dataclass
class Rule:
    """A what-if rule: who it applies to, and what changes for them."""
    label: str = "Rule 1"
    customers: list[str] = field(default_factory=list)
    portfolios: list[str] = field(default_factory=list)
    stages: list[int] = field(default_factory=list)
    stage_to: int | None = None
    rating_notches: int = 0
    collateral_pct: float = 100.0
    exposure_pct: float = 100.0
    maturity_years: float = 0.0
    pd_scenario: str = ""

    def matches(self, report: pd.DataFrame) -> pd.Series:
        keep = pd.Series(True, index=report.index)
        if self.customers:
            keep &= report["customer"].astype(str).isin([str(c) for c in self.customers])
        if self.portfolios:
            keep &= report["portfolio"].isin(self.portfolios)
        if self.stages:
            keep &= report["stage"].isin([float(s) for s in self.stages])
        return keep


# --------------------------------------------------------------- repricing --
def _ingredients(inputs: EngineInputs, report: pd.DataFrame) -> pd.DataFrame:
    """Join the report to the engine inputs on contract id.

    Only columns the inputs do NOT supply are taken from the report: carrying
    a duplicate (rating_type, say) makes the join produce suffixed columns and
    the lookups silently return nothing.
    """
    cols = [c for c in ("contract", "customer", "stage", "exposure", "ecl",
                        "collcov", "dpd", "watchlist", "restructured", "local2",
                        "local3", "local4", "local5", "local6", "default_flag",
                        "insolvency") if c in report.columns]
    # The account master is de-duplicated before the join. A contract id is not
    # unique in a LIC book -- an investment security held in two positions
    # appears in AccountMaster twice, and in the report twice -- and a plain
    # merge squares that: two report rows against two master rows is FOUR, so
    # the security was priced twice and every repriced total carried it twice.
    # De-duplicating the right side keeps one row per report row, which is what
    # a repricing is. The second position then prices on the first's balance;
    # that is a known limitation, and the alternative would be to pair them by
    # an order the files do not guarantee.
    facts = inputs.contracts
    if len(facts) and facts["contract"].duplicated().any():
        facts = facts.drop_duplicates("contract", keep="first")
    return report[cols].merge(facts, on="contract", how="left")


def reprice(
    inputs: EngineInputs,
    report: pd.DataFrame,
    *,
    stage: pd.Series | None = None,
    rating: pd.Series | None = None,
    on_balance: pd.Series | None = None,
    collateral_net: pd.Series | None = None,
    maturity_shift_months: pd.Series | None = None,
    pd_curves: dict[str, np.ndarray] | None = None,
    pd_multiplier: float = 1.0,
    advance_months: int = 0,
    cfg: EclConfig | None = None,
) -> pd.DataFrame:
    """Price every contract, with any input substituted.

    Returns one row per contract with the ECL and the LGD the engine used, so a
    caller can show what changed rather than only the total.
    """
    cfg = cfg or EclConfig()
    ing = _ingredients(inputs, report)
    n = len(ing)
    curves = pd_curves if pd_curves is not None else inputs.pd_curves

    def col(override, name, default=None):
        if override is not None:
            return pd.Series(list(override), index=ing.index)
        if name in ing.columns:
            return ing[name]
        return pd.Series([default] * n, index=ing.index)

    st = col(stage, "stage").fillna(2).astype(int)
    rt = col(rating, "rating")
    onb = col(on_balance, "on_balance")
    cnet = col(collateral_net, "collateral_net", 0.0).fillna(0.0)
    shift = (pd.Series(list(maturity_shift_months), index=ing.index)
             if maturity_shift_months is not None else pd.Series(0.0, index=ing.index))

    ecl = np.full(n, np.nan)
    lgd = np.full(n, np.nan)
    for i in range(n):
        row = ing.iloc[i]
        contract = row["contract"]
        s = int(st.iloc[i])
        bal = onb.iloc[i]

        l = compute_lgd(bal, cnet.iloc[i], base=cfg.lgd_base,
                        unsecured_floor=cfg.lgd_unsecured_floor,
                        zero_exposure_lgd=cfg.zero_exposure_lgd)
        lgd[i] = l

        if s == 3:
            ecl[i] = 0.0 if cfg.stage3_method == "zero" else (
                0.0 if not (bal and bal > 0) else float(bal))
            continue

        scale = inputs.scale_for(row.get("rating_type"))
        bucket = scale.bucket(str(rt.iloc[i])) if scale else None
        cum = curves.get(f"{row['portfolio']}|{bucket}") if bucket is not None else None
        if cum is None:
            continue
        if pd_multiplier != 1.0:
            cum = np.clip(np.asarray(cum, float) * pd_multiplier, 0, 1)
        if advance_months:
            cum = conditional_pd(cum, advance_months)

        curve = inputs.ead_curve(contract, s, row)
        if curve is None or len(curve) == 0:
            continue
        if shift.iloc[i]:
            curve = stretch_curve(curve, max(1, len(curve) + int(round(shift.iloc[i]))))
        if advance_months:
            curve = advance_curve(curve, advance_months)
        if bal is not None and row.get("on_balance") and row["on_balance"] != 0:
            curve = curve * (bal / row["on_balance"])

        v = sum_marginal_ecl(curve, l, cum, row.get("eir") or 0.0, horizon=len(curve))
        if np.isfinite(v) and cfg.cap_ecl_at_exposure and bal and bal > 0:
            v = min(v, float(bal))
        ecl[i] = v

    out = ing[["contract", "customer", "portfolio", "rating", "stage",
               "exposure"]].copy()
    out["stage_after"] = st.to_numpy()
    out["rating_after"] = rt.to_numpy()
    out["ecl"] = ecl
    out["lgd"] = lgd
    return out


# ------------------------------------------------------------ apply stress --
def apply_stress(inputs: EngineInputs, report: pd.DataFrame,
                 spec: StressSpec, cfg: EclConfig | None = None) -> dict:
    """Apply a stress package and reprice."""
    if not inputs.ok:
        return {"ok": False, "reason": f"Engine inputs missing: {', '.join(inputs.missing)}"}
    ing = _ingredients(inputs, report)
    if len(ing) == 0:
        return {"ok": False, "reason": "The report could not be matched to the inputs."}

    scoped = (ing["portfolio"].isin(spec.portfolios) if spec.portfolios
              else pd.Series(True, index=ing.index))

    base_cfg = cfg or EclConfig()
    before = reprice(inputs, report, cfg=base_cfg)

    # staging policy
    local_any = pd.Series(False, index=ing.index)
    if spec.local_triggers:
        for c in ("restructured", "local2", "local3", "local4", "local5", "local6"):
            if c in ing.columns:
                local_any |= (ing[c] == 1).fillna(False)
    watch = ing.get("watchlist") if spec.watchlist_triggers else None
    new_stage = classify_stage(
        ing.get("dpd"), ing.get("default_flag"), watch, local_any,
        ing["portfolio"], ing["customer"], dpd_threshold=spec.dpd_threshold,
        contagion=spec.contagion, tasdeer_collective=spec.tasdeer_collective)
    stage = ing["stage"].fillna(2).astype(int).to_numpy().copy()
    stage[scoped.to_numpy()] = new_stage[scoped.to_numpy()]

    # ratings
    rating = ing["rating"].astype(str).to_numpy().copy()
    if spec.rating_notches:
        for i in np.where(scoped.to_numpy())[0]:
            sc = inputs.scale_for(ing.iloc[i].get("rating_type"))
            if sc:
                rating[i] = sc.notch(rating[i], spec.rating_notches)

    # largest N forced to default: rank by EXPOSURE, and keep every one picked.
    # Ranking by ECL puts customers already in Stage 3 at the top -- they carry
    # a 100% provision by construction -- so the lever quietly delivered fewer
    # defaults than asked for.
    defaulted: list[str] = []
    if spec.default_top_n and spec.default_top_n > 0:
        elig = ing[scoped & (ing["exposure"] > 0)]
        rank = elig.groupby("customer")["exposure"].sum().sort_values(ascending=False)
        defaulted = list(rank.head(spec.default_top_n).index)
        hit = scoped & ing["customer"].isin(defaulted)
        stage[hit.to_numpy()] = 3

    on_balance = ing["on_balance"].to_numpy().copy()
    if spec.exposure_pct != 100:
        on_balance[scoped.to_numpy()] *= spec.exposure_pct / 100

    cnet = ing["collateral_net"].fillna(0.0).to_numpy().copy()
    if spec.collateral_pct != 100:
        cnet[scoped.to_numpy()] *= spec.collateral_pct / 100

    shift = np.zeros(len(ing))
    if spec.maturity_years:
        shift[scoped.to_numpy()] = spec.maturity_years * 12

    after = reprice(
        inputs, report, stage=stage, rating=rating, on_balance=on_balance,
        collateral_net=cnet, maturity_shift_months=shift,
        pd_multiplier=spec.pd_multiplier,
        cfg=EclConfig(stage3_method=base_cfg.stage3_method,
                      cap_ecl_at_exposure=base_cfg.cap_ecl_at_exposure,
                      lgd_base=spec.lgd_base, lgd_unsecured_floor=spec.lgd_floor),
    )

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
    return {
        "ok": True,
        "name": spec.name,
        "before": float(priced["ecl_before"].sum()),
        "after": float(priced["ecl_after"].sum()),
        "delta": float(priced["change"].sum()),
        "priced": int(ok.sum()),
        "moved": int(moved.sum()),
        "customers_moved": int(priced.loc[moved, "customer"].nunique()),
        "defaulted": defaulted,
        "by_customer": by_cust,
        "by_portfolio": by_pf.sort_values("change", key=abs, ascending=False),
        "detail": priced.reindex(priced["change"].abs()
                                 .sort_values(ascending=False).index),
    }


def compare_packages(inputs, report, specs: Iterable[StressSpec]) -> pd.DataFrame:
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
    "default_top_n": {"label": "Largest customers defaulting", "lo": 0, "hi": 50,
                      "unit": " customers", "integer": True},
}


def reverse_stress(inputs, report, lever: str, target_pct: float,
                   base: StressSpec | None = None, tol: float = 0.25,
                   max_iter: int = 14) -> dict:
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

    lo, up, best = L["lo"], L["hi"], hi
    for _ in range(max_iter):
        mid = (lo + up) / 2
        if L.get("integer"):
            mid = round(mid)
        r = at(mid)
        p = 100 * (r["after"] - base_prov) / base_prov
        if p >= target_pct:
            up, best = mid, r
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

    Three things move together and all three are needed: the balance advances
    along its own curve, the PD becomes conditional on surviving those months,
    and contracts maturing inside the window run off. Omitting either of the
    first two misstates the answer badly -- on a worked example, by -55% and
    +280% respectively.
    """
    if not inputs.ok:
        return {"ok": False, "reason": f"Engine inputs missing: {', '.join(inputs.missing)}"}
    before = reprice(inputs, report)
    ing = _ingredients(inputs, report)
    scoped = (ing["portfolio"].isin(portfolios) if portfolios
              else pd.Series(True, index=ing.index))

    on_balance = ing["on_balance"].to_numpy().copy()
    matured = np.zeros(len(ing), dtype=bool)
    for i in np.where(scoped.to_numpy())[0]:
        row = ing.iloc[i]
        curve = inputs.ead_curve(row["contract"], int(row.get("stage") or 2), row)
        if curve is None or months >= len(curve):
            matured[i] = True
            on_balance[i] = 0.0
        else:
            on_balance[i] = curve[months]

    after = reprice(inputs, report, on_balance=on_balance, advance_months=months)
    a = after["ecl"].to_numpy().copy()
    a[matured] = 0.0

    ok = before["ecl"].notna() & pd.Series(a).notna()
    return {
        "ok": True, "months": months,
        "before": float(before.loc[ok, "ecl"].sum()),
        "after": float(pd.Series(a)[ok].sum()),
        "delta": float(pd.Series(a)[ok].sum() - before.loc[ok, "ecl"].sum()),
        "matured": int(matured.sum()),
        "matured_exposure": float(ing.loc[matured, "exposure"].sum()),
        "exposure_before": float(ing.loc[ok, "exposure"].sum()),
        "exposure_after": float(pd.Series(on_balance)[ok].sum()),
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

    A run freezes its staging policy beside its outputs. Sweeping around a
    hardcoded 60 when the run used something else compares the book against a
    policy nobody applied.
    """
    if out_dir is None:
        return default
    p = Path(out_dir)
    f = p / "staging_thresholds.csv"
    if not f.is_file():
        f = p / "Output" / "staging_thresholds.csv"
    if not f.is_file():
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
                            reference: float = 60,
                            cfg: EclConfig | None = None) -> pd.DataFrame:
    """Reprice the book at each candidate Stage 2 DPD threshold.

    The curve this draws is rarely straight: most of the book is staged by
    watchlist, restructuring and contagion rather than by days past due, so
    lowering the threshold moves far less than people expect. Showing that is
    the point -- it is the difference between a policy debate and an argument
    about a number nobody has tested.

    Everything except the threshold is held at ``base``, so each row differs
    from its neighbours in exactly one thing.
    """
    rows = []
    for t in thresholds:
        spec = replace(base or StressSpec(), dpd_threshold=float(t),
                       name=f"DPD > {t:g}")
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
               cfg: EclConfig | None = None) -> dict:
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
        hold    pin them to what this run used, isolating the PD effect
        custom  use the weights supplied

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

    base_internal = (raw.get("internal_scenario_weights") or {}).get("explicit_weights")
    if weight_mode == "hold":
        for node in ("internal_scenario_weights", "external_scenario_weights"):
            if node in raw:
                raw[node]["mode"] = "explicit"
    elif weight_mode == "custom" and weights:
        for node in ("internal_scenario_weights", "external_scenario_weights"):
            raw.setdefault(node, {})
            raw[node]["mode"] = "explicit"
            raw[node]["explicit_weights"] = {k: float(v) for k, v in weights.items()}

    extract = str(report["extract_date"].dropna().iloc[0]) \
        if "extract_date" in report.columns and report["extract_date"].notna().any() \
        else ""
    try:
        stpd_new = build_stpd_from_static(static, model, raw, extract)
    except Exception as exc:
        return {"ok": False, "reason": f"Rebuilding the PD chain failed: {exc}"}
    curves = stpd_to_curves(stpd_new)
    if not curves:
        return {"ok": False, "reason": "The rebuilt StPD had no usable curves."}

    base_cfg = cfg or EclConfig()
    before = reprice(inputs, report, cfg=base_cfg)
    after = reprice(inputs, report, pd_curves=curves, cfg=base_cfg)

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

    return {
        "ok": True,
        "before": float(priced["ecl_before"].sum()),
        "after": float(priced["ecl_after"].sum()),
        "delta": float(priced["change"].sum()),
        "priced": int(ok.sum()),
        "weight_mode": weight_mode,
        "weights_base": base_internal,
        "by_portfolio": by_pf.reindex(
            by_pf["change"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True),
        "movers": customer_rows(priced, n=500),
        "path": path,
        "stpd": stpd_new,
    }


# ------------------------------------------------------------- what-if -----
def match_rules(report: pd.DataFrame, rules) -> dict:
    """Assign each contract to at most ONE rule, and name the conflicts.

    Rules do not stack. A contract caught by two of them has no defined
    answer -- applying both in some order would make the result depend on the
    order, and applying one silently would hide the other -- so it is excluded
    from repricing and listed instead. This mirrors the overlay feature's
    no-stacking rule, and for the same reason: an adjustment nobody can
    reconstruct is worse than one that was refused.

    Returns ``{assignment, conflicts}``: a Series of rule index or NA per
    contract, and a frame of the contracts more than one rule claimed.
    """
    if report is None or len(report) == 0 or not rules:
        return {"assignment": pd.Series(dtype="Int64"),
                "conflicts": pd.DataFrame()}

    hits = pd.DataFrame(
        {j: rule.matches(report).fillna(False).to_numpy()
         for j, rule in enumerate(rules)}, index=report.index)
    n_hit = hits.sum(axis=1)

    assignment = pd.Series(pd.NA, index=report.index, dtype="Int64")
    single = n_hit == 1
    if single.any():
        assignment[single] = hits[single].to_numpy().argmax(axis=1)

    clash = n_hit > 1
    conflicts = pd.DataFrame()
    if clash.any():
        labels = [r.label for r in rules]
        conflicts = pd.DataFrame({
            "contract": report.loc[clash, "contract"].to_numpy(),
            "customer": report.loc[clash, "customer"].to_numpy(),
            "rules": [" + ".join(labels[j] for j in range(len(rules)) if row[j])
                      for _, row in hits[clash].iterrows()],
        })
    return {"assignment": assignment, "conflicts": conflicts}


def reprice_rules(inputs: EngineInputs, report: pd.DataFrame, rules,
                  cfg: EclConfig | None = None) -> dict:
    """Apply a set of what-if rules and reprice, rule by rule.

    Both sides are priced through the same ``reprice``, so the baseline and
    the what-if agree by construction. Taking the baseline from the report's
    own ECL column instead would reintroduce a difference whenever the loaded
    report predates the current config -- a difference that reads as the
    what-if having done something.

    A rule naming a ``pd_scenario`` prices its contracts on that scenario's
    curves. Those contracts are repriced in their own pass, because a curve
    set applies to a whole repricing; stitching the passes keeps each
    contract on the curves its rule asked for.
    """
    if inputs is None or not inputs.ok:
        return {"ok": False,
                "reason": f"Engine inputs missing: {', '.join(inputs.missing)}"
                if inputs is not None else "No engine inputs."}
    if report is None or len(report) == 0 or not rules:
        return {"ok": False, "reason": "Nothing to reprice."}

    base_cfg = cfg or EclConfig()
    ing = _ingredients(inputs, report)
    if len(ing) == 0:
        return {"ok": False,
                "reason": "The report could not be matched to the inputs."}

    # Matched on the INGREDIENTS, not on the report. The two are row-aligned
    # only while every contract id is unique on both sides, and a LIC report
    # can repeat one -- an investment security held twice. Matching here keeps
    # the lever masks aligned with what is actually being priced.
    matched = match_rules(ing, rules)
    assignment = matched["assignment"].reset_index(drop=True)
    if assignment.notna().sum() == 0:
        return {"ok": False, "reason": "No contract matched any rule.",
                "conflicts": matched["conflicts"]}

    before = reprice(inputs, report, cfg=base_cfg)

    # The levers, built per contract from whichever rule claimed it.
    stage = ing["stage"].fillna(2).astype(int).to_numpy().copy()
    rating = ing["rating"].astype(str).to_numpy().copy()
    onbal = pd.to_numeric(ing["on_balance"], errors="coerce").to_numpy().copy()
    cnet = pd.to_numeric(ing.get("collateral_net"), errors="coerce").fillna(0.0)
    cnet = cnet.to_numpy().copy()
    shift = np.zeros(len(ing))
    scenario_of = np.array([""] * len(ing), dtype=object)

    for j, rule in enumerate(rules):
        # Int64 holds pd.NA for the unmatched, and NA has no truth value.
        sel = assignment.eq(j).fillna(False).to_numpy()
        if not sel.any():
            continue
        if rule.stage_to:
            stage[sel] = int(rule.stage_to)
        if rule.rating_notches:
            for i in np.where(sel)[0]:
                sc = inputs.scale_for(ing.iloc[i].get("rating_type"))
                if sc:
                    rating[i] = sc.notch(rating[i], rule.rating_notches)
        if rule.collateral_pct != 100:
            cnet[sel] = cnet[sel] * (rule.collateral_pct / 100.0)
        if rule.exposure_pct != 100:
            onbal[sel] = onbal[sel] * (rule.exposure_pct / 100.0)
        if rule.maturity_years:
            shift[sel] = rule.maturity_years * 12
        if rule.pd_scenario:
            scenario_of[sel] = rule.pd_scenario

    def price(curves=None, mask=None):
        return reprice(inputs, report, stage=pd.Series(stage),
                       rating=pd.Series(rating),
                       on_balance=pd.Series(onbal),
                       collateral_net=pd.Series(cnet),
                       maturity_shift_months=pd.Series(shift),
                       pd_curves=curves, cfg=base_cfg)

    after = price()
    # Contracts on a scenario's curves get their own pass, stitched back in.
    from .analytics.model_view import read_scenario_stpd
    for name in {s for s in scenario_of if s}:
        curves = read_scenario_stpd(inputs.out_dir, name)
        if not curves:
            continue
        curves = {k: np.asarray(v, dtype=float) for k, v in curves.items()}
        alt = price(curves)
        sel = scenario_of == name
        after.loc[sel, "ecl"] = alt.loc[sel, "ecl"].to_numpy()

    d = before[["contract", "customer", "portfolio", "rating", "stage",
                "exposure"]].copy()
    d["ecl_before"] = before["ecl"].to_numpy()
    d["ecl_after"] = after["ecl"].to_numpy()
    d["rating_after"] = after["rating_after"].to_numpy()
    d["stage_after"] = after["stage_after"].to_numpy()
    d["rule"] = [rules[int(j)].label if pd.notna(j) else ""
                 for j in assignment.to_numpy()]
    ok = d["ecl_before"].notna() & d["ecl_after"].notna()
    priced = d[ok].copy()
    priced["change"] = priced["ecl_after"] - priced["ecl_before"]
    touched = priced[priced["rule"] != ""]

    by_rule = (touched.groupby("rule")
               .agg(contracts=("contract", "size"),
                    customers=("customer", "nunique"),
                    exposure=("exposure", "sum"),
                    before=("ecl_before", "sum"), after=("ecl_after", "sum"))
               .reset_index())
    if len(by_rule):
        by_rule["change"] = by_rule["after"] - by_rule["before"]

    return {
        "ok": True,
        "before": float(priced["ecl_before"].sum()),
        "after": float(priced["ecl_after"].sum()),
        "delta": float(priced["change"].sum()),
        "priced": int(ok.sum()),
        "matched": int(assignment.notna().sum()),
        "conflicts": matched["conflicts"],
        "by_rule": by_rule,
        "movers": customer_rows(touched, n=500),
        "detail": touched.reindex(
            touched["change"].abs().sort_values(ascending=False).index),
    }


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
    base_cfg = cfg or EclConfig()
    out = {}
    for name in wanted:
        one_hot = {s: (1.0 if s == name else 0.0) for s in names}
        try:
            stpd = build_stpd_from_static(static, model, model_inputs, extract,
                                          scenario_weights=one_hot)
        except Exception as exc:
            return {"ok": False,
                    "reason": f"Rebuilding the PD chain for {name} failed: {exc}"}
        curves = stpd_to_curves(stpd)
        if not curves:
            return {"ok": False,
                    "reason": f"The rebuilt StPD for {name} had no curves."}
        priced = reprice(inputs, report, pd_curves=curves, cfg=base_cfg)
        out[name] = float(pd.to_numeric(priced["ecl"], errors="coerce").sum())

    return {"ok": True, "ecl": out, "scenarios": wanted,
            "rebuilt": True, "weighted": None}
