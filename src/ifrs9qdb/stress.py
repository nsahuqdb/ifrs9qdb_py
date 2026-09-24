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

from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .analytics.profile import classify_stage, customer_view
from .engine import EclConfig, compute_lgd, sum_marginal_ecl
from .inputs import EngineInputs

__all__ = [
    "StressSpec", "Rule", "reprice", "apply_stress", "compare_packages",
    "reverse_stress", "reverse_stress_all", "roll_forward", "tornado",
    "TORNADO_LEVERS", "stretch_curve", "advance_curve", "conditional_pd",
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
    m = report[cols].merge(inputs.contracts, on="contract", how="left")
    return m


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
