"""
Movement between two runs.

The walk decomposes the change in provision into causes that sum EXACTLY to the
closing balance. That exactness is the point: a walk with a plug is not a walk,
and the residual is asserted in the tests rather than hoped for.

The identity behind it, for a contract in both runs:

    E1*C1 - E0*C0 = (E1 - E0)*C0  +  E1*(C1 - C0)
                    ^ exposure       ^ coverage

Exposure movement is measured at the OLD coverage, then the coverage change at
the NEW exposure. The order is a convention -- the other order moves the
interaction term between the two -- and it is fixed so the walk is reproducible
quarter to quarter. The coverage part is then split by whether the contract's
stage changed.

Contracts with no exposure in the prior run have no prior coverage to split
with, so their whole change sits in "Other" rather than being spread across
terms that would be meaningless for them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ecl_walk", "ecl_walk_detail", "stage_transitions", "movement_by",
           "flow_profile"]


def _dedup(d: pd.DataFrame) -> pd.DataFrame:
    """One row per position, indexed by contract and occurrence.

    A contract the report lists twice is two positions, matched across runs
    in the order they appear -- as the ECL bridge does -- rather than one with
    the second dropped, which left the opening and closing short of the
    reports' totals.
    """
    d = d.copy()
    occ = d.groupby("contract", dropna=False, sort=False).cumcount()
    d.index = (d["contract"].astype(str) + "#" + occ.astype(str)).to_numpy()
    return d


def ecl_walk(prev: pd.DataFrame, curr: pd.DataFrame) -> dict:
    """Decompose the movement in provision between two runs.

    Returns opening, closing, the labelled steps and a residual which must be
    zero to floating-point tolerance.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return {}
    a, b = _dedup(prev), _dedup(curr)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    common = a.index.intersection(b.index)
    ca, cb = a.loc[common], b.loc[common]

    has_prior = (ca["exposure"] > 0).to_numpy()
    d_expo = ((cb["exposure"] - ca["exposure"]) * ca["coverage"]).to_numpy()
    d_cov = (cb["exposure"] * (cb["coverage"] - ca["coverage"])).to_numpy()
    moved = (ca["stage"].notna() & cb["stage"].notna()
             & (ca["stage"] != cb["stage"])).to_numpy()

    opening = float(a["ecl"].sum())
    closing = float(b["ecl"].sum())
    derecognised = -float(a.loc[only_a, "ecl"].sum())
    new_business = float(b.loc[only_b, "ecl"].sum())
    exposure_movement = float(d_expo[has_prior].sum())
    stage_migration = float(d_cov[has_prior & moved].sum())
    risk_model = float(d_cov[has_prior & ~moved].sum())
    other = float((cb["ecl"] - ca["ecl"]).to_numpy()[~has_prior].sum())

    steps = pd.DataFrame([
        {"label": "Opening", "amount": opening, "kind": "total"},
        {"label": "Derecognised", "amount": derecognised, "kind": "delta"},
        {"label": "New business", "amount": new_business, "kind": "delta"},
        {"label": "Exposure movement", "amount": exposure_movement, "kind": "delta"},
        {"label": "Stage migration", "amount": stage_migration, "kind": "delta"},
        {"label": "Risk & model", "amount": risk_model, "kind": "delta"},
        {"label": "Other", "amount": other, "kind": "delta"},
        {"label": "Closing", "amount": closing, "kind": "total"},
    ])
    deltas = steps.loc[steps["kind"] == "delta", "amount"].sum()
    return {
        "opening": opening,
        "closing": closing,
        "steps": steps,
        "residual": opening + deltas - closing,
        "counts": {
            "left": int(len(only_a)),
            "arrived": int(len(only_b)),
            "common": int(len(common)),
            "migrated": int(moved.sum()),
        },
    }


def ecl_walk_detail(prev: pd.DataFrame, curr: pd.DataFrame,
                    n: int = 200) -> dict[str, pd.DataFrame]:
    """The contracts behind each walk step, largest first.

    Each frame sums back to its step, which the tests assert: a drill-down that
    did not tie to the number above it would be worse than none.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return {}
    a, b = _dedup(prev), _dedup(curr)
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    common = a.index.intersection(b.index)
    ca, cb = a.loc[common], b.loc[common]
    has_prior = (ca["exposure"] > 0)
    d_expo = (cb["exposure"] - ca["exposure"]) * ca["coverage"]
    d_cov = cb["exposure"] * (cb["coverage"] - ca["coverage"])
    moved = ca["stage"].notna() & cb["stage"].notna() & (ca["stage"] != cb["stage"])

    def frame(idx, amount, src):
        if len(idx) == 0:
            return pd.DataFrame()
        out = pd.DataFrame({
            "contract": src.loc[idx, "contract"].to_numpy(),
            "customer": src.loc[idx, "customer"].to_numpy(),
            "portfolio": src.loc[idx, "portfolio"].to_numpy(),
            "amount": np.asarray(amount, dtype=float),
        })
        return out.reindex(out["amount"].abs().sort_values(ascending=False).index).head(n)

    return {
        "Derecognised": frame(only_a, -a.loc[only_a, "ecl"].to_numpy(), a),
        "New business": frame(only_b, b.loc[only_b, "ecl"].to_numpy(), b),
        "Exposure movement": frame(common[has_prior], d_expo[has_prior].to_numpy(), cb),
        "Stage migration": frame(common[has_prior & moved],
                                 d_cov[has_prior & moved].to_numpy(), cb),
        "Risk & model": frame(common[has_prior & ~moved],
                              d_cov[has_prior & ~moved].to_numpy(), cb),
        "Other": frame(common[~has_prior],
                       (cb["ecl"] - ca["ecl"])[~has_prior].to_numpy(), cb),
    }


def stage_transitions(prev: pd.DataFrame, curr: pd.DataFrame,
                      by_customer: bool = True) -> pd.DataFrame:
    """Stage migration.

    Counted as CUSTOMERS by default, because staging is a customer-level
    decision and a contract count over-weights large relationships.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return pd.DataFrame()
    if by_customer:
        from .profile import customer_view
        a = customer_view(prev).set_index("customer")
        b = customer_view(curr).set_index("customer")
        unit = "customers"
    else:
        a, b = _dedup(prev), _dedup(curr)
        unit = "contracts"
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return pd.DataFrame()
    m = pd.DataFrame({
        "from": a.loc[common, "stage"].to_numpy(),
        "to": b.loc[common, "stage"].to_numpy(),
        "exposure": b.loc[common, "exposure"].to_numpy(),
        "ecl": b.loc[common, "ecl"].to_numpy(),
    }).dropna(subset=["from", "to"])
    g = m.groupby(["from", "to"])
    out = pd.DataFrame({
        unit: g.size(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
    }).reset_index()
    out["from"] = out["from"].astype(int)
    out["to"] = out["to"].astype(int)
    return out.sort_values(["from", "to"]).reset_index(drop=True)


def movement_by(prev: pd.DataFrame, curr: pd.DataFrame,
                by: str = "portfolio") -> pd.DataFrame:
    """Change in provision by segment."""
    from .profile import run_profile
    if prev is None or curr is None:
        return pd.DataFrame()
    a = run_profile(prev, by).set_index("group")
    b = run_profile(curr, by).set_index("group")
    idx = a.index.union(b.index)
    out = pd.DataFrame({
        "group": idx,
        "ecl_prev": a["ecl"].reindex(idx).fillna(0).to_numpy(),
        "ecl_curr": b["ecl"].reindex(idx).fillna(0).to_numpy(),
        "exposure_prev": a["exposure"].reindex(idx).fillna(0).to_numpy(),
        "exposure_curr": b["exposure"].reindex(idx).fillna(0).to_numpy(),
    })
    out["change"] = out["ecl_curr"] - out["ecl_prev"]
    out["pct"] = np.where(out["ecl_prev"] != 0,
                          100 * out["change"] / out["ecl_prev"], np.nan)
    return out.reindex(out["change"].abs().sort_values(ascending=False).index
                       ).reset_index(drop=True)


def flow_profile(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """What came on and off the book, and at what coverage."""
    if prev is None or curr is None:
        return pd.DataFrame()
    a, b = _dedup(prev), _dedup(curr)
    left = a.loc[a.index.difference(b.index)]
    arrived = b.loc[b.index.difference(a.index)]

    def row(label, d):
        exp = float(d["exposure"].sum())
        ecl = float(d["ecl"].sum())
        return {"flow": label, "contracts": len(d), "exposure": exp, "ecl": ecl,
                "coverage": 100 * ecl / exp if exp > 0 else np.nan}

    return pd.DataFrame([row("New business", arrived),
                         row("Derecognised", left)])
