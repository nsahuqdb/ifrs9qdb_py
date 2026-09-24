"""The risk parameters themselves: PD, LGD, collateral and EAD.

The profile analytics say where the provision sits. These say what it is made
of, and are where a broken mapping shows up first -- a PD curve that did not
resolve, collateral that netted to nothing, an amortisation curve that runs
off in one month.

Weighted averages here are weighted by EXPOSURE, never by contract count. A
book's LGD is not the average of its facilities' LGDs; it is what it costs,
and a thousand tiny facilities should not outvote the single large one that
holds the loss.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["report_total", "pd_profile", "pd_term_structure",
           "lgd_floor_stats", "lgd_vs_collateral",
           "collateral_analysis", "ead_runoff", "segment_matrix"]

_PD_BY = ("stage", "portfolio", "rating", "account_type")
_LGD_BY = ("portfolio", "stage")
_MATRIX_VALUES = ("ecl", "exposure", "coverage", "contracts")


def _wmean(x, w) -> float:
    """Exposure-weighted mean, ignoring rows with no weight to carry."""
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    w = pd.to_numeric(pd.Series(w), errors="coerce")
    ok = x.notna() & w.notna() & (w > 0)
    if not ok.any():
        return float("nan")
    return float((x[ok] * w[ok]).sum() / w[ok].sum())


def _group(v: pd.Series) -> pd.Series:
    """A grouping column as text, with whole numbers kept whole."""
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().any() and (num.dropna() % 1 == 0).all():
        out = num.astype("Int64").astype("string")
    else:
        out = v.astype("string")
    return out.fillna("").replace("", "(unassigned)")


def _has(d: pd.DataFrame, col: str) -> bool:
    return (d is not None and len(d) > 0 and col in d.columns
            and pd.to_numeric(d[col], errors="coerce").notna().any())


def report_total(d: pd.DataFrame) -> float:
    """The provision a report carries. NaN for nothing, never 0."""
    if d is None or len(d) == 0:
        return float("nan")
    return float(pd.to_numeric(d["ecl"], errors="coerce").sum())


# ----------------------------------------------------------------- PD ------
def pd_profile(d: pd.DataFrame, by: str = "stage") -> pd.DataFrame:
    """Lifetime PD by segment, weighted and unweighted side by side.

    Both, because the gap between them is the finding: a weighted PD far above
    the mean says the large exposures sit in the worse grades, which is a
    different book from one where the two agree.
    """
    if by not in _PD_BY:
        raise ValueError(f"by must be one of {_PD_BY}, not {by!r}")
    if not _has(d, "pd"):
        return pd.DataFrame()
    g = _group(d[by])
    rows = []
    for k, idx in d.assign(_g=g).groupby("_g").groups.items():
        s = d.loc[idx]
        rows.append({
            "group": k, "contracts": len(s),
            "exposure": float(s["exposure"].sum()),
            "pd_w": _wmean(s["pd"], s["exposure"]),
            "pd_mean": float(pd.to_numeric(s["pd"], errors="coerce").mean()),
            "ecl": float(s["ecl"].sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("exposure", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- LGD ------
def lgd_floor_stats(d: pd.DataFrame, floor: float = 0.225,
                    by: str = "portfolio") -> pd.DataFrame:
    """How often the LGD floor binds.

    The floor is ``lgd_base x lgd_floor`` -- 0.45 x 0.5 = 0.225 as configured --
    and a segment sitting almost entirely on it is one where collateral, not
    the model, is setting the loss. That is worth knowing before anybody reads
    a movement in LGD as a change in credit quality.
    """
    if by not in _LGD_BY:
        raise ValueError(f"by must be one of {_LGD_BY}, not {by!r}")
    if not _has(d, "lgd"):
        return pd.DataFrame()
    s = d[pd.to_numeric(d["lgd"], errors="coerce").notna()]
    if len(s) == 0:
        return pd.DataFrame()
    g = _group(s[by])
    rows = []
    for k, idx in s.assign(_g=g).groupby("_g").groups.items():
        p = s.loc[idx]
        on_floor = (pd.to_numeric(p["lgd"], errors="coerce") - floor).abs() < 1e-9
        rows.append({
            "group": k, "contracts": len(p), "on_floor": int(on_floor.sum()),
            "pct_on_floor": 100 * float(on_floor.mean()),
            "lgd_w": _wmean(p["lgd"], p["exposure"]),
            "exposure": float(p["exposure"].sum()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("exposure", ascending=False).reset_index(drop=True)


def lgd_vs_collateral(d: pd.DataFrame, n: int = 900,
                      seed: int | None = 0) -> pd.DataFrame:
    """LGD against collateral coverage, for a scatter.

    Sampled down to ``n`` points because the shape, not every contract, is what
    a scatter shows -- and a browser asked to draw 6,000 of them stops being
    interactive. The sample is SEEDED, so the same run draws the same picture
    twice; an unseeded sample makes a chart that changes when nothing did.

    Coverage is capped at 200%: a handful of contracts secured many times over
    would otherwise flatten the axis where the whole story is, between 0 and
    100%.
    """
    if not _has(d, "lgd") or not _has(d, "collcov"):
        return pd.DataFrame()
    s = d[pd.to_numeric(d["lgd"], errors="coerce").notna()
          & pd.to_numeric(d["collcov"], errors="coerce").notna()
          & (d["exposure"] > 0)]
    if len(s) == 0:
        return pd.DataFrame()
    if len(s) > n:
        s = s.sample(n=n, random_state=seed)
    return pd.DataFrame({
        "collcov": 100 * np.minimum(pd.to_numeric(s["collcov"], errors="coerce"), 2),
        "lgd": pd.to_numeric(s["lgd"], errors="coerce"),
        "exposure": s["exposure"].to_numpy(),
        "stage": s["stage"].to_numpy(),
    }).reset_index(drop=True)


# --------------------------------------------------------- collateral ------
def collateral_analysis(inputs) -> dict:
    """What the collateral tables hold, and what did not join.

    An orphan allocation -- a contract pointing at a collateral record that is
    not in the extract -- is the quiet one. It does not error: the contract
    simply prices as unsecured and its LGD goes to the model maximum. Counting
    them is how that gets noticed.
    """
    if inputs is None or getattr(inputs, "alloc", None) is None:
        return {}
    al, co = inputs.alloc, getattr(inputs, "collateral", None)
    ct = getattr(inputs, "coll_type", None)
    if co is None or "CollateralId" not in al.columns:
        return {}

    from ..ids import as_id
    a_id = as_id(al["CollateralId"])
    c_id = set(as_id(co["CollateralId"])) if "CollateralId" in co.columns else set()
    orphan = ~a_id.isin(c_id)

    by_type = pd.DataFrame()
    tcol = next((c for c in ("CollateralTypeId", "CollateralType")
                 if c in co.columns), None)
    if tcol is not None:
        val = pd.to_numeric(co.get("CollateralValue"), errors="coerce")
        if val is None or val.isna().all():
            val = pd.to_numeric(co.get("Value"), errors="coerce")
        g = co[tcol].astype("string").fillna("").replace("", "(unknown)")
        if ct is not None and "CollateralTypeId" in ct.columns:
            nm = next((c for c in ("Name", "CollateralTypeName", "Description")
                       if c in ct.columns), None)
            if nm is not None:
                lut = dict(zip(ct["CollateralTypeId"].astype("string"),
                               ct[nm].astype("string")))
                g = g.map(lambda k: lut.get(k, k))
        by_type = (pd.DataFrame({"type": g, "value": val})
                   .groupby("type", dropna=False)
                   .agg(records=("value", "size"), value=("value", "sum"))
                   .reset_index()
                   .sort_values("value", ascending=False).reset_index(drop=True))

    c_col = "ContractId" if "ContractId" in al.columns else None
    return {
        "by_type": by_type,
        "allocations": len(al),
        "orphan_allocations": int(orphan.sum()),
        "orphan_contracts": int(as_id(al[c_col])[orphan].nunique()) if c_col else 0,
        "collateral_records": len(co),
    }


# ---------------------------------------------------------------- EAD ------
def ead_runoff(inputs, max_month: int = 120) -> pd.DataFrame:
    """How the book's exposure amortises, month by month.

    The count beside it is the point: exposure falling away because facilities
    MATURE is a run-off, exposure falling while the count holds is amortisation.
    A provision that depends on the second while the book is doing the first
    is being measured against the wrong term.
    """
    curves = getattr(inputs, "ead_curves", None) if inputs is not None else None
    if not curves:
        return pd.DataFrame()
    n = min(max(len(v) for v in curves.values()), int(max_month))
    if n < 1:
        return pd.DataFrame()
    total = np.zeros(n)
    count = np.zeros(n, dtype=int)
    for v in curves.values():
        k = min(len(v), n)
        total[:k] += np.nan_to_num(np.asarray(v, dtype=float)[:k])
        count[:k] += 1
    return pd.DataFrame({
        "month": np.arange(1, n + 1), "exposure": total, "contracts": count,
        "pct_of_today": 100 * total / total[0] if total[0] > 0 else np.nan,
    })


# ------------------------------------------------------------ segments -----
def segment_matrix(d: pd.DataFrame, rows: str = "portfolio",
                   cols: str = "stage", value: str = "ecl") -> pd.DataFrame:
    """Any two segmentations crossed, in long form.

    Long rather than pivoted so the caller decides the layout, and so an empty
    cell stays absent instead of becoming a zero that reads as "nothing here"
    when it means "nobody in this segment".
    """
    if value not in _MATRIX_VALUES:
        raise ValueError(f"value must be one of {_MATRIX_VALUES}, not {value!r}")
    if d is None or len(d) == 0:
        return pd.DataFrame()
    if rows not in d.columns or cols not in d.columns:
        return pd.DataFrame()
    g = (d.assign(_r=_group(d[rows]), _c=_group(d[cols]))
         .groupby(["_r", "_c"], dropna=False)
         .agg(exposure=("exposure", "sum"), ecl=("ecl", "sum"),
              contracts=("contract", "size"))
         .reset_index().rename(columns={"_r": "row", "_c": "col"}))
    if value == "coverage":
        g["value"] = np.where(g["exposure"] > 0,
                              100 * g["ecl"] / g["exposure"], np.nan)
    else:
        g["value"] = g[value]
    return g[["row", "col", "value", "exposure", "ecl", "contracts"]]


def pd_term_structure(inputs, portfolio: str | None = None,
                      max_month: int = 120) -> pd.DataFrame:
    """The cumulative and marginal PD curves a run priced on.

    Both, because they answer different questions. The cumulative curve is what
    the engine multiplies through; the marginal one is where a broken term
    structure shows -- a step, a flat stretch, a month where the marginal PD
    goes negative because the curve was built by interpolating the wrong way.

    In percent, since that is how a PD curve is read and discussed.
    """
    curves = getattr(inputs, "pd_curves", None) if inputs is not None else None
    if not curves:
        return pd.DataFrame()
    keys = list(curves)
    if portfolio:
        keys = [k for k in keys if k.startswith(f"{portfolio}|")]
    rows = []
    for k in keys:
        v = np.asarray(curves[k], dtype=float)
        n = min(len(v) - 1, int(max_month))
        if n < 1:
            continue
        rows.append(pd.DataFrame({
            "curve": k, "month": np.arange(1, n + 1),
            "cum_pd": 100 * v[1:n + 1],
            "marginal_pd": 100 * np.diff(v[:n + 1]),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
