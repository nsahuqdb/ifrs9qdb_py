"""Why the provision moved.

A movement analysis that only says "up 180m" invites the same question every
quarter. These split the move into the things that caused it, on the same
contracts, so the answer is "up 180m, of which 140m is PD".

Both are computed on the INTERSECTION of the two books. Contracts that arrived
or left are a real part of the movement but not an attribution -- they belong
in the flow analysis, and mixing them in here makes every factor look wrong.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["ecl_factor_attribution", "coverage_bridge",
           "factor_attribution_exact", "factor_attribution_diagnosis"]


def _dedup(d: pd.DataFrame, key: str = "contract") -> pd.DataFrame:
    return d.drop_duplicates(subset=[key])


def _as_label(v: pd.Series) -> pd.Series:
    """A grouping column as text, with whole numbers kept whole."""
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().any() and (num.dropna() % 1 == 0).all():
        out = num.astype("Int64").astype("string")
    else:
        out = v.astype("string")
    return out.fillna("").replace("", "(unassigned)")


def ecl_factor_attribution(prev: pd.DataFrame,
                           curr: pd.DataFrame) -> pd.DataFrame:
    """Split the ECL move into exposure, PD and LGD effects.

    A sequential decomposition: exposure is moved first at the old PD and LGD,
    then PD at the new exposure and old LGD, then LGD at the new exposure and
    PD. The order matters -- a different order gives different splits, because
    the interaction terms have to land somewhere.

    The three effects are then SCALED so they sum to the actual move. ECL is
    not exactly exposure x PD x LGD -- there is discounting, the exposure cap,
    the Stage 3 treatment -- so the raw effects do not add up on their own.
    That is why the result is marked ``indicative``: the split is a guide to
    where the movement came from, not an identity.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return pd.DataFrame()

    cols = ["contract", "exposure", "pd", "lgd", "ecl"]
    has_ov = "overlay" in prev.columns and "overlay" in curr.columns
    if has_ov:
        cols = cols + ["overlay"]
    a = _dedup(prev)[cols]
    b = _dedup(curr)[cols]
    m = a.merge(b, on="contract", suffixes=("_a", "_b"))

    for c in ("exposure_a", "exposure_b", "pd_a", "pd_b", "lgd_a", "lgd_b"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["exposure_a", "exposure_b", "pd_a", "pd_b",
                         "lgd_a", "lgd_b"])
    # A contract that started at zero on any factor has no proportional story
    # to tell: every effect would be the whole move.
    m = m[(m["exposure_a"] > 0) & (m["pd_a"] > 0) & (m["lgd_a"] > 0)]
    if len(m) == 0:
        return pd.DataFrame()

    e_eff = float(((m["exposure_b"] - m["exposure_a"])
                   * m["pd_a"] * m["lgd_a"]).sum())
    p_eff = float((m["exposure_b"] * (m["pd_b"] - m["pd_a"])
                   * m["lgd_a"]).sum())
    l_eff = float((m["exposure_b"] * m["pd_b"]
                   * (m["lgd_b"] - m["lgd_a"])).sum())

    raw = {"Exposure": e_eff, "PD": p_eff, "LGD": l_eff}
    actual = float((pd.to_numeric(m["ecl_b"], errors="coerce").fillna(0)
                    - pd.to_numeric(m["ecl_a"], errors="coerce").fillna(0)).sum())
    # A post-model overlay moves the provision without moving any factor, so
    # it is its own line: the factors explain the MODEL move, the overlay line
    # the management adjustment, and the two together the change. Without it
    # a run that differs from the last only by an overlay showed every factor
    # at zero against the whole movement.
    ov = 0.0
    if has_ov:
        ov = float((pd.to_numeric(m["overlay_b"], errors="coerce").fillna(0)
                    - pd.to_numeric(m["overlay_a"], errors="coerce").fillna(0)).sum())
    total = sum(raw.values())
    scale = (actual - ov) / total if abs(total) > 1e-9 else 1.0
    effects = {k: v * scale for k, v in raw.items()}
    if abs(ov) > 1e-9:
        raw["Overlay"] = ov
        effects["Overlay"] = ov

    return pd.DataFrame({
        "factor": list(raw),
        "effect": list(effects.values()),
        "raw_effect": list(raw.values()),
        "contracts": len(m),
        "actual_change": actual,
        "indicative": True,
    })


_BRIDGE_BY = ("portfolio", "stage", "rating")


def coverage_bridge(prev: pd.DataFrame, curr: pd.DataFrame,
                    by: str = "portfolio") -> pd.DataFrame:
    """Split the change in COVERAGE into mix and rate.

    Coverage can move without a single contract changing: if the book shifts
    toward a segment that was always provisioned more heavily, the headline
    rate rises while every segment's own rate is flat. The two are different
    facts and only one of them is a credit event.

        mix effect   the weight of each segment changed, at its OLD rate
        rate effect  each segment's own rate changed, at its NEW weight

    Both are in percentage POINTS of overall coverage, so they add to the
    total change and can be read straight off.
    """
    if by not in _BRIDGE_BY:
        raise ValueError(f"by must be one of {_BRIDGE_BY}, not {by!r}")
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return pd.DataFrame()
    if by not in prev.columns or by not in curr.columns:
        return pd.DataFrame()

    def profile(d: pd.DataFrame) -> pd.DataFrame:
        # Stage is numeric here, and a plain cast would label the groups
        # "1.0", "2.0", "3.0". The bridge is read by people.
        g = _as_label(d[by])
        out = (d.assign(_g=g).groupby("_g", dropna=False)
               .agg(exposure=("exposure", "sum"), ecl=("ecl", "sum"))
               .reset_index().rename(columns={"_g": "group"}))
        out["cov"] = np.where(out["exposure"] > 0,
                              out["ecl"] / out["exposure"], 0.0)
        return out

    a, b = profile(prev), profile(curr)
    m = a.merge(b, on="group", how="outer", suffixes=("_a", "_b"))
    for c in ("exposure_a", "exposure_b", "ecl_a", "ecl_b", "cov_a", "cov_b"):
        m[c] = m[c].fillna(0.0)

    ea, eb = m["exposure_a"].sum(), m["exposure_b"].sum()
    if ea <= 0 or eb <= 0:
        return pd.DataFrame()
    wa, wb = m["exposure_a"] / ea, m["exposure_b"] / eb

    m["mix_effect"] = 100 * (wb - wa) * m["cov_a"]
    m["rate_effect"] = 100 * wb * (m["cov_b"] - m["cov_a"])
    m["total_effect"] = m["mix_effect"] + m["rate_effect"]

    cols = ["group", "exposure_a", "exposure_b", "cov_a", "cov_b",
            "mix_effect", "rate_effect", "total_effect"]
    return (m[cols]
            .reindex(m["total_effect"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


# --------------------------------------------------------------- exact -----
_EXACT_ORDER = ("Horizon", "EAD", "PD", "LGD")


def _priced(inputs, report, cfg=None) -> pd.DataFrame:
    """The report joined to the engine inputs, with LGD as the engine sets it.

    LGD comes from the balance and the netted collateral through the engine's
    own ``compute_lgd``, not from the report's LGD column and not from a
    reimplementation of the formula. The floor, the base rate and the
    zero-exposure case then behave here exactly as they did in the run.
    """
    from ..engine import EclConfig, compute_lgd
    from ..stress import _ingredients

    cfg = cfg or EclConfig()
    m = _ingredients(inputs, report)
    if m is None or len(m) == 0:
        return pd.DataFrame()
    bal = pd.to_numeric(m.get("on_balance"), errors="coerce")
    cnet = pd.to_numeric(m.get("collateral_net"), errors="coerce").fillna(0.0)
    m["lgd"] = [compute_lgd(b, c, base=cfg.lgd_base,
                            unsecured_floor=cfg.lgd_unsecured_floor,
                            zero_exposure_lgd=cfg.zero_exposure_lgd)
                for b, c in zip(bal, cnet)]
    return m


def factor_attribution_exact(inputs_a, report_a, inputs_b, report_b,
                             cfg=None) -> dict:
    """The same decomposition, through the ENGINE rather than a product.

    ``ecl_factor_attribution`` multiplies exposure by PD by LGD, which is not
    what the engine does, so its effects have to be rescaled to fit. This one
    reprices each contract five times, substituting one ingredient at a time:

        f0  the prior run, as it was
        f1  + the current horizon          -> Horizon
        f2  + the current EAD curve        -> EAD
        f3  + the current PD curve         -> PD
        f4  + the current LGD and EIR      -> LGD

    Each effect is the difference between consecutive repricings, so they sum
    to closing minus opening by construction and the residual is zero. The
    order is fixed, as in the walk: a different order moves the interaction
    terms between the factors.

    Opening and closing are the REPRICED totals over the covered contracts,
    not the reports' headline provision. They exclude Stage 3 -- which the
    engine books at outstanding rather than from a curve -- and the exposure
    cap, so they will not equal the reports and are not meant to.

    A contract that cannot be priced in BOTH runs is not forced into a factor:
    its whole move goes to ``uncovered``, reported separately, because an
    unattributable change hidden inside "PD" is worse than one that says so.
    """
    from ..engine import sum_marginal_ecl

    if inputs_a is None or inputs_b is None:
        return {}
    a = _priced(inputs_a, report_a, cfg)
    b = _priced(inputs_b, report_b, cfg)
    if len(a) == 0 or len(b) == 0:
        return {}
    m = a.merge(b, on="contract", suffixes=("_a", "_b"))
    if len(m) == 0:
        return {}

    eff = dict.fromkeys(_EXACT_ORDER, 0.0)
    opening = closing = uncovered = 0.0
    covered = 0

    def curve(inputs, row: dict, side: str):
        """The EAD curve the engine would use, parametric fallback included."""
        stage = row.get(f"stage_{side}")
        stage = int(stage) if pd.notna(stage) else 1
        base = {"on_balance": row.get(f"on_balance_{side}"),
                "months_to_mat": row.get(f"months_to_mat_{side}"),
                "payment_type": row.get(f"payment_type_{side}"),
                "payment_frequency": row.get(f"payment_frequency_{side}"),
                "deferral": row.get(f"deferral_{side}")}
        return inputs.ead_curve(row["contract"], stage, base), stage

    for row in m.itertuples(index=False):
        d = row._asdict()
        ca = inputs_a.pd_curves.get(d.get("pd_key_a"))
        cb = inputs_b.pd_curves.get(d.get("pd_key_b"))
        ea, _ = curve(inputs_a, d, "a")
        eb, _ = curve(inputs_b, d, "b")
        move = float(d.get("ecl_b") or 0) - float(d.get("ecl_a") or 0)
        la, lb = d.get("lgd_a"), d.get("lgd_b")
        if (ca is None or cb is None or ea is None or eb is None
                or la is None or lb is None or pd.isna(la) or pd.isna(lb)):
            uncovered += move
            continue

        ha, hb = len(ea), len(eb)
        ia, ib = d.get("eir_a"), d.get("eir_b")
        ia = 0.0 if ia is None or pd.isna(ia) else float(ia)
        ib = 0.0 if ib is None or pd.isna(ib) else float(ib)

        f = [sum_marginal_ecl(ea, la, ca, ia, ha),
             sum_marginal_ecl(ea, la, ca, ia, hb),
             sum_marginal_ecl(eb, la, ca, ia, hb),
             sum_marginal_ecl(eb, la, cb, ia, hb),
             sum_marginal_ecl(eb, lb, cb, ib, hb)]
        if any(not np.isfinite(v) for v in f):
            uncovered += move
            continue

        for k, name in enumerate(_EXACT_ORDER):
            eff[name] += f[k + 1] - f[k]
        opening += f[0]
        closing += f[4]
        covered += 1

    if covered == 0:
        return {}
    return {
        "effects": pd.DataFrame({"factor": list(eff),
                                 "effect": list(eff.values())}),
        "covered": covered, "uncovered": uncovered,
        "opening": opening, "closing": closing, "contracts": len(m),
        "residual": (opening + sum(eff.values())) - closing,
    }


def factor_attribution_diagnosis(inputs_a, report_a, inputs_b,
                                 report_b) -> str | None:
    """Why the exact attribution produced nothing.

    An empty chart is not an answer. Every case here was a real one: a run
    whose Output folder predates a file the attribution needs, two runs with
    no contract in common, and a book whose PD buckets did not resolve.
    """
    def unusable(x, which: str) -> str | None:
        if x is None:
            return f"No engine inputs could be read for the {which} run."
        if not getattr(x, "ok", False):
            missing = ", ".join(getattr(x, "missing", []) or ["required files"])
            return (f"The {which} run is missing {missing} in "
                    f"{getattr(x, 'out_dir', 'its Output folder')}.")
        return None

    for x, which in ((inputs_a, "prior"), (inputs_b, "current")):
        msg = unusable(x, which)
        if msg:
            return msg

    a = _priced(inputs_a, report_a)
    b = _priced(inputs_b, report_b)
    if len(a) == 0 or len(b) == 0:
        return "The ECL report could not be matched to the engine inputs."
    common = set(a["contract"]) & set(b["contract"])
    if not common:
        return "No contract appears in both runs, so nothing can be attributed."

    a_no_pd = int(a["pd_key"].isna().sum()) if "pd_key" in a.columns else 0
    b_no_pd = int(b["pd_key"].isna().sum()) if "pd_key" in b.columns else 0
    a_no_ead = int((~a["contract"].isin(inputs_a.ead_curves)).sum())
    b_no_ead = int((~b["contract"].isin(inputs_b.ead_curves)).sum())
    return (f"{len(common):,} contracts are in both runs, but none could be "
            f"priced. Prior run: {a_no_pd:,} without a resolved PD bucket, "
            f"{a_no_ead:,} without a supplied EAD curve. Current run: "
            f"{b_no_pd:,} without a resolved PD bucket, {b_no_ead:,} without "
            f"a supplied EAD curve.")
