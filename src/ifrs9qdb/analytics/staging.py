"""Staging and migration, read from the outside.

The staging rule itself lives in the engine. These look at the OUTCOME and
answer the questions asked of it at a review: what pushed these names into
Stage 2, what is actually driving Stage 3, who moved, and whether the reported
stage still agrees with the rule that produced it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .profile import (_stage_override_set, classify_stage,
                      customer_view, stage2_reasons)

__all__ = ["dpd_by_stage", "stage2_trigger_overlap", "stage3_drivers",
           "rating_migration", "migration_summary", "customer_stage_migration",
           "stage_movers", "staging_consistency"]

_DPD_BREAKS = [-np.inf, 1, 31, 61, 91, np.inf]
_DPD_LABELS = ["0", "1-30", "31-60", "61-90", "90+"]
_LOCAL_FLAGS = ("local2", "local3", "local4", "local5", "local6")


def _flag(d: pd.DataFrame, key: str) -> pd.Series:
    """A flag column as numbers, or all-missing when the report omits it."""
    if key not in d.columns:
        return pd.Series(np.nan, index=d.index)
    return pd.to_numeric(d[key], errors="coerce")


def _local_any(d: pd.DataFrame, keys=_LOCAL_FLAGS) -> pd.Series:
    out = pd.Series(False, index=d.index)
    for k in keys:
        out |= _flag(d, k) == 1
    return out


# -------------------------------------------------------------- profile ----
def dpd_by_stage(d: pd.DataFrame) -> pd.DataFrame:
    """Days past due banded, by stage, at CUSTOMER level.

    Customer level because staging is decided per customer: counting contracts
    would show one customer once per facility and overstate whichever band
    they fall in.
    """
    cv = customer_view(d)
    if cv is None or len(cv) == 0:
        return pd.DataFrame()
    band = pd.cut(pd.to_numeric(cv["dpd"], errors="coerce"),
                  bins=_DPD_BREAKS, labels=_DPD_LABELS, right=False)
    stage = pd.to_numeric(cv["stage"], errors="coerce")
    rows = []
    for s in sorted(stage.dropna().unique()):
        counts = band[stage == s].value_counts().reindex(_DPD_LABELS,
                                                         fill_value=0)
        for lab in _DPD_LABELS:
            rows.append({"stage": f"Stage {int(s)}", "band": lab,
                         "customers": int(counts[lab])})
    return pd.DataFrame(rows)


def stage2_trigger_overlap(d: pd.DataFrame, dpd_threshold: float = 60,
                           stage_overrides=None) -> pd.DataFrame:
    """Which COMBINATIONS of reasons put contracts into Stage 2.

    Distinct from the per-reason view, which double-counts: a name that is
    both watchlisted and restructured appears under each. Here it appears once,
    under "Watchlist + Restructured", so the rows are additive and the overlaps
    are visible. The reasons are stage2_reasons(), so the two views agree.
    """
    r = stage2_reasons(d, dpd_threshold, stage_overrides)
    if len(r) == 0:
        return pd.DataFrame()
    s2 = d.loc[r.index]
    names = np.array(r.columns)
    vals = r.to_numpy()
    combo = [" + ".join(names[row]) or "No reason in the report" for row in vals]

    out = (s2.assign(_c=combo).groupby("_c")
           .agg(contracts=("contract", "size"),
                customers=("customer", "nunique"),
                exposure=("exposure", "sum"), ecl=("ecl", "sum"))
           .reset_index().rename(columns={"_c": "combination"}))
    return out.sort_values("contracts", ascending=False,
                           kind="stable").reset_index(drop=True)


def stage3_drivers(d: pd.DataFrame, stage_overrides=None) -> pd.DataFrame:
    """Why each Stage 3 customer is in Stage 3. Customer level, additive.

        DPD over 90       the customer's worst facility is more than 90 days
                          past due
        Stage override    not over 90 days, but a reviewed override put the
                          customer in Stage 3

    The default flag is not counted as a driver of its own: the ETL sets it
    exactly when the customer's worst DPD exceeds 90, or when a Stage 3
    override is applied, so on its own it restates the two rows above. A
    default flag WITHOUT DPD over 90 is how an override shows when the run's
    override file is not to hand. The insolvency flag is blank in QDB's
    extract.
    """
    cv = customer_view(d)
    if cv is None or len(cv) == 0:
        return pd.DataFrame()
    s3 = cv[pd.to_numeric(cv["stage"], errors="coerce") == 3]
    if len(s3) == 0:
        return pd.DataFrame()

    over90 = (pd.to_numeric(s3.get("dpd"), errors="coerce") > 90).fillna(False)
    in_file = s3["customer"].astype(str).isin(
        _stage_override_set(stage_overrides, "Stage 3"))
    ovr = ~over90 & ((_flag(s3, "default_flag") == 1).fillna(False) | in_file)
    other = ~over90 & ~ovr

    def row(label: str, sel: pd.Series) -> dict:
        return {"driver": label, "customers": int(sel.sum()),
                "pct_customers": 100 * float(sel.sum()) / len(s3),
                "exposure": float(s3.loc[sel, "exposure"].sum()),
                "ecl": float(s3.loc[sel, "ecl"].sum())}

    rows = [row("DPD over 90", over90), row("Stage override", ovr)]
    if other.any():
        rows.append(row("No reason in the report", other))
    return pd.DataFrame(rows)


# ------------------------------------------------------------ migration ----
def rating_migration(prev: pd.DataFrame, curr: pd.DataFrame, top: int = 12,
                     levels: list[str] | None = None,
                     by_customer: bool = True) -> pd.DataFrame:
    """A from/to rating matrix between two runs, valued at current exposure.

    By CUSTOMER by default, because the rating is a customer attribute: a
    contract view counts one customer once per facility and inflates every cell
    by that customer's facility count.

    Given ``levels`` -- the rating scale in its own order -- the matrix reads
    best-to-worst and ``direction`` is a genuine up- or downgrade. Without it
    the axes fall back to frequency order and direction would only be a string
    comparison, so the scale order is worth passing whenever it is known.

    Long, one row per populated cell. ``result.attrs["levels"]`` carries the
    axis order for anything that wants to pivot it.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return pd.DataFrame()

    if by_customer:
        pv, cv = customer_view(prev), customer_view(curr)
        if pv is None or cv is None or len(pv) == 0 or len(cv) == 0:
            return pd.DataFrame()
        a = pv[["customer", "rating", "exposure"]].rename(
            columns={"customer": "key"})
        b = cv[["customer", "rating", "exposure", "ecl"]].rename(
            columns={"customer": "key"})
    else:
        a = prev.drop_duplicates("contract")[
            ["contract", "rating", "exposure"]].rename(
            columns={"contract": "key"})
        b = curr.drop_duplicates("contract")[
            ["contract", "rating", "exposure", "ecl"]].rename(
            columns={"contract": "key"})

    m = a.merge(b, on="key", suffixes=("_prev", "_curr"))
    for c in ("rating_prev", "rating_curr"):
        m[c] = m[c].astype("string").fillna("").str.strip()
    m = m[(m["rating_prev"] != "") & (m["rating_curr"] != "")]
    if len(m) == 0:
        return pd.DataFrame()

    if levels:
        m = m[m["rating_prev"].isin(levels) & m["rating_curr"].isin(levels)]
        if len(m) == 0:
            return pd.DataFrame()
        seen = set(m["rating_prev"]) | set(m["rating_curr"])
        keep = [r for r in levels if r in seen]
    else:
        counts = pd.concat([m["rating_prev"], m["rating_curr"]]).value_counts()
        keep = list(counts.head(top).index)
        m = m[m["rating_prev"].isin(keep) & m["rating_curr"].isin(keep)]
        if len(m) == 0:
            return pd.DataFrame()

    rank = {r: i for i, r in enumerate(keep)}
    agg = (m.groupby(["rating_prev", "rating_curr"], observed=True)
           .agg(n=("key", "size"), exposure=("exposure_curr", "sum"),
                ecl=("ecl", "sum"))
           .reset_index().rename(columns={"rating_prev": "from",
                                          "rating_curr": "to"}))
    r_from = agg["from"].map(rank)
    r_to = agg["to"].map(rank)
    agg["direction"] = np.where(r_from == r_to, "stable",
                                np.where(r_from < r_to, "downgrade", "upgrade"))
    agg = (agg.assign(_f=r_from, _t=r_to).sort_values(["_f", "_t"])
           .drop(columns=["_f", "_t"]).reset_index(drop=True))
    agg.attrs["levels"] = keep
    return agg


def migration_summary(prev: pd.DataFrame, curr: pd.DataFrame,
                      levels: list[str] | None = None) -> pd.DataFrame:
    """Headline counts: upgraded, stable, downgraded, with what they carry.

    Built over the WHOLE scale (``top=100``), not the twelve busiest grades:
    a summary that silently dropped the thin tails would understate exactly the
    migrations that matter.
    """
    mg = rating_migration(prev, curr, top=100, levels=levels)
    if mg is None or len(mg) == 0:
        return pd.DataFrame()
    out = (mg.groupby("direction", observed=True)[["n", "exposure", "ecl"]]
           .sum().reset_index())
    order = {"upgrade": 0, "stable": 1, "downgrade": 2}
    return (out.assign(_o=out["direction"].map(order)).sort_values("_o")
            .drop(columns="_o").reset_index(drop=True))


def customer_stage_migration(prev: pd.DataFrame,
                             curr: pd.DataFrame) -> pd.DataFrame:
    """Customer-level stage migration: a from/to matrix, in long form.

    Customers present in both runs only. One that arrived or left has no
    migration to report -- it belongs in the flow analysis.
    """
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return pd.DataFrame()
    pv, cv = customer_view(prev), customer_view(curr)
    if pv is None or cv is None or len(pv) == 0 or len(cv) == 0:
        return pd.DataFrame()
    m = pv[["customer", "stage"]].merge(
        cv[["customer", "stage", "exposure", "ecl"]], on="customer",
        suffixes=("_prev", "_curr"))
    m["stage_prev"] = pd.to_numeric(m["stage_prev"], errors="coerce")
    m["stage_curr"] = pd.to_numeric(m["stage_curr"], errors="coerce")
    m = m.dropna(subset=["stage_prev", "stage_curr"])
    if len(m) == 0:
        return pd.DataFrame()
    out = (m.groupby(["stage_prev", "stage_curr"])
           .agg(customers=("customer", "size"), exposure=("exposure", "sum"),
                ecl=("ecl", "sum"))
           .reset_index().rename(columns={"stage_prev": "from",
                                          "stage_curr": "to"}))
    out["from"] = out["from"].astype(int)
    out["to"] = out["to"].astype(int)
    return out.sort_values(["from", "to"]).reset_index(drop=True)


_MOVER_COLUMNS = ["customer", "stage_prev", "exposure_prev", "ecl_prev",
                  "stage_curr", "exposure_curr", "ecl_curr", "direction",
                  "ecl_change"]


def stage_movers(prev: pd.DataFrame, curr: pd.DataFrame) -> pd.DataFrame:
    """The customers behind the migration matrix, one row each.

    The matrix says how many moved; a review asks which ones. Sorted by the
    size of the provision move, so the names that explain the headline come
    first.
    """
    # An empty answer keeps its columns, so a caller sorting or filtering on
    # them works whether or not anybody moved.
    empty = pd.DataFrame(columns=_MOVER_COLUMNS)
    if prev is None or curr is None or len(prev) == 0 or len(curr) == 0:
        return empty
    pv, cv = customer_view(prev), customer_view(curr)
    if pv is None or cv is None or len(pv) == 0 or len(cv) == 0:
        return empty
    m = pv[["customer", "stage", "exposure", "ecl"]].merge(
        cv[["customer", "stage", "exposure", "ecl"]], on="customer",
        suffixes=("_prev", "_curr"))
    m["stage_prev"] = pd.to_numeric(m["stage_prev"], errors="coerce")
    m["stage_curr"] = pd.to_numeric(m["stage_curr"], errors="coerce")
    moved = m.dropna(subset=["stage_prev", "stage_curr"])
    moved = moved[moved["stage_prev"] != moved["stage_curr"]].copy()
    if len(moved) == 0:
        return empty
    moved["direction"] = np.where(moved["stage_curr"] > moved["stage_prev"],
                                  "deteriorated", "improved")
    moved["ecl_change"] = moved["ecl_curr"] - moved["ecl_prev"]
    return (moved.reindex(moved["ecl_change"].abs()
                          .sort_values(ascending=False).index)
            .reset_index(drop=True))


# ---------------------------------------------------------- consistency ----
def staging_consistency(d: pd.DataFrame, dpd_threshold: float = 60) -> dict:
    """Check the reported stage by RE-APPLYING the rule that produced it.

    Approximating the rule and flagging whatever does not fit produces false
    warnings -- the approximation misses Tasdeer, local flags 2 to 6, stage
    overrides and cross-facility contagion. So this recomputes the expected
    stage with the engine's own ``classify_stage`` and compares.

    A difference is not automatically a defect. The report does not carry
    ``AccountMaster.Stage``, so a manual override cannot be replayed and shows
    up here; that is the usual explanation for a contract staged WORSE than the
    rule alone implies, which is why the two directions are reported
    separately. Staged BETTER is the one worth chasing.

    Returns ``{rule_available, checked, mismatches, findings}``.
    """
    if d is None or len(d) == 0:
        return {"rule_available": False, "checked": 0, "mismatches": 0,
                "findings": pd.DataFrame()}

    local_any = _local_any(d, ("watchlist", "restructured") + _LOCAL_FLAGS)
    try:
        expected = classify_stage(
            dpd=d.get("dpd"), default_flag=_flag(d, "default_flag"),
            watchlist=_flag(d, "watchlist"), local_any=local_any,
            portfolio=d.get("portfolio"), customer=d.get("customer"),
            dpd_threshold=dpd_threshold)
    except Exception:
        return {"rule_available": False, "checked": 0, "mismatches": 0,
                "findings": pd.DataFrame()}

    actual = pd.to_numeric(d["stage"], errors="coerce")
    expected = pd.Series(expected, index=d.index)
    comparable = actual.notna() & expected.notna()
    checked = int(comparable.sum())
    differs = comparable & (actual != expected)
    base = {"rule_available": True, "checked": checked,
            "mismatches": int(differs.sum())}
    if not differs.any():
        return {**base, "findings": pd.DataFrame()}

    def finding(label: str, sel: pd.Series, severity: str, note: str) -> dict:
        return {"check": label, "contracts": int(sel.sum()),
                "customers": int(d.loc[sel, "customer"].nunique()),
                "pct": 100 * float(sel.sum()) / max(checked, 1),
                "exposure": float(d.loc[sel, "exposure"].sum()),
                "ecl": float(d.loc[sel, "ecl"].sum()),
                "severity": severity, "note": note}

    rows = []
    worse = differs & (actual > expected)
    better = differs & (actual < expected)
    if worse.any():
        rows.append(finding(
            "Staged worse than the rule alone implies", worse, "info",
            "Normally a manual Stage override. The report does not carry "
            "AccountMaster.Stage, so overrides cannot be replayed here."))
    if better.any():
        rows.append(finding(
            "Staged better than the rule implies", better, "warn",
            "The rule would put these in a worse stage. Check for an "
            "override, or a flag that did not reach staging."))
    return {**base, "findings": pd.DataFrame(rows)}
