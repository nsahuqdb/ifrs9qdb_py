"""
Analytics over a completed run.

Pure functions on a normalised ECL report. Nothing here touches a UI, so the
same code serves the API, a notebook and a scheduled job.

Two conventions run through the whole module and are easy to get wrong:

  * Staging and rating are CUSTOMER attributes at QDB. A contract-level count
    over-weights customers holding many facilities -- Stage 2 is 35% of
    customers but 67% of contracts on a typical book. Anything that reports
    stage or rating aggregates to the customer first.
  * The internal and external rating scales reuse hierarchy numbers 1-21, so
    hierarchy 10 means a different grade on each. They are never combined.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

__all__ = [
    "normalise",
    "customer_view",
    "run_profile",
    "staging_distribution",
    "stage2_triggers",
    "stage2_reasons",
    "effective_pd",
    "classify_stage",
    "concentration",
    "lorenz_curve",
    "hhi",
    "hhi_band",
    "hhi_equivalent_n",
    "top_contributors",
    "data_quality",
    "maturity_profile",
    "vintage_profile",
    "dpd_profile",
    "exposure_bands",
    "collateral_bands",
    "lgd_distribution",
    "pd_distribution",
    "pd_by_rating",
]

# Report column -> analytics column. Matching is done on a squashed name
# (lowercase, punctuation removed) because the same field appears as
# "Contract Id", "ContractId" and "contract_id" depending on where it came from.
_COLUMNS = {
    "contractid": "contract",
    "customerid": "customer",
    "portfoliocode": "portfolio",
    "accounttype": "account_type",
    "rating": "rating",
    "ratingtype": "rating_type",
    "ifrsstage": "stage",
    "exposureonbal": "exposure",
    "exposureoffbal": "exposure_off",
    "ead": "ead",
    "claamountonbal": "ecl",
    # an overlaid report's post-model adjustment, per contract (zero or absent
    # on a model report); the attribution separates it from the model move
    "overlayamount": "overlay",
    "pdlifetimevalue": "pd",
    "pd12m": "pd12",
    "lgdrate": "lgd",
    "residlgdrate": "resid_lgd",
    "collcov": "collcov",
    "collateralvalue": "collateral",
    "eir": "eir",
    "pastduedays": "dpd",
    "mob": "mob",
    "defaultflag": "default_flag",
    "watchlistflag": "watchlist",
    "insolvencyflag": "insolvency",
    "defaultingccflag": "default_gcc",
    "localflag1": "restructured",
    "localflag2": "local2",
    "localflag3": "local3",
    "localflag4": "local4",
    "localflag5": "local5",
    "localflag6": "local6",
    "maturitydate": "maturity_date",
    "opendate": "open_date",
    "extractdate": "extract_date",
}


def _squash(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a LIC-format report into the frame the analytics work on.

    Missing columns are created empty rather than raising: LIC populates
    Origination Rating, Origination PD 12M, PD 12M and Time To Expected
    Maturity only in some configurations, and an analysis that needs them
    should say so itself rather than failing here.
    """
    from ..inputs import as_id

    if df is None or len(df) == 0:
        return pd.DataFrame()
    out = pd.DataFrame(index=df.index)
    lookup = {_squash(c): c for c in df.columns}
    for squashed, target in _COLUMNS.items():
        src = lookup.get(squashed)
        out[target] = df[src] if src is not None else np.nan

    for c in ("stage", "exposure", "exposure_off", "ead", "ecl", "overlay", "pd", "pd12",
              "lgd", "resid_lgd", "collcov", "collateral", "eir", "dpd", "mob",
              "default_flag", "watchlist", "insolvency", "restructured",
              "default_gcc", "local2", "local3", "local4", "local5", "local6"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("portfolio", "account_type", "rating", "rating_type"):
        out[c] = out[c].astype("string")
    # Ids are join keys, so they go through the same normalisation the engine
    # inputs use. A column of whole numbers reads as float the moment one value
    # is blank, and "548840.0" never matches "548840" - silently.
    for c in ("contract", "customer"):
        out[c] = as_id(out[c]).astype("string")

    out = out[out["contract"].notna() & (out["contract"].str.len() > 0)].copy()
    out["exposure"] = out["exposure"].fillna(0.0)
    out["ecl"] = out["ecl"].fillna(0.0)
    out["overlay"] = out["overlay"].fillna(0.0)
    out["coverage"] = np.where(out["exposure"] > 0,
                               out["ecl"] / out["exposure"].replace(0, np.nan), 0.0)
    out["coverage"] = out["coverage"].fillna(0.0)

    mat = pd.to_datetime(out["maturity_date"], errors="coerce", format="mixed")
    ext = pd.to_datetime(out["extract_date"], errors="coerce", format="mixed")
    if ext.notna().any():
        ref = ext
    else:
        ref = pd.Series(mat.max(), index=out.index)
    out["months_to_mat"] = (mat - ref).dt.days / 30.4375
    out["open_year"] = pd.to_datetime(out["open_date"], errors="coerce",
                                      format="mixed").dt.year
    return out.reset_index(drop=True)


# ------------------------------------------------------------- customer ----
def customer_view(d: pd.DataFrame) -> pd.DataFrame:
    """One row per customer.

    Stage, rating, DPD and the flags are customer attributes, taken as the
    worst across the customer's facilities; exposure and ECL are summed. The
    portfolio is only named when the customer genuinely sits in one -- naming
    the first of several was misleading, since a customer commonly holds both
    Business Finance and Off BS facilities.
    """
    if d is None or len(d) == 0:
        return pd.DataFrame()
    d = d.copy()
    d["customer"] = d["customer"].fillna("(unknown)")

    def portfolios(s):
        u = sorted({x for x in s.dropna().tolist() if str(x)})
        if not u:
            return "(unassigned)"
        return u[0] if len(u) == 1 else f"{len(u)} portfolios"

    g = d.groupby("customer", dropna=False)
    out = pd.DataFrame({
        "contracts": g["contract"].size(),
        "portfolio": g["portfolio"].apply(portfolios),
        "portfolios": g["portfolio"].apply(
            lambda s: ", ".join(sorted({str(x) for x in s.dropna() if str(x)}))),
        "rating": g["rating"].first(),
        "stage": g["stage"].max(),
        "dpd": g["dpd"].max(),
        "watchlist": g["watchlist"].max(),
        "restructured": g["restructured"].max(),
        "default_flag": g["default_flag"].max(),
        "insolvency": g["insolvency"].max(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
    }).reset_index()
    out["coverage"] = np.where(out["exposure"] > 0,
                               100 * out["ecl"] / out["exposure"], np.nan)
    return out.sort_values("ecl", ascending=False).reset_index(drop=True)


def run_profile(d: pd.DataFrame, by: str = "portfolio") -> pd.DataFrame:
    """Contracts, exposure, ECL and coverage by segment."""
    if d is None or len(d) == 0:
        return pd.DataFrame()
    g = d.assign(_g=d[by].fillna("(unassigned)")).groupby("_g")
    out = pd.DataFrame({
        "contracts": g["contract"].size(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
    }).reset_index().rename(columns={"_g": "group"})
    out["coverage"] = np.where(out["exposure"] > 0,
                               100 * out["ecl"] / out["exposure"], np.nan)
    return out.sort_values("exposure", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------- staging ----
def classify_stage(
    dpd, default_flag=None, watchlist=None, local_any=None, portfolio=None,
    customer=None, dpd_threshold: float = 60, contagion: bool = True,
    tasdeer_collective: bool = True,
) -> np.ndarray:
    """Re-apply the staging rule.

        Stage 3  DPD > 90, or the default flag
        Stage 2  threshold < DPD <= 90, or watchlist, or ANY local flag,
                 or the portfolio is Tasdeer (collectively assessed)
        then     contagion: a customer with any Stage 2 or worse facility has
                 its Stage 1 facilities bumped, Tasdeer excluded

    Every mask is forced to a real boolean. The R original builds its contagion
    mask arithmetically and throws on an NA subscript when a customer's group
    cannot be resolved -- a failure that blanked four screens before it was
    found.
    """
    n = max(len(dpd) if dpd is not None else 0,
            len(customer) if customer is not None else 0,
            len(portfolio) if portfolio is not None else 0)
    if n == 0:
        return np.array([], dtype=int)

    def fit(v, fill):
        if v is None:
            return pd.Series([fill] * n)
        s = pd.Series(list(v))
        return s if len(s) == n else s.reindex(range(n), fill_value=fill)

    dpd = pd.to_numeric(fit(dpd, np.nan), errors="coerce")
    default_flag = pd.to_numeric(fit(default_flag, np.nan), errors="coerce")
    watchlist = pd.to_numeric(fit(watchlist, np.nan), errors="coerce")
    local_any = fit(local_any, False).fillna(False).astype(bool)
    portfolio = fit(portfolio, "").astype(str)
    customer = fit(customer, "(unknown)").astype(str).replace("", "(unknown)")

    st = np.ones(n, dtype=int)
    s2 = ((dpd > dpd_threshold) & (dpd <= 90)).fillna(False).to_numpy() \
        | (watchlist == 1).fillna(False).to_numpy() | local_any.to_numpy()
    st[s2] = 2
    if tasdeer_collective:
        st[(portfolio == "Tasdeer").to_numpy()] = 2
    s3 = (dpd > 90).fillna(False).to_numpy() | (default_flag == 1).fillna(False).to_numpy()
    st[s3] = 3

    if contagion:
        worst = pd.Series(st).groupby(customer.to_numpy()).transform("max").to_numpy()
        bump = (st == 1) & (worst >= 2) & (portfolio != "Tasdeer").to_numpy()
        st[bump] = 2
    return st


def staging_distribution(d: pd.DataFrame) -> pd.DataFrame:
    """Stage split by customer AND by contract.

    Both are shown because they tell different stories, and quoting the
    contract figure as if it were the customer figure overstates Stage 2.
    """
    if d is None or len(d) == 0:
        return pd.DataFrame()
    cv = customer_view(d)
    rows = []
    total_ecl = max(cv["ecl"].sum(), 1)
    for st in sorted(set(cv["stage"].dropna()) | set(d["stage"].dropna())):
        c = cv[cv["stage"] == st]
        k = d[d["stage"] == st]
        exp, ecl = c["exposure"].sum(), c["ecl"].sum()
        rows.append({
            "stage": f"Stage {int(st)}",
            "customers": len(c), "contracts": len(k),
            "exposure": exp, "ecl": ecl,
            "coverage": 100 * ecl / exp if exp > 0 else np.nan,
            "pct_customers": 100 * len(c) / max(len(cv), 1),
            "pct_ecl": 100 * ecl / total_ecl,
        })
    return pd.DataFrame(rows)


def _stage_override_set(stage_overrides, stage: str) -> set:
    """Customers a reviewed override put in ``stage`` ("Stage 2" / "Stage 3").

    Takes the run's ``overrides/stage_overrides.csv`` as a frame (customer_id,
    value) or a {customer: value} dict; None for none."""
    if stage_overrides is None:
        return set()
    if isinstance(stage_overrides, pd.DataFrame):
        if len(stage_overrides) == 0 or "customer_id" not in stage_overrides:
            return set()
        pairs = zip(stage_overrides["customer_id"], stage_overrides["value"])
    else:
        pairs = dict(stage_overrides).items()
    want = re.sub(r"[^0-9]", "", stage)
    return {str(c) for c, v in pairs
            if re.sub(r"[^0-9]", "", str(v)) == want}


def stage2_reasons(d: pd.DataFrame, dpd_threshold: float = 60,
                   stage_overrides=None) -> pd.DataFrame:
    """One row per Stage 2 contract, one True/False column per reason.

    The reasons are the ones the staging actually acts on, named:

        Tasdeer (collective)  the portfolio is assessed as Stage 2 outright
        DPD <thr+1>-90        this contract is past due beyond the threshold
        Watchlist             the customer is on the watchlist
        Restructured          Local Flag 1 -- QDB's restructuring flag,
                              including restructuring overrides
        Stage override        a reviewed override put the customer in Stage 2
        Another facility of the customer
                              nothing on this contract, but another facility
                              of the same customer is in Stage 2 or 3

    Local Flag 3 is NOT a reason of its own. The ETL sets it on every customer
    its rule puts in Stage 2 (watchlist, restructured, worst DPD over the
    threshold) and on every Stage 2 override, so it restates the outcome.
    What it adds is the override: a customer carrying it with none of the
    rule's triggers is there by override, which is how an override is found
    when the run's override file is not to hand. Local Flags 2, 4, 5 and 6
    are blank or FALSE in QDB's extract.
    """
    if d is None or len(d) == 0:
        return pd.DataFrame()
    st = pd.to_numeric(d["stage"], errors="coerce")
    s2 = d[st == 2]
    if len(s2) == 0:
        return pd.DataFrame()

    def flag(frame, key):
        if key not in frame.columns:
            return pd.Series(False, index=frame.index)
        return (pd.to_numeric(frame[key], errors="coerce") == 1).fillna(False)

    cust = d["customer"].astype(str)
    dpd_all = pd.to_numeric(d["dpd"], errors="coerce")
    worst_dpd = dpd_all.groupby(cust).transform("max").reindex(s2.index)
    worst_stage = st.groupby(cust).transform("max").reindex(s2.index)

    dpd = dpd_all.reindex(s2.index)
    t_tas = (s2["portfolio"].astype(str) == "Tasdeer")
    t_dpd = ((dpd > dpd_threshold) & (dpd <= 90)).fillna(False)
    t_watch = flag(s2, "watchlist")
    t_restr = flag(s2, "restructured")
    # the ETL's own Stage 2 reasons for the customer: what Local Flag 3
    # restates, so a Local Flag 3 beyond them is an override
    rule = (t_watch | t_restr
            | ((worst_dpd > dpd_threshold) & (worst_dpd <= 90)).fillna(False))
    in_file = s2["customer"].astype(str).isin(
        _stage_override_set(stage_overrides, "Stage 2"))
    t_ovr = in_file | (flag(s2, "local3") & ~rule)
    own = t_tas | t_dpd | t_watch | t_restr | t_ovr
    t_cust = ~own & (worst_stage >= 2).fillna(False)

    return pd.DataFrame({
        "Tasdeer (collective)": t_tas.to_numpy(),
        f"DPD {int(dpd_threshold) + 1}-90": t_dpd.to_numpy(),
        "Watchlist": t_watch.to_numpy(),
        "Restructured": t_restr.to_numpy(),
        "Stage override": t_ovr.to_numpy(),
        "Another facility of the customer": t_cust.to_numpy(),
    }, index=s2.index)


def stage2_triggers(d: pd.DataFrame, dpd_threshold: float = 60,
                    stage_overrides=None) -> pd.DataFrame:
    """Why each Stage 2 contract is in Stage 2, reason by reason.

    A contract counts under every reason it has, so the shares add to more
    than 100%; the trigger combinations (stage2_trigger_overlap) are the
    additive view. Contract level, because Tasdeer and contagion act per
    contract. A last row counts any Stage 2 contract the report gives no
    reason for -- normally none.
    """
    r = stage2_reasons(d, dpd_threshold, stage_overrides)
    if len(r) == 0:
        return pd.DataFrame()
    s2 = d.loc[r.index]
    total_ecl = max(s2["ecl"].sum(), 1)

    def row(label, sel):
        sel = pd.Series(np.asarray(sel, dtype=bool), index=s2.index)
        return {
            "trigger": label,
            "contracts": int(sel.sum()),
            "customers": int(s2.loc[sel, "customer"].nunique()),
            "pct_contracts": 100 * sel.sum() / len(s2),
            "exposure": s2.loc[sel, "exposure"].sum(),
            "ecl": s2.loc[sel, "ecl"].sum(),
            "pct_ecl": 100 * s2.loc[sel, "ecl"].sum() / total_ecl,
        }

    rows = [row(c, r[c]) for c in r.columns]
    none = ~r.any(axis=1)
    if none.any():
        rows.append(row("No reason in the report", none))
    return pd.DataFrame(rows)


# -------------------------------------------------------- concentration ----
def hhi(d: pd.DataFrame, by: str = "customer") -> float:
    """Herfindahl-Hirschman index of ECL concentration, 0 to 10,000."""
    if d is None or len(d) == 0:
        return float("nan")
    v = d[d["ecl"] > 0].groupby(by)["ecl"].sum()
    if len(v) == 0 or v.sum() <= 0:
        return float("nan")
    share = v / v.sum()
    return float((share ** 2).sum() * 10000)


def hhi_equivalent_n(h: float) -> float:
    """Number of equally sized exposures giving the same concentration.

    Far easier to read than the index: an HHI of 144 is "about 69 equal names".
    """
    if h is None or not np.isfinite(h) or h <= 0:
        return float("nan")
    return 10000 / h


def hhi_band(h: float) -> dict:
    """Interpret an HHI.

    The 1,500 and 2,500 cut-offs are the competition-authority convention, a
    market reference rather than a regulatory limit for a loan book. Single
    obligor and large-exposure limits are not tested here.
    """
    if h is None or not np.isfinite(h):
        return {"band": "-", "tone": "muted", "note": "Not enough data."}
    if h < 1500:
        return {"band": "Diversified", "tone": "ok",
                "note": "Below 1,500. No single name dominates the provision."}
    if h < 2500:
        return {"band": "Moderately concentrated", "tone": "warn",
                "note": "Between 1,500 and 2,500. A few names carry a noticeable share."}
    return {"band": "Highly concentrated", "tone": "err",
            "note": "Above 2,500. The provision depends on a small number of names."}


def concentration(d: pd.DataFrame, tops=(10, 25, 50, 100),
                  level: str = "customer") -> pd.DataFrame:
    """Share of the provision held by the largest N.

    Customer is the right unit: one borrower with ten facilities is a single
    exposure, not ten.
    """
    if d is None or len(d) == 0:
        return pd.DataFrame()
    if level == "customer":
        e = customer_view(d)["ecl"]
    else:
        e = d["ecl"]
    e = np.sort(e[e > 0].to_numpy())[::-1]
    if e.size == 0:
        return pd.DataFrame()
    total = e.sum()
    rows = [{"top_n": n, "ecl": e[:n].sum(),
             "share": 100 * e[:n].sum() / total}
            for n in tops if n <= e.size]
    return pd.DataFrame(rows)


def lorenz_curve(d: pd.DataFrame, points: int = 60,
                 level: str = "customer") -> pd.DataFrame:
    """Cumulative share of the provision held by the largest exposures."""
    if d is None or len(d) == 0:
        return pd.DataFrame()
    e = (customer_view(d)["ecl"] if level == "customer" else d["ecl"])
    e = np.sort(e[e > 0].to_numpy())[::-1]
    if e.size < 2:
        return pd.DataFrame()
    cum = np.cumsum(e) / e.sum()
    idx = np.unique(np.linspace(0, e.size - 1, min(points, e.size)).astype(int))
    return pd.DataFrame({
        "pct_contracts": 100 * (idx + 1) / e.size,
        "pct_ecl": 100 * cum[idx],
    })


def top_contributors(d: pd.DataFrame, n: int = 25,
                     level: str = "customer") -> pd.DataFrame:
    """Largest single contributors to the provision."""
    if d is None or len(d) == 0:
        return pd.DataFrame()
    src = customer_view(d) if level == "customer" else d
    src = src[src["ecl"] > 0].nlargest(n, "ecl").copy()
    total = max((customer_view(d) if level == "customer" else d)["ecl"].clip(lower=0).sum(), 1)
    src["share"] = 100 * src["ecl"] / total
    return src.reset_index(drop=True)


# ------------------------------------------------------------ dimensions ----
def _banded(d, col, breaks, labels, extra=None):
    """Band a column into labelled buckets, losing nothing at either end.

    The bins are half-open, ``[lo, hi)``, so a value sitting exactly ON the
    final edge falls outside every bin and is silently dropped. That is not
    hypothetical: LGD is capped at 1.0 and the top edge is 1.0, so the 114
    zero-exposure contracts at exactly 1.0 vanished from the most severe
    bucket — the one anybody reading the chart is looking for. The last edge
    is nudged up by one float so the maximum lands in the last bin; the labels
    are the lower edges, so they are unaffected.
    """
    if d is None or len(d) == 0 or d[col].isna().all():
        return pd.DataFrame()
    breaks = np.asarray(breaks, dtype=float).copy()
    if np.isfinite(breaks[-1]):
        breaks[-1] = np.nextafter(breaks[-1], np.inf)
    b = pd.cut(d[col], bins=breaks, labels=labels, right=False,
               include_lowest=True)
    g = d.assign(_b=b).groupby("_b", observed=False)
    out = pd.DataFrame({
        "contracts": g["contract"].size(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
    }).reset_index().rename(columns={"_b": "band"})
    out["coverage"] = np.where(out["exposure"] > 0,
                               100 * out["ecl"] / out["exposure"], np.nan)
    return out


def maturity_profile(d):
    """Run-off: exposure and ECL by remaining maturity."""
    return _banded(d, "months_to_mat",
                   [-np.inf, 0, 3, 6, 12, 24, 36, 60, 120, np.inf],
                   ["Past due/matured", "0-3m", "3-6m", "6-12m", "1-2y",
                    "2-3y", "3-5y", "5-10y", "10y+"])


def dpd_profile(d):
    """Delinquency: coverage should climb steeply across the buckets."""
    return _banded(d, "dpd", [-np.inf, 1, 30, 60, 90, 180, 360, np.inf],
                   ["Current", "1-29", "30-59", "60-89", "90-179",
                    "180-359", "360+"])


def exposure_bands(d):
    """Ticket size: many small facilities, or a few large ones."""
    return _banded(d, "exposure",
                   [-np.inf, 0, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, np.inf],
                   ["Zero/negative", "<100k", "100k-500k", "500k-1m",
                    "1m-5m", "5m-10m", "10m-50m", ">50m"])


def collateral_bands(d):
    """How secured the book is. Over 100% still sits on the LGD floor."""
    if d is None or len(d) == 0:
        return pd.DataFrame()
    x = d.assign(collcov=d["collcov"].fillna(0.0))
    return _banded(x, "collcov", [0, 0.0001, 0.25, 0.5, 0.75, 1, np.inf],
                   ["Unsecured", "0-25%", "25-50%", "50-75%", "75-100%",
                    "Over 100%"])


def vintage_profile(d: pd.DataFrame) -> pd.DataFrame:
    """By origination year: whether a particular year's lending performs worse."""
    if d is None or len(d) == 0 or d["open_year"].isna().all():
        return pd.DataFrame()
    x = d[d["open_year"].between(1950, 2100)]
    g = x.groupby("open_year")
    out = pd.DataFrame({
        "contracts": g["contract"].size(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
    }).reset_index().rename(columns={"open_year": "vintage"})
    out["coverage"] = np.where(out["exposure"] > 0,
                               100 * out["ecl"] / out["exposure"], np.nan)
    return out.sort_values("vintage").reset_index(drop=True)


def effective_pd(d: pd.DataFrame) -> pd.Series:
    """The lifetime PD a contract is provisioned on: 100% in Stage 3.

    A Stage 3 contract has defaulted, so its PD is 1 and its provision is the
    full outstanding. The report's "PD Lifetime Value" column still carries
    the rating curve's lifetime PD for those rows -- LIC writes it, the
    provision does not use it -- and averaging that figure made Stage 3 look
    like a PD of about 0.8.
    """
    p = pd.to_numeric(d["pd"], errors="coerce")
    if "stage" in d.columns:
        p = p.where(pd.to_numeric(d["stage"], errors="coerce") != 3, 1.0)
    return p


def pd_distribution(d: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if d is None or len(d) == 0 or d["pd"].isna().all():
        return pd.DataFrame()
    edges = np.linspace(0, 1, bins + 1)
    labels = [f"{100*edges[i]:.0f}-{100*edges[i+1]:.0f}%" for i in range(bins)]
    return _banded(d.assign(pd=effective_pd(d).clip(0, 1)), "pd", edges, labels)


def pd_by_rating(d: pd.DataFrame) -> pd.DataFrame:
    """PD and coverage by rating.

    Coverage should rise as the rating worsens; a kink usually means a rating
    or curve mapping is wrong rather than a real risk pattern.
    """
    if d is None or len(d) == 0:
        return pd.DataFrame()
    x = d[d["rating"].notna()].assign(pd=lambda f: effective_pd(f))
    if len(x) == 0:
        return pd.DataFrame()
    g = x.groupby("rating")

    def wmean(s):
        w = x.loc[s.index, "exposure"]
        ok = s.notna() & (w > 0)
        return float((s[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan

    out = pd.DataFrame({
        "contracts": g["contract"].size(),
        "exposure": g["exposure"].sum(),
        "ecl": g["ecl"].sum(),
        "pd_weighted": g["pd"].apply(wmean),
    }).reset_index().rename(columns={"rating": "group"})
    out["coverage"] = np.where(out["exposure"] > 0,
                               100 * out["ecl"] / out["exposure"], np.nan)
    return out.sort_values("group").reset_index(drop=True)


def lgd_distribution(d: pd.DataFrame, bins: int = 16) -> pd.DataFrame:
    if d is None or len(d) == 0 or d["lgd"].isna().all():
        return pd.DataFrame()
    hi = max(1.0, float(np.nanmax(d["lgd"])))
    edges = np.linspace(0, hi, bins + 1)
    labels = [f"{edges[i]:.2f}" for i in range(bins)]
    return _banded(d, "lgd", edges, labels)


# ------------------------------------------------------------- quality -----
def _quality_checks(d: pd.DataFrame) -> list[tuple]:
    """The checks themselves: (label, mask, severity, note).

    One list, read by both the summary and the detail. Written twice they
    drift, and a summary that counts one thing while the drill-down lists
    another is worse than either alone.
    """
    return [
        ("Exposure but zero ECL",
         (d["exposure"] > 0) & (d["ecl"] == 0) & (d["stage"] != 3), "warn",
         "Stage 1 or 2 with a balance but no provision - usually a missing PD "
         "curve or collateral that could not be resolved."),
        ("Negative or zero exposure", d["exposure"] <= 0, "info",
         "EAD is zero for these, so ECL is zero by construction."),
        ("ECL exceeds exposure", (d["ecl"] > d["exposure"]) & (d["exposure"] > 0),
         "warn",
         "The engine caps this, so rows here mean the cap is off or the run "
         "predates it. The usual cause is an EAD curve carrying expected "
         "drawdowns on an undrawn commitment."),
        ("Missing rating",
         d["rating"].isna() | (d["rating"].astype("string").fillna("").str.strip() == ""),
         "error",
         "No rating means no PD bucket, so no ECL can be computed."),
        ("Missing lifetime PD", d["pd"].isna() & (d["stage"] != 3), "warn",
         "Stage 1 or 2 with no lifetime PD reported."),
        ("Missing LGD", d["lgd"].isna() & (d["stage"] != 3), "info",
         "LIC blanks Stage 2 LGD by design; our report populates it."),
        ("Collateral coverage missing", d["collcov"].isna(), "warn",
         "Often an allocation pointing at a collateral record that does not exist."),
        ("Collateral over 100%", d["collcov"].notna() & (d["collcov"] > 1), "info",
         "Over-collateralised; the LGD floor applies."),
        ("Implausible months on book",
         d["mob"].notna() & ((d["mob"] < 0) | (d["mob"] > 1200)), "error",
         "A months-on-book outside a sensible range points at a bad open date."),
    ]


def data_quality(d: pd.DataFrame) -> pd.DataFrame:
    """Conditions that silently distort a provision. Reported, never corrected."""
    if d is None or len(d) == 0:
        return pd.DataFrame()
    n = len(d)
    checks = _quality_checks(d)
    rows = []
    for label, sel, severity, note in checks:
        sel = sel.fillna(False)
        if not sel.any():
            continue
        rows.append({
            "check": label, "contracts": int(sel.sum()),
            "pct": 100 * sel.sum() / n,
            "exposure": d.loc[sel, "exposure"].sum(),
            "ecl": d.loc[sel, "ecl"].sum(),
            "severity": severity, "note": note,
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    order = {"error": 0, "warn": 1, "info": 2}
    return out.sort_values(["severity", "contracts"],
                           key=lambda s: s.map(order) if s.name == "severity" else -s
                           ).reset_index(drop=True)


def data_quality_detail(d: pd.DataFrame, n: int = 200) -> dict:
    """The contracts behind each data-quality finding.

    A count is an argument; a list of contract ids is something somebody can
    act on. Largest exposure first and capped at ``n``, because the point is to
    give an analyst somewhere to start, not to export the book.

    Only checks that actually fired appear, matching the summary.
    """
    if d is None or len(d) == 0:
        return {}
    cols = [c for c in ("contract", "customer", "portfolio", "rating", "stage",
                        "exposure", "ecl") if c in d.columns]
    out = {}
    for label, sel, _severity, _note in _quality_checks(d):
        sel = sel.fillna(False)
        if not sel.any():
            continue
        out[label] = (d.loc[sel, cols]
                      .sort_values("exposure", ascending=False)
                      .head(n).reset_index(drop=True))
    return out


def customer_lookup(report: pd.DataFrame, customer_ids, spaces: bool = True) -> dict:
    """Look several customers up at once, and say which were not found.

    Ids arrive pasted from a spreadsheet or an email, so any of commas,
    semicolons, newlines and spaces separate them. ``spaces=False`` reads them
    as a stress's scope does (``stress.stress_scope``): only commas, semicolons
    and newlines separate them, so a customer named with spaces is one id. The
    ones that matched and the ones that did not are BOTH returned: a lookup
    that quietly drops an unknown id lets somebody conclude a name is not in
    the book when they simply mistyped it.

    PD and LGD are exposure-weighted across the customer's facilities, which is
    the only aggregation of them that means anything.
    """
    from ..ids import as_id

    if report is None or len(report) == 0 or customer_ids is None:
        return {}
    if isinstance(customer_ids, str):
        raw = customer_ids
    else:
        raw = ",".join(str(c) for c in customer_ids)
    sep = r"[,;\s]+" if spaces else r"[,;\n]+"
    ids = [t.strip() for t in re.split(sep, raw.strip()) if t.strip()]
    if not ids:
        return {}

    key = as_id(report["customer"]).astype("string")
    sub = report[key.isin(ids)]
    if len(sub) == 0:
        return {"found": pd.DataFrame(), "missing": ids}

    cv = customer_view(sub)
    expo = sub.groupby("customer")["exposure"].sum().clip(lower=1)
    for col in ("pd", "lgd"):
        if col in sub.columns:
            w = (sub["exposure"] * pd.to_numeric(sub[col], errors="coerce")
                 .fillna(0)).groupby(sub["customer"]).sum()
            cv[col] = cv["customer"].map(w / expo)
    found = cv.sort_values("exposure", ascending=False).reset_index(drop=True)
    return {"found": found,
            "missing": [i for i in ids if i not in set(found["customer"])]}
