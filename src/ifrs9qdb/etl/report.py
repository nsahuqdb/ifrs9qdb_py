"""
The final ECL report.

A column-for-column port of R/final_ecl_report.R: prices a run's LIC input
files (the Output folder) and writes the report in the 76-column LIC layout,
plus the four overlay waterfall columns R appends.

The report is built from the OUTPUT folder, not from the raw extracts: by this
point the ETL has resolved contract ids, collateral allocations and rating
buckets, and re-deriving any of that here would give two answers to the same
question.

WHY A LINE-BY-LINE PORT
    The previous version priced the same numbers as R but filled only about
    thirty of the seventy-six columns and derived several of them differently:
    Time To Maturity in months where R reports years, the rating type as 1/2
    where R writes Internal/External, LGD shown on Stage 3 rows R blanks, the
    staging DPD and flags of the INVESTMENT book looked up in the LENDING
    files, and the manual stage override in AccountMaster.Stage ignored
    altogether. A reviewer comparing the two reports saw a different report.
    Each rule below cites the R line it reproduces, so the next difference is
    a diff against R rather than an argument.

HOW R READS A COLUMN MATTERS
    R reads each output CSV with readr, which guesses a type per column: a
    column of TRUE/FALSE (or one that is entirely blank) becomes logical, a
    column of numbers becomes double, anything else character. The report's
    flag and id columns depend on that guess -- a staging flag missing for a
    customer is NA when the column is logical and 0 when it is numeric -- so
    :func:`_readr_type` reproduces it rather than letting pandas decide.
"""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..dates import calendar_months, fer_parse_date, months_between, years_between
from ..engine import (MIN_HORIZON_MONTHS, EclConfig, compute_lgd,
                      ead_fallback_rules, fallback_ead_curve, months_to_maturity,
                      resolve_ead_shape, sum_marginal_ecl)
from ..ids import as_id

__all__ = ["REPORT_COLUMNS", "OVERLAY_COLUMNS", "build_final_ecl_report",
           "classify_stage_report", "read_output_csv", "DEFAULT_PORTFOLIOS",
           "engine_frame", "ecl_frame", "EclContext", "build_ecl_context",
           "compute_ecl", "ead_path", "run_pricing_config"]

REPORT_COLUMNS = [
    "Run Id", "Enterprise Entity Id", "Extract Date", "Contract Id",
    "Account Code", "Portfolio Code", "Customer Id", "Customer Code",
    "Customer Name", "Account Type", "Past Due Days", "Lim Id", "Open Date",
    "Time From Open Date", "Maturity Date", "Expected Maturity Date",
    "Time To Expected Maturity", "Rating", "Origination Rating", "Rating Type",
    "MOB", "Ifrs Stage", "Time To Maturity", "Is Individual Assessment", "EAD",
    "EIR", "Is Initial Recognition", "PD 12M", "Origination PD 12M",
    "PD Lifetime Value", "LGD Rate", "Is POCI", "Resid Lgd Rate", "CCF",
    "Coll Cov", "Is Cla Simplified Approach", "Is Cla Loss Rate",
    "Is Cla Renewable Credit Facility", "Exposure On Bal", "Exposure Off Bal",
    "Cla Amount Onbal", "Delta Cla Amount Onbal", "Cla Amount Offbal",
    "Delta Cla Amount Offbal", "Cla Amount Principal",
    "Cla Amount Principal Overdue", "Cla Amount Interest Accrued",
    "Cla Amount Interest Overdue", "Cla Amount Fee", "Cla Amount Fee Overdue",
    "Cla Amount Penalty", "Cla Amount Penalty Overdue", "Cla Amount Commission",
    "Cla Amount Commission Overdue", "Cla Amount Other",
    "Cla Amount Other Overdue", "CLA Calculation Approach Id", "CLA Currency",
    "Impairment Coverage Off Bal", "Impairment Coverage On Bal",
    "Poci Cla Amount At Origination Offbal",
    "Poci Cla Amount At Origination Onbal", "Original Ecl Offbal",
    "Original Ecl Onbal", "Customer Organizational Unit Code",
    "Collateral Value", "Watchlist Flag", "Default Flag",
    "Default In GCC Flag", "Insolvency Flag", "Local Flag 1", "Local Flag 2",
    "Local Flag 3", "Local Flag 4", "Local Flag 5", "Local Flag 6",
]

# R's apply_overlays() always appends the waterfall, overlays or not, so the
# report has one schema.
OVERLAY_COLUMNS = ["Ecl Model Onbal", "Overlay Id", "Overlay Amount",
                   "Ecl Final Onbal"]

# R's .default_stpd_portfolios(): the portfolio -> rating type the report uses.
DEFAULT_PORTFOLIOS = {
    "Business Finance": "Internal", "Off BS": "Internal",
    "Al Dhameen": "Internal", "Tasdeer": "Internal",
    "Banks and Fis": "External", "Investments": "External",
}

_KEYSEP = " ::> "


# ------------------------------------------------------------ reading ------
_LOGICAL = {"TRUE", "FALSE", "T", "F", "true", "false", "True", "False"}


def _readr_type(raw: pd.Series) -> str:
    """The type readr would guess for a column: logical, double or character.

    readr looks at the first 1000 values; so does this.
    """
    v = raw.iloc[:1000]
    v = v[v.notna() & (v.astype(str).str.strip() != "")].astype(str).str.strip()
    if v.empty:
        return "logical"
    if v.isin(_LOGICAL).all():
        return "logical"
    if pd.to_numeric(v.str.replace(",", "", regex=False), errors="coerce").notna().all():
        return "double"
    return "character"


def read_output_csv(out_dir, name: str) -> pd.DataFrame | None:
    """One Output CSV, every column as text, or None when absent.

    Kept as text so each use site can apply readr's typing for that column;
    see :func:`_chr`, :func:`_num` and :func:`_flag01`.
    """
    for fn in (name, name + ".csv"):
        p = Path(out_dir) / fn
        if p.is_file():
            return pd.read_csv(p, dtype=str, keep_default_na=False,
                               na_values=[""], low_memory=False)
    return None


def _col(df: pd.DataFrame | None, name: str):
    """R's .fer_col(): case-insensitive, then punctuation-insensitive."""
    if df is None:
        return None
    low = [c.lower() for c in df.columns]
    if name.lower() in low:
        return df[df.columns[low.index(name.lower())]]
    sq = [re.sub(r"[^a-z0-9]", "", c) for c in low]
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key in sq:
        return df[df.columns[sq.index(key)]]
    return None


def _r_num_str(x: float) -> str:
    """as.character() of an R double: 15 significant digits, no trailing .0."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if float(x).is_integer() and abs(x) < 1e15:
        return str(int(x))
    return format(float(x), ".15g")


def _chr(raw) -> pd.Series:
    """as.character() of a readr column: numbers lose leading zeros and '.0'."""
    if raw is None:
        return None
    s = pd.Series(raw)
    t = _readr_type(s)
    if t == "double":
        x = pd.to_numeric(s.astype(str).str.replace(",", "", regex=False),
                          errors="coerce")
        return x.map(_r_num_str).where(x.notna(), None)
    if t == "logical":
        up = s.astype(str).str.strip().str.upper()
        out = pd.Series([None] * len(s), index=s.index, dtype=object)
        out[up.isin(["TRUE", "T"])] = "TRUE"
        out[up.isin(["FALSE", "F"])] = "FALSE"
        return out
    return s.where(s.notna(), None)


def _num(raw) -> pd.Series:
    if raw is None:
        return None
    return pd.to_numeric(pd.Series(raw).astype(str).str.replace(",", "", regex=False)
                         .str.strip(), errors="coerce")


def _flag01(raw, n: int) -> pd.Series:
    """R's .fer_flag01(): logical columns keep NA, anything else maps NA to 0."""
    if raw is None:
        return pd.Series([np.nan] * n, dtype=float)
    s = pd.Series(raw).reset_index(drop=True)
    t = _readr_type(s)
    up = s.astype(object).where(s.notna(), None)
    if t == "logical":
        out = pd.Series(np.nan, index=s.index)
        u = up.map(lambda v: str(v).strip().upper() if v is not None else None)
        out[u.isin(["TRUE", "T"])] = 1
        out[u.isin(["FALSE", "F"])] = 0
        return out
    xs = up.map(lambda v: str(v).strip().lower() if v is not None else None)
    out = pd.to_numeric(xs, errors="coerce").apply(
        lambda v: float(int(v)) if pd.notna(v) else np.nan)
    out[xs.isin(["true", "t", "yes", "y"])] = 1
    out[xs.isin(["false", "f", "no", "n", ""])] = 0
    return out.fillna(0)


def _flag_lookup(key_col, val_col, keys: pd.Series) -> pd.Series:
    """R's .fer_flag01(val[match(keys, key)]): look up, THEN convert.

    The order matters. A customer with no row comes back NA; a logical column
    keeps that NA, any other type turns it into 0.
    """
    n = len(keys)
    if key_col is None or val_col is None:
        return pd.Series([np.nan] * n, dtype=float)
    typed = _flag01(val_col, len(val_col))
    out = _lookup(key_col, typed, keys).astype(float)
    if _readr_type(pd.Series(val_col)) != "logical":
        out = out.fillna(0.0)
    return out


def _lookup(key_col, val_col, keys: pd.Series) -> pd.Series:
    """R's val[match(keys, key)]: the FIRST matching row, NA when none."""
    if key_col is None or val_col is None:
        return pd.Series([None] * len(keys))
    k = _chr(key_col).reset_index(drop=True)
    v = pd.Series(val_col).reset_index(drop=True)
    first = pd.Series(range(len(k)), index=k).groupby(level=0).first()
    pos = keys.reset_index(drop=True).map(first)
    out = pd.Series([None] * len(keys), dtype=object)
    hit = pos.notna()
    out[hit] = v.iloc[pos[hit].astype(int)].to_numpy()
    return out


# ------------------------------------------------------------ staging ------
def classify_stage_report(dpd, default_flag, watchlist, local_any, portfolio,
                          customer, dpd_stage2_threshold: float = 60,
                          contagion: bool = True,
                          stage_override=None) -> np.ndarray:
    """R's .fer_classify_stage(), rule for rule.

        threshold < DPD <= 90, the watchlist flag or any local flag -> Stage 2
        portfolio is Tasdeer                                    -> Stage 2
        DPD > 90 or the default flag                             -> Stage 3
        AccountMaster.Stage (a manual override, "Stage 2" or 2)  -> that stage,
            except that Stage 3 stays Stage 3
        contagion: a customer with any Stage 2 or worse facility has its
            Stage 1 facilities lifted to Stage 2, Tasdeer excluded

    Tasdeer is assessed collectively, which is why it is set outright and left
    out of the contagion sweep.
    """
    d = pd.to_numeric(pd.Series(dpd), errors="coerce").reset_index(drop=True)
    n = len(d)
    st = np.ones(n, dtype=int)
    pf = pd.Series(portfolio).reset_index(drop=True).astype(object)
    wf = pd.to_numeric(pd.Series(watchlist).reset_index(drop=True), errors="coerce")
    df_ = pd.to_numeric(pd.Series(default_flag).reset_index(drop=True), errors="coerce")
    la = pd.Series(local_any).reset_index(drop=True).fillna(False).astype(bool)

    s2 = ((d > dpd_stage2_threshold) & (d <= 90)).fillna(False) | (wf == 1) | la
    st[s2.to_numpy()] = 2
    st[(pf == "Tasdeer").to_numpy()] = 2
    s3 = (d > 90).fillna(False) | (df_ == 1)
    st[s3.to_numpy()] = 3

    if stage_override is not None:
        ov = pd.Series(stage_override).reset_index(drop=True).astype(object)
        digits = ov.map(lambda v: re.sub(r"[^0-9]", "", str(v))
                        if v is not None and not (isinstance(v, float) and math.isnan(v))
                        else "")
        ov_num = pd.to_numeric(digits, errors="coerce")
        has = ov_num.isin([1, 2, 3])
        st[has.to_numpy()] = ov_num[has].astype(int).to_numpy()
        st[s3.to_numpy()] = 3

    if contagion and customer is not None:
        cid = pd.Series(customer).reset_index(drop=True).astype(object)
        known = cid.notna()
        worst = pd.Series(st)[known].groupby(cid[known]).transform("max")
        cw = pd.Series(np.nan, index=cid.index)
        cw[known] = worst
        bump = (st == 1) & (cw >= 2).to_numpy() & (pf != "Tasdeer").to_numpy()
        st[bump] = 2
    return st


# ------------------------------------------------------------- engine ------
class EclContext:
    """R's build_ecl_context(): the segment-independent pricing lookups.

    Everything that prices a contract -- the run's report, the comparison
    bridge, every stress, what-if and roll-forward -- prices on one of these
    and :func:`compute_ecl`. A lever changes the context (``copy``) or the
    contract facts, never the answer.
    """

    def __init__(self, out_dir, model_cfg=None, stpd=None, cfg=None):
        cfg = cfg or EclConfig()
        self.cfg = cfg
        self.rules = ead_fallback_rules(model_cfg)
        ecl_node = (model_cfg or {}).get("ecl", {}) if isinstance(model_cfg, dict) else {}
        self.stage3 = str(ecl_node.get("stage3_method") or cfg.stage3_method)
        if self.stage3 not in ("full_outstanding", "zero"):
            self.stage3 = "full_outstanding"
        cap = ecl_node.get("cap_ecl_at_exposure")
        self.cap = cfg.cap_ecl_at_exposure if cap is None else bool(cap)
        self.lgd_base = float(ecl_node.get("lgd_base", cfg.lgd_base))
        self.lgd_floor = float(ecl_node.get("lgd_unsecured_floor",
                                            cfg.lgd_unsecured_floor))
        self.collnet = self._collateral(out_dir)
        self.schedules = self._schedules(out_dir)
        self.stpd = self._stpd(out_dir if stpd is None else stpd)
        self.rat2bucket = self._buckets(out_dir)

    @staticmethod
    def _collateral(out_dir) -> dict:
        """R's build_collateral_net(): value x share x (1 - haircut), summed.

        Named-vector lookups in R return the FIRST match, so a duplicated
        collateral or type id resolves to its first row here too.
        """
        alloc = read_output_csv(out_dir, "AccountCollateralAllocation.csv")
        coll = read_output_csv(out_dir, "Collateral.csv")
        ctype = read_output_csv(out_dir, "CollateralType.csv")
        if alloc is None or coll is None or ctype is None:
            return {}
        hc = pd.Series(_num(_col(ctype, "HaircutGeneral")).to_numpy(),
                       index=_chr(_col(ctype, "CollateralTypeId")).to_numpy())
        hc = hc[~hc.index.duplicated(keep="first")]
        cid = _chr(_col(coll, "CollateralId")).to_numpy()
        cval = pd.Series(_num(_col(coll, "CollateralValue")).to_numpy(), index=cid)
        cval = cval[~cval.index.duplicated(keep="first")]
        ctyp = pd.Series(_chr(_col(coll, "CollateralTypeId")).to_numpy(), index=cid)
        ctyp = ctyp[~ctyp.index.duplicated(keep="first")]
        a_cid = _chr(_col(alloc, "ContractId"))
        a_clid = _chr(_col(alloc, "CollateralId"))
        a_pct = _num(_col(alloc, "AllocationPercentage"))
        v = a_clid.map(cval).astype(float)
        h = a_clid.map(ctyp).map(hc).astype(float)
        contrib = v * a_pct * (1 - h)
        contrib[~np.isfinite(contrib)] = 0.0
        keep = a_cid.notna()
        return contrib[keep].groupby(a_cid[keep]).sum().to_dict()

    @staticmethod
    def _schedules(out_dir) -> dict:
        lt = read_output_csv(out_dir, "LifeTimeParameterOther.csv")
        if lt is None or len(lt) == 0:
            return {}
        d = pd.DataFrame({"c": _chr(_col(lt, "ContractId")),
                          "m": _num(_col(lt, "MonthLifetime")),
                          "e": _num(_col(lt, "EADLifetime"))})
        d = d[d["c"].notna()].sort_values(["c", "m"], kind="mergesort")
        return {c: g["e"].to_numpy(dtype=float) for c, g in d.groupby("c", sort=False)}

    @staticmethod
    def _stpd(src) -> dict:
        """R's build_stpd_curves(): v[m] = cumulative PD at month m, v[0] = 0."""
        if isinstance(src, pd.DataFrame):
            stpd = src.astype(str).where(src.notna(), None)
        else:
            stpd = read_output_csv(src, "StPD.csv")
        if stpd is None or len(stpd) == 0:
            return {}
        pf = _chr(_col(stpd, "PortfolioCode"))
        bk = _chr(_col(stpd, "PDBucketDim1"))
        mon = _num(_col(stpd, "MonthLifetime"))
        val = _num(_col(stpd, "PDLifetime"))
        key = pf.astype(str) + _KEYSEP + bk.astype(str)
        out = {}
        d = pd.DataFrame({"k": key, "m": mon, "v": val})
        d = d[d["m"].notna()]
        for k, g in d.groupby("k", sort=False):
            mm = g["m"].astype(int).to_numpy()
            v = np.zeros(int(mm.max()) + 1)
            v[mm] = g["v"].to_numpy(dtype=float)
            out[k] = v
        return out

    @staticmethod
    def _buckets(out_dir) -> dict:
        r = read_output_csv(out_dir, "Ratings.csv")
        if r is None:
            return {}
        key = _chr(_col(r, "RatingType")).astype(str) + _KEYSEP + \
            _chr(_col(r, "Rating")).astype(str)
        return dict(zip(key, _chr(_col(r, "Hierarchy"))))

    def copy(self, **changes) -> "EclContext":
        """A shallow copy with some lookups replaced -- a stress's context.
        The run's own lookups are never changed in place."""
        out = copy.copy(self)
        for k, v in changes.items():
            if not hasattr(out, k):
                raise AttributeError(f"EclContext has no {k!r}")
            setattr(out, k, v)
        return out

    def pd_curve(self, portfolio, rating, rating_type):
        """R's lookup_pd_curve(): the external scale (named "External" or
        "External Rating", or type 2) -> type 2, anything else -> 1."""
        t = str(rating_type)
        rt = "2" if (t.startswith("External") or t == "2") else "1"
        bucket = self.rat2bucket.get(f"{rt}{_KEYSEP}{rating}")
        if bucket is None or (isinstance(bucket, float) and math.isnan(bucket)):
            return None
        return self.stpd.get(f"{portfolio}{_KEYSEP}{bucket}")


_Ctx = EclContext       # the name the rest of the package grew up with


def build_ecl_context(out_dir, model_cfg=None, stpd=None,
                      cfg: EclConfig | None = None) -> EclContext:
    """R's build_ecl_context(): a run's pricing context, from its Output
    folder and model config (the ``ecl`` block: EAD fallback rules, Stage 3
    method, exposure cap, LGD). ``stpd`` prices on another set of PD curves
    (a scenario's, a rebuilt chain's) instead of the run's StPD.csv."""
    return EclContext(out_dir, model_cfg=model_cfg, stpd=stpd, cfg=cfg)


def _months_to_maturity(d_mat, d_ext, min_months: int = 3) -> int:
    """R's months_to_maturity(): LIC's count (a part month is a whole one;
    ``min_months`` only for a facility at or past maturity). One count for
    every caller: :func:`engine.months_to_maturity`."""
    return months_to_maturity(d_mat, d_ext, min_months)


def _pd_lifetime_at(cum, stage: int, months: int) -> float:
    if cum is None:
        return float("nan")
    h = min(12, months) if stage == 1 else months
    if h is None or h < 0:
        h = 0
    return float(cum[min(h, len(cum) - 1)])


def _ead_curve(ctx: EclContext, cid, stage, on_bal, months, payment_type,
               portfolio, nir, deferral, pay_freq):
    """R's resolve_ead_curve(): the schedule when there is one, else LIC's
    parametric fallback (:func:`engine.fallback_ead_curve`, the one
    fallback); Stage 1 capped at twelve months. Returns (curve, horizon,
    shape), shape None for a supplied schedule."""
    curve = ctx.schedules.get(cid)
    if curve is not None and len(curve) > 0:
        H = min(12, len(curve)) if stage == 1 else len(curve)
        return curve[:H], H, None
    rules, default = ctx.rules
    shape = resolve_ead_shape(payment_type, portfolio, rules=rules, default=default)
    if months is not None and months <= MIN_HORIZON_MONTHS:
        shape = "bullet"
    full = fallback_ead_curve(on_bal, months, payment_type, pay_freq, deferral,
                              portfolio=portfolio, nir=nir, rules=rules,
                              default=default)
    H = min(12, len(full)) if stage == 1 else len(full)
    return full[:H], H, shape


def ead_path(ctx: EclContext, row) -> np.ndarray | None:
    """A contract's whole EAD path, month by month from the extract date:
    its supplied schedule, else its fallback over its full remaining term --
    never cut to the twelve months a Stage 1 provision looks at. What a
    balance is after k months is read from this (roll-forward)."""
    curve, _, _ = _ead_curve(ctx, row["cid"], 2, row["on_balance"],
                             _months_to_maturity(row["d_maturity"], row["d_extract"]),
                             row["payment_type"], row["portfolio"], row["nir"],
                             row["deferral"], row["pay_freq"])
    return curve


def compute_ecl(ctx: EclContext, frame: pd.DataFrame, lgd=None,
                cum=None) -> pd.DataFrame:
    """Price every contract of ``frame`` on ``ctx``: R's compute_ecl().

    THE pricing function. The run's report, the comparison bridge and every
    stress, what-if, roll-forward and macro or scenario repricing call it, so
    the ECL is calculated one way everywhere:

        ECL = sum_t EAD(t) x LGD x [cumPD(t+1) - cumPD(t)] / (1 + EIR)^(t/12)

    with the EAD curve from :func:`_ead_curve` (Stage 1 capped at twelve
    months), LGD from :func:`engine.compute_lgd` on the contract's net
    collateral, the PD curve of its segment and rating, Stage 3 per the
    context's method, and the provision capped at the balance when the
    context says so.

    ``frame`` is :func:`engine_frame` / :func:`ecl_frame`, or a copy with
    some facts changed (stage, rating, balance, EIR, dates). ``lgd`` and
    ``cum`` -- one value per row, NaN / None to look it up -- hold a
    contract's LGD or PD curve at another run's, which is how the bridge
    changes one ingredient at a time.

    Returns one row per contract: ecl, lgd, collcov, impcov, pdlife,
    collval, shape (the fallback shape, None for a supplied schedule) and
    horizon.
    """
    n = len(frame)
    out = {k: np.full(n, np.nan) for k in ("ecl", "lgd", "collcov", "impcov",
                                           "pdlife")}
    out["collval"] = np.zeros(n)
    shape = np.empty(n, dtype=object)
    horizon = np.zeros(n, dtype=int)
    cid = (frame["cid"] if "cid" in frame.columns else frame["contract"]).to_numpy()
    stage = frame["stage"].to_numpy()
    onbal = pd.to_numeric(frame["on_balance"], errors="coerce").to_numpy(dtype=float)
    if "d_maturity" in frame.columns and "d_extract" in frame.columns:
        months = [_months_to_maturity(m, e) for m, e in
                  zip(frame["d_maturity"], frame["d_extract"])]
    else:
        months = [int(m) for m in frame["months"]]
    pt, pf = frame["payment_type"].to_numpy(), frame["portfolio"].to_numpy()
    nir, dfr = frame["nir"].to_numpy(), frame["deferral"].to_numpy()
    pfq, eir = frame["pay_freq"].to_numpy(), frame["eir"].to_numpy()
    rat, rtp = frame["rating"].to_numpy(), frame["rating_type"].to_numpy()
    lgd_in = None if lgd is None else np.asarray(lgd, dtype=float)
    for i in range(n):
        c = cid[i]
        onb = 0.0 if not np.isfinite(onbal[i]) else float(onbal[i])
        st = int(stage[i])
        cnet = ctx.collnet.get(c, 0.0) if c is not None else 0.0
        cnet = 0.0 if cnet is None or not np.isfinite(cnet) else float(cnet)
        out["collval"][i] = cnet
        out["collcov"][i] = cnet / onb if onb > 0 else np.nan
        matm = int(months[i])
        curve, H, shape[i] = _ead_curve(ctx, c, st, onb, matm, pt[i], pf[i],
                                        nir[i], dfr[i], pfq[i])
        horizon[i] = H
        if lgd_in is not None and np.isfinite(lgd_in[i]):
            lg = float(lgd_in[i])
        else:
            lg = compute_lgd(onb, cnet, base=ctx.lgd_base,
                             unsecured_floor=ctx.lgd_floor)
        out["lgd"][i] = lg
        cm = cum[i] if cum is not None else ctx.pd_curve(pf[i], rat[i], rtp[i])
        if st == 3:
            if ctx.stage3 == "zero":
                out["ecl"][i] = 0.0
                out["impcov"][i] = 0.0 if onb > 0 else np.nan
            else:
                out["ecl"][i] = onb if onb > 0 else 0.0
                out["impcov"][i] = 1.0 if onb > 0 else np.nan
        else:
            tot = (float("nan") if cm is None else
                   sum_marginal_ecl(curve, lg, cm, eir[i], H))
            if np.isfinite(tot) and ctx.cap and onb > 0:
                tot = min(tot, onb)
            out["ecl"][i] = tot
            if np.isfinite(tot):
                out["impcov"][i] = tot / onb if onb > 0 else np.nan
        out["pdlife"][i] = _pd_lifetime_at(cm, st, matm)
    res = pd.DataFrame(out, index=frame.index)
    res["shape"] = shape
    res["horizon"] = horizon
    return res


def engine_frame(out_dir, suffix: str, portfolio_map: dict | None = None,
                 dpd_stage2_threshold: float = 60) -> pd.DataFrame | None:
    """The per-contract inputs the engine prices one book with.

    Shared by the report and the readiness check, so the stage, portfolio,
    rating type and maturity the check predicts are exactly the ones the
    report then prices.
    """
    am = read_output_csv(out_dir, f"AccountMaster{suffix}.csv")
    if am is None or len(am) == 0:
        return None
    csf = read_output_csv(out_dir, f"CustomerStagingFlag{suffix}.csv")
    n = len(am)
    cust = _chr(_col(am, "CustomerId")).reset_index(drop=True)
    contract = _chr(_col(am, "ContractId")).reset_index(drop=True)

    def csf_get(field):
        if csf is None:
            return pd.Series([None] * n)
        return _lookup(_col(csf, "CustomerId"), _col(csf, field), cust)

    def flag(field):
        if csf is None:
            return pd.Series([np.nan] * n)
        return _flag_lookup(_col(csf, "CustomerId"), _col(csf, field), cust)

    f = {k: flag(k) for k in ("IsWatchlist", "IsDefault", "IsDefaultInGCC",
                              "IsInsolvency", "IsLocal1", "IsLocal2", "IsLocal3",
                              "IsLocal4", "IsLocal5", "IsLocal6")}
    local_any = (f["IsWatchlist"] == 1)
    for i in range(1, 7):
        local_any = local_any | (f[f"IsLocal{i}"] == 1)

    acct = _chr(_col(am, "AccountType")).reset_index(drop=True)
    pmap = portfolio_map or {}
    portfolio = pd.Series(["Business Finance"] * n, dtype=object)
    mapped = acct.map(lambda a: pmap.get(a) if a is not None else None)
    ok_ = mapped.notna() & (mapped.astype(str) != "")
    portfolio[ok_] = mapped[ok_]
    if suffix == "_2":
        portfolio[:] = "Investments"
    rating_type = portfolio.map(DEFAULT_PORTFOLIOS)
    rating_type = rating_type.where(rating_type.notna(),
                                    "External" if suffix == "_2" else "Internal")

    d_ext = fer_parse_date(_col(am, "ExtractDate")).reset_index(drop=True)
    d_mat = fer_parse_date(_col(am, "MaturityDate")).reset_index(drop=True)
    dpd = _num(_col(am, "PastDueDays")).reset_index(drop=True)
    stage = classify_stage_report(
        dpd, f["IsDefault"], f["IsWatchlist"], local_any, portfolio, cust,
        dpd_stage2_threshold, stage_override=_chr(_col(am, "Stage")))

    def num_or(name, fill):
        c = _col(am, name)
        return _num(c).reset_index(drop=True) if c is not None else \
            pd.Series([fill] * n, dtype=float)

    defer = num_or("DeferralPeriod", np.nan)
    if defer.isna().all():
        defer = pd.Series([0.0] * n)
    pay_freq = num_or("PaymentFrequency", np.nan)
    pay_freq = pay_freq.where(pay_freq.notna() & (pay_freq >= 1), 1.0).fillna(1.0)

    raw_months = calendar_months(d_ext, d_mat)
    return pd.DataFrame({
        "book": "Lending" if suffix == "_1" else "Investments",
        "suffix": suffix,
        "contract": contract, "customer": cust, "account_type": acct,
        "portfolio": portfolio, "rating_type": rating_type,
        "rating": _chr(_col(am, "Rating")).reset_index(drop=True),
        "on_balance": num_or("OnBalance", np.nan),
        "off_balance": num_or("OffBalance", np.nan),
        "eir": num_or("EIR", np.nan),
        "dpd": dpd, "stage": stage,
        "d_extract": d_ext, "d_maturity": d_mat,
        "raw_months": raw_months,
        "months": [_months_to_maturity(m, e) for m, e in zip(d_mat, d_ext)],
        "payment_type": _chr(_col(am, "PaymentTypeId")).reset_index(drop=True)
        if _col(am, "PaymentTypeId") is not None else pd.Series([None] * n),
        "nir": num_or("NominalInterestRate", np.nan),
        "deferral": defer, "pay_freq": pay_freq,
        "has_flags_row": csf_get("CustomerId").notna() if csf is not None
        else pd.Series([False] * n),
        **{k: v for k, v in f.items()},
        "local_any": local_any,
    })


def ecl_frame(out_dir, portfolio_map: dict | None = None,
              dpd_stage2_threshold: float | None = None) -> pd.DataFrame | None:
    """Both books' :func:`engine_frame`, as the report prices them, one row
    per position: ``cid`` is the run's own contract id (what its collateral
    and EAD tables are keyed on), ``contract`` the normalised one the report
    uses, and ``key`` the contract with its occurrence -- a contract id is not
    unique (an investment held in two positions), and the report and the
    frame both keep the AccountMaster's order, so the n-th row of a contract
    in one is the n-th in the other.

    ``dpd_stage2_threshold`` defaults to the one the run froze with its
    config, as its report was staged on."""
    if dpd_stage2_threshold is None:
        from ..stress import staging_threshold
        dpd_stage2_threshold = staging_threshold(out_dir)
    parts = [engine_frame(out_dir, s, portfolio_map, dpd_stage2_threshold)
             for s in ("_1", "_2")]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    f = pd.concat(parts, ignore_index=True)
    f["cid"] = f["contract"]
    f["contract"] = as_id(f["contract"]).astype("string")
    f["occ"] = f.groupby("contract").cumcount()
    f["key"] = f["contract"].astype(str) + "#" + f["occ"].astype(str)
    return f


def _segment(out_dir, suffix, ctx: _Ctx, portfolio_map, entity_id,
             run_id_label, dpd_stage2_threshold) -> pd.DataFrame | None:
    """R's .fer_build_segment(): one book's rows of the report."""
    am = read_output_csv(out_dir, f"AccountMaster{suffix}.csv")
    if am is None or len(am) == 0:
        return None
    e = engine_frame(out_dir, suffix, portfolio_map, dpd_stage2_threshold)
    n = len(am)
    cm = read_output_csv(out_dir, f"CustomerMaster{suffix}.csv")
    org = read_output_csv(out_dir, f"Origination{suffix}.csv")
    cust, contract = e["customer"], e["contract"]

    def cm_get(field):
        if cm is None:
            return pd.Series([None] * n)
        return _lookup(_col(cm, "CustomerId"), _chr(_col(cm, field))
                       if _col(cm, field) is not None else None, cust)

    def org_get(field):
        if org is None:
            return pd.Series([None] * n)
        return _lookup(_col(org, "ContractId"), _chr(_col(org, field))
                       if _col(org, field) is not None else None, contract)

    is_indiv = (_flag_lookup(_col(cm, "CustomerId"),
                             _col(cm, "IsIndividualAssessment"), cust)
                if cm is not None else pd.Series([np.nan] * n))

    d_ext = e["d_extract"]
    d_open = fer_parse_date(_col(am, "OpenDate")).reset_index(drop=True)
    d_exp = fer_parse_date(_col(am, "ExpectedMaturityDate")).reset_index(drop=True)
    d_mat = e["d_maturity"]
    tfo = months_between(d_open, d_ext).map(lambda v: round(v, 2) if pd.notna(v) else v)
    tte = years_between(d_ext, d_exp).map(lambda v: round(v, 4) if pd.notna(v) else v)
    ttm = years_between(d_ext, d_mat).map(lambda v: round(v, 4) if pd.notna(v) else v)

    def am_num(name):
        c = _col(am, name)
        return _num(c).reset_index(drop=True) if c is not None else \
            pd.Series([np.nan] * n)

    on_bal = e["on_balance"]
    stage = e["stage"].to_numpy()
    is_init = pd.Series(0, index=range(n))
    both = d_open.notna() & d_ext.notna()
    is_init[both] = ((d_open[both].dt.year == d_ext[both].dt.year)
                     & (d_open[both].dt.month == d_ext[both].dt.month)).astype(int)

    # ---- the engine: R's compute_ecl() -----------------------------------
    priced = compute_ecl(ctx, e)
    lgd, collcov = priced["lgd"].to_numpy(), priced["collcov"].to_numpy()
    ecl, impcov = priced["ecl"].to_numpy(), priced["impcov"].to_numpy()
    pdlife, collval = priced["pdlife"].to_numpy(), priced["collval"].to_numpy()

    lgd_rate = np.where(np.isin(stage, [1, 2]), lgd, np.nan)
    resid_lgd = np.where(stage == 3, 1.0, np.nan)
    coll_value = np.where(stage == 3, collval, np.nan)

    def chr_col(name):
        c = _col(am, name)
        return _chr(c).reset_index(drop=True) if c is not None else \
            pd.Series([None] * n, dtype=object)

    rep = pd.DataFrame(index=range(n))
    rep["Run Id"] = run_id_label
    rep["Enterprise Entity Id"] = entity_id
    rep["Extract Date"] = chr_col("ExtractDate")
    rep["Contract Id"] = contract
    rep["Account Code"] = chr_col("AccountCode")
    rep["Portfolio Code"] = e["portfolio"]
    rep["Customer Id"] = cust
    rep["Customer Code"] = cm_get("CustomerCode")
    rep["Customer Name"] = cm_get("CustomerName")
    rep["Account Type"] = e["account_type"]
    rep["Past Due Days"] = e["dpd"]
    rep["Lim Id"] = chr_col("LimId")
    rep["Open Date"] = chr_col("OpenDate")
    rep["Time From Open Date"] = tfo
    rep["Maturity Date"] = chr_col("MaturityDate")
    rep["Expected Maturity Date"] = chr_col("ExpectedMaturityDate")
    rep["Time To Expected Maturity"] = tte
    rep["Rating"] = e["rating"]
    rep["Origination Rating"] = org_get("OriginationRating")
    rep["Rating Type"] = e["rating_type"]
    rep["MOB"] = tfo
    rep["Ifrs Stage"] = stage
    rep["Time To Maturity"] = ttm
    rep["Is Individual Assessment"] = is_indiv
    rep["EAD"] = on_bal
    rep["EIR"] = e["eir"]
    rep["Is Initial Recognition"] = is_init
    rep["PD 12M"] = am_num("PD12M")
    rep["Origination PD 12M"] = _num(org_get("OriginationPD12M"))
    rep["PD Lifetime Value"] = pdlife
    rep["LGD Rate"] = lgd_rate
    rep["Is POCI"] = _flag01(_col(am, "IsPOCI"), n) if _col(am, "IsPOCI") is not None \
        else np.nan
    rep["Resid Lgd Rate"] = resid_lgd
    rep["CCF"] = am_num("CCF")
    rep["Coll Cov"] = collcov
    rep["Is Cla Simplified Approach"] = 0
    rep["Is Cla Loss Rate"] = 0
    rep["Is Cla Renewable Credit Facility"] = 0
    rep["Exposure On Bal"] = on_bal
    rep["Exposure Off Bal"] = e["off_balance"]
    rep["Cla Amount Onbal"] = ecl
    for c in ("Delta Cla Amount Onbal", "Cla Amount Offbal",
              "Delta Cla Amount Offbal", "Cla Amount Principal",
              "Cla Amount Principal Overdue", "Cla Amount Interest Accrued",
              "Cla Amount Interest Overdue", "Cla Amount Fee",
              "Cla Amount Fee Overdue", "Cla Amount Penalty",
              "Cla Amount Penalty Overdue", "Cla Amount Commission",
              "Cla Amount Commission Overdue", "Cla Amount Other",
              "Cla Amount Other Overdue"):
        rep[c] = np.nan
    rep["CLA Calculation Approach Id"] = "C"
    rep["CLA Currency"] = chr_col("CurrencyCode")
    rep["Impairment Coverage Off Bal"] = np.nan
    rep["Impairment Coverage On Bal"] = impcov
    for c in ("Poci Cla Amount At Origination Offbal",
              "Poci Cla Amount At Origination Onbal",
              "Original Ecl Offbal", "Original Ecl Onbal"):
        rep[c] = np.nan
    rep["Customer Organizational Unit Code"] = cm_get("OrganizationalUnitCode")
    rep["Collateral Value"] = coll_value
    rep["Watchlist Flag"] = e["IsWatchlist"]
    rep["Default Flag"] = e["IsDefault"]
    rep["Default In GCC Flag"] = e["IsDefaultInGCC"]
    rep["Insolvency Flag"] = e["IsInsolvency"]
    for i in range(1, 7):
        rep[f"Local Flag {i}"] = e[f"IsLocal{i}"]
    return rep[REPORT_COLUMNS]


def _overlay_waterfall(report: pd.DataFrame, overlays) -> pd.DataFrame:
    """R's apply_overlays(): Ecl Model -> Overlay Amount -> Ecl Final.

    With no overlays the waterfall is appended unchanged, so the report has
    one schema. With overlays, the engine in ``overlays.py`` resolves them and
    the ECL column takes the final figure, the model figure kept beside it.
    """
    ecl = pd.to_numeric(report["Cla Amount Onbal"], errors="coerce")
    report["Ecl Model Onbal"] = ecl
    report["Overlay Id"] = None
    report["Overlay Amount"] = 0.0
    report["Ecl Final Onbal"] = ecl
    if not overlays:
        return report
    from ..overlays import _lic_to_normalised, apply_overlays
    norm = _lic_to_normalised(report)
    res = apply_overlays(norm, overlays)
    if not res.get("ok"):
        raise ValueError("Overlay conflict - resolve overlapping overlays and "
                         "re-run; no report written. "
                         + "; ".join(res.get("errors", [])))
    d = res["report"]
    amt = pd.to_numeric(d["overlay_amount"], errors="coerce").fillna(0.0).to_numpy()
    oid = d["overlay_id"].replace("", None).to_numpy()
    report["Overlay Amount"] = amt
    report["Overlay Id"] = oid
    report["Ecl Final Onbal"] = ecl + amt
    report["Cla Amount Onbal"] = report["Ecl Final Onbal"]
    return report


def _format_for_csv(rep: pd.DataFrame) -> pd.DataFrame:
    """Numbers written as readr writes them: 12, not 12.0."""
    out = rep.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s):
            out[c] = s.map(lambda v: "" if pd.isna(v) else (
                str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))))
    return out


def _portfolio_map(static) -> dict:
    """product_type -> portfolio, from the static mapping (R's portfolio_map)."""
    if static is None:
        return {}
    ppm = static.get("product_portfolio_mapping") if hasattr(static, "get") else None
    if ppm is None or len(ppm) == 0:
        return {}
    pt = _col(ppm, "product_type")
    pf = _col(ppm, "portfolio")
    if pt is None or pf is None:
        return {}
    out = {}
    for k, v in zip(pt.astype(str).str.strip(), pf):
        if k not in out:
            out[k] = v
    return out


def run_pricing_config(run_dir):
    """(static, model_cfg) a run is priced on: its frozen ``config_used`` when
    it has one, else the package defaults -- the same for the report, the
    comparison bridge and every stress, so all of them price one way."""
    run_dir = Path(run_dir)
    if run_dir.name == "Output" and not (run_dir / "Output").is_dir():
        run_dir = run_dir.parent
    try:
        from .static_ref import load_static_reference
        frozen = run_dir / "config_used" / "static"
        static = load_static_reference(frozen if frozen.is_dir() else None)
    except Exception:
        static = None
    try:
        from .pipeline import load_model_config
        frozen = run_dir / "config_used" / "config"
        model_cfg, _ = load_model_config(frozen if frozen.is_dir() else None)
    except Exception:
        model_cfg = None
    return static, model_cfg


def build_final_ecl_report(run_dir, entity_id=None, run_id_label=None,
                           dpd_stage2_threshold: float = 60,
                           cfg: EclConfig | None = None,
                           write: bool = True,
                           out_name: str = "FinalEclReport.csv",
                           static=None, model_cfg: dict | None = None,
                           overlays=None, stpd: pd.DataFrame | None = None,
                           include_investments: bool = True) -> pd.DataFrame:
    """Price a run and write its ECL report (R's build_final_ecl_report)."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir
    if static is None or model_cfg is None:
        st, mc = run_pricing_config(run_dir)
        static = st if static is None else static
        model_cfg = mc if model_cfg is None else model_cfg
    pmap = _portfolio_map(static)
    ctx = build_ecl_context(out_dir, model_cfg=model_cfg, stpd=stpd, cfg=cfg)
    segs = [_segment(out_dir, "_1", ctx, pmap, entity_id, run_id_label,
                     dpd_stage2_threshold)]
    if include_investments:
        segs.append(_segment(out_dir, "_2", ctx, pmap, entity_id, run_id_label,
                             dpd_stage2_threshold))
    segs = [s for s in segs if s is not None]
    if not segs:
        raise FileNotFoundError(f"No AccountMaster output found in {out_dir}")
    rep = pd.concat(segs, ignore_index=True)
    rep = _overlay_waterfall(rep, overlays or [])
    if write:
        _format_for_csv(rep).to_csv(out_dir / out_name, index=False, na_rep="")
    return rep
