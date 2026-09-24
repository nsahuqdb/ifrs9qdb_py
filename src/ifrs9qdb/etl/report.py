"""
The final ECL report.

Prices a run's LIC input files and writes the 76-column report the analytics
read. This is the file that turns a set of inputs into a provision, so it is
also where the config settings that change a reported number take effect.

The report is built from the OUTPUT folder, not from the raw extracts: by this
point the ETL has resolved contract ids, collateral allocations and rating
buckets, and re-deriving any of that here would give two answers to the same
question.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..engine import EclConfig, compute_lgd, sum_marginal_ecl
from ..inputs import load_engine_inputs
from .customer import apply_staging_rule
from .transform import pick
from ..ids import as_id

__all__ = ["REPORT_COLUMNS", "build_final_ecl_report", "classify_stage_report"]

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


def _read(out_dir: Path, name: str) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.is_file():
        return None
    return pd.read_csv(p, low_memory=False)


def classify_stage_report(dpd, default_flag, watchlist, local_any, portfolio,
                          customer, dpd_stage2_threshold: float = 60,
                          contagion: bool = True) -> np.ndarray:
    """Stage per CONTRACT, with the collective and contagion rules applied.

        DPD > 90 or the default flag                       -> Stage 3
        threshold < DPD <= 90, watchlist, or any local flag -> Stage 2
        portfolio is Tasdeer                                -> Stage 2 always
        then contagion: a customer with any Stage 2 or worse facility has its
        Stage 1 facilities lifted, Tasdeer excluded

    Tasdeer is assessed collectively rather than facility by facility, which is
    why it is set unconditionally and then left out of the contagion sweep --
    it is already at its collective floor.
    """
    n = len(portfolio)
    stage = apply_staging_rule(dpd, local_any, watchlist, dpd_stage2_threshold)
    st = np.array([3 if s == "Stage 3" else (2 if s == "Stage 2" else 1)
                   for s in stage], dtype=int)

    df = pd.Series(default_flag).fillna(0)
    st[(pd.to_numeric(df, errors="coerce") == 1).to_numpy()] = 3

    pf = pd.Series(portfolio).astype(str).str.strip()
    st[(pf == "Tasdeer").to_numpy() & (st < 2)] = 2

    if contagion:
        cid = pd.Series(customer).astype(str)
        cid[cid.isin(["", "nan", "None"])] = "(unknown)"
        worst = pd.Series(st).groupby(cid.to_numpy()).transform("max").to_numpy()
        bump = (st == 1) & (worst >= 2) & (pf != "Tasdeer").to_numpy()
        st[bump] = 2
    return st


def build_final_ecl_report(run_dir, entity_id: str = "", run_id_label: str = "",
                           dpd_stage2_threshold: float = 60,
                           cfg: EclConfig | None = None,
                           write: bool = True,
                           out_name: str = "FinalEclReport.csv") -> pd.DataFrame:
    """Price a run and write its ECL report."""
    run_dir = Path(run_dir)
    out_dir = run_dir / "Output" if (run_dir / "Output").is_dir() else run_dir
    cfg = cfg or EclConfig()

    inputs = load_engine_inputs(out_dir)
    if not inputs.ok:
        raise FileNotFoundError(
            "Cannot price this run: missing " + ", ".join(inputs.missing))
    con = inputs.contracts
    if len(con) == 0:
        raise ValueError("No contracts in this run's account master")

    extract = _read(out_dir, "AccountMaster_1.csv")
    extract_date = (str(extract["ExtractDate"].iloc[0])
                    if extract is not None and len(extract) else "")

    # customer-level attributes, joined on the customer
    flags = _read(out_dir, "CustomerStagingFlag_1.csv")
    cust = _read(out_dir, "CustomerMaster_1.csv")
    orig = _read(out_dir, "Origination_1.csv")

    def flag_map(df, col):
        if df is None or col not in df.columns:
            return {}
        cid = pick(df, "CustomerId", "CUSTOMERID")
        if cid is None:
            return {}
        v = df[col].astype(str).str.strip().str.upper().isin(("TRUE", "1"))
        return dict(zip(as_id(cid), v))

    watch = flag_map(flags, "IsWatchlist")
    dflt = flag_map(flags, "IsDefault")
    insol = flag_map(flags, "IsInsolvency")
    locals_ = {f"Local Flag {i}": flag_map(flags, f"IsLocal{i}")
               for i in range(1, 7)}

    cust_key = con["contract"].map(
        dict(zip(con["contract"], con["customer"]))) \
        if "customer" in con.columns else None
    if cust_key is None:
        am = _read(out_dir, "AccountMaster_1.csv")
        lut = dict(zip(as_id(am["ContractId"]),
                       am["CustomerId"].astype(str))) if am is not None else {}
        cust_key = con["contract"].map(lut)
    cust_key = cust_key.fillna("").astype(str)

    dpd = pd.Series(0.0, index=con.index)
    if extract is not None:
        lut = dict(zip(extract["ContractId"].astype(str),
                       pd.to_numeric(extract["PastDueDays"], errors="coerce")))
        dpd = con["contract"].map(lut).fillna(0.0)

    local_any = pd.Series(False, index=con.index)
    for m in locals_.values():
        local_any |= cust_key.map(m).fillna(False)

    stage = classify_stage_report(
        dpd, cust_key.map(dflt).fillna(False), cust_key.map(watch).fillna(False),
        local_any, con["portfolio"], cust_key, dpd_stage2_threshold)

    # ---- price -------------------------------------------------------
    n = len(con)
    ecl = np.full(n, np.nan)
    lgd_out = np.full(n, np.nan)
    pd_life = np.full(n, np.nan)
    for i in range(n):
        row = con.iloc[i]
        s = int(stage[i])
        bal = row.get("on_balance")
        coll = float(row.get("collateral_net") or 0.0)
        lgd = compute_lgd(bal, coll, base=cfg.lgd_base,
                          unsecured_floor=cfg.lgd_unsecured_floor,
                          zero_exposure_lgd=cfg.zero_exposure_lgd)
        lgd_out[i] = lgd
        if s == 3:
            ecl[i] = 0.0 if cfg.stage3_method == "zero" else (
                float(bal) if bal and bal > 0 else 0.0)
            continue
        cum = inputs.pd_curve(row["portfolio"], row.get("bucket"))
        curve = inputs.ead_curve(row["contract"], s, row)
        if cum is None or curve is None or len(curve) == 0:
            continue
        pd_life[i] = float(cum[min(len(curve), len(cum) - 1)])
        v = sum_marginal_ecl(curve, lgd, cum, row.get("eir") or 0.0,
                             horizon=len(curve))
        if np.isfinite(v) and cfg.cap_ecl_at_exposure and bal and bal > 0:
            v = min(v, float(bal))
        ecl[i] = v

    on_bal = pd.to_numeric(con["on_balance"], errors="coerce").fillna(0.0)
    coll_val = pd.to_numeric(con["collateral_net"], errors="coerce").fillna(0.0)
    coverage = np.where(on_bal > 0, ecl / on_bal.replace(0, np.nan), np.nan)

    rep = pd.DataFrame({c: "" for c in REPORT_COLUMNS}, index=range(n))
    rep["Run Id"] = run_id_label or run_dir.name
    rep["Enterprise Entity Id"] = entity_id
    rep["Extract Date"] = extract_date
    rep["Contract Id"] = con["contract"].to_numpy()
    rep["Portfolio Code"] = con["portfolio"].to_numpy()
    rep["Customer Id"] = cust_key.to_numpy()
    rep["Past Due Days"] = dpd.to_numpy()
    rep["Rating"] = con["rating"].to_numpy()
    rep["Rating Type"] = con.get("rating_type", pd.Series([1] * n)).to_numpy()
    rep["Ifrs Stage"] = stage
    rep["EIR"] = pd.to_numeric(con["eir"], errors="coerce").to_numpy()
    rep["PD Lifetime Value"] = pd_life
    rep["LGD Rate"] = lgd_out
    rep["Exposure On Bal"] = on_bal.to_numpy()
    rep["Cla Amount Onbal"] = ecl
    rep["Impairment Coverage On Bal"] = coverage
    rep["Collateral Value"] = coll_val.to_numpy()
    rep["Coll Cov"] = np.where(on_bal > 0,
                               coll_val / on_bal.replace(0, np.nan), 0.0)
    rep["Maturity Date"] = con["maturity_date"].astype(str).to_numpy()
    rep["Time To Maturity"] = pd.to_numeric(
        con.get("months_to_mat"), errors="coerce").to_numpy()

    if cust is not None:
        nm = pick(cust, "CustomerName")
        if nm is not None:
            rep["Customer Name"] = cust_key.map(
                dict(zip(pick(cust, "CustomerId").astype(str), nm))).fillna("")
    if orig is not None and "OriginationRating" in orig.columns:
        rep["Origination Rating"] = con["contract"].map(
            dict(zip(orig["ContractId"].astype(str),
                     orig["OriginationRating"]))).fillna("")

    for i, key in enumerate(locals_, start=1):
        rep[key] = cust_key.map(locals_[key]).fillna(False).astype(int).to_numpy()
    rep["Watchlist Flag"] = cust_key.map(watch).fillna(False).astype(int).to_numpy()
    rep["Default Flag"] = cust_key.map(dflt).fillna(False).astype(int).to_numpy()
    rep["Insolvency Flag"] = cust_key.map(insol).fillna(False).astype(int).to_numpy()

    rep = rep[REPORT_COLUMNS]
    if write:
        rep.to_csv(out_dir / out_name, index=False, na_rep="")
    return rep
