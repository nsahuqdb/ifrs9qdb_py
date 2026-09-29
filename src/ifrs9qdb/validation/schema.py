"""The canonical input schema, as the R package applies it before validating.

R reads every extract through ``apply_input_schema`` (``R/input_schemas.R``):
each canonical column is found by one of its known header spellings, and failing
that by its POSITION, because the SQL*Plus exports truncate aliases --
``COLLATERALTYPEID`` arrives as ``CO``, ``MATURITYDATE`` as ``MATURITYDAT``.
Every R validator then reads canonical snake_case names.

The Python checks used to read the raw headers instead, and a check that could
not find its column returned a PASS. That is how the port reported the June
extract clean on two coverage checks R failed (three unmapped collateral types,
35 unmapped industry codes) and failed three others R passed. Canonicalising
first gives every check the columns R's checks see.

Repeated SQL*Plus header rows, the "N rows selected." trailer and blank rows are
stripped at the same point, and counted per file, as ``.strip_duplicate_header_row``
does -- left in, a header row reaches the cross-file checks as an allocation of
collateral "COLLATERALID".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ids import as_id

__all__ = ["INPUT_SCHEMAS", "canonicalise", "CanonicalInputs", "strip_junk_rows"]


def _c(canonical, sources, position, type_="character", required=True):
    return {"canonical": canonical, "sources": tuple(sources), "position": position,
            "type": type_, "required": required}


# Ported from R/input_schemas.R, INPUT_SCHEMAS. Positions are 1-based, as in R.
INPUT_SCHEMAS: dict[str, list[dict]] = {
    "AccountMaster": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("contract_id", ["CONTRACTID"], 2),
        _c("lim_id", ["LIMID"], 3, required=False),
        _c("customer_id", ["CUSTOMERID"], 4),
        _c("portfolio_code", ["PORTFOLIOCODE"], 5),
        _c("account_type", ["ACCOUNTTYPE"], 6),
        _c("impairment_amount", ["IMPAIRMENTAMOUNT"], 7, "numeric", False),
        _c("stage", ["STAGE"], 8, required=False),
        _c("open_date", ["OPENDATE"], 9, "date"),
        _c("past_due_days", ["PASTDUEDAYS", "PASTDUE_DAYS"], 11, "integer"),
        _c("off_balance", ["OFFBALANCE"], 17, "numeric"),
        _c("on_balance", ["ONBALANCE"], 18, "numeric"),
        _c("ead", ["EAD"], 19, "numeric", False),
        _c("ccf", ["CCF"], 20, "numeric", False),
        _c("maturity_date", ["MATURITYDATE", "MATURITYDAT"], 21, "date"),
        _c("eir", ["EIR"], 23, "numeric", False),
        _c("payment_frequency", ["PAYMENTFREQUENCY", "PAYMENT_FRE"], 36, "integer", False),
        _c("currency_code", ["CURRENCYCODE", "CUR"], 37, required=False),
        _c("deferral_period", ["DEFERRALPERIOD"], 38, "numeric", False),
        _c("nominal_int_rate", ["NOMINALINTERESTRATE"], 39, "numeric", False),
        _c("payment_type_id", ["PAYMENTTYPEID"], 40, "integer", False),
    ],
    "AccountMasterInvestments": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("account_id", ["CONTRACTID"], 2),
        _c("lim_id", ["LIMID"], 3, required=False),
        _c("customer_id", ["CUSTOMERID"], 4),
        _c("portfolio_code", ["PORTFOLIOCODE"], 5),
        _c("account_type", ["ACCOUNTTYPE"], 6),
        _c("open_date", ["OPENDATE"], 9, "date"),
        _c("rating", ["RATING"], 10),
        _c("past_due_days", ["PASTDUEDAYS"], 11, "integer", False),
        _c("on_balance", ["ONBALANCE"], 18, "numeric"),
        _c("maturity_date", ["MATURITYDATE", "MATURITYDAT"], 21, "date"),
        _c("eir", ["EIR"], 23, "numeric", False),
        _c("payment_frequency", ["PAYMENTFREQUENCY", "PAYMENT_FRE"], 36, "integer", False),
        _c("currency_code", ["CURRENCYCODE", "CUR"], 37, required=False),
        _c("nominal_int_rate", ["NOMINALINTERESTRATE"], 39, "numeric", False),
        _c("payment_type_id", ["PAYMENTTYPEID"], 40, "integer", False),
        _c("ccf", ["CCF"], 20, "numeric", False),
    ],
    "CustomerMaster": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("customer_id", ["CUSTOMERID"], 2),
        _c("portfolio_code", ["PORTFOLIOCODE"], 3, required=False),
        _c("customer_name", ["CUSTOMERNAME"], 4, required=False),
        _c("customer_limit", ["CUSTOMERLIMIT"], 5, "numeric", False),
        _c("rating", ["RATING"], 6),
        _c("past_due_days", ["PASTDUE_DAYS", "PASTDUEDAYS"], 7, "integer", False),
        _c("is_individual_assessment", ["ISINDIVIDUALASSESSMENT"], 10, "logical", False),
    ],
    "CustomerMasterInvestments": [
        _c("extract_date", ["EXTRACTDA"], 1, "date", False),
        _c("customer_id", ["CUSTOMERID"], 2),
        _c("rating", ["RATING"], 6, required=False),
        _c("customer_name", ["CUSTOMERNAME"], 4, required=False),
    ],
    "CustomerStagingFlag": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("customer_id", ["CUSTOMERID"], 2),
        _c("is_default", ["ISDEFAULT"], 3, "integer", False),
        _c("is_watchlist", ["ISWATCHLIST"], 4, "integer"),
        _c("is_insolvency", ["ISINSOLVENCY"], 5, "integer", False),
        _c("is_default_in_gcc", ["ISDEFAULTINGCC"], 6, "integer", False),
        _c("is_local1", ["ISLOCAL1"], 7, "integer"),
    ],
    "CustomerStagingFlagInvestments": [
        _c("customer_id", ["CUSTOMERID"], 2),
        _c("is_default", ["ISDEFAULT"], 3, "integer", False),
        _c("is_watchlist", ["ISWATCHLIST"], 4, "integer", False),
        _c("is_local1", ["ISLOCAL1"], 7, "integer", False),
    ],
    "Collateral": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("collateral_id", ["COLLATERALID"], 2),
        # readxl misreads column C as logical; column D ("CO") carries the type.
        _c("collateral_type_id", ["CO"], 4, "integer"),
        _c("currency", ["COL"], 5),
        _c("value", ["COLLATERALVALUE"], 6, "numeric"),
    ],
    "AccountCollateralAllocation": [
        _c("extract_date", ["EXTRACTDA"], 1, "date"),
        _c("collateral_id", ["COLLATERALID"], 2),
        _c("contract_id", ["CONTRACTID"], 3),
        _c("allocation_percentage", ["ALLOCATIONPERCENTAGE"], 4, "numeric"),
    ],
    "RepaymentSchedule": [
        _c("contract_id", ["KEY_1"], 1),
        _c("post_date", ["POST_DATE"], 2, "date"),
        _c("start_date", ["START_DATE", "START_DAT"], 3, "date"),
        _c("principal_due", ["PRINCE_DUE"], 4, "numeric"),
        _c("projected_interest", ["PROJ_INT"], 5, "numeric"),
        _c("repayment", ["REPAYMENT"], 6, "numeric"),
        _c("balance", ["BALANCE"], 7, "numeric"),
    ],
    "IndustryCode": [
        _c("customer_id", ["CUSTOMERID"], 1),
        _c("industry_code", ["INDUSTRYCODE"], 2, required=False),
        _c("industry_description", ["INDUSTRYDESCRIPTION"], 3, required=False),
    ],
    # R/input_schemas.R lists these positions as 1..5, but the extract carries
    # EXTRACTDA first and CONTRACTID second, so everything after the id is read
    # from the wrong column (origination_pd_12m would hold the contract id).
    # Mirrored exactly for parity; recorded in INPUT_DATA_ISSUES.md. Harmless
    # only because both writers emit origination values blank.
    "Origination": [
        _c("contract_id", ["CONTRACTID", "KEY_1"], 1),
        _c("origination_pd_12m", ["PD12M"], 2, "numeric", False),
        _c("origination_rating", ["RATING"], 3, required=False),
        _c("origination_dpd", ["PASTDUEDAYS"], 4, "integer", False),
        _c("origination_watchlist", ["ISWATCHLIST"], 5, "integer", False),
    ],
    "OriginationInvestments": [
        _c("contract_id", ["CONTRACTID", "KEY_1"], 1),
        _c("origination_pd_12m", ["PD12M"], 2, "numeric", False),
        _c("origination_rating", ["RATING"], 3, required=False),
    ],
}

_ID_COLUMNS = {"contract_id", "customer_id", "collateral_id", "account_id",
               "lim_id", "collateral_type_id"}


def _original_header(name) -> str:
    """The header as SQL*Plus wrote it, before pandas de-duplicated it (I.1)."""
    return re.sub(r"\.\d+$", "", str(name)).strip().lower()


def strip_junk_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove repeated headers, the row-count trailer and blank rows.

    Mirrors ``.strip_duplicate_header_row``: a header repeat is a row whose
    every non-blank cell equals its column's ORIGINAL heading, with at least two
    non-blank cells.
    """
    if df is None or len(df) == 0 or df.shape[1] == 0:
        return df, 0
    heads = [_original_header(c) for c in df.columns]
    n = len(df)
    nonblank = np.zeros((n, df.shape[1]), dtype=bool)
    match = np.zeros((n, df.shape[1]), dtype=bool)
    for j, c in enumerate(df.columns):
        cell = df[c].astype(str).str.strip().str.lower()
        nb = (df[c].notna() & (cell != "") & (cell != "nan")).to_numpy()
        nonblank[:, j] = nb
        match[:, j] = nb & (cell.to_numpy() == heads[j])
    n_nb = nonblank.sum(axis=1)
    n_match = match.sum(axis=1)
    is_header = (n_nb >= 2) & (n_match == n_nb)
    first = df.iloc[:, 0].astype(str).str.strip().str.lower()
    is_footer = ((n_nb == 1) & nonblank[:, 0]
                 & first.str.match(r"^[0-9][0-9,]*\s+rows?\s+selected\.?$",
                                   na=False).to_numpy())
    is_blank = n_nb == 0
    drop = is_header | is_footer | is_blank
    return df.loc[~drop].reset_index(drop=True), int(drop.sum())


def _coerce(values: pd.Series, type_: str, canonical: str) -> pd.Series:
    if type_ in ("numeric", "integer"):
        v = pd.to_numeric(values, errors="coerce")
        return v.round() if type_ == "integer" else v
    if type_ == "date":
        from ._helpers import parse_any_date
        return parse_any_date(values)
    if type_ == "logical":
        s = values.astype(str).str.strip().str.upper()
        out = pd.Series(np.nan, index=values.index, dtype=object)
        out[s.isin(["TRUE", "T", "1", "Y", "YES"])] = True
        out[s.isin(["FALSE", "F", "0", "N", "NO"])] = False
        return out
    if canonical in _ID_COLUMNS:
        return as_id(values).replace("", np.nan)
    s = values.where(values.isna(), values.astype(str).str.strip())
    return s.replace({"": np.nan, "nan": np.nan})


def _canonical_frame(raw: pd.DataFrame, schema: list[dict]) -> pd.DataFrame:
    """Raw columns kept, canonical columns added -- so a check written against
    either spelling finds its data."""
    out = raw.copy()
    upper = {str(c).strip().upper(): c for c in raw.columns}
    for spec in schema:
        src = None
        for name in spec["sources"]:
            if name.upper() in upper:
                src = upper[name.upper()]
                break
        if src is None and spec["position"] and raw.shape[1] >= spec["position"]:
            src = raw.columns[spec["position"] - 1]
        if src is None:
            out[spec["canonical"]] = np.nan
            continue
        out[spec["canonical"]] = _coerce(raw[src], spec["type"], spec["canonical"])
    return out


@dataclass
class CanonicalInputs:
    """An InputSet whose tables carry the canonical columns as well."""
    tables: dict[str, pd.DataFrame]
    paths: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    strip_log: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.missing

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.tables.get(name, pd.DataFrame())

    def get(self, name, default=None):
        return self.tables.get(name, default)


def canonicalise(inputs) -> CanonicalInputs:
    """Strip junk rows and add the canonical columns to every table."""
    if isinstance(inputs, CanonicalInputs):
        return inputs
    tables = getattr(inputs, "tables", None)
    if tables is None:
        tables = dict(inputs) if isinstance(inputs, dict) else {}
    paths = dict(getattr(inputs, "paths", {}) or {})
    out, log = {}, {}
    for name, raw in tables.items():
        if raw is None:
            continue
        clean, n = strip_junk_rows(raw)
        if n:
            p = paths.get(name)
            log[p.name if p is not None else name] = n
        schema = INPUT_SCHEMAS.get(name)
        out[name] = _canonical_frame(clean, schema) if schema else clean
    return CanonicalInputs(tables=out, paths=paths,
                           missing=list(getattr(inputs, "missing", []) or []),
                           strip_log=log)
