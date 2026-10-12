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

Repeated SQL*Plus header rows, the "N rows selected." trailer, blank rows, total
lines at the end of a file and exact duplicate rows are removed when the file is
read (``read_all_inputs`` -> ``clean_rows``), and counted per file, as R's
``.strip_duplicate_header_row`` does -- left in, a header row reaches the
cross-file checks as an allocation of collateral "COLLATERALID".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..ids import as_id

__all__ = ["INPUT_SCHEMAS", "canonicalise", "CanonicalInputs", "strip_junk_rows",
           "clean_rows", "describe_clean_log", "CLEAN_KINDS", "NO_DEDUPE",
           "schema_unread_values"]


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
    # The extract carries EXTRACTDA first and CONTRACTID second, as every
    # other file does; the origination fields follow under truncated headers
    # ("O", "I"), so they are found by position. The positions used to start
    # at 1 in both engines, which read the contract id as the origination PD
    # (INPUT_values_typed reported 1,065 "unreadable" ids); corrected in both.
    "Origination": [
        _c("contract_id", ["CONTRACTID", "KEY_1"], 2),
        _c("origination_pd_12m", ["PD12M"], 3, "numeric", False),
        _c("origination_rating", ["RATING"], 4, required=False),
        _c("origination_dpd", ["PASTDUEDAYS"], 5, "integer", False),
        _c("origination_watchlist", ["ISWATCHLIST"], 6, "integer", False),
    ],
    "OriginationInvestments": [
        _c("contract_id", ["CONTRACTID", "KEY_1"], 2),
        _c("origination_pd_12m", ["PD12M"], 3, "numeric", False),
        _c("origination_rating", ["RATING"], 4, required=False),
    ],
}

_ID_COLUMNS = {"contract_id", "customer_id", "collateral_id", "account_id",
               "lim_id", "collateral_type_id"}


def _original_header(name) -> str:
    """The header as SQL*Plus wrote it, before pandas de-duplicated it (I.1)."""
    return re.sub(r"\.\d+$", "", str(name)).strip().lower()


# What the reader removes from an extract before anything reads it, by kind.
CLEAN_KINDS = ("header", "trailer", "blank", "total", "duplicate")
CLEAN_LABELS = {"header": "repeated header", "trailer": "row-count trailer",
                "blank": "blank", "total": "total line",
                "duplicate": "exact duplicate"}
# Files whose identical rows can be real: a repayment schedule may carry two
# equal instalments on one date, so its rows are never de-duplicated.
NO_DEDUPE = frozenset({"RepaymentSchedule"})
# The label of a total line: "Total", "Totals", "Grand total", "Sub-total",
# "Sum", with or without a colon -- the whole cell, nothing else in it.
_TOTAL_LABEL = re.compile(r"^(?:(?:grand|sub)[\s-]*)?totals?\s*:?$|^sum\s*:?$",
                          re.IGNORECASE)


def clean_rows(df: pd.DataFrame, dedupe: bool = False
               ) -> tuple[pd.DataFrame, dict]:
    """Remove the rows an export adds that are not data, and count them.

    header     every non-blank cell equals its column's ORIGINAL heading, with
               at least two non-blank cells (SQL*Plus repeats its headings each
               page).
    trailer    one non-blank cell, in the first column, reading "7250 rows
               selected."
    blank      no non-blank cell.
    total      at the END of the file only: a row whose first non-blank cell is
               a total label ("Total", "Grand total", "Sum" ...). A row like
               that above the last data row is left alone.
    duplicate  (``dedupe``) a row identical, cell for cell, to an earlier one.
               The first is kept. Two rows with the same id that differ are
               NOT touched -- which of them is right is a question for the
               source, and the uniqueness checks report it.

    Mirrors R's ``.strip_duplicate_header_row`` and ``.drop_exact_duplicates``.
    """
    counts = dict.fromkeys(CLEAN_KINDS, 0)
    if df is None or len(df) == 0 or df.shape[1] == 0:
        return df, counts
    heads = [_original_header(c) for c in df.columns]
    n, m = len(df), df.shape[1]
    nonblank = np.zeros((n, m), dtype=bool)
    match = np.zeros((n, m), dtype=bool)
    cells = []
    for j, c in enumerate(df.columns):
        txt = df[c].astype(str).str.strip()
        nb = (df[c].notna() & (txt != "") & (txt.str.lower() != "nan")).to_numpy()
        nonblank[:, j] = nb
        low = txt.str.lower().to_numpy()
        match[:, j] = nb & (low == heads[j])
        cells.append(np.where(nb, txt.to_numpy(), ""))
    n_nb = nonblank.sum(axis=1)
    n_match = match.sum(axis=1)
    is_header = (n_nb >= 2) & (n_match == n_nb)
    first_col = pd.Series(cells[0]).str.lower()
    is_trailer = ((n_nb == 1) & nonblank[:, 0]
                  & first_col.str.match(r"^[0-9][0-9,]*\s+rows?\s+selected\.?$",
                                        na=False).to_numpy())
    is_blank = n_nb == 0
    first_nb = np.full(n, "", dtype=object)
    for j in range(m - 1, -1, -1):
        first_nb = np.where(nonblank[:, j], cells[j], first_nb)
    looks_total = (n_nb >= 1) & ~is_header & np.array(
        [bool(_TOTAL_LABEL.match(str(v))) for v in first_nb])
    data_row = ~(is_header | is_trailer | is_blank | looks_total)
    last_data = int(np.flatnonzero(data_row).max()) if data_row.any() else -1
    is_total = looks_total & (np.arange(n) > last_data)
    drop = is_header | is_trailer | is_blank | is_total
    is_dup = np.zeros(n, dtype=bool)
    if dedupe and (~drop).any():
        keep = np.flatnonzero(~drop)
        frame = pd.DataFrame({j: col[keep] for j, col in enumerate(cells)})
        is_dup[keep] = frame.duplicated(keep="first").to_numpy()
    counts.update(header=int(is_header.sum()), trailer=int(is_trailer.sum()),
                  blank=int(is_blank.sum()), total=int(is_total.sum()),
                  duplicate=int(is_dup.sum()))
    gone = drop | is_dup
    if not gone.any():
        return df, counts
    return df.loc[~gone].reset_index(drop=True), counts


def strip_junk_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove repeated headers, the row-count trailer, blank rows and trailing
    total lines (``clean_rows`` without de-duplication); the count removed."""
    out, counts = clean_rows(df)
    return out, int(sum(counts.values()))


def describe_clean_log(detail: dict) -> str:
    """'OriginationInvestments.xls: 1 exact duplicate row; ...' for a report."""
    parts = []
    for f, kinds in (detail or {}).items():
        bits = [f"{int(v)} {CLEAN_LABELS.get(k, k)} row{'s' if int(v) != 1 else ''}"
                for k, v in kinds.items() if int(v) > 0]
        if bits:
            parts.append(f"{f}: " + ", ".join(bits))
    return "; ".join(parts)


def _coerce(values: pd.Series, type_: str, canonical: str) -> pd.Series:
    """R's .coerce_to_type(), type by type."""
    if type_ in ("numeric", "integer"):
        v = pd.to_numeric(values, errors="coerce")
        if type_ == "integer":
            # R's as.integer(): truncates toward zero, NA beyond 32 bits
            v = np.trunc(v).where(v.abs() <= 2147483647)
        return v
    if type_ == "date":
        from ..dates import schema_parse_dates
        return schema_parse_dates(values)
    if type_ == "logical":
        from ..dates import r_text
        s = values.map(r_text).astype(object).where(values.notna(), "")
        s = s.astype(str).str.strip().str.upper()
        out = pd.Series(np.nan, index=values.index, dtype=object)
        out[s.isin(["TRUE", "T", "1", "Y", "YES"])] = True
        out[s.isin(["FALSE", "F", "0", "N", "NO"])] = False
        return out
    if canonical in _ID_COLUMNS:
        return as_id(values).replace("", np.nan)
    s = values.where(values.isna(), values.astype(str).str.strip())
    return s.replace({"": np.nan, "nan": np.nan})


# Missing-value markers: a value the typing reads as blank, not as a value it
# could not read. The markers pandas' readers turn into NaN, so both engines
# count the same values whichever reader met the file (R keeps them as text).
_BLANK_MARKERS = {"", "NA", "N/A", "#N/A", "#N/A N/A", "#NA", "NULL", "NAN",
                  "-NAN", "NONE", "<NA>", "-1.#IND", "-1.#QNAN", "1.#IND",
                  "1.#QNAN"}


def _unread(raw: pd.Series, typed: pd.Series, type_: str, source) -> dict | None:
    """R's .unread_values(): the non-blank raw values a date or number column
    holds that did not read as its type."""
    if type_ not in ("date", "numeric", "integer", "logical") or len(raw) == 0:
        return None
    from ..dates import r_text
    txt = raw.map(r_text)
    blank = txt.isna() | txt.astype(str).str.strip().str.upper().isin(_BLANK_MARKERS)
    lost = pd.Series(typed).isna().to_numpy() & ~blank.to_numpy()
    if not lost.any():
        return None
    vals = txt[lost].astype(str).str.strip()
    # the header as the file wrote it: pandas' "O.1" is R's "O...2"
    return {"source": re.sub(r"\.\d+$", "", str(source)).strip(),
            "type": type_, "n": int(lost.sum()),
            "sample": list(dict.fromkeys(vals.tolist()))[:5]}


def _canonical_frame(raw: pd.DataFrame, schema: list[dict]) -> pd.DataFrame:
    """Raw columns kept, canonical columns added -- so a check written against
    either spelling finds its data. What the typing could not read is kept in
    ``attrs["schema_unread"]``, as R keeps it in attr(, "schema_unread")."""
    out = raw.copy()
    upper = {str(c).strip().upper(): c for c in raw.columns}
    unread = {}
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
        typed = _coerce(raw[src], spec["type"], spec["canonical"])
        out[spec["canonical"]] = typed
        u = _unread(raw[src], typed, spec["type"], src)
        if u is not None:
            unread[spec["canonical"]] = u
    out.attrs = {k: v for k, v in raw.attrs.items() if k != "schema_unread"}
    # typed by the schema: the checks read its typed columns, as R's do
    out.attrs["canonical"] = True
    if unread:
        out.attrs["schema_unread"] = unread
    return out


def schema_unread_values(inputs) -> pd.DataFrame:
    """The values the schema could not type, per file and column (R's
    schema_unread_values): a date in a format the schema does not know, text
    in an amount, a number with a thousands separator. The typed column holds
    a blank there, and every check and calculation after it sees a blank."""
    rows = []
    tables = getattr(inputs, "tables", None)
    if tables is None:
        tables = dict(inputs) if isinstance(inputs, dict) else {}
    for key, df in tables.items():
        un = getattr(df, "attrs", {}).get("schema_unread") if df is not None else None
        for cn, u in (un or {}).items():
            rows.append({"file": key, "column": cn, "source": u.get("source", cn),
                         "type": u.get("type"), "n": int(u["n"]),
                         "sample": " | ".join(u.get("sample", []))})
    return pd.DataFrame(rows, columns=["file", "column", "source", "type", "n",
                                       "sample"])


@dataclass
class CanonicalInputs:
    """An InputSet whose tables carry the canonical columns as well."""
    tables: dict[str, pd.DataFrame]
    paths: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    strip_log: dict = field(default_factory=dict)
    clean_log: dict = field(default_factory=dict)   # file -> {kind: rows}

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
    # what the reader already removed, then anything left (tables built
    # without read_all_inputs)
    detail = {f: dict(k) for f, k in
              (getattr(inputs, "clean_log", None) or {}).items()}
    out = {}
    for name, raw in tables.items():
        if raw is None:
            continue
        clean, counts = clean_rows(raw)
        if sum(counts.values()):
            p = paths.get(name)
            d = detail.setdefault(p.name if p is not None else name, {})
            for k, v in counts.items():
                if v:
                    d[k] = d.get(k, 0) + int(v)
        schema = INPUT_SCHEMAS.get(name)
        out[name] = _canonical_frame(clean, schema) if schema else clean
    detail = {f: k for f, k in detail.items() if sum(k.values())}
    log = {f: int(sum(k.values())) for f, k in detail.items()}
    return CanonicalInputs(tables=out, paths=paths,
                           missing=list(getattr(inputs, "missing", []) or []),
                           strip_log=log, clean_log=detail)
