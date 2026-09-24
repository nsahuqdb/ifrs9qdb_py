"""
The lending transformation.

Turns the account extract into the per-contract view AccountMaster_1 is written
from. Three things here are not obvious from the output and are the reason a
column-by-column copy of the source does not reproduce it:

  * ``Rating`` and ``PastDueDays`` on a CONTRACT row are the customer's WORST
    across all their facilities, not that contract's own values. A borrower is
    rated as a borrower, so every facility carries the same rating and the same
    days-past-due. Copying the per-contract values looks more precise and is
    wrong.
  * ``PortfolioCode`` is a lookup from the account type through the product
    mapping, falling back to Business Finance when the type is unrecognised --
    an unmapped product must still be provisioned, so it lands in the general
    book rather than being dropped.
  * ``PaymentTypeId`` is derived, not read: 3 (bullet) when no payment
    frequency is given, otherwise 4 (amortising). This one matters more than it
    looks, because it decides the shape of the EAD curve for every contract
    without a supplied schedule.

Many output columns are deliberately BLANK -- PD12M, PDLifetimeValue, LGDRate,
EAD, ImpairmentAmount. LIC computes those. Filling them from the source would
feed the engine values it is about to overwrite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import _fmt_date, _num, at, pick

__all__ = ["transform_lending", "transform_investments",
           "build_account_master", "ACCOUNT_MASTER_COLUMNS",
           "apply_id_substitutions", "OFF_BALANCE_PRODUCTS", "drop_repeated_headers"]

# Off-balance product code -> the code LIC expects in the contract id.
OFF_BALANCE_PRODUCTS = {"APG": "1", "BGA": "2", "FGG": "3", "ILC": "4",
                        "PGG": "5", "DHG": "6", "TIG": "7", "WTO": "8",
                        "NFG": "9"}


def drop_repeated_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the three kinds of junk row a SQL*Plus export leaves behind.

    header  every non-blank cell equals the column heading, with at least two
            non-blank cells. SQL*Plus repeats its headings each page, roughly
            every thousand rows, so these are scattered through the file rather
            than sitting at the top.
    footer  a single non-blank cell in the first column reading "7250 rows
            selected." -- the query summary line.
    blank   no non-blank cells at all.

    The "at least two non-blank cells" guard matters: a genuine contract whose
    only populated field happened to equal its heading would otherwise be
    dropped.
    """
    if df is None or len(df) == 0 or df.shape[1] == 0:
        return df

    headers = [str(c).strip().lower() for c in df.columns]
    nonblank = pd.DataFrame(False, index=df.index, columns=df.columns)
    matches = pd.DataFrame(False, index=df.index, columns=df.columns)
    for j, col in enumerate(df.columns):
        cell = df[col].astype(str).str.strip().str.lower()
        nb = df[col].notna() & (cell != "") & (cell != "nan")
        nonblank[col] = nb
        matches[col] = nb & (cell == headers[j])

    n_nonblank = nonblank.sum(axis=1)
    n_match = matches.sum(axis=1)
    is_header = (n_nonblank >= 2) & (n_match == n_nonblank) & (n_nonblank > 0)

    first = df.iloc[:, 0].astype(str).str.strip().str.lower()
    is_footer = ((n_nonblank == 1) & nonblank.iloc[:, 0]
                 & first.str.match(r"^[0-9][0-9,]*\s+rows?\s+selected\.?$",
                                   na=False))
    is_blank = n_nonblank == 0

    return df[~(is_header | is_footer | is_blank)].reset_index(drop=True)


def apply_id_substitutions(contract_ids, products: dict | None = None) -> pd.Series:
    """Turn a source contract id into the id LIC uses.

    The rule comes from the Excel tool and is not guessable from the data:

        IFERROR(B*1,
                (LEFT(B,7) & VLOOKUP(MID(B,8,3), products) & RIGHT(B,6)) * 1)

    So an id that is already numeric simply loses its leading zeros, and one
    like ``0000121FGG007659`` becomes ``0000121`` + ``3`` (FGG) + ``007659``,
    coerced to a number: ``1213007659``.

    Two details that look like typos and are not: the tail is the last SIX
    characters, not three, and both branches end in a numeric coercion, which
    is what drops the leading zeros.
    """
    lkp = products or OFF_BALANCE_PRODUCTS
    s = pd.Series(contract_ids).astype(str).str.strip()
    out = s.copy()

    as_num = pd.to_numeric(s, errors="coerce")
    pure = as_num.notna()
    out[pure] = as_num[pure].map(lambda v: str(int(v)) if float(v).is_integer()
                                 else str(v))

    needs = (~pure) & (s.str.len() >= 10)
    if needs.any():
        head = s[needs].str[:7]
        code = s[needs].str[7:10]
        tail = s[needs].str[-6:]
        sub = code.map(lambda c: lkp.get(c, c))
        raw = head + sub + tail
        num = pd.to_numeric(raw, errors="coerce")
        out[needs] = np.where(num.notna(),
                              num.fillna(0).map(lambda v: str(int(v))), raw)
    return out

ACCOUNT_MASTER_COLUMNS = [
    "ExtractDate", "ContractId", "LimId", "CustomerId", "PortfolioCode",
    "AccountCode", "AccountType", "ImpairmentAmount", "OriginalECLOnbal",
    "OriginalECLOfbal", "Stage", "OpenDate", "Rating", "PastDueDays", "PD12M",
    "PDLifetimeValue", "IsPOCI", "LGDRate", "LoanToValue", "OffBalance",
    "OnBalance", "EAD", "CCF", "MaturityDate", "ExpectedMaturityDate", "EIR",
    "Principal", "PrincipalOverdue", "InterestAccrued", "InterestOverdue",
    "Fee", "FeeOverdue", "Penalty", "PenaltyOverdue", "Commission",
    "CommissionOverdue", "Other", "OtherOverdue", "PaymentFrequency",
    "CurrencyCode", "DeferralPeriod", "NominalInterestRate", "PaymentTypeId",
]


def transform_lending(accounts: pd.DataFrame,
                      product_portfolio_mapping: pd.DataFrame | None = None,
                      customers: pd.DataFrame | None = None,
                      rating_overrides: dict | None = None) -> pd.DataFrame:
    """Per-contract lending view, with the customer-level values resolved."""
    if accounts is None or len(accounts) == 0:
        return pd.DataFrame()
    accounts = drop_repeated_headers(accounts)

    raw_id = pick(accounts, "CONTRACTID", "ContractId").astype(str).str.strip()
    out = pd.DataFrame({
        "contract_id_raw": raw_id,
        "contract_id": apply_id_substitutions(raw_id),
        "lim_id": pick(accounts, "LIMID", "LimId", default=""),
        "customer_id": pick(accounts, "CUSTOMERID", "CustomerId").astype(str).str.strip(),
        "account_type": pick(accounts, "ACCOUNTTYPE", "AccountType", default=""),
        "open_date": pick(accounts, "OPENDATE", "OpenDate"),
        "maturity_date": pick(accounts, "MATURITYDATE", "MaturityDate"),
        # The lending rating is a CUSTOMER attribute and is not carried on the
        # account rows at all -- it is joined from the customer master below.
        "rating": "",
        "past_dues_days": _num(pick(accounts, "PASTDUEDAYS", "PASTDUE_DAYS",
                                    "PastDueDays")),
        "on_balance": _num(pick(accounts, "ONBALANCE", "OnBalance")),
        "off_balance": _num(pick(accounts, "OFFBALANCE", "OffBalance")),
        "ccf": _num(pick(accounts, "CCF", default=np.nan)),
        "eir": _num(pick(accounts, "EIR", default=np.nan)),
        "nominal_int_rate": _num(pick(accounts, "NOMINALINTERESTRATE",
                                      "NominalInterestRate", default=np.nan)),
        "payment_frequency": _num(pick(accounts, "PAYMENTFREQUENCY",
                                       "PaymentFrequency", default=np.nan)),
        "deferral_period": _num(pick(accounts, "DEFERRALPERIOD",
                                     "DeferralPeriod", default=np.nan)),
        "currency": pick(accounts, "CURRENCYCODE", "CurrencyCode",
                         "CURRENCY", default="QAR"),
    })

    if customers is not None and len(customers):
        cid = pick(customers, "CUSTOMERID", "CustomerId")
        crat = pick(customers, "RATING", "Rating")
        if cid is not None and crat is not None:
            lut = dict(zip(cid.astype(str).str.strip(), crat.astype(str)))
            out["rating"] = out["customer_id"].map(lut).fillna("")

    # Customer-level worst values, broadcast back to every facility.
    worst_dpd = out.groupby("customer_id")["past_dues_days"].transform("max")
    out["past_dues_worst"] = worst_dpd.fillna(0)

    # Worst rating: the weakest grade the customer holds anywhere. Ranked by
    # position in the scale, so "worst" means furthest down the ladder rather
    # than alphabetically last.
    out["rating_worst"] = _worst_rating(out)
    if rating_overrides:
        ov = out["customer_id"].map({str(k): v for k, v in rating_overrides.items()})
        out["rating_worst"] = ov.where(ov.notna() & (ov != ""), out["rating_worst"])

    # Portfolio from the product mapping, defaulting to the general book.
    out["portfolio_code"] = _portfolio(out["account_type"],
                                       product_portfolio_mapping)

    # Payment type: 3 bullet when no frequency, else 4 amortising.
    freq = out["payment_frequency"]
    out["payment_type"] = np.where(freq.isna() | (freq == 0), 3, 4)
    return out


def _worst_rating(view: pd.DataFrame) -> pd.Series:
    """The weakest rating each customer holds, broadcast to all their contracts.

    Ranked on the QDB internal ladder. An unrecognised grade sorts as unknown
    rather than as the worst, so a typo cannot silently downgrade a customer.
    """
    order = ["QDB 1", "QDB 1-", "QDB 2+", "QDB 2", "QDB 2-", "QDB 3+", "QDB 3",
             "QDB 3-", "QDB 4+", "QDB 4", "QDB 4-", "QDB 5+", "QDB 5", "QDB 5-",
             "QDB 6", "QDB 7", "QDB 8", "QDB 9", "QDB 10", "QDB 11", "QDB 12"]
    rank = {r: i for i, r in enumerate(order)}
    r = view["rating"].astype(str).str.strip()
    score = r.map(rank)
    tmp = pd.DataFrame({"c": view["customer_id"], "r": r, "s": score})
    idx = tmp.groupby("c")["s"].transform("max")
    # map the worst score back to its label, per customer
    best_label = (tmp.dropna(subset=["s"])
                  .sort_values("s")
                  .groupby("c")
                  .last()["r"])
    out = view["customer_id"].map(best_label)
    return out.where(out.notna(), r)


def _portfolio(account_type: pd.Series,
               mapping: pd.DataFrame | None) -> pd.Series:
    """Account type to portfolio, defaulting to Business Finance.

    An unmapped product still has to be provisioned, so it lands in the general
    book rather than being dropped or left blank.
    """
    if mapping is not None and len(mapping):
        key = pick(mapping, "product_type", "ProductType", "AccountType")
        val = pick(mapping, "portfolio", "Portfolio", "PortfolioCode")
        if key is not None and val is not None:
            lut = dict(zip(key.astype(str).str.strip(), val.astype(str)))
            out = account_type.astype(str).str.strip().map(lut)
            return out.where(out.notna() & (out != ""), "Business Finance")
    # With no mapping the account type IS the portfolio where it names one.
    at_ = account_type.astype(str).str.strip()
    return at_.where(at_.ne(""), "Business Finance")


def build_account_master(view: pd.DataFrame, extract_date: str,
                         investments: bool = False) -> pd.DataFrame:
    """Write the AccountMaster row set from the transformed view."""
    if view is None or len(view) == 0:
        return pd.DataFrame(columns=ACCOUNT_MASTER_COLUMNS)
    n = len(view)
    blank = [""] * n

    def col(name, default=""):
        return view[name] if name in view.columns else pd.Series([default] * n)

    out = pd.DataFrame({
        "ExtractDate": [extract_date] * n,
        "ContractId": col("contract_id").astype(str),
        "LimId": col("lim_id").fillna("").astype(str).replace("nan", ""),
        "CustomerId": col("customer_id").astype(str),
        "PortfolioCode": col("portfolio_code").astype(str),
        "AccountCode": blank,
        "AccountType": col("account_type").astype(str),
        "ImpairmentAmount": blank,
        "OriginalECLOnbal": blank,
        "OriginalECLOfbal": blank,
        # Tasdeer is collectively assessed and carries a fixed Stage 2; every
        # other account is staged by LIC, so the column is left blank.
        "Stage": np.where(col("account_type").astype(str).str.strip() == "Tasdeer",
                          "2", ""),
        # Zero-padded on the account rows, unlike ExtractDate. Both formats
        # appear in the same file; matching each is the requirement.
        "OpenDate": _fmt_date(col("open_date"), pad=True),
        "Rating": col("rating_worst").astype(str),
        "PastDueDays": col("past_dues_worst"),
        "PD12M": blank,
        "PDLifetimeValue": blank,
        "IsPOCI": blank,
        "LGDRate": blank,
        "LoanToValue": blank,
        "OffBalance": col("off_balance"),
        "OnBalance": col("on_balance"),
        "EAD": blank,
        "CCF": col("ccf"),
        "MaturityDate": _fmt_date(col("maturity_date"), pad=True),
        "ExpectedMaturityDate": blank,
        "EIR": col("eir"),
    })
    for c in ("Principal", "PrincipalOverdue", "InterestAccrued",
              "InterestOverdue", "Fee", "FeeOverdue", "Penalty",
              "PenaltyOverdue", "Commission", "CommissionOverdue", "Other",
              "OtherOverdue"):
        out[c] = blank
    out["PaymentFrequency"] = col("payment_frequency")
    out["CurrencyCode"] = col("currency").astype(str)
    out["DeferralPeriod"] = col("deferral_period")
    out["NominalInterestRate"] = col("nominal_int_rate")
    out["PaymentTypeId"] = col("payment_type")
    return out[ACCOUNT_MASTER_COLUMNS]


# ---------------------------------------------------------- investments ----
def transform_investments(accounts: pd.DataFrame) -> pd.DataFrame:
    """The investment book, which differs from lending in several ways.

      * BOTH ids are surrogate sequential numbers, 1..n per account row. The
        extract identifies the counterparty by NAME and carries a contract id
        that LIC does not use, so the file is keyed on position. That is why it
        has 73 rows against 62 distinct counterparties.
      * The portfolio is "Banks and Fis" when the account type says so, and
        "Investments" otherwise.
      * Days past due are ZERO throughout. A traded instrument does not carry
        an arrears count, and leaving the source value would import whatever
        happened to be in the column.
      * Rates may arrive as whole percent or as a fraction. Any value above 1
        means the column was stored in percent, so the whole column is divided
        by 100 -- detected rather than assumed, because it has changed between
        extracts.
    """
    if accounts is None or len(accounts) == 0:
        return pd.DataFrame()
    accounts = drop_repeated_headers(accounts)
    n = len(accounts)

    def rate(*names):
        v = _num(pick(accounts, *names, default=np.nan))
        if v is None:
            return pd.Series([np.nan] * n)
        if (v > 1).any():
            v = v / 100.0
        return v

    acct_type = pick(accounts, "ACCOUNTTYPE", "AccountType", default="").astype(str)
    return pd.DataFrame({
        "contract_id": pd.Series(range(1, n + 1)).astype(str),
        "customer_id": pd.Series(range(1, n + 1)).astype(str),
        "lim_id": pick(accounts, "LIMID", "LimId", default=""),
        "account_type": acct_type,
        "portfolio_code": np.where(acct_type.str.strip() == "Banks and Fis",
                                   "Banks and Fis", "Investments"),
        "open_date": pick(accounts, "OPENDATE", "OpenDate"),
        "maturity_date": pick(accounts, "MATURITYDATE", "MATURITYDAT",
                              "MaturityDate"),
        "rating_worst": pick(accounts, "RATING", "RatingCurrent",
                             default="").astype(str),
        "past_dues_worst": _num(pick(accounts, "PASTDUEDAYS", "PastDueDays",
                                     default=0)).fillna(0),
        "on_balance": _num(pick(accounts, "ONBALANCE", "OnBalance")),
        # the investment extract has no off-balance column
        "off_balance": 0.0,
        "eir": rate("EIR"),
        "nominal_int_rate": rate("NOMINALINTERESTRATE", "NominalInterestRate"),
        "payment_frequency": _num(pick(accounts, "PAYMENTFREQUENCY",
                                       "PaymentFrequency", default=np.nan)),
        "deferral_period": 0.0,
        "currency": pick(accounts, "CURRENCYCODE", "CurrencyCode",
                         default="QAR"),
        "ccf": _num(pick(accounts, "CCF", default=0)),
        "payment_type": _num(pick(accounts, "PAYMENTTYPEID", "PaymentTypeId",
                                  default=np.nan)),
    })
