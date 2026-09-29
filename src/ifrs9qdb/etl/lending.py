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

import re

import numpy as np
import pandas as pd

from .transform import _fmt_date, _num, at, pick, r_format_numeric
from ..ids import as_id

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
                      rating_overrides: dict | None = None,
                      industry: pd.DataFrame | None = None,
                      static: dict | None = None,
                      reporting_date=None) -> pd.DataFrame:
    """Per-contract lending view, with the customer-level values resolved.

    Mirrors ``R/transform_lending.R``. ``static`` is the reference data
    (master rating scale, collective-assessment rules, segment fallbacks,
    industry-to-sector mapping, staging thresholds); when omitted the packaged
    copy is used, which is byte-identical to the R package's ``inst/static``.
    ``industry`` is the IndustryCode extract, which the rating chain needs to
    find a customer's sector.
    """
    if accounts is None or len(accounts) == 0:
        return pd.DataFrame()
    accounts = drop_repeated_headers(accounts)

    raw_id = as_id(pick(accounts, "CONTRACTID", "ContractId"))
    out = pd.DataFrame({
        "contract_id_raw": raw_id,
        "contract_id": apply_id_substitutions(raw_id),
        "lim_id": pick(accounts, "LIMID", "LimId", default=""),
        "customer_id": as_id(pick(accounts, "CUSTOMERID", "CustomerId")),
        "account_type": pick(accounts, "ACCOUNTTYPE", "AccountType", default=""),
        "open_date": pick(accounts, "OPENDATE", "OpenDate"),
        # The SQL export truncates the alias to eleven characters, so the
        # column arrives as MATURITYDAT. Missing it left every maturity blank.
        "maturity_date": pick(accounts, "MATURITYDATE", "MATURITYDAT",
                              "MaturityDate"),
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

    if static is None:
        from .static_ref import load_static_reference
        static = load_static_reference()
    scale = _master_scale(static)

    cust_rating = pd.Series(np.nan, index=out.index, dtype=object)
    if customers is not None and len(customers):
        cid = pick(customers, "CUSTOMERID", "CustomerId")
        crat = pick(customers, "RATING", "Rating")
        if cid is not None and crat is not None:
            lut = dict(zip(as_id(cid), crat.astype(str).str.strip()))
            cust_rating = out["customer_id"].map(lut)

    # Customer-level worst DPD, broadcast back to every facility.
    worst_dpd = out.groupby("customer_id")["past_dues_days"].transform("max")
    out["past_dues_worst"] = worst_dpd.fillna(0)

    # The rating chain, V4 Transformation!Z. A customer grade counts only if it
    # is on the master scale; "Unrated" is not, and must not reach the engine
    # as a grade, because it has no PD bucket and prices the facility at zero.
    sector = _customer_sector(out["customer_id"], industry,
                              static.get("industry_sector_mapping"))
    external = cust_rating.where(cust_rating.isin(set(scale["rating"])))
    out["rating"] = _derive_rating_lending(
        external, sector, out["past_dues_worst"], out["account_type"],
        static.get("collective_assessment_rules"),
        static.get("segment_fallback_ratings"))

    # Worst rating per customer, on the HIERARCHY, written back as the internal
    # label at that position -- Transformation!AB/AC.
    out["rating_worst"] = _worst_rating(out, scale)
    if rating_overrides:
        ov = out["customer_id"].map({str(k): v for k, v in rating_overrides.items()})
        out["rating_worst"] = ov.where(ov.notna() & (ov != ""), out["rating_worst"])

    # Portfolio from the product mapping, defaulting to the general book.
    out["portfolio_code"] = _portfolio(out["account_type"],
                                       product_portfolio_mapping)

    # EIR arrives as a RAW PERCENT ("4.2" means 4.2% a year) and every consumer
    # wants a decimal. Leaving it unscaled discounts at (1 + 4.2) rather than
    # (1 + 0.042): a loss forty-two months out is divided by 128 instead of
    # 1.11, which prices every long-dated month at nothing. The investments
    # transform already scaled; this path did not, so 7,259 of 7,334 contracts
    # on a June book carried the percent straight into the discount factor.
    out["eir"], out["nominal_int_rate"] = _effective_rates(
        out["eir"], out["nominal_int_rate"], out["account_type"],
        product_portfolio_mapping)

    out["maturity_date"] = _extend_maturity(
        out["maturity_date"], reporting_date,
        static.get("staging_thresholds"))

    # A zero frequency is "no schedule", written blank, as R does.
    freq = out["payment_frequency"].where(out["payment_frequency"] != 0)
    out["payment_frequency"] = freq
    # Payment type: 3 bullet when no frequency, else 4 amortising.
    out["payment_type"] = np.where(freq.isna(), 3, 4)
    return out


def _effective_rates(eir, nir, account_type, mapping):
    """EIR as a decimal, with the workbook's fallback for a missing rate.

    Mirrors ``R/transform_lending.R:335-402``. A zero or blank EIR is not a
    zero-interest loan, it is a gap, and the V4 workbook fills it in two tiers:

    1. the mean EIR of that account type, but only for the seven Business
       Finance product types the workbook itself listed;
    2. otherwise the mean of those seven self-means.

    Team-added Business Finance products and the off-balance products both fall
    to tier 2, because V4's VLOOKUP missed them and dropped through to
    ``AVERAGE(AM9:AM15)``. Matching that is the whole point.
    """
    raw = pd.to_numeric(eir, errors="coerce")
    rate = raw.where(raw.notna() & (raw != 0)) / 100.0

    at = account_type.astype(str).str.strip()
    v4_bf: list[str] = []
    if mapping is not None and len(mapping):
        key = pick(mapping, "product_type", "ProductType", "AccountType")
        val = pick(mapping, "portfolio", "Portfolio", "PortfolioCode")
        src = pick(mapping, "source", "Source")
        if key is not None and val is not None:
            is_bf = val.astype(str).str.strip() == "Business Finance"
            # No source column: treat every BF row as V4, which is what the R
            # port does for older snapshots.
            is_v4 = (src.astype(str).str.strip() == "v4") if src is not None \
                else pd.Series(True, index=key.index)
            v4_bf = list(key[is_bf & is_v4].astype(str).str.strip())

    self_means = rate.groupby(at).mean()
    bf_means = self_means[self_means.index.isin(v4_bf)].dropna()
    mean_of_means = float(bf_means.mean()) if len(bf_means) else float(rate.mean())
    if not np.isfinite(mean_of_means):
        mean_of_means = 0.0

    fill = at.map(lambda a: self_means.get(a, np.nan) if a in v4_bf else np.nan)
    fill = pd.to_numeric(fill, errors="coerce").fillna(mean_of_means)
    rate = rate.fillna(fill)

    # NominalInterestRate is the same quantity on the same scale, and the R
    # writer emits the filled EIR for it.
    nir_raw = pd.to_numeric(nir, errors="coerce")
    nir_out = nir_raw.where(nir_raw.notna() & (nir_raw != 0)) / 100.0
    return rate, nir_out.fillna(rate)


def _master_scale(static: dict) -> pd.DataFrame:
    """The master rating scale: rating, rating_type, hierarchy."""
    ms = static.get("master_rating_scale")
    if ms is None or not len(ms):
        return pd.DataFrame(columns=["rating", "rating_type", "hierarchy"])
    return pd.DataFrame({
        "rating": pick(ms, "rating", "Rating").astype(str).str.strip(),
        "rating_type": pick(ms, "rating_type", "RatingType").astype(str).str.strip(),
        "hierarchy": pd.to_numeric(pick(ms, "hierarchy", "Hierarchy"),
                                   errors="coerce"),
    })


def _customer_sector(customer_id: pd.Series, industry: pd.DataFrame | None,
                     mapping: pd.DataFrame | None) -> pd.Series:
    """Sector per facility, from the customer's industry DESCRIPTION.

    Mirrors ``lookup_sector``: the first four characters of the description
    are the ISIC code with its leading zero ("0113 Growing of vegetables"),
    which the numeric INDUST column has lost (113).
    """
    none = pd.Series(np.nan, index=customer_id.index, dtype=object)
    if industry is None or not len(industry) or mapping is None or not len(mapping):
        return none
    cid = pick(industry, "CUSTOMERID", "CustomerId")
    desc = pick(industry, "DESCRIPTION", "INDUSTRYDESCRIPTION",
                "IndustryDescription")
    if cid is None or desc is None:
        return none
    by_customer = dict(zip(as_id(cid), desc.astype(str)))
    code = pick(mapping, "industry_code", "IndustryCode").astype(str).str.strip()
    code = code.str.replace(r"\.0$", "", regex=True).str.zfill(4)
    sector_of = dict(zip(code, pick(mapping, "sector", "Sector").astype(str)))
    prefix = customer_id.map(by_customer).astype(str).str[:4]
    return prefix.map(sector_of)


def _derive_rating_lending(external: pd.Series, sector: pd.Series,
                           dpd_worst: pd.Series, account_type: pd.Series,
                           rules: pd.DataFrame | None,
                           fallback: pd.DataFrame | None) -> pd.Series:
    """The per-facility rating, V4 Transformation!Z (``derive_rating_lending``).

    1. The customer's own grade, when it is on the master scale.
    2. Otherwise, for Agriculture, Fisheries and Livestock customers, the
       collective-assessment grade for their worst days past due
       (``VLOOKUP(dpd, ..., TRUE)``: the largest threshold not above it).
    3. Otherwise the segment fallback: Al Dhameen facilities take the Al
       Dhameen grade, everything else the unrated-customer grade.
    """
    out = external.astype(object).copy()
    has = out.notna() & (out.astype(str).str.strip() != "")
    out[~has] = np.nan
    needs = ~has
    sec = sector.astype(str).str.strip().str.lower().where(sector.notna())
    dpd = pd.to_numeric(dpd_worst, errors="coerce").fillna(0).to_numpy()

    if rules is not None and len(rules):
        r_sec = pick(rules, "sector", "Sector").astype(str)
        r_thr = pd.to_numeric(pick(rules, "dpd_threshold"), errors="coerce")
        r_rat = pick(rules, "rating", "Rating").astype(str)
        for s in r_sec.unique():
            short = re.sub(r"\s*Sector\s*$", "", s, flags=re.I)
            names = {short.lower(),
                     re.sub("Lifestock", "Livestock", short, flags=re.I).lower()}
            rows = (needs & sec.isin(names)).to_numpy()
            if not rows.any():
                continue
            sel = (r_sec == s).to_numpy()
            order = np.argsort(r_thr[sel].to_numpy(), kind="stable")
            thr = r_thr[sel].to_numpy()[order]
            grades = r_rat[sel].to_numpy()[order]
            idx = np.searchsorted(thr, dpd[rows], side="right")
            idx[idx == 0] = 1
            out.iloc[np.flatnonzero(rows)] = grades[idx - 1]

    fb = {}
    if fallback is not None and len(fallback):
        fb = dict(zip(pick(fallback, "segment", "Segment").astype(str),
                      pick(fallback, "fallback_rating", "FallbackRating").astype(str)))
    still = needs & out.isna()
    dhameen = account_type.astype(str).str.strip().str.lower() == "al dhameen"
    out[still & dhameen] = fb.get("Al Dhameen Customers", np.nan)
    out[still & ~dhameen] = fb.get("Unrated Customer (Internal Rating)", np.nan)
    return out


def _worst_rating(view: pd.DataFrame, scale: pd.DataFrame) -> pd.Series:
    """The customer's worst grade, as the INTERNAL label at that hierarchy.

    Mirrors Transformation!AB/AC: rank every facility's grade on the master
    scale, take the customer's maximum hierarchy, and write the internal grade
    sitting at that position. An externally rated customer therefore comes out
    on the internal ladder (Aa2, hierarchy 3, becomes QDB 1-).

    This used to rank on a hard-coded ladder that omitted QDB 1+, 6+ and 6-
    and listed QDB 10 to 12, which do not exist.
    """
    hier_of = dict(zip(scale["rating"], scale["hierarchy"]))
    h = view["rating"].astype(str).str.strip().map(hier_of)
    worst = h.groupby(view["customer_id"]).transform("max")
    internal = scale[scale["rating_type"] == "Internal"].sort_values(
        "hierarchy", kind="stable")["rating"].tolist()
    def label(v):
        if pd.isna(v):
            return ""
        i = int(v)
        return internal[i - 1] if 1 <= i <= len(internal) else ""
    return worst.map(label)


def _extend_maturity(maturity, reporting_date, thresholds) -> pd.Series:
    """A maturity already passed is pushed out, as V4 Transformation!AM does.

    ``IF(raw < reporting, reporting + maturity_extension_days, raw)``. Strictly
    less than: a facility maturing ON the reporting date is not extended. The
    description in staging_thresholds.csv says ``<=``; the R port and the V4
    formula both use ``<``, and this follows them.
    """
    raw = pd.to_datetime(maturity, errors="coerce", format="mixed")
    if reporting_date is None:
        return raw
    ref = pd.Timestamp(reporting_date).normalize()
    days = 365
    if thresholds is not None and len(thresholds):
        k = pick(thresholds, "key", "Key")
        v = pick(thresholds, "value", "Value")
        if k is not None and v is not None:
            m = dict(zip(k.astype(str), v))
            try:
                days = int(float(m.get("maturity_extension_days", days)))
            except (TypeError, ValueError):
                pass
    needs = raw.notna() & (raw < ref)
    return raw.where(~needs, ref + pd.Timedelta(days=days))


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
        # M/D/YYYY with no leading zeros, as R's format_date_col writes it and
        # as both delivered runs carry it. An earlier version zero-padded the
        # account-row dates on the belief that the reference did; the R output
        # LIC actually received does not.
        "OpenDate": _fmt_date(col("open_date")),
        "Rating": col("rating_worst").astype(str),
        # Investments are written with PastDueDays 0, as R does
        # (R/output_writers.R: past_due_days = rep(0, nrow(ia)), the V4
        # convention): the investment book is staged on rating deterioration,
        # not days past due. Carrying the raw DPD here staged a bond 100 days
        # past due at Stage 3 in Python only -- and pulled five sister
        # holdings of the same issuer to Stage 2 by contagion.
        # INPUT_AccountMasterInvestments_dpd_ignored reports any such DPD.
        "PastDueDays": ([0] * n if investments else col("past_dues_worst")),
        "PD12M": blank,
        "PDLifetimeValue": blank,
        "IsPOCI": blank,
        "LGDRate": blank,
        "LoanToValue": blank,
        # Lending balances as R's fmt_numeric writes them (see
        # r_format_numeric); investments are written at four fixed decimals.
        "OffBalance": (col("off_balance") if investments
                       else r_format_numeric(col("off_balance"))),
        "OnBalance": (col("on_balance") if investments
                      else r_format_numeric(col("on_balance"))),
        "EAD": blank,
        "CCF": col("ccf"),
        "MaturityDate": _fmt_date(col("maturity_date")),
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
def transform_investments(accounts: pd.DataFrame, static: dict | None = None,
                          reporting_date=None) -> pd.DataFrame:
    """The investment book, which differs from lending in several ways.

      * The contract id is the extract's own account id, and the customer id
        is the counterparty NAME, as ``R/transform_investments.R`` writes them
        and as both delivered runs carry them (``1028, DUKHAN BANK``). An
        earlier version replaced both with 1..n on the belief that LIC keys the
        file on position; the output LIC actually received does not. One row
        per account, so a counterparty with several holdings repeats.
      * The rating is the extract's grade when it is on the EXTERNAL scale,
        otherwise the investment-segment fallback (Baa3) --
        ``derive_rating_investment``. A grade the scale does not know has no PD
        bucket and would price the holding at zero.
      * Maturity takes the same extension rule as lending.
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

    if static is None:
        from .static_ref import load_static_reference
        static = load_static_reference()
    scale = _master_scale(static)
    external = set(scale.loc[scale["rating_type"] == "External", "rating"])
    fb = static.get("segment_fallback_ratings")
    fallback = ""
    if fb is not None and len(fb):
        m = dict(zip(pick(fb, "segment", "Segment").astype(str),
                     pick(fb, "fallback_rating", "FallbackRating").astype(str)))
        fallback = m.get("Investment Portfolio", "")
    raw_rating = pick(accounts, "RATING", "RatingCurrent", default="").astype(str).str.strip()
    rating = raw_rating.where(raw_rating.isin(external), fallback)

    acct_type = pick(accounts, "ACCOUNTTYPE", "AccountType", default="").astype(str)
    return pd.DataFrame({
        "contract_id": as_id(pick(accounts, "CONTRACTID", "ContractId", "ACCOUNTID")),
        "customer_id": pick(accounts, "CUSTOMERID", "CustomerId",
                            default="").astype(str).str.strip(),
        "lim_id": pick(accounts, "LIMID", "LimId", default=""),
        "account_type": acct_type,
        "portfolio_code": np.where(acct_type.str.strip() == "Banks and Fis",
                                   "Banks and Fis", "Investments"),
        "open_date": pick(accounts, "OPENDATE", "OpenDate"),
        "maturity_date": _extend_maturity(
            pick(accounts, "MATURITYDATE", "MATURITYDAT", "MaturityDate"),
            reporting_date, static.get("staging_thresholds")),
        "rating_worst": rating,
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
        # Blank when the extract leaves it blank, as R writes it.
        "ccf": _num(pick(accounts, "CCF", default=np.nan)),
        "payment_type": _num(pick(accounts, "PAYMENTTYPEID", "PaymentTypeId",
                                  default=np.nan)),
    })
