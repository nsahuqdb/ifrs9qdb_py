"""
Building the eighteen LIC input files.

They fall into three kinds, and knowing which is which matters when something
looks wrong:

  reference   Six files copied from the static reference: Portfolios, Ratings,
              RatingTypes, PortfolioRatingType, CollateralType, FxRate. They
              change only when the model changes, not per quarter.
  transform   Ten files derived from the Oracle extracts, mostly a rename and
              a type coercion away from the source.
  computed    Two files the model produces: LifeTimeParameterOther, the monthly
              EAD curve from the repayment schedule, and StPD, the PD term
              structure from the macro model.

Column ORDER is part of the contract: LIC reads these positionally in places,
so the writers emit the columns in a fixed order rather than whatever the
transformation happened to leave.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["OUTPUT_SPECS", "write_outputs", "transform_collateral",
           "transform_allocation", "transform_customer_master",
           "transform_staging_flags", "transform_origination",
           "build_ead_curves"]


# The eighteen, with the kind of thing each is. Used by the writer and by the
# reconciliation report, so a missing file can be explained rather than merely
# noted.
OUTPUT_SPECS = {
    "Portfolios.csv": "reference",
    "Ratings.csv": "reference",
    "RatingTypes.csv": "reference",
    "PortfolioRatingType.csv": "reference",
    "CollateralType.csv": "reference",
    "FxRate.csv": "reference",
    "AccountMaster_1.csv": "transform",
    "AccountMaster_2.csv": "transform",
    "CustomerMaster_1.csv": "transform",
    "CustomerMaster_2.csv": "transform",
    "CustomerStagingFlag_1.csv": "transform",
    "CustomerStagingFlag_2.csv": "transform",
    "Origination_1.csv": "transform",
    "Origination_2.csv": "transform",
    "Collateral.csv": "transform",
    "AccountCollateralAllocation.csv": "transform",
    "LifeTimeParameterOther.csv": "computed",
    "StPD.csv": "computed",
}


def _squash(name) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def pick(df: pd.DataFrame, *candidates, default=None):
    """Find a column by any of several names, ignoring punctuation and case.

    The SQL aliases differ between extracts -- CONTRACTID, ContractId,
    contract_id -- and one changing should not break a run.
    """
    if df is None or len(df.columns) == 0:
        return None
    lookup = {_squash(c): c for c in df.columns}
    for cand in candidates:
        hit = lookup.get(_squash(cand))
        if hit is not None:
            return df[hit]
    if default is not None:
        return pd.Series([default] * len(df), index=df.index)
    return None


def at(df: pd.DataFrame, i: int, name: str | None = None):
    """Column by POSITION, with a name as a fallback.

    This is the mapping the original Excel tool used and the R port kept, and
    it is not a shortcut: the SQL*Plus exports truncate their aliases to
    whatever fits, so CustomerStagingFlag arrives as
    ``I, ISWATCHLIST, I.1, I.2, ISLOCAL1, I.3 ...`` where every ``I.n`` is a
    different flag. Only the position says which. Matching on names here would
    silently put IsInsolvency in the IsDefaultInGCC column.
    """
    if df is None or len(df.columns) <= i:
        if name is not None:
            got = pick(df, name)
            if got is not None:
                return got
        return pd.Series([np.nan] * (0 if df is None else len(df)))
    return df.iloc[:, i]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _flag(s):
    """A 0/1 source flag as the boolean LIC reads.

    The extract writes 1 for set and leaves the cell empty otherwise -- not 0 --
    so a blank is False, and only a genuinely absent column stays empty.
    """
    if s is None:
        return ""
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().sum() == 0:
        return ""
    return (n == 1).map({True: "True", False: "False"})


def _bool(s):
    """Write a flag the way LIC reads it: the strings TRUE and FALSE."""
    v = pd.Series(s).fillna(False)
    if v.dtype == object:
        v = v.astype(str).str.strip().str.lower().isin(("1", "true", "t", "yes"))
    else:
        v = pd.to_numeric(v, errors="coerce").fillna(0) == 1
    return v.map({True: "TRUE", False: "FALSE"}).to_numpy()


def _date(s):
    if s is None:
        return None
    return pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=False)



def r_format_numeric(values, digits: int = 7) -> pd.Series:
    """Numbers as R's ``format(x, scientific = FALSE, trim = TRUE)`` writes them.

    R's ``fmt_numeric`` with no ``decimals`` formats the whole COLUMN to a
    common number of decimal places: enough for each value to show ``digits``
    significant figures, maximised over the column. So a balance column holding
    7852.9968 is written with three decimals throughout -- 7852.997 and
    160012.500 -- and the fourth decimal is lost. That is an R artefact rather
    than a rule anybody chose, and it moves a balance by less than 0.0005; it is
    matched here so the two ports' outputs can be compared byte for byte.
    """
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    ok = x.notna() & np.isfinite(x)
    decimals = 0
    for v in x[ok]:
        a = abs(float(v))
        if a == 0:
            continue
        int_digits = int(np.floor(np.log10(a))) + 1
        need = digits
        for s in range(1, digits + 1):
            if float(f"{v:.{s}g}") == float(f"{v:.{digits}g}"):
                need = s
                break
        decimals = max(decimals, need - int_digits)
    out = pd.Series([""] * len(x), index=x.index, dtype=object)
    out[ok] = [f"{float(v):.{decimals}f}" for v in x[ok]]
    return out

def _fmt_date(s, pad: bool = False):
    """Dates as the reference writes them.

    Two formats appear in the SAME file: ExtractDate is M/D/YYYY with no
    leading zeros, while OpenDate and MaturityDate on the account rows are
    zero-padded MM/DD/YYYY. Matching each is the requirement; normalising both
    to one form looked tidier and did not match.

    Built from the components rather than with strftime: the unpadded form
    needs `%-m` on Linux and `%#m` on Windows, and `%-m` raises
    "Invalid format string" there rather than falling back. Formatting the
    numbers directly works everywhere.
    """
    d = _date(s)
    if d is None:
        return None
    if pad:
        return d.dt.strftime("%m/%d/%Y")
    out = pd.Series([""] * len(d), index=d.index, dtype=object)
    ok = d.notna()
    if ok.any():
        out[ok] = (d[ok].dt.month.astype(str) + "/"
                   + d[ok].dt.day.astype(str) + "/"
                   + d[ok].dt.year.astype(str))
    return out


# ------------------------------------------------------------- transforms --
def transform_collateral(raw: pd.DataFrame) -> pd.DataFrame:
    """Collateral master.

    The SQL*Plus export aliases these to single letters, so they are matched by
    several possible names rather than one.
    """
    return pd.DataFrame({
        "ExtractDate": _fmt_date(pick(raw, "EXTRACTDA", "ExtractDate")),
        "CollateralId": pick(raw, "COLLATERALID", "CollateralId"),
        # The SQL aliases are truncated, and only the truncation tells you
        # which is which:  P -> ParentCollateralId (never populated),
        # CO -> CollateralTypeId, COL -> CollateralCurrency. CollateralCode has
        # no source column at all and is emitted empty, as the R writer does.
        "ParentCollateralId": pick(raw, "PARENTCOLLATERALID",
                                   "ParentCollateralId", "P", default=""),
        "CollateralCode": pick(raw, "COLLATERALCODE", "CollateralCode",
                               default=""),
        "CollateralTypeId": pick(raw, "COLLATERALTYPEID", "CollateralTypeId",
                                 "CO", default=""),
        "CollateralCurrency": pick(raw, "COLLATERALCURRENCY",
                                   "CollateralCurrency", "COL", default=""),
        "CollateralValue": _num(pick(raw, "COLLATERALVALUE", "CollateralValue")),
    })


def transform_allocation(raw: pd.DataFrame) -> pd.DataFrame:
    """Which collateral is allocated to which contract.

    An allocation pointing at a collateral record that does not exist makes LIC
    return NaN coverage, which blanks the whole contract's ECL. The transform
    does not drop those rows -- the validators report them, so a silent
    correction here cannot hide a source-data problem.
    """
    # The export repeats its headings every page. R strips them before this
    # point; left in, one arrives as an allocation of collateral "COLLATERALID"
    # to contract "CONTRACTID".
    from .lending import drop_repeated_headers
    raw = drop_repeated_headers(raw)
    return pd.DataFrame({
        "ExtractDate": _fmt_date(at(raw, 0, "EXTRACTDA")),
        "CollateralId": at(raw, 1, "COLLATERALID"),
        "ContractId": at(raw, 2, "CONTRACTID"),
        # The source writes 10.09 for ten per cent; LIC wants the fraction.
        # Getting this wrong scales every collateral allocation by a hundred,
        # which would show up as coverage far above 100% rather than as an error.
        "AllocationPercentage": _num(at(raw, 3, "ALLOCATIONPERCENTAGE")) / 100.0,
    })


def transform_customer_master(raw: pd.DataFrame, customer_ids=None,
                              extract_date: str | None = None,
                              investments: bool = False) -> pd.DataFrame:
    # Every field except CustomerId is written BLANK, and
    # IsIndividualAssessment is the literal "FALSE". That is not an omission:
    # LIC derives the customer attributes itself from the account file, and
    # populating them here would feed it values it is going to overwrite. The
    # source columns exist and are deliberately not used.
    #
    # The investment file differs in one header only: BusinessUnitCode where
    # the lending file has OrganizationalUnitCode.
    unit_col = "BusinessUnitCode" if investments else "OrganizationalUnitCode"
    # Rows follow the CUSTOMER extract for lending, in its own order. The
    # investment file is keyed on the account instead, so its ids are passed in.
    ids = (pd.Series(customer_ids).astype(str).reset_index(drop=True)
           if customer_ids is not None
           else at(raw, 1, "CUSTOMERID").astype(str).str.strip().reset_index(drop=True))
    n = len(ids)
    out = pd.DataFrame({
        "ExtractDate": [extract_date] * n if extract_date else
                       list(_fmt_date(at(raw, 0, "EXTRACTDA"))[:n]),
        "CustomerId": ids.to_numpy(),
    })
    for c in ("PortfolioCode", unit_col, "CustomerCode", "CustomerName",
              "CustomerLimit", "Rating", "PastDueDays", "PD12M",
              "PDLifetimeValue"):
        out[c] = ""
    out["IsIndividualAssessment"] = "FALSE"
    return out


def transform_staging_flags(raw: pd.DataFrame, flags: pd.DataFrame | None = None,
                            extract_date: str | None = None,
                            investments: bool = False) -> pd.DataFrame:
    """Customer staging flags.

    Only IsWatchlist and IsLocal1 are genuinely populated at source; the rest
    are emitted empty because LIC expects the columns to exist. They are kept
    rather than dropped so the file shape matches what LIC reads.
    """
    # These flags are DERIVED, not copied, which is why matching the source
    # column by name produced the wrong answer:
    #
    #   IsDefault       the customer's worst DPD exceeds 90 -- computed, and
    #                   blank in the source
    #   IsWatchlist     the one genuinely populated source flag
    #   IsInsolvency    blank for every row
    #   IsDefaultInGCC  blank for every row
    #   IsLocal1        the restructured override
    #   IsLocal2        always FALSE
    #   IsLocal3        the Stage 2 override; the source has no per-customer
    #                   sheet for it, so the computed Stage 2 indicator stands
    #                   in, and that substitution is worth knowing about
    #   IsLocal4..6     blank
    n = len(raw) if flags is None else len(flags)
    f = flags if flags is not None else pd.DataFrame(index=range(n))

    def col(name, default=None):
        if name in f.columns:
            return f[name]
        return pd.Series([default] * n)

    ids = (f["customer_id"].astype(str).reset_index(drop=True)
           if "customer_id" in f.columns
           else at(raw, 1, "CUSTOMERID").astype(str).str.strip().reset_index(drop=True))
    ids = ids.iloc[:n].reset_index(drop=True) if len(ids) >= n else ids
    n = len(ids)
    dates = ([extract_date] * n if extract_date
             else list(_fmt_date(at(raw, 0, "EXTRACTDA"))[:n]))
    out = pd.DataFrame({
        "ExtractDate": dates,
        "CustomerId": ids.to_numpy(),
        "IsDefault": _bool(col("is_default", False))[:n],
    })
    if investments:
        # The investment book carries no watchlist or override flags, so only
        # IsDefault and IsLocal1 are written and the rest stay blank. Writing
        # FALSE where the reference writes nothing is a real difference to LIC.
        out["IsWatchlist"] = ""
        out["IsInsolvency"] = ""
        out["IsDefaultInGCC"] = ""
        out["IsLocal1"] = _bool(col("is_local1", False))
        for i in range(2, 7):
            out[f"IsLocal{i}"] = ""
        return out

    out["IsWatchlist"] = _bool(col("is_watchlist", False))
    out["IsInsolvency"] = ""
    out["IsDefaultInGCC"] = ""
    out["IsLocal1"] = _bool(col("is_local1", False))
    out["IsLocal2"] = "FALSE"
    # IsLocal3 is the Stage 2 override, and in practice it tracks the
    # watchlist: on the reference run every watchlisted customer carries it.
    # Two more principled-looking guesses -- a computed Stage 2 indicator, and
    # following IsLocal1 -- each disagreed with the reference on hundreds of
    # customers, so this follows what the output actually contains.
    out["IsLocal3"] = _bool(col("is_local3", None) if "is_local3" in f.columns
                            else col("is_watchlist", False))
    for i in range(4, 7):
        out[f"IsLocal{i}"] = ""
    return out


def transform_origination_investments(raw: pd.DataFrame,
                                      accounts: pd.DataFrame,
                                      extract_date: str) -> pd.DataFrame:
    """Origination for the investment book.

    Keyed on the ACCOUNT, with the same sequential surrogate ids as
    AccountMaster_2, so the two files line up row for row. The origination
    extract can carry more rows than there are accounts -- a duplicate, or a
    position closed since -- and joining on the source contract id then
    matching back to account order is what keeps the counts equal.
    """
    n = 0 if accounts is None else len(accounts)
    acct_ids = pick(accounts, "CONTRACTID", "ContractId")
    acct_ids = (acct_ids.astype(str).str.strip().reset_index(drop=True)
                if acct_ids is not None else pd.Series([], dtype=str))

    lookup_pd, lookup_rating = {}, {}
    if raw is not None and len(raw):
        rid = pick(raw, "CONTRACTID", "ContractId")
        if rid is not None:
            rid = rid.astype(str).str.strip()
            p12 = pick(raw, "ORIGINATIONPD12M", "OriginationPD12M", default="")
            rat = pick(raw, "ORIGINATIONRATING", "OriginationRating", default="")
            lookup_pd = dict(zip(rid, p12))
            lookup_rating = dict(zip(rid, rat))

    out = pd.DataFrame({
        "EXTRACTDATE": [extract_date] * n,
        "ContractId": [str(i) for i in range(1, n + 1)],
        "OriginationPD12M": [lookup_pd.get(c, "") for c in acct_ids],
        "OriginationRating": [lookup_rating.get(c, "") for c in acct_ids],
    })
    for flag in ("PastDueDays", "IsWatchlist", "IsInsolvency", "IsDefaultInGCC",
                 "IsLocal1", "IsLocal2", "IsLocal3", "IsLocal4", "IsLocal5",
                 "IsLocal6"):
        out[f"IsOrigination{flag}Stage2"] = ""
    return out


ORIGINATION_FLAGS = ("PastDueDays", "IsWatchlist", "IsInsolvency",
                     "IsDefaultInGCC", "IsLocal1", "IsLocal2", "IsLocal3",
                     "IsLocal4", "IsLocal5", "IsLocal6")


def build_origination_rows(extract_date: str, contract_ids) -> pd.DataFrame:
    """One origination row per account, as ``.build_origination_rows`` in R.

    The ids are the ACCOUNT MASTER's ids -- after the off-balance substitution
    -- in account order, so the two files line up row for row. Every
    origination value is written BLANK. That is what R does and what both
    delivered runs carry: the origination PD and rating are never passed to
    LIC, even when the extract supplies them. The relative SICR test is
    therefore unavailable by construction, not only because the source is
    empty -- see INPUT_DATA_ISSUES.md, I3.
    """
    ids = pd.Series(contract_ids).astype(str).reset_index(drop=True)
    n = len(ids)
    out = pd.DataFrame({
        "EXTRACTDATE": [extract_date] * n,
        "ContractId": ids,
        "OriginationPD12M": [""] * n,
        "OriginationRating": [""] * n,
    })
    for flag in ORIGINATION_FLAGS:
        out[f"IsOrigination{flag}Stage2"] = [""] * n
    return out


def transform_origination(raw: pd.DataFrame, contracts=None) -> pd.DataFrame:
    """Origination view.

    The header is EXTRACTDATE here, not ExtractDate: LIC reads this file with a
    different spelling from the others, and matching it matters more than
    consistency. The Stage 2 origination flags are emitted empty because the
    source does not populate them.

    Restricted to contracts present in the account master when one is supplied:
    the origination extract carries rows for contracts that have since closed,
    and LIC expects the two files to line up.
    """
    out = pd.DataFrame({
        "EXTRACTDATE": _fmt_date(pick(raw, "EXTRACTDA", "EXTRACTDATE",
                                      "ExtractDate")),
        "ContractId": pick(raw, "CONTRACTID", "ContractId"),
        "OriginationPD12M": pick(raw, "ORIGINATIONPD12M", "OriginationPD12M",
                                 default=""),
        "OriginationRating": pick(raw, "ORIGINATIONRATING", "OriginationRating",
                                  default=""),
    })
    for flag in ("PastDueDays", "IsWatchlist", "IsInsolvency", "IsDefaultInGCC",
                 "IsLocal1", "IsLocal2", "IsLocal3", "IsLocal4", "IsLocal5",
                 "IsLocal6"):
        out[f"IsOrigination{flag}Stage2"] = pick(
            raw, f"ISORIGINATION{flag.upper()}STAGE2", default="")
    if contracts is not None and len(contracts):
        keep = set(contracts.astype(str))
        out = out[out["ContractId"].astype(str).isin(keep)].reset_index(drop=True)
    return out


# --------------------------------------------------------------- computed --
def build_ead_curves(schedule: pd.DataFrame, accounts: pd.DataFrame,
                     extract_date=None, max_month: int = 600) -> pd.DataFrame:
    """Monthly exposure at default, from the contractual repayment schedule.

    One row per contract per month: the balance still outstanding at that
    month, which is what the ECL sum prices against. Contracts with no schedule
    are ABSENT here by design -- the engine falls back to a parametric shape
    for them, and inventing a flat curve would hide that.
    """
    if schedule is None or len(schedule) == 0:
        return pd.DataFrame(columns=["ExtractDate", "ContractId", "MonthLifetime",
                                     "EADLifetime", "PrincipalDue", "InterestDue",
                                     "Balance"])

    key = pick(schedule, "KEY_1", "ContractId", "CONTRACTID")
    post = _date(pick(schedule, "POST_DATE", "PostDate"))
    principal = _num(pick(schedule, "PRINCE_DUE", "PrincipalDue", default=0))
    interest = _num(pick(schedule, "PROJ_INT", "InterestDue", default=0))

    sched = pd.DataFrame({
        "contract": key.astype(str),
        "date": post,
        "principal": principal.fillna(0.0),
        "interest": interest.fillna(0.0),
    }).dropna(subset=["date"])

    ref = (pd.to_datetime(extract_date) if extract_date is not None
           else sched["date"].min())
    sched = sched[sched["date"] >= ref]
    if len(sched) == 0:
        return pd.DataFrame(columns=["ExtractDate", "ContractId", "MonthLifetime",
                                     "EADLifetime", "PrincipalDue", "InterestDue",
                                     "Balance"])

    sched["month"] = ((sched["date"].dt.year - ref.year) * 12
                      + (sched["date"].dt.month - ref.month)).clip(lower=0)
    sched = sched[sched["month"] < max_month]

    # opening balance per contract, from the account master where available
    opening = {}
    if accounts is not None and len(accounts):
        cid = pick(accounts, "CONTRACTID", "ContractId")
        bal = _num(pick(accounts, "ONBALANCE", "OnBalance", "PRINCIPALOUTSTANDING"))
        if cid is not None and bal is not None:
            opening = dict(zip(cid.astype(str), bal.fillna(0.0)))

    rows = []
    for contract, g in sched.groupby("contract", sort=False):
        g = g.sort_values("month")
        by_month = g.groupby("month", as_index=False)[["principal", "interest"]].sum()
        start = opening.get(contract, float(by_month["principal"].sum()))
        balance = float(start)
        for r in by_month.itertuples():
            rows.append((contract, int(r.month), balance, r.principal, r.interest))
            balance = max(0.0, balance - float(r.principal))

    out = pd.DataFrame(rows, columns=["ContractId", "MonthLifetime",
                                      "EADLifetime", "PrincipalDue", "InterestDue"])
    out.insert(0, "ExtractDate", ref.strftime("%Y-%m-%d"))
    out["Balance"] = out["EADLifetime"]
    return out


# ---------------------------------------------------------------- writing --
def write_outputs(out_dir, tables: dict[str, pd.DataFrame],
                  verbose: bool = False) -> dict[str, Path]:
    """Write the LIC input files.

    Blank rather than "nan" for missing values: LIC reads these as text in
    places, and the string "nan" is not the same as an empty cell.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, df in tables.items():
        if df is None:
            continue
        path = out_dir / name
        if name.startswith("StPD") and "PDLifetime" in df.columns:
            # Fixed sixteen decimals, as R writes it (output_writers.R:917).
            # pandas otherwise writes small PDs in scientific notation --
            # 6.27e-05 -- and a CSV reader that does not take exponents would
            # misread the whole term structure.
            df = df.copy()
            v = pd.to_numeric(df["PDLifetime"], errors="coerce")
            df["PDLifetime"] = [("" if pd.isna(x) else f"{x:.16f}") for x in v]
        df.to_csv(path, index=False, na_rep="")
        written[name] = path
        if verbose:
            print(f"  {name:<36}{len(df):>8,} rows")
    return written
