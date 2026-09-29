"""Agreement BETWEEN files, and coverage of the config against the data.

This is the group that matters most and the easiest to skip, because every
file passes its own checks. A collateral allocation pointing at a collateral
record that does not exist is valid in both files and produces NaN coverage,
which zeroes a contract's provision with no error anywhere.

Ids match the R package exactly.
"""
from __future__ import annotations

import pandas as pd

from ..ids import as_id
from ._helpers import col, fail, has, ok, pad4, text
from .framework import Severity, Validator

__all__ = ["CROSS_FILE_STAGE_VALIDATORS", "CONFIG_COVERAGE_VALIDATORS"]


_WS = str.maketrans({" ": " ", "​": " ", "‌": " ",
                     "‍": " ", "﻿": " "})


def _xf_chr(df, *names) -> pd.Series:
    """R's .xf_chr(): a key column as text that joins across extracts.

    Non-breaking and zero-width spaces become spaces, runs of whitespace
    collapse, the ends are trimmed, and blanks are dropped -- Oracle HTML
    exports render spaces as U+00A0, and issuer names used as ids carry
    different internal spacing between two files.
    """
    c = col(df, *names) if df is not None else None
    if c is None:
        return pd.Series([], dtype=object)
    x = as_id(c).map(lambda v: " ".join(str(v).translate(_WS).split()))
    return x[x != ""].reset_index(drop=True)


def _missing_from(child_df, child_cols, parent_df, parent_cols):
    """setdiff(unique(child), unique(parent)), in R's first-seen order."""
    have = set(_xf_chr(parent_df, *parent_cols))
    return [v for v in pd.unique(_xf_chr(child_df, *child_cols)) if v not in have]


def _sample(values, n=15) -> str:
    return ", ".join(list(values)[:n])


def _v_aca_collateral_exists(inputs):
    if not (has(inputs, "AccountCollateralAllocation") and has(inputs, "Collateral")):
        return ok()
    aca, coll = inputs["AccountCollateralAllocation"], inputs["Collateral"]
    have = set(_xf_chr(coll, "collateral_id"))
    ref = _xf_chr(aca, "collateral_id")
    bad = [v for v in pd.unique(ref) if v not in have]
    if not bad:
        return ok()
    nrows = int(ref.isin(bad).sum())
    rows = aca[as_id(col(aca, "collateral_id")).str.strip().isin(bad).to_numpy()]
    ncontracts = int(pd.Series(_xf_chr(rows, "contract_id")).nunique())
    return {"passed": False, "count": nrows,
            "detail": (f"{len(bad)} CollateralId(s) referenced by {nrows} row(s) "
                       f"are MISSING from Collateral.xlsx. Sample: {_sample(bad)}"
                       f" ({ncontracts} contract(s) affected - LIC may omit their "
                       "provision)"),
            "examples": bad[:10]}


def _v_collateral_unallocated(inputs):
    if not (has(inputs, "AccountCollateralAllocation") and has(inputs, "Collateral")):
        return ok()
    used = set(_xf_chr(inputs["AccountCollateralAllocation"], "collateral_id"))
    all_ids = list(pd.unique(_xf_chr(inputs["Collateral"], "collateral_id")))
    idle = [v for v in all_ids if v not in used]
    if not idle:
        return ok()
    return {"passed": False, "count": len(idle),
            "detail": (f"{len(idle)} of {len(all_ids)} collateral id(s) have no "
                       f"allocation row (no benefit taken). Sample: {_sample(idle)}"),
            "examples": idle[:10]}


def _v_collateral_value_valid(inputs, static=None):
    """Allocated collateral with no value -- and how much of it matters.

    Most collateral types carry a 100% haircut and give no benefit whatever
    their value, so a zero value on those changes nothing. The count that
    moves the provision is the zero-valued collateral of a type that WOULD
    reduce the loss (haircut below 100%: property, bank guarantees).
    """
    if not (has(inputs, "AccountCollateralAllocation") and has(inputs, "Collateral")):
        return ok()
    aca, coll = inputs["AccountCollateralAllocation"], inputs["Collateral"]
    val = col(coll, "value", "collateral_value")
    if val is None:
        return ok()
    used = set(_xf_chr(aca, "collateral_id"))
    cid = as_id(col(coll, "collateral_id")).str.strip()
    v = pd.to_numeric(val, errors="coerce")
    m = cid.isin(used) & (v.isna() | (v <= 0))
    bad = list(pd.unique(cid[m]))
    if not bad:
        return ok()
    n_benefit = 0
    ct = static.get("collateral_types") if static is not None else None
    tcol = col(coll, "collateral_type_id")
    if ct is not None and tcol is not None:
        hc = dict(zip(as_id(col(ct, "collateral_type_id")).str.strip(),
                      pd.to_numeric(col(ct, "haircut_general"), errors="coerce")))
        h = pd.Series([hc.get(x) for x in as_id(tcol)[m]], dtype=float)
        n_benefit = int(pd.Series(cid[m].to_numpy())[(h < 1).to_numpy()].nunique())
    return {"passed": False, "count": len(bad),
            "detail": (f"{len(bad)} allocated collateral id(s) have missing/zero "
                       f"CollateralValue (no LGD benefit); {n_benefit} of them "
                       "are of a type that would reduce the loss (haircut < 100%). "
                       f"Sample: {_sample(bad)}"),
            "examples": bad[:10]}


def _v_alloc_sum_per_collateral(inputs):
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    df = inputs["AccountCollateralAllocation"]
    pcol = col(df, "allocation_percentage")
    if pcol is None:
        return ok()
    cid = as_id(col(df, "collateral_id")).str.strip()
    pct = pd.to_numeric(pcol, errors="coerce")
    keep = (cid != "") & pct.notna()
    sums = pct[keep].groupby(cid[keep]).sum()
    over = sums[sums > 100.5]
    if over.empty:
        return ok()
    order = sorted(over.index, key=lambda k: (-over[k], k))
    return {"passed": False, "count": len(over),
            "detail": (f"{len(over)} collateral id(s) allocated above 100% (max "
                       f"{over.max():.1f}%) - benefit double-counted. Sample: "
                       f"{', '.join(order[:10])}"),
            "examples": order[:10]}


def _missing_check(child, child_cols, parent, parent_cols, template):
    def check(inputs, _c=child, _cc=child_cols, _p=parent, _pc=parent_cols,
              _t=template):
        if not (has(inputs, _c) and has(inputs, _p)):
            return ok()
        bad = _missing_from(inputs[_c], _cc, inputs[_p], _pc)
        if not bad:
            return ok()
        return {"passed": False, "count": len(bad),
                "detail": _t.format(n=len(bad), sample=_sample(bad)),
                "examples": bad[:10]}
    return check


def _xv(id, severity, description, fn, context, rationale, remediation):
    return Validator(id, severity, description, fn, context=context,
                     rationale=rationale, remediation=remediation,
                     tags=("pre_run", "cross_file"))


# The eleven checks of R/validators_cross_file.R, in its order, rule for rule
# and message for message.
CROSS_FILE_STAGE_VALIDATORS: list[Validator] = [
    _xv("XFILE_ACA_collateral_exists", Severity.ERROR,
        "Every CollateralId in AccountCollateralAllocation exists in Collateral",
        _v_aca_collateral_exists, "AccountCollateralAllocation",
        "Both files are cut from the same collateral module on the same night, "
        "so every allocated CollateralId must appear in the Collateral extract. "
        "When one is missing, the ETL still runs but the LIC engine cannot value "
        "the allocation and has been observed to DROP THE PROVISION for the "
        "affected facilities entirely - a silent understatement.",
        "Raise with the data team: the Collateral extract is missing rows that "
        "AccountCollateralAllocation references. Regenerate both files from the "
        "same business date."),
    _xv("XFILE_collateral_unallocated", Severity.WARN,
        "Every CollateralId in Collateral is referenced by at least one allocation",
        _v_collateral_unallocated, "Collateral",
        "A collateral with no allocation row signals the two extracts have "
        "drifted apart. The collateral gives no benefit (conservative), but the "
        "drift itself should be raised.",
        "Ask the data team to confirm both spools ran on the same business date."),
    _xv("XFILE_collateral_value_valid", Severity.WARN,
        "Allocated collateral has a positive CollateralValue",
        _v_collateral_value_valid, "Collateral",
        "CollateralValue comes from nvl(appraisal_value, market_value). An "
        "allocated collateral with a missing/zero value contributes no benefit, "
        "so LGD silently rises to the unsecured 45%.",
        "Send the listed collateral ids to the collateral unit to fix the "
        "appraisal or market value at source."),
    _xv("XFILE_ACA_allocation_sum_per_collateral", Severity.WARN,
        "Sum of AllocationPercentage per CollateralId is <= 100.5%",
        _v_alloc_sum_per_collateral, "AccountCollateralAllocation",
        "Each collateral's value is split across the exposures it secures "
        "(loan_bal / sum_loan_bal), so per COLLATERAL the percentages must sum "
        "to about 100%.",
        "Review the allocations for the named collateral."),
    _xv("XFILE_AM_customer_in_staging_flags", Severity.WARN,
        "Every lending CustomerId has a CustomerStagingFlag row",
        _missing_check("AccountMaster", ("customer_id",), "CustomerStagingFlag",
                       ("customer_id",),
                       "{n} lending customer(s) missing from CustomerStagingFlag "
                       "(flags default to 0). Sample: {sample}"),
        "AccountMaster",
        "No staging row means no watchlist or restructuring flag, so the "
        "customer cannot be staged above Stage 1 except by DPD.",
        "Check the staging extract's filter."),
    _xv("XFILE_AM_customer_in_industry", Severity.WARN,
        "Every lending CustomerId has an IndustryCode row",
        _missing_check("AccountMaster", ("customer_id",), "IndustryCode",
                       ("customer_id",),
                       "{n} lending customer(s) have no IndustryCode row "
                       "(sector = NA). Sample: {sample}"),
        "AccountMaster",
        "Without an industry the sector is NA: the customer drops out of the "
        "sector collective-assessment rules and every sector concentration.",
        "Ask IT to extend the industry extract."),
    _xv("XFILE_AM_contract_in_origination", Severity.WARN,
        "Every lending ContractId has an Origination row",
        _missing_check("AccountMaster", ("contract_id",), "Origination",
                       ("contract_id",),
                       "{n} lending contract(s) missing from Origination "
                       "(no SICR baseline). Sample: {sample}"),
        "AccountMaster",
        "The origination rating is what the SICR test compares against.",
        "Check the origination extract's filter."),
    _xv("XFILE_RS_orphans", Severity.WARN,
        "Every RepaymentSchedule ContractId exists in AccountMaster",
        _missing_check("RepaymentSchedule", ("contract_id",), "AccountMaster",
                       ("contract_id",),
                       "{n} contract(s) in RepaymentSchedule are not in "
                       "AccountMaster (schedules ignored). Sample: {sample}"),
        "RepaymentSchedule",
        "A schedule for a contract that is not on the book is usually a closed "
        "account still in the extract; the EAD-curve builder leaves it out.",
        "Confirm the account extract's closing filter."),
    _xv("XFILE_AMI_customer_in_CMI", Severity.ERROR,
        "Every investment CustomerId exists in CustomerMasterInvestments",
        _missing_check("AccountMasterInvestments", ("customer_id",),
                       "CustomerMasterInvestments", ("customer_id",),
                       "{n} investment customer(s) missing from "
                       "CustomerMasterInvestments (fallback rating/PD used). "
                       "Sample: {sample}"),
        "AccountMasterInvestments",
        "The counterparty carries the external rating; without it the holding "
        "falls back to the segment rating.",
        "Check the counterparty extract."),
    _xv("XFILE_AMI_customer_in_CSFI", Severity.WARN,
        "Every investment CustomerId has a CustomerStagingFlagInvestments row",
        _missing_check("AccountMasterInvestments", ("customer_id",),
                       "CustomerStagingFlagInvestments", ("customer_id",),
                       "{n} investment customer(s) missing from "
                       "CustomerStagingFlagInvestments (flags default to 0). "
                       "Sample: {sample}"),
        "AccountMasterInvestments",
        "No staging row means the holding's flags default to 0.",
        "Check the investment staging extract."),
    _xv("XFILE_AMI_account_in_origination", Severity.WARN,
        "Every investment AccountId has an OriginationInvestments row",
        _missing_check("AccountMasterInvestments", ("account_id",),
                       "OriginationInvestments", ("contract_id",),
                       "{n} investment account(s) missing from "
                       "OriginationInvestments (no SICR baseline). "
                       "Sample: {sample}"),
        "AccountMasterInvestments",
        "Without an origination grade the SICR test cannot run on the "
        "investment book.",
        "Check the investment origination extract."),
]


# ------------------------------------------------- config coverage ---------
def _static_keys(static, table, column):
    if static is None:
        return None
    df = static.get(table) if hasattr(static, "get") else None
    if df is None or len(df) == 0:
        return None
    c = col(df, column)
    if c is None:
        return None
    return set(text(c)) - {""}


def _coverage(bad, n, message):
    if not bad:
        return ok()
    return {"passed": False, "count": int(n), "detail": message,
            "examples": list(bad)[:10]}


# The five coverage checks below mirror R's validators_config_coverage.R rule
# for rule and message for message: which values count, in what order they
# are listed, and how blanks are treated. A coverage check that disagrees with
# its R twin by one value is how the two reports stop being comparable.

def _v_product_portfolio_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    have = _static_keys(static, "product_portfolio_mapping", "product_type")
    if have is None:
        return ok()
    c = col(inputs["AccountMaster"], "account_type")
    if c is None:
        return ok()
    bad = [v for v in _first_seen(c) if v not in have]
    return _coverage(bad, text(c).isin(bad).sum(),
                     f"{len(bad)} product type(s) with no portfolio mapping: "
                     + ", ".join(bad))


def _v_internal_rating_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    scale = static.get("master_rating_scale")
    if scale is None or len(scale) == 0:
        return ok()
    rt = col(scale, "rating_type")
    names = col(scale, "rating")
    if names is None or rt is None:
        return ok()
    have = set(text(names[text(rt) == "Internal"]))
    c = col(inputs["AccountMaster"], "RATING", "rating")
    if c is None:
        return ok()
    bad = [v for v in _first_seen(c) if v not in have]
    return _coverage(bad, text(c).isin(bad).sum(),
                     f"{len(bad)} internal rating(s) not in master_rating_scale: "
                     + ", ".join(bad))


def _v_portfolio_referential(static=None):
    if static is None:
        return ok()
    pf = static.get("portfolios")
    mapping = static.get("product_portfolio_mapping")
    if pf is None or mapping is None or len(pf) == 0:
        return ok()
    have = set(text(col(pf, "portfolio_code", "portfolio")))
    mapped = col(mapping, "portfolio")
    if mapped is None:
        return ok()
    bad = [v for v in _first_seen(mapped) if v not in have]
    return _coverage(bad, len(bad),
                     f"{len(bad)} portfolio(s) mapped but missing from "
                     "portfolios.csv: " + ", ".join(bad))


def _first_seen(values) -> list:
    """Distinct non-blank values in the order they first appear, as R's
    unique() returns them -- so a message lists them in the same order."""
    v = text(values)
    return list(pd.unique(v[(v != "") & (v.str.upper() != "NA")]))


def _v_collateral_type_coverage(inputs, static=None):
    """Mirrors R's CONFIG_collateral_type_coverage exactly.

    The haircut is looked up by collateral_type_id, so an unmapped type has no
    haircut. The R and Python engines then treat that collateral as worth
    nothing; what LIC does with it is not documented, which is the reason to
    stop it here rather than find out from the provision.
    """
    if not has(inputs, "Collateral") or static is None:
        return ok()
    ct = static.get("collateral_types") if hasattr(static, "get") else None
    if ct is None or len(ct) == 0:
        return ok()
    have = set(as_id(col(ct, "collateral_type_id")))
    df = inputs["Collateral"]
    c = col(df, "collateral_type_id")
    if c is None:
        # the positional column the Collateral schema puts it in
        c = df.iloc[:, 3] if df.shape[1] >= 4 else pd.Series([], dtype=object)
    need = _first_seen(as_id(c))
    bad = [v for v in need if v not in have]
    if not bad:
        return ok()
    n = int(as_id(c).isin(bad).sum())
    return {"passed": False, "count": n,
            "detail": (f"{len(bad)} collateral type id(s) not in "
                       f"collateral_types: {', '.join(bad)}"),
            "examples": bad[:10]}


def _v_industry_sector_coverage(inputs, static=None):
    """Mirrors R's CONFIG_industry_sector_coverage exactly.

    The sector comes from the leading 4-digit code of the industry
    DESCRIPTION, because the numeric INDUST column drops the leading zero
    (113 for 0113). Both sides are compared as zero-padded 4-digit codes.
    """
    if not has(inputs, "IndustryCode") or static is None:
        return ok()
    ism = static.get("industry_sector_mapping") if hasattr(static, "get") else None
    if ism is None or len(ism) == 0:
        return ok()
    have = set(pad4(col(ism, "industry_code"))) - {""}
    df = inputs["IndustryCode"]
    src = col(df, "industry_description")
    if src is None:
        src = col(df, "industry_code")
    if src is None:
        return ok()
    codes = pad4(src)
    need = list(pd.unique(codes[codes != ""]))
    bad = [v for v in need if v not in have]
    if not bad:
        return ok()
    n = int(codes.isin(bad).sum())
    return {"passed": False, "count": n,
            "detail": (f"{len(bad)} industry code(s) not in "
                       f"industry_sector_mapping: {', '.join(bad[:25])}"),
            "examples": bad[:10]}


def _v_off_balance_products_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    have = _static_keys(static, "off_balance_products", "product_code")
    if have is None:
        return ok()
    c = col(inputs["AccountMaster"], "CONTRACTID", "contract_id")
    if c is None:
        return ok()
    ids = text(c)
    ids = ids[ids != ""]
    # A non-numeric id of at least ten characters embeds the product code at
    # characters 8-10 ("0000112FGG000471" -> "FGG"); a numeric id carries none.
    numeric = pd.to_numeric(ids, errors="coerce").notna()
    cand = ids[~numeric & (ids.str.len() >= 10)]
    codes = cand.str[7:10]
    bad = [v for v in pd.unique(codes) if v not in have]
    return _coverage(bad, codes.isin(bad).sum(),
                     f"{len(bad)} off-balance product code(s) not in "
                     "off_balance_products: " + ", ".join(bad))


CONFIG_COVERAGE_VALIDATORS: list[Validator] = [
    Validator("CONFIG_product_portfolio_coverage", Severity.ERROR,
              "Every AccountMaster account_type maps to a portfolio",
              _v_product_portfolio_coverage, context="AccountMaster",
              tags=("config",),
              rationale="The portfolio is a LOOKUP from the product type. "
                        "Without it every product becomes its own portfolio and "
                        "no PD curve resolves - the run completes and prices "
                        "almost nothing.",
              remediation="Add the product to product_portfolio_mapping.csv."),
    Validator("CONFIG_off_balance_products_coverage", Severity.ERROR,
              "Every off-balance product code in a contract id is mapped",
              _v_off_balance_products_coverage, context="AccountMaster",
              tags=("config",),
              rationale="The contract id transformation substitutes the product "
                        "code. An unmapped code leaves the id untransformed and "
                        "it then matches nothing in LIC.",
              remediation="Add the code to off_balance_products.csv."),
    Validator("CONFIG_internal_rating_coverage", Severity.ERROR,
              "Every AccountMaster internal rating is in the master scale",
              _v_internal_rating_coverage, context="AccountMaster",
              tags=("config",),
              rationale="An unrecognised grade resolves to no bucket, so the "
                        "contract prices to zero.",
              remediation="Add the grade to master_rating_scale.csv, or correct "
                          "it at source."),
    Validator("CONFIG_portfolio_referential", Severity.ERROR,
              "Every mapped portfolio exists in portfolios.csv with a rating_type",
              _v_portfolio_referential, context="static", tags=("config",),
              rationale="The rating_type decides which scale a portfolio uses. "
                        "A portfolio missing from portfolios.csv gets no scale "
                        "and no curve.",
              remediation="Add the portfolio to portfolios.csv."),
    Validator("CONFIG_collateral_type_coverage", Severity.WARN,
              "Every Collateral collateral_type_id is in collateral_types.csv",
              _v_collateral_type_coverage, context="Collateral",
              tags=("config",),
              rationale="An unknown type has no haircut, so the security is "
                        "treated as worthless and the provision is over-stated.",
              remediation="Add the type with its haircut to collateral_types.csv."),
    Validator("CONFIG_industry_sector_coverage", Severity.WARN,
              "Every customer industry code maps to a sector",
              _v_industry_sector_coverage, context="IndustryCode",
              tags=("config",),
              rationale="Unmapped codes drop out of sector concentration.",
              remediation="Add the code to industry_sector_mapping.csv."),
]
