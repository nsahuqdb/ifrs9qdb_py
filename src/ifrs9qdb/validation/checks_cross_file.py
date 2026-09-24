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
from ._helpers import col, fail, fk_detail, has, ok
from .framework import Severity, Validator

__all__ = ["CROSS_FILE_STAGE_VALIDATORS", "CONFIG_COVERAGE_VALIDATORS"]


def _fk(child_table, child_col, parent_table, parent_col, label, parent_label):
    def check(inputs, _ct=child_table, _cc=child_col, _pt=parent_table,
              _pc=parent_col, _l=label, _pl=parent_label):
        if not (has(inputs, _ct) and has(inputs, _pt)):
            return ok()
        c = col(inputs[_ct], *_cc)
        p = col(inputs[_pt], *_pc)
        if c is None or p is None:
            return fail(0, f"key column missing on {_ct} or {_pt}")
        return fk_detail(c, p, _l, _pl)
    return check


def _v_collateral_unallocated(inputs):
    if not (has(inputs, "Collateral") and has(inputs, "AccountCollateralAllocation")):
        return ok()
    coll = as_id(col(inputs["Collateral"], "COLLATERALID", "collateral_id"))
    alloc = set(as_id(col(inputs["AccountCollateralAllocation"],
                          "COLLATERALID", "collateral_id")))
    orphan = sorted(set(coll[coll != ""]) - alloc)
    if not orphan:
        return ok()
    return fail(len(orphan),
                f"{len(orphan)} collateral record(s) are allocated to nothing, "
                "so their value is never applied",
                examples=orphan[:10])


def _v_collateral_value_valid(inputs):
    if not has(inputs, "Collateral"):
        return ok()
    df = inputs["Collateral"]
    cid = col(df, "COLLATERALID", "collateral_id")
    val = pd.to_numeric(col(df, "COLLATERALVALUE", "collateral_value"),
                        errors="coerce")
    if cid is None or val is None:
        return fail(0, "CollateralId or CollateralValue column missing")
    allocated = set()
    if has(inputs, "AccountCollateralAllocation"):
        allocated = set(as_id(col(inputs["AccountCollateralAllocation"],
                                  "COLLATERALID", "collateral_id")))
    ids = as_id(cid)
    used = ids.isin(allocated) if allocated else pd.Series(True, index=ids.index)
    bad = used & (val.isna() | (val <= 0))
    n = int(bad.sum())
    if n == 0:
        return ok()
    return fail(n, f"{n} allocated collateral record(s) have no positive value",
                examples=list(ids[bad].head(10)))


def _v_alloc_sum_per_collateral(inputs):
    if not has(inputs, "AccountCollateralAllocation"):
        return ok()
    df = inputs["AccountCollateralAllocation"]
    cid = col(df, "COLLATERALID", "collateral_id")
    pct = pd.to_numeric(col(df, "ALLOCATIONPERCENTAGE", "allocation_percentage"),
                        errors="coerce")
    if cid is None or pct is None:
        return fail(0, "CollateralId or AllocationPercentage column missing")
    tot = pd.DataFrame({"c": as_id(cid), "p": pct}).groupby("c")["p"].sum()
    over = tot[tot > 100.5]
    if over.empty:
        return ok()
    return fail(len(over),
                f"{len(over)} collateral record(s) are allocated more than once "
                "over, so the same security covers more than it is worth",
                examples=[f"{k}={v:.2f}%" for k, v in over.head(10).items()])


CROSS_FILE_STAGE_VALIDATORS: list[Validator] = [
    Validator("XFILE_ACA_collateral_exists", Severity.ERROR,
              "Every CollateralId in AccountCollateralAllocation exists in Collateral",
              _fk("AccountCollateralAllocation", ("COLLATERALID", "collateral_id"),
                  "Collateral", ("COLLATERALID", "collateral_id"),
                  "allocation CollateralId", "Collateral"),
              context="AccountCollateralAllocation", tags=("cross_file",),
              rationale="An allocation pointing at a collateral record that does "
                        "not exist makes coverage NaN, which zeroes the "
                        "contract's provision with no error anywhere.",
              remediation="Ask IT whether the collateral extract is filtered "
                          "differently from the allocation extract."),
    Validator("XFILE_collateral_unallocated", Severity.WARN,
              "Every CollateralId in Collateral is referenced by an allocation",
              _v_collateral_unallocated, context="Collateral",
              tags=("cross_file",),
              rationale="Unallocated collateral is value the bank holds and does "
                        "not get credit for.",
              remediation="Confirm with Credit whether the allocation is missing "
                          "or the security is genuinely unassigned."),
    Validator("XFILE_collateral_value_valid", Severity.WARN,
              "Allocated collateral has a positive CollateralValue",
              _v_collateral_value_valid, context="Collateral",
              tags=("cross_file",),
              rationale="A zero or missing value contributes nothing, so the "
                        "allocation looks like cover and is not.",
              remediation="Get the valuation from Credit Admin."),
    Validator("XFILE_ACA_allocation_sum_per_collateral", Severity.WARN,
              "Sum of AllocationPercentage per CollateralId is <= 100.5%",
              _v_alloc_sum_per_collateral,
              context="AccountCollateralAllocation", tags=("cross_file",),
              rationale="Allocating one security more than once over gives more "
                        "cover than the security is worth.",
              remediation="Review the allocations for the named collateral."),
    Validator("XFILE_AM_customer_in_staging_flags", Severity.WARN,
              "Every lending CustomerId has a CustomerStagingFlag row",
              _fk("AccountMaster", ("CUSTOMERID", "customer_id"),
                  "CustomerStagingFlag", ("CUSTOMERID", "customer_id"),
                  "AccountMaster.CustomerId", "CustomerStagingFlag"),
              context="AccountMaster", tags=("cross_file",),
              rationale="No staging row means no watchlist or restructuring "
                        "flag, so the customer cannot be staged above Stage 1 "
                        "except by DPD.",
              remediation="Check the staging extract's filter."),
    Validator("XFILE_AM_customer_in_industry", Severity.WARN,
              "Every lending CustomerId has an IndustryCode row",
              _fk("AccountMaster", ("CUSTOMERID", "customer_id"),
                  "IndustryCode", ("CUSTOMERID", "customer_id"),
                  "AccountMaster.CustomerId", "IndustryCode"),
              context="AccountMaster", tags=("cross_file",),
              rationale="Without an industry the exposure is missing from every "
                        "sector concentration figure.",
              remediation="Ask IT to extend the industry extract."),
    Validator("XFILE_AM_contract_in_origination", Severity.WARN,
              "Every lending ContractId has an Origination row",
              _fk("AccountMaster", ("CONTRACTID", "contract_id"),
                  "Origination", ("CONTRACTID", "contract_id"),
                  "AccountMaster.ContractId", "Origination"),
              context="AccountMaster", tags=("cross_file",),
              rationale="The origination rating is what the SICR test compares "
                        "against. Without it the contract cannot migrate to "
                        "Stage 2 on rating deterioration.",
              remediation="Check the origination extract's filter."),
    Validator("XFILE_RS_orphans", Severity.WARN,
              "Every RepaymentSchedule ContractId exists in AccountMaster",
              _fk("RepaymentSchedule", ("CONTRACTID", "contract_id"),
                  "AccountMaster", ("CONTRACTID", "contract_id"),
                  "RepaymentSchedule.ContractId", "AccountMaster"),
              context="RepaymentSchedule", tags=("cross_file",),
              rationale="A schedule for a contract that is not on the book is "
                        "usually a closed account still in the extract.",
              remediation="Confirm the account extract's closing filter."),
    Validator("XFILE_AMI_customer_in_CMI", Severity.ERROR,
              "Every investment CustomerId exists in CustomerMasterInvestments",
              _fk("AccountMasterInvestments", ("CUSTOMERID", "customer_id"),
                  "CustomerMasterInvestments", ("CUSTOMERID", "customer_id"),
                  "AccountMasterInvestments.CustomerId",
                  "CustomerMasterInvestments"),
              context="AccountMasterInvestments", tags=("cross_file",),
              rationale="The counterparty carries the external rating. Without "
                        "it the holding has no PD bucket.",
              remediation="Check the counterparty extract."),
    Validator("XFILE_AMI_customer_in_CSFI", Severity.WARN,
              "Every investment CustomerId has a CustomerStagingFlagInvestments row",
              _fk("AccountMasterInvestments", ("CUSTOMERID", "customer_id"),
                  "CustomerStagingFlagInvestments", ("CUSTOMERID", "customer_id"),
                  "AccountMasterInvestments.CustomerId",
                  "CustomerStagingFlagInvestments"),
              context="AccountMasterInvestments", tags=("cross_file",),
              rationale="No staging row means the holding stays Stage 1 whatever "
                        "the counterparty's condition.",
              remediation="Check the investment staging extract."),
    Validator("XFILE_AMI_account_in_origination", Severity.WARN,
              "Every investment AccountId has an OriginationInvestments row",
              _fk("AccountMasterInvestments", ("CONTRACTID", "contract_id",
                                               "ACCOUNTID", "account_id"),
                  "OriginationInvestments", ("CONTRACTID", "contract_id",
                                             "ACCOUNTID", "account_id"),
                  "AccountMasterInvestments.AccountId", "OriginationInvestments"),
              context="AccountMasterInvestments", tags=("cross_file",),
              rationale="Without an origination grade the SICR test cannot run "
                        "on the investment book.",
              remediation="Check the investment origination extract."),
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
    return set(pd.Series(c).astype(str).str.strip())


def _v_product_portfolio_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    known = _static_keys(static, "product_portfolio_mapping", "product_type")
    if known is None:
        known = _static_keys(static, "product_portfolio_mapping", "account_type")
    if known is None:
        return ok()
    c = col(inputs["AccountMaster"], "ACCOUNTTYPE", "account_type")
    if c is None:
        return ok()
    seen = pd.Series(c).astype(str).str.strip()
    missing = sorted({v for v in seen.unique() if v and v.lower() != "nan"} - known)
    if not missing:
        return ok()
    n = int(seen.isin(missing).sum())
    return fail(n, f"{len(missing)} product type(s) map to no portfolio, so no "
                   "PD curve resolves for them",
                examples=missing[:10])


def _v_internal_rating_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    scale = static.get("master_rating_scale")
    if scale is None or len(scale) == 0:
        return ok()
    rt = col(scale, "rating_type")
    names = col(scale, "rating")
    if names is None:
        return ok()
    if rt is not None:
        internal = scale[pd.Series(rt).astype(str).str.lower().str.startswith("int")]
        known = set(pd.Series(col(internal, "rating")).astype(str).str.strip())
    else:
        known = set(pd.Series(names).astype(str).str.strip())
    c = col(inputs["AccountMaster"], "RATING", "rating")
    if c is None:
        return ok()
    seen = pd.Series(c).astype(str).str.strip()
    bad = {v for v in seen.unique() if v and v.lower() != "nan"} - known
    if not bad:
        return ok()
    n = int(seen.isin(bad).sum())
    return fail(n, f"{len(bad)} internal rating(s) are not in the master scale",
                examples=sorted(bad)[:10])


def _v_portfolio_referential(static=None):
    if static is None:
        return ok()
    pf = static.get("portfolios")
    mapping = static.get("product_portfolio_mapping")
    if pf is None or mapping is None or len(pf) == 0:
        return ok()
    known = set(pd.Series(col(pf, "portfolio", "portfolio_code")
                          ).astype(str).str.strip())
    mapped = col(mapping, "portfolio", "portfolio_code")
    if mapped is None:
        return ok()
    missing = sorted({v for v in pd.Series(mapped).astype(str).str.strip().unique()
                      if v and v.lower() != "nan"} - known)
    if not missing:
        return ok()
    return fail(len(missing),
                f"{len(missing)} mapped portfolio(s) are not in portfolios.csv",
                examples=missing[:10])


def _v_collateral_type_coverage(inputs, static=None):
    if not has(inputs, "Collateral") or static is None:
        return ok()
    known = _static_keys(static, "collateral_types", "collateral_type_id")
    if known is None:
        return ok()
    c = col(inputs["Collateral"], "COLLATERALTYPEID", "collateral_type_id")
    if c is None:
        return ok()
    seen = as_id(c)
    known = {str(k).strip() for k in known}
    known |= {k[:-2] for k in known if k.endswith(".0")}
    missing = sorted({v for v in seen.unique() if v} - known)
    if not missing:
        return ok()
    n = int(seen.isin(missing).sum())
    return fail(n, f"{len(missing)} collateral type(s) have no haircut defined, "
                   "so they are treated as unsecured",
                examples=missing[:10])


def _v_industry_sector_coverage(inputs, static=None):
    if not has(inputs, "IndustryCode") or static is None:
        return ok()
    known = _static_keys(static, "industry_sector_mapping", "industry_code")
    if known is None:
        return ok()
    c = col(inputs["IndustryCode"], "INDUSTRYCODE", "industry_code")
    if c is None:
        return ok()
    seen = pd.Series(c).astype(str).str.strip()
    missing = sorted({v for v in seen.unique() if v and v.lower() != "nan"} - known)
    if not missing:
        return ok()
    n = int(seen.isin(missing).sum())
    return fail(n, f"{len(missing)} industry code(s) map to no sector",
                examples=missing[:10])


def _v_off_balance_products_coverage(inputs, static=None):
    if not has(inputs, "AccountMaster") or static is None:
        return ok()
    known = _static_keys(static, "off_balance_products", "product_code")
    if known is None:
        return ok()
    c = col(inputs["AccountMaster"], "CONTRACTID", "contract_id")
    if c is None:
        return ok()
    ids = pd.Series(c).astype(str).str.strip()
    # The product code is the three characters after the first seven, and only
    # in the alphanumeric contract ids - a numeric id carries no product code.
    codes = ids[ids.str.len() >= 10].str[7:10]
    codes = codes[codes.str.isalpha()]
    if codes.empty:
        return ok()
    missing = sorted(set(codes.unique()) - known)
    if not missing:
        return ok()
    n = int(codes.isin(missing).sum())
    return fail(n, f"{len(missing)} off-balance product code(s) in contract ids "
                   "are not mapped, so the id transformation cannot resolve them",
                examples=missing[:10])


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
