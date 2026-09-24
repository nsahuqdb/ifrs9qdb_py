"""
The monthly EAD curve, one row per contract per month.

This is the exposure the ECL sum prices against, and getting it wrong moves
every provision. The rule, from the Excel `RepaymentScheduleTransform` sheet:

  1. For each scheduled payment, EAD is ``BALANCE + REPAYMENT`` -- the balance
     BEFORE that payment is applied. Using the balance after would understate
     exposure by exactly one instalment at every point on the curve.
  2. ``MonthLifetime`` is whole months from the reporting date to the payment's
     start date.
  3. Per contract, the curve runs from month 0 to the last scheduled month
     EXCLUSIVE, written in descending month order. The exclusive bound is not
     an off-by-one: a contract whose whole schedule falls in the current month
     produces no curve at all, which is right -- there is no future exposure to
     price. On a typical extract that silently removes about 250 contracts,
     and including them inflates the file by six thousand rows.
  4. For a month BEFORE the first scheduled payment, EAD is the current
     outstanding from AccountMaster. For any later month, it is the balance at
     the FIRST scheduled payment falling after that month -- a step function
     that holds flat between instalments rather than interpolating.

``LGDLifetime``, ``PaymentScheduleLifetime`` and ``TotalLimitLifetime`` are
always blank. They belong to a different downstream engine; the columns exist
because LIC reads the schema positionally.

Contracts with no schedule are ABSENT from this file by design -- the engine
falls back to a parametric shape for them. Emitting a flat curve here instead
would hide which contracts have a real schedule and which are estimated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import pick

__all__ = ["build_lifetime_parameter_other", "LPO_COLUMNS"]

LPO_COLUMNS = ["ExtractDate", "ContractId", "MonthLifetime", "EADLifetime",
               "LGDLifetime", "PaymentScheduleLifetime", "TotalLimitLifetime"]


def build_lifetime_parameter_other(
    repayment_schedule: pd.DataFrame,
    accounts: pd.DataFrame,
    reporting_date,
    extract_date: str,
    id_map: dict[str, str] | None = None,
    contracts: set[str] | None = None,
) -> pd.DataFrame:
    """Build the monthly EAD curve for every contract with a schedule.

    ``id_map`` maps the raw source contract id to the id LIC uses; the schedule
    is keyed on the raw id while the output must carry the transformed one.

    ``contracts`` restricts the output to the live book. The repayment schedule
    carries rows for facilities that have since closed or matured -- a couple
    of hundred on a typical extract -- and a curve for a contract LIC has never
    heard of is at best ignored and at worst joined to the wrong thing.
    """
    if repayment_schedule is None or len(repayment_schedule) == 0:
        return pd.DataFrame(columns=LPO_COLUMNS)

    ref = pd.to_datetime(reporting_date)
    key = pick(repayment_schedule, "KEY_1", "ContractId", "CONTRACTID")
    start = pd.to_datetime(pick(repayment_schedule, "START_DAT", "START_DATE",
                                "StartDate"), errors="coerce", format="mixed")
    balance = pd.to_numeric(pick(repayment_schedule, "BALANCE", "BAL"),
                            errors="coerce")
    repayment = pd.to_numeric(pick(repayment_schedule, "REPAYMENT",
                                   "PRINCE_DUE", "PrincipalDue"),
                              errors="coerce")
    if key is None or start is None:
        return pd.DataFrame(columns=LPO_COLUMNS)
    if balance is None:
        balance = pd.Series(np.nan, index=repayment_schedule.index)
    if repayment is None:
        repayment = pd.Series(0.0, index=repayment_schedule.index)

    sched = pd.DataFrame({
        "contract": key.astype(str).str.strip(),
        "start": start,
        # the balance BEFORE the payment is applied
        "ead": balance.fillna(0.0) + repayment.fillna(0.0),
    }).dropna(subset=["start"])
    sched["month"] = ((sched["start"].dt.year - ref.year) * 12
                      + (sched["start"].dt.month - ref.month))
    sched = sched[sched["month"] >= 0]
    if contracts is not None:
        sched = sched[sched["contract"].isin(contracts)]
    if len(sched) == 0:
        return pd.DataFrame(columns=LPO_COLUMNS)

    # current outstanding, for the months before the first scheduled payment
    outstanding: dict[str, float] = {}
    if accounts is not None and len(accounts):
        acid = pick(accounts, "CONTRACTID", "ContractId")
        bal = pd.to_numeric(pick(accounts, "ONBALANCE", "OnBalance"),
                            errors="coerce")
        if acid is not None and bal is not None:
            outstanding = dict(zip(acid.astype(str).str.strip(),
                                   bal.fillna(0.0)))

    rows: list[tuple] = []
    for contract, g in sched.groupby("contract", sort=False):
        g = g.sort_values("month")
        months = g["month"].to_numpy()
        eads = g["ead"].to_numpy()
        end_month = int(months.max())
        first_month = int(months.min())
        current = float(outstanding.get(contract, eads[0]))

        # searchsorted with side="right" gives the FIRST payment strictly after
        # month m, which is the step-function rule above.
        for m in range(0, end_month):
            # Month 0 is TODAY'S balance from the account master, not the
            # schedule's first figure. The two differ whenever a payment falls
            # in the current month: the schedule already nets it off, while the
            # exposure at the reporting date does not. Taking the schedule
            # value here understates month 0 on about 2,300 curves.
            if m == 0 or m < first_month:
                rows.append((contract, m, current))
                continue
            idx = int(np.searchsorted(months, m, side="right"))
            rows.append((contract, m, float(eads[min(idx, len(eads) - 1)])))

    if not rows:
        return pd.DataFrame(columns=LPO_COLUMNS)

    out = pd.DataFrame(rows, columns=["contract_raw", "MonthLifetime",
                                      "EADLifetime"])
    out["ContractId"] = (out["contract_raw"].map(id_map)
                         if id_map else out["contract_raw"])
    out["ContractId"] = out["ContractId"].fillna(out["contract_raw"])
    out["ExtractDate"] = extract_date
    # Descending month, as the reference writes it.
    out = out.sort_values(["contract_raw", "MonthLifetime"],
                          ascending=[True, False]).reset_index(drop=True)
    for c in ("LGDLifetime", "PaymentScheduleLifetime", "TotalLimitLifetime"):
        out[c] = ""
    return out[LPO_COLUMNS]
