"""
Deriving the customer-level view from the account extract.

Several of the LIC input files are keyed on the CUSTOMER but built from the
ACCOUNT file, not the customer extract. That is why the row counts never lined
up when the customer extract was used directly: the investment customer extract
has 62 rows while the output needs 73, because the output follows the accounts.

The staging flags are computed here too. Only IsWatchlist arrives populated
from source; IsDefault is the customer's worst days-past-due exceeding 90, and
IsLocal1 and IsLocal3 are override flags. Copying the source columns across
gives the wrong answer, which is what the reconciliation caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .transform import at, pick
from ..ids import as_id

__all__ = ["customer_ids_from_accounts", "derive_customer_flags",
           "investment_customer_ids"]


def investment_customer_ids(investment_accounts: pd.DataFrame) -> pd.Series:
    """Customer ids for the investment book: the counterparty NAME, one per row.

    As ``R/transform_investments.R`` writes it (``customer_id_inv``) and as
    both delivered runs carry it -- ``DUKHAN BANK`` repeated once per holding.
    Keyed on the account, so a counterparty with several holdings repeats.

    This used to return 1, 2, 3 ..., on the belief that LIC will not accept a
    name as a key. The output LIC actually received uses the names.
    """
    if investment_accounts is None or not len(investment_accounts):
        return pd.Series([], dtype=str)
    from .lending import drop_repeated_headers
    accts = drop_repeated_headers(investment_accounts)
    name = pick(accts, "CUSTOMERID", "CustomerId", default="")
    return name.astype(str).str.strip().reset_index(drop=True)


def customer_ids_from_accounts(accounts: pd.DataFrame,
                               customer_col: str = "CUSTOMERID") -> pd.Series:
    """Distinct customers, in the order the account file first mentions them.

    Order matters: LIC joins these files positionally in places, and sorting
    would silently reorder rows relative to the reference output.
    """
    if accounts is None or len(accounts) == 0:
        return pd.Series([], dtype=str)
    cid = pick(accounts, customer_col, "CustomerId")
    if cid is None:
        cid = at(accounts, 3)
    ids = as_id(cid)
    return ids[~ids.duplicated()].reset_index(drop=True)


def apply_staging_rule(dpd, restructured, watchlist,
                       dpd_stage2_threshold: float = 60) -> np.ndarray:
    """The staging rule, as the engine applies it.

        DPD > 90                                            -> Stage 3
        restructured, or watchlisted, or threshold < DPD <= 90 -> Stage 2
        otherwise                                            -> Stage 1

    Order matters: Stage 3 is applied last so a defaulted customer cannot be
    pulled back to Stage 2 by also being watchlisted.
    """
    n = len(dpd)
    out = np.full(n, "Stage 1", dtype=object)
    d = pd.to_numeric(pd.Series(dpd), errors="coerce")
    r = pd.Series(restructured).fillna(False).astype(bool).to_numpy()
    w = pd.Series(watchlist).fillna(False).astype(bool).to_numpy()
    s2 = ((d > dpd_stage2_threshold) & (d <= 90)).fillna(False).to_numpy()
    out[r | w | s2] = "Stage 2"
    out[(d > 90).fillna(False).to_numpy()] = "Stage 3"
    return out


def derive_customer_flags(accounts: pd.DataFrame,
                          staging_raw: pd.DataFrame | None = None,
                          dpd_threshold: float = 60) -> pd.DataFrame:
    """One row per customer, with the flags the staging file needs.

    Every flag is derived from the customer's FINAL STAGE, which is why
    copying the source columns gave the wrong answer:

        is_default    stage is Stage 3
        is_watchlist  the source flag, any flagged facility flags the customer
        is_local1     restructured, any flagged facility flags the customer
        is_local3     stage is Stage 2

    The stage itself follows apply_staging_rule() on the customer's WORST
    days-past-due, so a customer is staged by its worst facility rather than
    an average.

    One subtlety worth knowing, because it looks like a bug and is not: the
    STAGE uses the watchlist flag carried on the ACCOUNT records, while the
    IsWatchlist column written to the file comes from the staging extract. The
    two normally agree, and where they do not the customer can be watchlisted
    in the file yet staged Stage 1. Reconciling against the reference run, that
    happens for exactly one customer.
    """
    ids = customer_ids_from_accounts(accounts)
    out = pd.DataFrame({"customer_id": ids})
    n = len(out)
    if n == 0:
        return out

    cid = pick(accounts, "CUSTOMERID", "CustomerId")
    dpd = pick(accounts, "PASTDUEDAYS", "PASTDUE_DAYS", "PastDueDays")
    worst = {}
    if cid is not None and dpd is not None:
        w = pd.DataFrame({"c": as_id(cid),
                          "d": pd.to_numeric(dpd, errors="coerce")})
        worst = w.groupby("c")["d"].max().to_dict()
    out["worst_dpd"] = out["customer_id"].map(worst).fillna(0)

    # restructured: any facility flagged makes the customer restructured
    restr = {}
    rcol = pick(accounts, "ISRESTRUCTURED", "IsRestructured", "RESTRUCTURED")
    if cid is not None and rcol is not None:
        w = pd.DataFrame({"c": as_id(cid),
                          "v": pd.to_numeric(rcol, errors="coerce").fillna(0)})
        restr = w.groupby("c")["v"].max().to_dict()
    out["is_restructured"] = out["customer_id"].map(restr).fillna(0).eq(1)

    # source flags, keyed on the customer
    def source_flag(*names) -> pd.Series:
        if staging_raw is None or len(staging_raw) == 0:
            return pd.Series([False] * n)
        scid = pick(staging_raw, "CUSTOMERID", "CustomerId")
        col = None
        for nm in names:
            col = pick(staging_raw, nm)
            if col is not None:
                break
        if scid is None or col is None:
            return pd.Series([False] * n)
        m = pd.DataFrame({"c": as_id(scid),
                          "v": pd.to_numeric(col, errors="coerce").fillna(0)})
        lookup = m.groupby("c")["v"].max().to_dict()
        return out["customer_id"].map(lookup).fillna(0).astype(float).eq(1)

    out["is_watchlist"] = source_flag("ISWATCHLIST", "IsWatchlist")
    # IsLocal1 in the source IS the restructuring flag for this book
    src_restr = source_flag("ISLOCAL1", "IsLocal1")
    out["is_restructured"] = out["is_restructured"] | src_restr

    stage = apply_staging_rule(out["worst_dpd"], out["is_restructured"],
                               out["is_watchlist"], dpd_threshold)
    out["stage_final"] = stage
    out["is_default"] = stage == "Stage 3"
    out["is_local1"] = out["is_restructured"]
    out["is_local3"] = stage == "Stage 2"
    return out
