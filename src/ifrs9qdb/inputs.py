"""
A run's engine inputs.

The ECL report says what the provision IS. To answer "what would it be if…"
the book has to be repriced, and that needs the inputs the run priced against:
the PD term structures, the supplied EAD curves, the rating scales and the
collateral.

All of it comes from the eighteen CSVs a run writes, so nothing here depends on
the ETL having been ported -- a run produced by the R pipeline works unchanged.

Two things that look like details and are not:

  * The rating scales reuse hierarchy numbers 1-21, so a rating must be looked
    up on (rating_type, rating), never on the code alone. Which scale applies
    is a property of the PORTFOLIO.
  * Roughly a quarter of contracts have no supplied EAD curve -- including
    every Off BS, Al Dhameen and Tasdeer facility on a typical book. Looking
    the curve up directly drops them silently; they must fall back to the
    parametric shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .engine import fallback_ead_curve, months_to_maturity
from .ids import as_id

__all__ = ["EngineInputs", "load_engine_inputs", "as_id"]


def _read(out_dir: Path, name: str) -> pd.DataFrame | None:
    p = out_dir / name
    if not p.is_file():
        hits = list(out_dir.rglob(name))
        if not hits:
            return None
        p = hits[0]
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return None


def _col(df: pd.DataFrame | None, name: str):
    """Column lookup that survives punctuation differences."""
    if df is None:
        return None
    want = "".join(c for c in name.lower() if c.isalnum())
    for c in df.columns:
        if "".join(ch for ch in str(c).lower() if ch.isalnum()) == want:
            return df[c]
    return None


@dataclass
class RatingScale:
    """One rating scale, ordered best first."""
    rating_type: int
    name: str
    ratings: list[str]
    hierarchy: list[int]

    def notch(self, rating: str, steps: int) -> str:
        """Move a rating along its own scale, clamped at both ends."""
        if rating not in self.ratings or not steps:
            return rating
        i = self.ratings.index(rating)
        return self.ratings[min(max(i + steps, 0), len(self.ratings) - 1)]

    def bucket(self, rating: str):
        if rating not in self.ratings:
            return None
        return self.hierarchy[self.ratings.index(rating)]


@dataclass
class EngineInputs:
    """Everything needed to reprice a run's book."""
    ok: bool
    out_dir: Path
    missing: list[str] = field(default_factory=list)
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    pd_curves: dict[str, np.ndarray] = field(default_factory=dict)
    ead_curves: dict[str, np.ndarray] = field(default_factory=dict)
    collateral_net: dict[str, float] = field(default_factory=dict)
    scales: dict[int, RatingScale] = field(default_factory=dict)
    rating_type_of_portfolio: dict[str, int] = field(default_factory=dict)

    # -- lookups -------------------------------------------------------
    def scale_for(self, rating_type) -> RatingScale | None:
        if not self.scales:
            return None
        try:
            rt = int(rating_type)
        except (TypeError, ValueError):
            rt = None
        return self.scales.get(rt) or next(iter(self.scales.values()))

    def internal_portfolios(self) -> list[str]:
        """Portfolios on the internal scale.

        Externally-rated portfolios carry agency ratings and do not resolve
        against the internal PD curves, so a stress leaves them untouched.
        Defaulting a stress to these keeps the numbers meaningful.
        """
        return sorted(p for p, t in self.rating_type_of_portfolio.items() if t == 1)

    def external_portfolios(self) -> list[str]:
        return sorted(p for p, t in self.rating_type_of_portfolio.items() if t == 2)

    def pd_curve(self, portfolio: str, bucket) -> np.ndarray | None:
        if bucket is None or (isinstance(bucket, float) and np.isnan(bucket)):
            return None
        return self.pd_curves.get(f"{portfolio}|{int(bucket)}")

    def ead_curve(self, contract: str, stage: int, row=None) -> np.ndarray | None:
        """The supplied curve, else the parametric fallback.

        This is the one place an EAD curve should be looked up. Reading
        ``ead_curves`` directly is what silently dropped a quarter of the book.
        """
        got = self.ead_curves.get(str(contract))
        if got is not None and len(got):
            return got[: min(12, len(got))] if stage == 1 else got
        if row is None:
            return None
        curve = fallback_ead_curve(
            row.get("on_balance"), int(row.get("months_to_mat") or 3),
            row.get("payment_type"), row.get("payment_frequency"),
            row.get("deferral"),
        )
        return curve[: min(12, len(curve))] if stage == 1 else curve


def load_engine_inputs(out_dir) -> EngineInputs:
    """Read a run's Output folder."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return EngineInputs(ok=False, out_dir=out_dir, missing=["Output directory"])

    stpd = _read(out_dir, "StPD.csv")
    ratings = _read(out_dir, "Ratings.csv")
    am1 = _read(out_dir, "AccountMaster_1.csv")
    am2 = _read(out_dir, "AccountMaster_2.csv")
    lp = _read(out_dir, "LifeTimeParameterOther.csv")
    prt = _read(out_dir, "PortfolioRatingType.csv")
    rtypes = _read(out_dir, "RatingTypes.csv")
    coll = _read(out_dir, "Collateral.csv")
    alloc = _read(out_dir, "AccountCollateralAllocation.csv")

    missing = [n for n, v in [("StPD.csv", stpd), ("Ratings.csv", ratings),
                              ("AccountMaster_1.csv", am1)] if v is None]
    if missing:
        return EngineInputs(ok=False, out_dir=out_dir, missing=missing)

    # ---- rating scales, keyed on (type, rating) ----------------------
    rt_col = _col(ratings, "RatingType")
    rt_vals = pd.to_numeric(rt_col, errors="coerce").fillna(1).astype(int) \
        if rt_col is not None else pd.Series([1] * len(ratings))
    rname = {1: "Internal Rating", 2: "External Rating"}
    if rtypes is not None:
        ids = pd.to_numeric(_col(rtypes, "RatingType"), errors="coerce")
        desc = _col(rtypes, "Description")
        if ids is not None and desc is not None:
            rname.update({int(i): str(d) for i, d in zip(ids, desc) if pd.notna(i)})

    scales: dict[int, RatingScale] = {}
    rdf = ratings.assign(_rt=rt_vals.to_numpy(),
                         _r=_col(ratings, "Rating").astype(str).to_numpy(),
                         _h=pd.to_numeric(_col(ratings, "Hierarchy"),
                                          errors="coerce").to_numpy())
    for rt, g in rdf.groupby("_rt"):
        g = g.sort_values("_h")
        scales[int(rt)] = RatingScale(int(rt), rname.get(int(rt), f"Type {rt}"),
                                      list(g["_r"]), [int(x) for x in g["_h"]])

    hier = {(int(t), str(r)): int(h)
            for t, r, h in zip(rdf["_rt"], rdf["_r"], rdf["_h"]) if pd.notna(h)}

    rt_of_pf: dict[str, int] = {}
    if prt is not None:
        pf = _col(prt, "PortfolioCode")
        tp = pd.to_numeric(_col(prt, "RatingType"), errors="coerce")
        if pf is not None and tp is not None:
            rt_of_pf = {str(p): int(t) for p, t in zip(pf, tp) if pd.notna(t)}

    # ---- PD curves, zero-prepended -----------------------------------
    pd_curves: dict[str, np.ndarray] = {}
    if stpd is not None:
        pf = _col(stpd, "PortfolioCode")
        bk = pd.to_numeric(_col(stpd, "PDBucketDim1"), errors="coerce")
        mon = pd.to_numeric(_col(stpd, "MonthLifetime"), errors="coerce")
        val = pd.to_numeric(_col(stpd, "PDLifetime"), errors="coerce")
        if all(x is not None for x in (pf, bk, mon, val)):
            t = pd.DataFrame({"pf": pf.astype(str), "bk": bk, "m": mon, "v": val}).dropna()
            for (p, b), g in t.groupby(["pf", "bk"]):
                g = g.sort_values("m")
                pd_curves[f"{p}|{int(b)}"] = np.concatenate([[0.0], g["v"].to_numpy()])

    # ---- supplied EAD curves ----------------------------------------
    ead_curves: dict[str, np.ndarray] = {}
    if lp is not None:
        cid = _col(lp, "ContractId")
        mon = pd.to_numeric(_col(lp, "MonthLifetime"), errors="coerce")
        val = pd.to_numeric(_col(lp, "EADLifetime"), errors="coerce")
        if all(x is not None for x in (cid, mon, val)):
            t = pd.DataFrame({"c": as_id(cid), "m": mon, "v": val}).dropna()
            for c, g in t.groupby("c"):
                ead_curves[c] = g.sort_values("m")["v"].to_numpy()

    # ---- collateral net per contract --------------------------------
    collateral_net: dict[str, float] = {}
    if alloc is not None and coll is not None:
        cval = {}
        cid = _col(coll, "CollateralId")
        cv = _col(coll, "CollateralValue")
        if cv is None:
            cv = _col(coll, "Value")
        if cid is not None and cv is not None:
            cval = {a: float(b) for a, b in
                    zip(as_id(cid), pd.to_numeric(cv, errors="coerce"))
                    if pd.notna(b)}
        acid = _col(alloc, "ContractId")
        acol = _col(alloc, "CollateralId")
        if acid is not None and acol is not None:
            for c, k in zip(as_id(acid), as_id(acol)):
                if k in cval:
                    collateral_net[c] = collateral_net.get(c, 0.0) + cval[k]

    # ---- contracts, lending and investments -------------------------
    def contracts_from(df):
        if df is None:
            return None
        mat = pd.to_datetime(_col(df, "MaturityDate"), errors="coerce", format="mixed")
        ext = pd.to_datetime(_col(df, "ExtractDate"), errors="coerce", format="mixed")
        out = pd.DataFrame({
            "contract": as_id(_col(df, "ContractId")),
            "portfolio": _col(df, "PortfolioCode").astype(str),
            "rating": _col(df, "Rating").astype(str),
            "on_balance": pd.to_numeric(_col(df, "OnBalance"), errors="coerce"),
            "eir": pd.to_numeric(_col(df, "EIR"), errors="coerce"),
            "payment_type": _col(df, "PaymentTypeId"),
            "payment_frequency": pd.to_numeric(_col(df, "PaymentFrequency"),
                                               errors="coerce").fillna(1),
            "deferral": pd.to_numeric(_col(df, "DeferralPeriod"),
                                      errors="coerce").fillna(0),
        })
        out["maturity_date"] = mat
        out["extract_date"] = ext
        out["months_to_mat"] = [months_to_maturity(m, e) for m, e in zip(mat, ext)]
        return out

    parts = [c for c in (contracts_from(am1), contracts_from(am2)) if c is not None]
    contracts = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(contracts):
        contracts["rating_type"] = contracts["portfolio"].map(rt_of_pf).fillna(1).astype(int)
        contracts["bucket"] = [
            hier.get((int(t), str(r))) for t, r in
            zip(contracts["rating_type"], contracts["rating"])
        ]
        contracts["pd_key"] = [
            f"{p}|{int(b)}" if pd.notna(b) else None
            for p, b in zip(contracts["portfolio"], contracts["bucket"])
        ]
        contracts["collateral_net"] = contracts["contract"].map(collateral_net).fillna(0.0)

    return EngineInputs(
        ok=True, out_dir=out_dir, contracts=contracts, pd_curves=pd_curves,
        ead_curves=ead_curves, collateral_net=collateral_net, scales=scales,
        rating_type_of_portfolio=rt_of_pf,
    )
