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
    """A run, ready to reprice: its pricing context and contract facts --
    what its own report was priced on (``etl.report.build_ecl_context`` and
    ``ecl_frame``) -- with the rating scales a notch moves along and the
    collateral tables behind the netted figure.

    ``pd_curves``, ``ead_curves``, ``collateral_net`` and ``contracts`` are
    views of that context and frame keyed the way the analytics look them up
    (``portfolio|bucket``; the normalised contract id), not a second reading
    of the run's files."""
    ok: bool
    out_dir: Path
    missing: list[str] = field(default_factory=list)
    contracts: pd.DataFrame = field(default_factory=pd.DataFrame)
    pd_curves: dict[str, np.ndarray] = field(default_factory=dict)
    ead_curves: dict[str, np.ndarray] = field(default_factory=dict)
    collateral_net: dict[str, float] = field(default_factory=dict)
    scales: dict[int, RatingScale] = field(default_factory=dict)
    rating_type_of_portfolio: dict[str, int] = field(default_factory=dict)
    # The raw collateral tables, kept beside the netted figure. The net is what
    # the engine prices with; these are what answers "netted down from what,
    # and did every allocation find a record".
    collateral: pd.DataFrame | None = None
    alloc: pd.DataFrame | None = None
    coll_type: pd.DataFrame | None = None
    # The EAD fallback rules the run was priced with: its frozen model.yml's
    # ecl.ead_fallback (R's rules when absent), so a repricing uses the same
    # shapes as the run's own report.
    ead_rules: tuple | None = None
    # what every repricing prices on: the run's own pricing context and its
    # contract facts, one row per position (etl.report)
    ctx: object = None
    frame: pd.DataFrame | None = None
    model_cfg: dict | None = None

    # -- lookups -------------------------------------------------------
    def scale_for(self, rating_type) -> RatingScale | None:
        """The scale of a rating type: 1 / 2, or the report's "Internal" /
        "External" ("External Rating")."""
        if not self.scales:
            return None
        t = str(rating_type)
        if t.startswith("External"):
            rt = 2
        elif t.startswith("Internal"):
            rt = 1
        else:
            try:
                rt = int(float(t))
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
        """The EAD curve the engine prices a contract on: its supplied
        schedule, else the parametric fallback -- ``etl.report._ead_curve``,
        on the run's own context. Reading ``ead_curves`` directly is what
        silently dropped a quarter of the book."""
        from .etl.report import _ead_curve
        if self.ctx is None or self.frame is None:
            return None
        hit = self.frame[self.frame["contract"] == str(contract)]
        if len(hit) == 0:
            return None
        r = hit.iloc[0]
        curve, _, _ = _ead_curve(self.ctx, r["cid"], int(stage), r["on_balance"],
                                 int(r["months"]), r["payment_type"],
                                 r["portfolio"], r["nir"], r["deferral"],
                                 r["pay_freq"])
        return curve


def load_engine_inputs(out_dir) -> EngineInputs:
    """Read a run's Output folder."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return EngineInputs(ok=False, out_dir=out_dir, missing=["Output directory"])

    stpd = _read(out_dir, "StPD.csv")
    ratings = _read(out_dir, "Ratings.csv")
    am1 = _read(out_dir, "AccountMaster_1.csv")
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

    rt_of_pf: dict[str, int] = {}
    if prt is not None:
        pf = _col(prt, "PortfolioCode")
        tp = pd.to_numeric(_col(prt, "RatingType"), errors="coerce")
        if pf is not None and tp is not None:
            rt_of_pf = {str(p): int(t) for p, t in zip(pf, tp) if pd.notna(t)}

    # ---- what the run is priced on: its own context and contract facts --
    # One reading of the run's files for every repricing (etl/report.py);
    # what follows are views of it keyed as the analytics look them up.
    from .etl.report import (_KEYSEP, _portfolio_map, build_ecl_context,
                             ecl_frame, run_pricing_config)
    static, model_cfg = run_pricing_config(out_dir.parent if out_dir.name == "Output"
                                           else out_dir)
    ctx = build_ecl_context(out_dir, model_cfg=model_cfg)
    frame = ecl_frame(out_dir, _portfolio_map(static))
    if frame is None:
        return EngineInputs(ok=False, out_dir=out_dir,
                            missing=["AccountMaster_1.csv"])

    pd_curves = {k.replace(_KEYSEP, "|"): v for k, v in ctx.stpd.items()}

    def by_id(d: dict) -> dict:
        if not d:
            return {}
        ids = as_id(pd.Series(list(d.keys()), dtype=object))
        return {k2: d[k1] for k1, k2 in zip(d.keys(), ids)}

    ead_curves = by_id(ctx.schedules)
    collateral_net = by_id(ctx.collnet)
    ctype = _read(out_dir, "CollateralType.csv")

    rt_code = frame["rating_type"].map(
        lambda t: 2 if str(t).startswith("External") else 1).astype(int)
    contracts = pd.DataFrame({
        "contract": frame["contract"].astype(str),
        "portfolio": frame["portfolio"].astype(str),
        "rating": frame["rating"].astype(str),
        "on_balance": pd.to_numeric(frame["on_balance"], errors="coerce"),
        "eir": pd.to_numeric(frame["eir"], errors="coerce"),
        "payment_type": frame["payment_type"],
        "payment_frequency": frame["pay_freq"],
        "deferral": frame["deferral"],
        "nir": frame["nir"],
        "maturity_date": frame["d_maturity"],
        "extract_date": frame["d_extract"],
        "months_to_mat": frame["months"],
        "rating_type": rt_code,
    })
    contracts["bucket"] = [
        (lambda b: int(b) if b is not None and str(b) not in ("", "nan", "None") else None)(
            ctx.rat2bucket.get(f"{t}{_KEYSEP}{r}"))
        for t, r in zip(contracts["rating_type"], contracts["rating"])]
    contracts["pd_key"] = [f"{p}|{b}" if b is not None else None
                           for p, b in zip(contracts["portfolio"], contracts["bucket"])]
    contracts["collateral_net"] = contracts["contract"].map(collateral_net).fillna(0.0)

    return EngineInputs(
        ok=True, out_dir=out_dir, contracts=contracts, pd_curves=pd_curves,
        ead_curves=ead_curves, collateral_net=collateral_net, scales=scales,
        rating_type_of_portfolio=rt_of_pf,
        collateral=coll, alloc=alloc, coll_type=ctype,
        ead_rules=ctx.rules, ctx=ctx, frame=frame, model_cfg=model_cfg,
    )
