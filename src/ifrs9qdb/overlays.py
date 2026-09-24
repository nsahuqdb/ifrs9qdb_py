"""
Post-model overlays: management adjustments applied after the engine.

An overlay never overwrites the model figure. The report gains three columns --
``Ecl Model Onbal``, ``Overlay Amount``, ``Ecl Final Onbal`` -- and every
overlay writes an audit row recording what it matched, what it moved and why.
That is the whole point: an adjustment nobody can trace back is indistinguishable
from an error.

Three types, and the choice changes what the number means:

    uplift_pct    ECL_final = ECL_model * (1 + value)
                  proportional, so it scales with the model's own view
    higher_of     ECL_final = max(ECL_model, value * exposure)
                  a floor, which binds only where the model is below it
    absolute_add  a fixed total spread across matched contracts PRO RATA BY
                  EXPOSURE, so a large facility takes more of it than a small one

Two rules that are not negotiable:

  * **Stage 3 is never touched.** Those provisions are booked manually outside
    the tool, and an overlay on top would double-count.
  * **At most one overlay per contract.** Overlapping overlays are an error
    rather than a silent compounding, because the order of application would
    then decide the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Overlay", "validate_overlays", "apply_overlays", "read_overlays",
           "OVERLAY_TYPES"]

OVERLAY_TYPES = ("uplift_pct", "higher_of", "absolute_add")


@dataclass
class Overlay:
    id: str
    type: str
    value: float
    level: str = "contract"
    rationale: str = ""
    approved_by: str = ""
    enabled: bool = True
    # selectors, all of which must match
    stage: list = field(default_factory=list)
    portfolio: list = field(default_factory=list)
    rating: list = field(default_factory=list)
    flag: list = field(default_factory=list)
    customer: list = field(default_factory=list)
    contract_id: list = field(default_factory=list)
    whole_book: bool = False

    def selectors(self) -> dict:
        return {k: getattr(self, k) for k in
                ("stage", "portfolio", "rating", "flag", "customer",
                 "contract_id") if getattr(self, k)}


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set)):
        return [x for x in v if x is not None and str(x) != ""]
    return [v]


def validate_overlays(overlays: list[Overlay]) -> list[str]:
    """Structural problems, listed rather than raised one at a time."""
    errors = []
    seen = set()
    for ov in overlays:
        if not ov.id:
            errors.append("an overlay has no id")
            continue
        if ov.id in seen:
            errors.append(f"{ov.id}: duplicate id")
        seen.add(ov.id)
        if ov.type not in OVERLAY_TYPES:
            errors.append(f"{ov.id}: type must be one of {', '.join(OVERLAY_TYPES)}")
        if ov.value is None or not np.isfinite(ov.value):
            errors.append(f"{ov.id}: value must be a number")
        elif ov.type == "higher_of" and not (0 <= ov.value <= 1):
            errors.append(f"{ov.id}: higher_of takes a fraction of exposure "
                          "between 0 and 1")
        elif ov.type == "uplift_pct" and ov.value < -1:
            errors.append(f"{ov.id}: an uplift below -100% would give a "
                          "negative provision")
        if ov.level not in ("contract", "customer"):
            errors.append(f"{ov.id}: level must be contract or customer")
        if not ov.selectors() and not ov.whole_book:
            errors.append(f"{ov.id}: no selector and whole_book is not set, so "
                          "it would match nothing")
        # An unexplained adjustment cannot be defended in a review.
        if not ov.rationale:
            errors.append(f"{ov.id}: no rationale")
    return errors


def _match(report: pd.DataFrame, ov: Overlay) -> pd.Series:
    """Contracts an overlay applies to.

    Stage 3 is excluded here rather than in the caller, so no overlay type can
    reach it by a different path.
    """
    keep = pd.Series(True, index=report.index)
    if not ov.whole_book:
        if ov.stage:
            keep &= report["stage"].isin([float(s) for s in ov.stage])
        if ov.portfolio:
            keep &= report["portfolio"].isin(ov.portfolio)
        if ov.rating:
            keep &= report["rating"].astype(str).isin([str(r) for r in ov.rating])
        if ov.contract_id:
            keep &= report["contract"].astype(str).isin(
                [str(c) for c in ov.contract_id])
        if ov.customer:
            keep &= report["customer"].astype(str).isin(
                [str(c) for c in ov.customer])
        if ov.flag:
            flags = pd.Series(False, index=report.index)
            for f in ov.flag:
                col = {"watchlist": "watchlist", "restructured": "restructured",
                       "default": "default_flag",
                       "insolvency": "insolvency"}.get(str(f).lower())
                if col and col in report.columns:
                    flags |= (report[col] == 1).fillna(False)
            keep &= flags

    # An overlay set at customer level applies to ALL that customer's
    # facilities, not only the ones that matched.
    if ov.level == "customer" and keep.any():
        custs = set(report.loc[keep, "customer"].dropna())
        keep = report["customer"].isin(custs)

    # Stage 3 is booked manually; an overlay on top would double-count.
    keep &= report["stage"] != 3
    return keep.fillna(False)


def apply_overlays(report: pd.DataFrame, overlays: list[Overlay]) -> dict:
    """Apply the overlays and return the adjusted report plus an audit trail.

    Raises on overlapping overlays rather than compounding them: if two
    overlays could both touch a contract, the order of application would decide
    the provision, and nobody would be able to say why.
    """
    errors = validate_overlays(overlays)
    if errors:
        return {"ok": False, "errors": errors}

    active = [o for o in overlays if o.enabled]
    d = report.copy()
    d["ecl_model"] = pd.to_numeric(d["ecl"], errors="coerce").fillna(0.0)
    d["overlay_amount"] = 0.0
    d["overlay_id"] = ""

    matches = {ov.id: _match(d, ov) for ov in active}

    # overlapping overlays: an error, not a silent compounding
    hit_count = pd.Series(0, index=d.index)
    for m in matches.values():
        hit_count += m.astype(int)
    if (hit_count > 1).any():
        clash = d.loc[hit_count > 1, "contract"].astype(str).tolist()
        overlapping = [oid for oid, m in matches.items()
                       if (m & (hit_count > 1)).any()]
        return {"ok": False, "errors": [
            f"{len(clash)} contracts are matched by more than one overlay "
            f"({', '.join(overlapping)}). Overlapping overlays are rejected "
            "because the order of application would decide the provision."],
            "contracts": clash[:20]}

    audit = []
    for ov in active:
        m = matches[ov.id]
        n = int(m.sum())
        model = d.loc[m, "ecl_model"]
        exposure = pd.to_numeric(d.loc[m, "exposure"], errors="coerce").fillna(0.0)

        if n == 0:
            amount = pd.Series(dtype=float)
        elif ov.type == "uplift_pct":
            amount = model * ov.value
        elif ov.type == "higher_of":
            floor = exposure * ov.value
            amount = (floor - model).clip(lower=0.0)
        else:  # absolute_add, pro rata by exposure
            total = exposure.sum()
            amount = (exposure / total * ov.value if total > 0
                      else pd.Series(0.0, index=model.index))

        if n:
            d.loc[m, "overlay_amount"] = amount.to_numpy()
            d.loc[m, "overlay_id"] = ov.id

        audit.append({
            "overlay_id": ov.id, "type": ov.type, "value": ov.value,
            "level": ov.level, "contracts": n,
            "customers": int(d.loc[m, "customer"].nunique()) if n else 0,
            "exposure": float(exposure.sum()) if n else 0.0,
            "ecl_model": float(model.sum()) if n else 0.0,
            "overlay_amount": float(amount.sum()) if n else 0.0,
            "ecl_final": float(model.sum() + amount.sum()) if n else 0.0,
            "rationale": ov.rationale, "approved_by": ov.approved_by,
        })

    d["ecl_final"] = d["ecl_model"] + d["overlay_amount"]
    d["ecl"] = d["ecl_final"]

    return {
        "ok": True,
        "report": d,
        "audit": pd.DataFrame(audit),
        "ecl_model_total": float(d["ecl_model"].sum()),
        "overlay_total": float(d["overlay_amount"].sum()),
        "ecl_final_total": float(d["ecl_final"].sum()),
        "contracts_touched": int((d["overlay_amount"] != 0).sum()),
    }


def read_overlays(path) -> list[Overlay]:
    """Read overlays from YAML. A malformed entry is skipped, not guessed at."""
    p = Path(path)
    if not p.is_file():
        return []
    import yaml
    data = yaml.safe_load(p.read_text()) or {}
    items = data.get("overlays", []) if isinstance(data, dict) else data
    out = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        try:
            out.append(Overlay(
                id=str(it.get("id", "")), type=str(it.get("type", "")),
                value=float(it.get("value", 0)),
                level=str(it.get("level", "contract")),
                rationale=str(it.get("rationale", "")),
                approved_by=str(it.get("approved_by", "")),
                enabled=bool(it.get("enabled", True)),
                stage=_as_list(it.get("stage")),
                portfolio=_as_list(it.get("portfolio")),
                rating=_as_list(it.get("rating")),
                flag=_as_list(it.get("flag")),
                customer=_as_list(it.get("customer")),
                contract_id=_as_list(it.get("contract_id")),
                whole_book=bool(it.get("whole_book", False))))
        except (TypeError, ValueError):
            continue
    return out
