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


# ===========================================================================
# Overlay BUNDLES: how an overlay is authored, approved and applied to a run.
#
# A bundle is one management adjustment as a person describes it: an id, an
# owner, a status, and a list of RULES. Each rule picks a level (stage,
# portfolio, rating, flag, customer, contract, or the whole book), a target, a
# method and a value. The engine above works on flattened single overlays, so a
# bundle is expanded into those before it is applied.
#
# The separation matters because the two have different lifetimes: a bundle is
# reviewed and approved by people and lives in config, while the flattened
# overlays exist only for the length of one calculation.
# ===========================================================================

OVERLAY_STATUSES = ("draft", "pending", "approved", "rejected")
OVERLAY_LEVELS = ("stage", "portfolio", "rating", "flag", "sector",
                  "customer", "contract", "whole_book")

# The columns the LIC-format report carries for overlays. They already exist on
# a priced report; an overlay writes back into them rather than adding columns,
# so an overlaid report stays the same shape as the model one.
LIC_COLUMNS = {
    "contract": "Contract Id", "customer": "Customer Id",
    "stage": "Stage", "portfolio": "Portfolio Code", "rating": "Rating",
    "exposure": "Exposure On Bal", "ecl": "Cla Amount Onbal",
    "ecl_model": "Ecl Model Onbal", "overlay_id": "Overlay Id",
    "overlay_amount": "Overlay Amount", "ecl_final": "Ecl Final Onbal",
}

AUDIT_COLUMNS = ["overlay_id", "name", "type", "level", "value", "contracts",
                 "exposure", "ecl_model", "overlay_amount", "ecl_final",
                 "comment", "owner", "approval_ref", "effective_date",
                 "expiry"]

__all__ += ["OVERLAY_STATUSES", "OVERLAY_LEVELS", "LIC_COLUMNS",
            "rule_selector", "bundle_to_overlays", "validate_overlay_bundle",
            "read_overlay_bundles", "write_overlay_bundles", "get_overlay",
            "upsert_overlay", "remove_overlay", "set_overlay_status",
            "apply_overlay_bundle", "apply_overlay_to_run",
            "list_applied_overlays", "remove_applied_overlay",
            "preview_overlays"]


def _num(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _id_safe(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(text))


def rule_selector(level: str, target) -> dict:
    """Turn a rule's (level, target) into the engine's selector form.

    A target typed as "a, b, c" is split, because that is how a person writes a
    list into a text box and refusing it would teach them to write three rules
    instead of one.
    """
    if level == "whole_book":
        return {"whole_book": True}
    if isinstance(target, str) and "," in target:
        items = [t.strip() for t in target.split(",")]
    elif isinstance(target, (list, tuple, set)):
        items = [str(t).strip() for t in target]
    else:
        items = [str(target).strip()] if not _blank(target) else []
    items = [t for t in items if t]

    if level == "stage":
        return {"stage": [int(float(t)) for t in items]}
    key = {"contract": "contract_id"}.get(level, level)
    return {key: items}


def bundle_to_overlays(bundle: dict) -> list[Overlay]:
    """Flatten a bundle's rules into the single overlays the engine applies."""
    rules = bundle.get("rules") or []
    out = []
    for i, r in enumerate(rules, start=1):
        level = r.get("level") or "whole_book"
        sel = rule_selector(level, r.get("target"))
        out.append(Overlay(
            id=f"{bundle.get('id')}::{i}",
            type=str(r.get("method") or ""),
            value=_num(r.get("value"), 0.0),
            level="customer" if level == "customer" else "contract",
            rationale=str(r.get("comment") or bundle.get("id") or ""),
            approved_by=str(bundle.get("owner") or ""),
            enabled=True,
            stage=_as_list(sel.get("stage")),
            portfolio=_as_list(sel.get("portfolio")),
            rating=_as_list(sel.get("rating")),
            flag=_as_list(sel.get("flag")),
            customer=_as_list(sel.get("customer")),
            contract_id=_as_list(sel.get("contract_id")),
            whole_book=bool(sel.get("whole_book", False))))
    return out


def validate_overlay_bundle(bundle: dict) -> list[str]:
    """Everything wrong with a bundle, rather than the first thing.

    Returning the whole list matters for a form: fixing one error at a time,
    with a round trip each, is how people give up and write the YAML by hand.
    """
    errors = []
    if _blank(bundle.get("id")):
        errors.append("an overlay needs an id")
    rules = bundle.get("rules") or []
    if not rules:
        errors.append(f"{bundle.get('id') or 'overlay'}: at least one rule is "
                      "required")
    for i, r in enumerate(rules, start=1):
        tag = f"{bundle.get('id') or 'overlay'} rule {i}"
        method = r.get("method")
        if _blank(method) or method not in OVERLAY_TYPES:
            errors.append(f"{tag}: method must be one of "
                          f"{', '.join(OVERLAY_TYPES)}")
        level = r.get("level")
        if _blank(level) or level not in OVERLAY_LEVELS:
            errors.append(f"{tag}: level must be one of "
                          f"{', '.join(OVERLAY_LEVELS)}")
        if level != "whole_book" and _blank(r.get("target")):
            errors.append(f"{tag}: a target is required for level {level!r}")
        v = _num(r.get("value"))
        if v != v:                       # NaN
            errors.append(f"{tag}: value must be a number")
        if _blank(r.get("comment")):
            errors.append(f"{tag}: a comment is mandatory - an adjustment "
                          "nobody can trace back is indistinguishable from an "
                          "error")
    return errors


def read_overlay_bundles(path) -> list[dict]:
    """The bundles in an overlays.yml, as written."""
    import yaml
    p = Path(path)
    if not p.is_file():
        return []
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    items = data.get("overlays") if isinstance(data, dict) else data
    return [b for b in (items or []) if isinstance(b, dict)]


def write_overlay_bundles(path, bundles: list[dict]) -> Path:
    """Write the file atomically, so a failed write cannot truncate it."""
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump({"overlays": list(bundles)},
                                  sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def get_overlay(overlay_id: str, path) -> dict | None:
    for b in read_overlay_bundles(path):
        if str(b.get("id")) == str(overlay_id):
            return b
    return None


def upsert_overlay(bundle: dict, path, validate: bool = True) -> dict:
    """Add or replace a bundle, keeping its place in the file.

    Keeping the position matters only because a file that reorders itself on
    every save is unreviewable in a diff.
    """
    if validate:
        errors = validate_overlay_bundle(bundle)
        if errors:
            raise ValueError("; ".join(errors))
    bundles = read_overlay_bundles(path)
    for i, b in enumerate(bundles):
        if str(b.get("id")) == str(bundle.get("id")):
            bundles[i] = bundle
            break
    else:
        bundles.append(bundle)
    write_overlay_bundles(path, bundles)
    return bundle


def remove_overlay(overlay_id: str, path) -> bool:
    bundles = read_overlay_bundles(path)
    kept = [b for b in bundles if str(b.get("id")) != str(overlay_id)]
    if len(kept) == len(bundles):
        return False
    write_overlay_bundles(path, kept)
    return True


def set_overlay_status(overlay_id: str, to: str, by: str, reason: str,
                       path) -> dict:
    """Move a bundle's approval status, recording who and why.

    The transition list is append-only: an overlay that was rejected and later
    approved should read as exactly that, not as one that was always approved.
    """
    if to not in OVERLAY_STATUSES:
        raise ValueError(f"invalid status {to!r}; allowed: "
                         f"{', '.join(OVERLAY_STATUSES)}")
    if _blank(by):
        raise ValueError("a user is required to change an overlay's status")
    if _blank(reason):
        raise ValueError("a reason is required to change an overlay's status")
    bundle = get_overlay(overlay_id, path)
    if bundle is None:
        raise FileNotFoundError(f"overlay {overlay_id!r} not found")
    import datetime as _d
    bundle.setdefault("transitions", []).append({
        "to": to, "by": by,
        "at": _d.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": reason})
    bundle["status"] = to
    # Not validated: a bundle already on file can be rejected even if it would
    # no longer pass, and refusing to record that would be worse.
    upsert_overlay(bundle, path, validate=False)
    return bundle


def _lic_to_normalised(report: pd.DataFrame) -> pd.DataFrame:
    """A LIC-format report in the shape the overlay engine works on."""
    from .analytics import normalise
    return normalise(report)


def apply_overlay_bundle(report: pd.DataFrame, bundle: dict) -> dict:
    """Apply a bundle to a normalised report."""
    res = apply_overlays(report, bundle_to_overlays(bundle))
    res["overlay_id"] = bundle.get("id")
    res["status"] = bundle.get("status") or "draft"
    return res


def preview_overlays(report: pd.DataFrame, bundle_or_overlays) -> dict:
    """What a bundle WOULD do, without writing anything.

    The point of a preview is that it can be wrong safely, so this never
    touches a file and returns the conflicts rather than raising on them.
    """
    overlays = (bundle_to_overlays(bundle_or_overlays)
                if isinstance(bundle_or_overlays, dict)
                else list(bundle_or_overlays))
    model_total = float(pd.to_numeric(report["ecl"], errors="coerce")
                        .fillna(0).sum())
    res = apply_overlays(report, overlays)
    if not res.get("ok"):
        return {"ok": False, "errors": res.get("errors", []),
                "contracts": res.get("contracts", []),
                "total": {"model": model_total, "overlay": 0.0,
                          "final": model_total}}
    return {"ok": True, "summary": res["audit"], "errors": [],
            "total": {"model": res["ecl_model_total"],
                      "overlay": res["overlay_total"],
                      "final": res["ecl_final_total"]},
            "contracts_touched": res["contracts_touched"]}


def _output_dir(run_path) -> Path:
    p = Path(run_path)
    out = p / "Output"
    return out if out.is_dir() else p


def apply_overlay_to_run(run_path, bundle: dict) -> dict:
    """Apply a bundle to a completed run, without touching what it produced.

    The model report is never modified. The overlaid figures are written
    alongside it as ``FinalEclReport_overlay_<id>.csv`` with an
    ``OverlayAuditLog_<id>.csv`` beside them, so the model number and the
    adjusted one both remain on disk and the difference between them is a file
    anybody can open.
    """
    out_dir = _output_dir(run_path)
    src = out_dir / "FinalEclReport.csv"
    if not src.is_file():
        raise FileNotFoundError(f"no FinalEclReport.csv in {run_path}")

    lic = pd.read_csv(src, low_memory=False)
    normalised = _lic_to_normalised(lic)
    res = apply_overlay_bundle(normalised, bundle)
    if not res.get("ok"):
        return {"ok": False, "errors": res.get("errors", []),
                "contracts": res.get("contracts", []),
                "overlay_id": bundle.get("id")}

    adjusted = res["report"]
    by_contract = adjusted.set_index(adjusted["contract"].astype(str))

    out = lic.copy()
    key = out[LIC_COLUMNS["contract"]].astype(str)
    from .ids import as_id
    key = as_id(key)

    def pull(col):
        return key.map(by_contract[col]).astype(float).fillna(0.0)

    model = pull("ecl_model")
    amount = pull("overlay_amount")
    out[LIC_COLUMNS["ecl_model"]] = model
    out[LIC_COLUMNS["overlay_amount"]] = amount
    out[LIC_COLUMNS["ecl_final"]] = model + amount
    out[LIC_COLUMNS["ecl"]] = model + amount
    if LIC_COLUMNS["overlay_id"] in out.columns:
        out[LIC_COLUMNS["overlay_id"]] = key.map(
            by_contract["overlay_id"]).fillna("")

    idsafe = _id_safe(bundle.get("id"))
    report_path = out_dir / f"FinalEclReport_overlay_{idsafe}.csv"
    audit_path = out_dir / f"OverlayAuditLog_{idsafe}.csv"
    out.to_csv(report_path, index=False, na_rep="")

    audit = res["audit"].copy()
    if len(audit):
        audit["name"] = [f"{bundle.get('id')} rule {i}"
                         for i in range(1, len(audit) + 1)]
        audit["comment"] = audit.get("rationale", "")
        audit["owner"] = bundle.get("owner", "")
        audit["approval_ref"] = bundle.get("approval_ref", bundle.get("id", ""))
        audit["effective_date"] = bundle.get("effective_date",
                                             bundle.get("created_at", ""))
        audit["expiry"] = bundle.get("expiry", "")
        for c in AUDIT_COLUMNS:
            if c not in audit.columns:
                audit[c] = ""
        audit = audit[AUDIT_COLUMNS]
        audit.to_csv(audit_path, index=False, na_rep="")

    return {"ok": True, "errors": [],
            "out_report": str(report_path),
            "out_audit": str(audit_path) if len(audit) else None,
            "overlay_id": bundle.get("id"),
            "status": bundle.get("status") or "draft",
            "totals": {"model": float(model.sum()),
                       "overlay": float(amount.sum()),
                       "final": float((model + amount).sum())},
            "audit": audit if len(audit) else pd.DataFrame(columns=AUDIT_COLUMNS)}


def list_applied_overlays(run_path) -> pd.DataFrame:
    """Which overlays have been applied to a run, found from what is on disk.

    Read from the files rather than from a register, so an overlay whose
    outputs were copied in by hand still shows up and one whose files were
    deleted stops showing up.
    """
    import datetime as _d
    out_dir = _output_dir(run_path)
    cols = ["overlay_id", "report_file", "report_path", "audit_path",
            "applied_at"]
    if not out_dir.is_dir():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    rows = []
    for f in sorted(out_dir.glob("FinalEclReport_overlay_*.csv")):
        oid = f.stem[len("FinalEclReport_overlay_"):]
        audit = out_dir / f"OverlayAuditLog_{oid}.csv"
        rows.append({
            "overlay_id": oid, "report_file": f.name, "report_path": str(f),
            "audit_path": str(audit) if audit.is_file() else None,
            "applied_at": _d.datetime.fromtimestamp(
                f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows, columns=cols)


def remove_applied_overlay(run_path, overlay_id: str) -> dict:
    """Remove an overlay's outputs from a run. The model report is untouched."""
    out_dir = _output_dir(run_path)
    idsafe = _id_safe(overlay_id)
    removed = []
    for name in (f"FinalEclReport_overlay_{idsafe}.csv",
                 f"OverlayAuditLog_{idsafe}.csv"):
        p = out_dir / name
        if p.is_file():
            p.unlink()
            removed.append(name)
    return {"ok": bool(removed), "removed": removed}
