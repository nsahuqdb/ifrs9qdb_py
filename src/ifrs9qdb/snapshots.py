"""Config snapshots: a frozen, versioned copy of everything a run is told.

A snapshot holds the config and the static reference together, under a label,
with a lifecycle of its own. It exists because "what changed?" is the first
question asked of a provision that moved, and the answer has to be a file
rather than a memory.

The lifecycle, and it is one-way except where marked:

    draft ──► tested ──► pending_final ──► approved ──► archived
      ▲         │            │   │
      └─────────┘            │   └──► rejected ──► (clone to a new draft)
                             └──────► tested

Only a DRAFT is editable. That is the point of `tested`: the creator locks
their own snapshot before impact-testing it, so the numbers being tested
cannot move underneath the test.

Separation of duties applies to the FINAL approval only. The creator may test
and submit their own snapshot -- those are their own work -- but somebody else
must approve it.

A snapshot is layered: a new version can be CLONED from an existing one rather
than from the live config, so "v3 based on v2" literally starts from v2's
content at any depth of the chain.
"""
from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import os
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

from .audit_log import audit_event

__all__ = ["SNAPSHOT_STATUSES", "ALLOWED_TRANSITIONS", "EDITABLE_FILES",
           "snapshot_dir", "snapshot_paths", "read_snapshot_metadata",
           "list_snapshots", "create_snapshot", "clone_snapshot",
           "promote_snapshot", "diff_snapshots", "editable_snapshot_files",
           "save_snapshot_yaml", "save_snapshot_csv",
           "read_static_csv_with_header", "write_static_csv_with_header"]

SNAPSHOT_STATUSES = ("draft", "tested", "pending_final", "approved",
                     "archived", "rejected", "pending")

# "pending" is the legacy name for pending_final and is read, never written.
ALLOWED_TRANSITIONS = {
    "draft": ("tested",),
    "tested": ("pending_final", "draft"),
    "pending_final": ("approved", "rejected", "tested"),
    "approved": ("archived",),
    "rejected": ("draft",),
    "archived": (),
}

_LABEL = re.compile(r"^[A-Za-z0-9._-]+$")

# The files a draft may be edited through, with what each one is for. A file
# present on disk but not registered here still appears, as a generic entry --
# nothing is ever hidden. A registered file absent from the snapshot is
# dropped, so nothing phantom appears either.
EDITABLE_FILES: list[dict] = [
    {"relpath": "config/model_inputs.yml", "group": "Model",
     "label": "Per-quarter model inputs",
     "help": "Scenario weights and the MEV forecasts. This is the file that "
             "changes every quarter."},
    {"relpath": "config/model.yml", "group": "Model", "advanced": True,
     "label": "PD model definition",
     "help": "Regression coefficients, standard deviations and weights. "
             "Changing these is a recalibration, not an update."},
    {"relpath": "config/config.yml", "group": "Model", "advanced": True,
     "label": "Run configuration",
     "help": "Paths, the Stage 3 method and the EAD fallback shapes."},
    {"relpath": "config/validation_suppressions.yml", "group": "Model",
     "label": "Accepted findings",
     "help": "Validator findings accepted for this snapshot, each with a "
             "reason and an approver."},
    {"relpath": "static/ttc_pd_table.csv", "group": "Rating scales",
     "label": "Through-the-cycle PDs",
     "help": "One PD per rating. A zero is valid and must be kept."},
    {"relpath": "static/master_rating_scale.csv", "group": "Rating scales",
     "label": "Rating scale (internal + external)",
     "help": "Both scales reuse hierarchy 1-21, so a rating is identified by "
             "(rating_type, rating), never by the code alone."},
    {"relpath": "static/scenario_severity.csv", "group": "Rating scales",
     "label": "Scenario severities",
     "help": "The z-score for each scenario."},
    {"relpath": "static/portfolios.csv", "group": "Mappings",
     "label": "Portfolio register",
     "help": "Each portfolio and the rating scale it uses."},
    {"relpath": "static/product_portfolio_mapping.csv", "group": "Mappings",
     "label": "Product → portfolio mapping",
     "help": "The portfolio is a LOOKUP from the product type. An unmapped "
             "product resolves to no PD curve."},
    {"relpath": "static/off_balance_products.csv", "group": "Mappings",
     "label": "Off-balance product codes",
     "help": "The codes the contract id transformation substitutes."},
    {"relpath": "static/industry_sector_mapping.csv", "group": "Mappings",
     "label": "Industry → sector mapping",
     "help": "Unmapped codes drop out of sector concentration."},
    {"relpath": "static/segment_fallback_ratings.csv", "group": "Mappings",
     "label": "Fallback ratings per segment",
     "help": "Used where a customer carries no rating."},
    {"relpath": "static/collateral_types.csv", "group": "Credit",
     "label": "Collateral types and haircuts",
     "help": "An unmapped type has no haircut, so the security counts for "
             "nothing and the provision is over-stated."},
    {"relpath": "static/staging_thresholds.csv", "group": "Credit",
     "label": "Staging thresholds (DPD days)",
     "help": "The days-past-due boundary between Stage 1 and Stage 2."},
    {"relpath": "static/collective_assessment_rules.csv", "group": "Credit",
     "label": "Collective assessment ladder",
     "help": "The Agri, Fisheries and Livestock DPD ladder."},
    {"relpath": "static/fx_rates.csv", "group": "Credit",
     "label": "FX rates", "help": "One rate per non-QAR currency."},
    {"relpath": "static/non_oil_gdp_history.csv", "group": "Macro",
     "label": "Non-oil GDP history",
     "help": "The domestic series the internal scenario weights are read "
             "against."},
    {"relpath": "static/gcc_real_gdp_growth.csv", "group": "Macro",
     "label": "GCC real GDP growth by country",
     "help": "The regional series. It is GDP-weighted at runtime and starts "
             "at 1982."},
    {"relpath": "static/gcc_gdp_current_prices.csv", "group": "Macro",
     "label": "GCC GDP at current prices",
     "help": "The weights for the regional series."},
]


def _now() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _who() -> str:
    for env in ("IFRS9_USER", "USER", "USERNAME"):
        if os.environ.get(env):
            return os.environ[env]
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _root(snapshots_root=None) -> Path:
    if snapshots_root:
        return Path(snapshots_root)
    env = os.environ.get("IFRS9_SNAPSHOTS_DIR")
    return Path(env) if env else Path.cwd() / "config_snapshots"


def snapshot_dir(label: str, snapshots_root=None) -> Path:
    return _root(snapshots_root) / str(label)


def snapshot_paths(label: str, snapshots_root=None) -> dict:
    """The paths a run needs to be pointed at this snapshot's frozen content."""
    d = snapshot_dir(label, snapshots_root)
    return {"snapshot_dir": d, "config_dir": d / "config",
            "static_dir": d / "static", "meta": d / "snapshot.yml"}


def read_snapshot_metadata(label: str, snapshots_root=None) -> dict | None:
    p = snapshot_dir(label, snapshots_root) / "snapshot.yml"
    if not p.is_file():
        return None
    try:
        meta = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    meta.setdefault("label", label)
    meta["path"] = str(p.parent)
    return meta


def _normalise_status(s) -> str:
    s = str(s or "draft")
    return "pending_final" if s == "pending" else s


def list_snapshots(snapshots_root=None) -> pd.DataFrame:
    """Every snapshot, newest first, with where each one stands."""
    cols = ["label", "status", "description", "created_by", "created_at",
            "parent", "base_source", "approved_by", "approved_at",
            "tested_by", "n_transitions", "path"]
    root = _root(snapshots_root)
    if not root.is_dir():
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = read_snapshot_metadata(d.name, root)
        if meta is None:
            # A directory with no snapshot.yml is shown rather than skipped:
            # something created it and did not finish, which is worth seeing.
            rows.append({"label": d.name, "status": "unknown",
                         "description": "", "created_by": "", "created_at": "",
                         "parent": None, "base_source": "", "approved_by": "",
                         "approved_at": "", "tested_by": "",
                         "n_transitions": 0, "path": str(d)})
            continue
        rows.append({
            "label": meta.get("label", d.name),
            "status": _normalise_status(meta.get("status")),
            "description": meta.get("description") or "",
            "created_by": meta.get("created_by") or "",
            "created_at": meta.get("created_at") or "",
            "parent": meta.get("parent"),
            "base_source": meta.get("base_source") or "",
            "approved_by": meta.get("approved_by") or "",
            "approved_at": meta.get("approved_at") or "",
            "tested_by": meta.get("tested_by") or "",
            "n_transitions": len(meta.get("transitions") or []),
            "path": str(d),
            "code_sha_at_creation": meta.get("code_sha_at_creation") or "",
            "approval_reason": meta.get("approval_reason") or "",
        })
    cols = cols + ["code_sha_at_creation", "approval_reason"]
    for r in rows:
        r.setdefault("code_sha_at_creation", "")
        r.setdefault("approval_reason", "")
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("created_at", ascending=False).reset_index(drop=True)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def _rewrite_snapshot_config(src: Path, dst: Path) -> None:
    """Point a copied config.yml at its own frozen siblings.

    In the live project the model files sit under ``config/``; inside a
    snapshot they sit beside config.yml. Without this rewrite the snapshot
    would read the LIVE model files, which is the one thing a frozen copy must
    not do.
    """
    try:
        cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except Exception:
        shutil.copy2(src, dst)
        return
    paths = cfg.get("paths")
    if isinstance(paths, dict):
        for k in ("variable_dictionary", "models", "model_inputs",
                  "model_config"):
            v = paths.get(k)
            if isinstance(v, str) and v.startswith("config/"):
                paths[k] = v[len("config/"):]
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")


def _write_meta(meta: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    out = {k: v for k, v in meta.items() if k != "path"}
    tmp.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(path)


def create_snapshot(label: str, description: str = "", created_by=None,
                    parent: str | None = None, config_dir="config",
                    static_dir="data-raw/static", run_config_path=None,
                    snapshots_root=None, code_sha=None, audit=None, _cloning=False) -> Path:
    """Freeze the current config and static reference under a label.

    With a ``parent`` the new snapshot is cloned from THAT snapshot's frozen
    content rather than from the live project, so a chain of versions each
    builds on the last.
    """
    label = str(label or "").strip()
    if not label:
        raise ValueError("a snapshot label is required")
    if not _LABEL.match(label):
        raise ValueError("a label may hold only letters, digits, '.', '_' "
                         "and '-'")
    root = _root(snapshots_root)
    out = root / label
    if out.exists():
        raise FileExistsError(f"snapshot {label!r} already exists at {out}")

    parent_meta = None
    if parent:
        parent_meta = read_snapshot_metadata(parent, root)
        if parent_meta is None:
            raise FileNotFoundError(
                f"parent snapshot {parent!r} not found under {root}")

    (out / "config").mkdir(parents=True)
    (out / "static").mkdir(parents=True)

    if parent_meta is not None:
        pp = snapshot_paths(parent, root)
        _copy_tree(pp["config_dir"], out / "config")
        _copy_tree(pp["static_dir"], out / "static")
        base_source = f"snapshot:{parent}"
    else:
        cfg_src, static_src = Path(config_dir), Path(static_dir)
        if not cfg_src.is_dir():
            raise FileNotFoundError(f"config_dir not found: {cfg_src}")
        if not static_src.is_dir():
            raise FileNotFoundError(f"static_dir not found: {static_src}")
        _copy_tree(cfg_src, out / "config")
        _copy_tree(static_src, out / "static")
        rc = Path(run_config_path) if run_config_path else Path.cwd() / "config.yml"
        if rc.is_file():
            _rewrite_snapshot_config(rc, out / "config" / "config.yml")
        base_source = "live"

    if code_sha is None:
        try:
            from .code_version import get_current_code_sha
            code_sha = get_current_code_sha()
        except Exception:
            code_sha = None

    meta = {
        "schema_version": "1.0",
        "label": label,
        "status": "draft",
        "description": description or "",
        "created_by": created_by or _who(),
        "created_at": _now(),
        "parent": parent,
        "base_source": base_source,
        "code_sha_at_creation": code_sha,
        "approved_by": None,
        "approved_at": None,
        "approval_reason": None,
        "transitions": [],
    }
    _write_meta(meta, out / "snapshot.yml")

    if audit is not None:
        try:
            audit.record("snapshot_create", description, snapshot=label,
                         parent=parent, user=meta["created_by"])
        except Exception:
            pass
    if not _cloning:
        audit_event({"event": "snapshot_create", "snapshot": label,
                     "parent": parent, "user": meta["created_by"],
                     "description": description})
    return out


def clone_snapshot(source_label: str, new_label: str, description: str = "",
                   created_by=None, snapshots_root=None, audit=None) -> Path:
    """Start a new draft from an existing snapshot's content.

    This is how a rejected snapshot is fixed: the rejected one stays as the
    record of what was refused, and the work continues on a copy.
    """
    root = _root(snapshots_root)
    if read_snapshot_metadata(source_label, root) is None:
        raise FileNotFoundError(f"source snapshot {source_label!r} not found")
    out = create_snapshot(new_label, description, created_by,
                          parent=source_label, snapshots_root=root,
                          audit=audit, _cloning=True)
    meta = read_snapshot_metadata(new_label, root) or {}
    audit_event({"event": "snapshot_clone", "snapshot": new_label,
                 "cloned_from": source_label,
                 "user": meta.get("created_by") or created_by})
    return out


def _approval_config(config_path=None) -> dict:
    from .run_status import approval_config
    return approval_config(config_path)


def promote_snapshot(label: str, status: str, approved_by=None, reason: str = "",
                     snapshots_root=None, config_path=None,
                     audit=None) -> dict:
    """Move a snapshot to the next status.

    Every transition needs a user and a reason, not just the approval: knowing
    who marked a snapshot tested, and why, is what makes the trail readable
    afterwards.
    """
    if status == "pending":
        status = "pending_final"
    if status not in SNAPSHOT_STATUSES:
        raise ValueError(f"unknown status {status!r}; allowed: "
                         f"{', '.join(SNAPSHOT_STATUSES)}")
    root = _root(snapshots_root)
    meta = read_snapshot_metadata(label, root)
    if meta is None:
        raise FileNotFoundError(f"snapshot {label!r} not found under {root}")

    by = str(approved_by or "").strip()
    if not by:
        raise ValueError("a user is required when transitioning a snapshot")
    if not str(reason or "").strip():
        raise ValueError("a reason is required when transitioning a snapshot")

    current = _normalise_status(meta.get("status"))
    allowed = ALLOWED_TRANSITIONS.get(current, ())
    if status not in allowed:
        raise ValueError(
            f"illegal transition {current} -> {status}. From {current} the "
            f"allowed moves are: {', '.join(allowed) or '(none)'}")

    if status == "approved" and \
            _approval_config(config_path)["enforce_separation_of_duties"]:
        creator = str(meta.get("created_by") or "").strip()
        if creator and creator.lower() == by.lower():
            raise PermissionError(
                f"separation of duties is enforced: {by!r} created this "
                "snapshot and cannot also approve it. A different user must "
                "approve. Marking it tested and submitting it are their own "
                "work and remain allowed.")

    record = {"at": _now(), "by": by, "from": current, "to": status,
              "reason": str(reason).strip()}
    meta["status"] = status
    meta.setdefault("transitions", []).append(record)
    if status == "approved":
        meta["approved_by"] = by
        meta["approved_at"] = record["at"]
        meta["approval_reason"] = record["reason"]
    if status == "tested":
        meta["tested_by"] = by
        meta["tested_at"] = record["at"]

    _write_meta(meta, Path(meta["path"]) / "snapshot.yml")
    if audit is not None:
        try:
            audit.record("snapshot_promote", record["reason"], snapshot=label,
                         from_status=current, to_status=status, user=by)
        except Exception:
            pass
    audit_event({"event": "snapshot_promote", "snapshot": label,
                 "from_status": current, "to_status": status, "user": by,
                 "reason": record["reason"]})
    return meta


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_files(d: Path) -> list[str]:
    out = []
    for sub in ("config", "static"):
        s = d / sub
        if not s.is_dir():
            continue
        out += [f"{sub}/{f.relative_to(s).as_posix()}"
                for f in sorted(s.rglob("*")) if f.is_file()]
    return out


def diff_snapshots(a: str, b: str, snapshots_root=None) -> dict:
    """What differs between two snapshots, file by file.

    Compared by content hash rather than by timestamp, so a file copied
    forward unchanged does not read as a change.
    """
    root = _root(snapshots_root)
    da, db = root / a, root / b
    fa, fb = set(_snapshot_files(da)), set(_snapshot_files(db))
    modified = []
    for rel in sorted(fa & fb):
        pa, pb = da / rel, db / rel
        sa, sb = _sha256(pa), _sha256(pb)
        if sa == sb:
            continue
        modified.append({"file": rel, "a_sha": sa, "b_sha": sb,
                         "a_size": pa.stat().st_size,
                         "b_size": pb.stat().st_size})
    return {"a": a, "b": b, "files_added": sorted(fb - fa),
            "files_removed": sorted(fa - fb), "files_modified": modified}


def editable_snapshot_files(label_or_dir, snapshots_root=None) -> pd.DataFrame:
    """What may be edited in this snapshot, with what each file is for.

    Registered files absent from the snapshot are dropped so nothing phantom
    appears; unregistered files present on disk are listed generically so
    nothing is hidden.
    """
    d = Path(label_or_dir)
    if not d.is_dir():
        d = snapshot_dir(str(label_or_dir), snapshots_root)
    present = set(_snapshot_files(d))
    rows = [dict(r, advanced=bool(r.get("advanced")))
            for r in EDITABLE_FILES if r["relpath"] in present]
    known = {r["relpath"] for r in EDITABLE_FILES}
    for rel in sorted(present - known):
        if not rel.endswith((".csv", ".yml", ".yaml")):
            continue
        rows.append({"relpath": rel, "group": "Other",
                     "label": rel.split("/")[-1], "help": "", "advanced": True})
    cols = ["relpath", "group", "label", "help", "advanced"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return pd.DataFrame(rows, columns=cols)


def _editable_guard(label: str, relpath: str, root: Path) -> dict | None:
    meta = read_snapshot_metadata(label, root)
    if meta is None:
        return {"ok": False, "message": f"snapshot not found: {label}"}
    if _normalise_status(meta.get("status")) != "draft":
        return {"ok": False,
                "message": (f"this snapshot is {meta.get('status')!r} and "
                            "cannot be edited. Only a draft is editable - "
                            "that is what `tested` is for, so the numbers "
                            "being impact-tested cannot move underneath the "
                            "test.")}
    allowed = set(editable_snapshot_files(root / label)["relpath"])
    if relpath not in allowed:
        return {"ok": False,
                "message": f"{relpath!r} is not one of this snapshot's "
                           "editable files"}
    return None


def save_snapshot_yaml(label: str, relpath: str, text: str,
                       validate: bool = True, snapshots_root=None,
                       edited_by=None, audit=None) -> dict:
    """Write a YAML file into a draft snapshot, parsing it first.

    Parsed before writing because an unparseable config does not fail at save
    time -- it fails at the next run, by which point nobody remembers editing
    it.
    """
    root = _root(snapshots_root)
    bad = _editable_guard(label, relpath, root)
    if bad:
        return bad
    if validate:
        try:
            yaml.safe_load(text)
        except Exception as exc:
            return {"ok": False, "message": f"YAML parse error: {exc}"}

    full = root / label / relpath
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = full.with_suffix(full.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(full)

    if audit is not None:
        try:
            audit.record("snapshot_edit", relpath, snapshot=label,
                         user=edited_by or _who())
        except Exception:
            pass
    audit_event({"event": "snapshot_edit", "snapshot": label,
                 "relpath": relpath, "user": edited_by or _who()})
    return {"ok": True, "message": f"saved {relpath}", "path": str(full)}


def read_static_csv_with_header(path) -> dict:
    """Read a static CSV, keeping its comment header.

    The static tables carry their provenance in leading ``#`` lines -- which
    variable, which source, when it was last refreshed. A plain read-and-write
    round trip silently deletes all of it, so the header is carried separately
    and put back on save.
    """
    p = Path(path)
    if not p.is_file():
        return {"comment_header": [], "data": pd.DataFrame()}
    lines = p.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines) and (lines[i].lstrip().startswith("#")
                              or not lines[i].strip()):
        i += 1
    header = lines[:i]
    try:
        data = pd.read_csv(p, comment="#", skip_blank_lines=True)
    except Exception:
        data = pd.DataFrame()
    return {"comment_header": header, "data": data}


def write_static_csv_with_header(path, df: pd.DataFrame,
                                 comment_header=None) -> Path:
    """Write a static CSV, putting its comment header back on top."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    body = df.to_csv(index=False, na_rep="")
    text = ("\n".join(comment_header) + "\n" + body) if comment_header else body
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return p


def save_snapshot_csv(label: str, relpath: str, df: pd.DataFrame,
                      snapshots_root=None, edited_by=None,
                      audit=None) -> dict:
    """Write a CSV into a draft snapshot, preserving its comment header."""
    root = _root(snapshots_root)
    bad = _editable_guard(label, relpath, root)
    if bad:
        return bad
    full = root / label / relpath
    existing = read_static_csv_with_header(full)
    write_static_csv_with_header(full, df, existing["comment_header"])
    if audit is not None:
        try:
            audit.record("snapshot_edit", relpath, snapshot=label,
                         user=edited_by or _who(), rows=len(df))
        except Exception:
            pass
    audit_event({"event": "snapshot_edit", "snapshot": label,
                 "relpath": relpath, "user": edited_by or _who(),
                 "n_rows": len(df)})
    return {"ok": True, "message": f"saved {relpath} ({len(df):,} rows)",
            "path": str(full)}
