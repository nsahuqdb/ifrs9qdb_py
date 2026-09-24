"""The calculator registry: which version of the calculation code a run used.

Deployment does not always go through git -- a bank release can be a copied
directory -- so the code version is tracked here explicitly rather than being
inferred from a SHA.

Each run records two things, and the pair is the point:

  * the version it SELECTED, from this registry; and
  * a FINGERPRINT of the code that actually executed.

Recording only the first lets a run claim v1.0 while running something else.
Recording only the second gives a hash nobody can name. Together they let any
run be traced to an exact calculator, and they make it visible when deployed
code has drifted from the version it claims to be.

One difference from the R, and it is deliberate. The R can execute an archived
version by sourcing its files into a fresh environment. Python has no clean
equivalent -- rebinding a live package's modules mid-process gives a state that
matches neither version -- so an archived version here is a directory you
install or put on the path, and :func:`calc_version_code_dir` returns it. The
registry, the fingerprint and the drift check all work the same.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["registry_path", "read_calculator_versions",
           "list_calculator_versions", "current_calculator_version",
           "get_calculator_version", "compute_code_fingerprint",
           "calculator_version_for_run", "register_calculator_version",
           "set_active_calculator_version", "calc_version_code_dir",
           "PACKAGED_FINGERPRINT"]

PACKAGED_FINGERPRINT = Path(__file__).parent / "CODE_FINGERPRINT"


def registry_path(root=None) -> Path:
    root = Path(root) if root else Path.cwd()
    return root / "config" / "calculator_versions.yml"


def read_calculator_versions(root=None) -> dict:
    """The registry, tolerant of a missing or unparseable file.

    An empty registry is a normal state -- a fresh deployment has not
    registered anything -- so it is not an error.
    """
    p = registry_path(root)
    if not p.is_file():
        return {"active": None, "versions": []}
    try:
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"active": None, "versions": []}
    return {"active": y.get("active"), "versions": list(y.get("versions") or [])}


def list_calculator_versions(root=None) -> pd.DataFrame:
    """One row per registered version, with the active one flagged."""
    reg = read_calculator_versions(root)
    cols = ["id", "label", "description", "created_at", "created_by",
            "code_hash", "archived", "active"]
    if not reg["versions"]:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    rows = []
    for v in reg["versions"]:
        code_dir = v.get("code_dir")
        rows.append({
            "id": v.get("id"),
            "label": v.get("label") or v.get("id"),
            "description": v.get("description") or "",
            "created_at": v.get("created_at") or "",
            "created_by": v.get("created_by") or "",
            "code_hash": v.get("code_hash") or "",
            "archived": bool(v.get("archived")) or bool(code_dir),
            "active": v.get("id") == reg["active"],
        })
    return pd.DataFrame(rows, columns=cols)


def current_calculator_version(root=None) -> dict | None:
    reg = read_calculator_versions(root)
    if not reg["active"]:
        return None
    for v in reg["versions"]:
        if v.get("id") == reg["active"]:
            return v
    return None


def get_calculator_version(id: str, root=None) -> dict | None:
    for v in read_calculator_versions(root)["versions"]:
        if v.get("id") == id:
            return v
    return None


def compute_code_fingerprint(code_dir=None) -> str | None:
    """An md5 over the calculation sources, stable across machines.

    Each file is hashed, the (name, hash) pairs are sorted, and the digest is
    taken over that listing -- so the result does not depend on directory
    order, and a renamed file changes it as much as an edited one.

    With no directory given, a fingerprint baked in at build time wins.
    Fingerprinting an installed package's directory would otherwise pick up
    .pyc files and whatever else the installer left there.
    """
    if code_dir is None:
        if PACKAGED_FINGERPRINT.is_file():
            baked = PACKAGED_FINGERPRINT.read_text(encoding="utf-8").strip()
            if baked:
                return baked.splitlines()[0].strip()
        code_dir = Path(__file__).parent

    d = Path(code_dir)
    if not d.is_dir():
        return None
    files = sorted(p for p in d.rglob("*.py")
                   if "__pycache__" not in p.parts)
    if not files:
        return None
    lines = []
    for f in files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        lines.append(f"{f.relative_to(d).as_posix()}:{h}")
    return hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()


def calculator_version_for_run(id: str | None = None, root=None,
                               code_dir=None) -> dict:
    """The record a run stamps into its manifest.

    Reports the declared version AND whether the deployed code matched the
    fingerprint registered for it. ``matches_registered`` is None when there is
    nothing to compare against, which is different from False and should not be
    displayed as a mismatch.
    """
    reg = read_calculator_versions(root)
    if not id:
        id = reg["active"]
    entry = get_calculator_version(id, root) if id else None
    current = compute_code_fingerprint(code_dir)
    registered = (entry or {}).get("code_hash") or ""
    matches = None if (current is None or not registered) \
        else current == registered
    return {
        "id": id,
        "label": (entry or {}).get("label") or id,
        "code_hash": current,
        "registered_hash": registered or None,
        "matches_registered": matches,
    }


def _write(reg: dict, root=None) -> Path:
    p = registry_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True),
                   encoding="utf-8")
    tmp.replace(p)
    return p


def _who() -> str:
    return (os.environ.get("IFRS9_USER") or os.environ.get("USER")
            or os.environ.get("USERNAME") or "unknown")


def register_calculator_version(id: str, label: str | None = None,
                                description: str = "",
                                created_by: str | None = None,
                                make_active: bool = True, root=None,
                                code_dir=None) -> dict:
    """Register a version, archiving an immutable copy of the code.

    The archive is what makes the registry worth having: a fingerprint alone
    tells you the deployed code has changed, not what it changed from.
    """
    if not id:
        raise ValueError("a calculator version id is required")
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "0123456789._-" for c in id):
        raise ValueError("a version id may hold only letters, digits, "
                         "'.', '_' and '-'")
    reg = read_calculator_versions(root)
    if any(v.get("id") == id for v in reg["versions"]):
        raise ValueError(f"calculator version {id!r} already exists")

    root_p = Path(root) if root else Path.cwd()
    src = Path(code_dir) if code_dir else Path(__file__).parent

    archive_rel = Path("calculator_versions") / id / "src"
    archived = False
    if src.is_dir():
        dest = root_p / archive_rel
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.rglob("*.py"):
            if "__pycache__" in f.parts:
                continue
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
        archived = any(dest.rglob("*.py"))

    entry = {
        "id": id,
        "label": label or id,
        "description": description or "",
        "created_at": _dt.date.today().isoformat(),
        "created_by": created_by or _who(),
        "code_hash": compute_code_fingerprint(src),
        "code_dir": archive_rel.as_posix() if archived else None,
        "archived": archived,
    }
    reg["versions"].append(entry)
    if make_active:
        reg["active"] = id
    _write(reg, root)
    return entry


def set_active_calculator_version(id: str, root=None) -> str:
    reg = read_calculator_versions(root)
    if not any(v.get("id") == id for v in reg["versions"]):
        raise ValueError(f"unknown calculator version {id!r}")
    reg["active"] = id
    _write(reg, root)
    return id


def calc_version_code_dir(id: str, root=None) -> Path | None:
    """Where a version's archived code lives, or None if it was not archived.

    Put this on ``sys.path`` (or install from it) to run that version. It is
    deliberately not imported for you: swapping a package's modules inside a
    running process leaves a state that matches neither version.
    """
    root_p = Path(root) if root else Path.cwd()
    entry = get_calculator_version(id, root)
    rel = (entry or {}).get("code_dir") or f"calculator_versions/{id}/src"
    p = root_p / rel
    return p if p.is_dir() else None
