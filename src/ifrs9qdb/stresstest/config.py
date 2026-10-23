"""The stress config: how a macro scenario reaches the provision.

``config/stress.yml`` in the project when there is one -- frozen with the
config like any other file -- else the packaged default
(``stresstest/data/stress.yml``, byte for byte the R package's
``inst/stress/stress.yml``). The booked provision never reads it.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

__all__ = ["DEFAULT_PATH", "read_stress_config", "stress_config_path",
           "segment_loadings", "config_hash"]

DEFAULT_PATH = Path(__file__).parent / "data" / "stress.yml"


def stress_config_path(project_root=None, config_dir=None) -> Path:
    for d in [config_dir, Path(project_root) / "config" if project_root else None]:
        if d and (Path(d) / "stress.yml").is_file():
            return Path(d) / "stress.yml"
    return DEFAULT_PATH


def read_stress_config(project_root=None, config_dir=None, path=None) -> dict:
    """The stress config with every section present (missing keys take the
    packaged default's), and ``_path`` saying which file it came from."""
    dflt = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8")) or {}
    p = Path(path) if path else stress_config_path(project_root, config_dir)
    got = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.is_file() else {}

    def merge(a, b):
        out = copy.deepcopy(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "segments":
                out[k] = merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    cfg = merge(dflt, got)
    cfg["_path"] = str(p)
    return cfg


def config_hash(cfg: dict) -> str:
    body = yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                          sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:12]


def segment_loadings(cfg: dict, segment: str, external: bool) -> dict:
    """A segment's factor loadings: ``default`` plus its own for an
    internal segment; the external segment's own alone for the external
    book."""
    f = cfg.get("factor") or {}
    segs = f.get("segments") or {}
    if external:
        return {k: float(v) for k, v in (_seg(segs, f.get("external_segment",
                                                         "External book")) or {}).items()}
    out = {k: float(v) for k, v in (segs.get("default") or {}).items()}
    for k, v in (_seg(segs, segment) or {}).items():
        out[k] = out.get(k, 0.0) + float(v)
    return out


def _seg(segs: dict, name: str) -> dict | None:
    """A segment's entry: a key names one segment or several separated by
    ``|`` (a mapping's own spellings: ``Manufacturing|Industry``), matched
    ignoring case and spaces at the ends."""
    want = str(name or "").strip().lower()
    for k, v in segs.items():
        if want in [a.strip().lower() for a in str(k).split("|")]:
            return v
    return None
