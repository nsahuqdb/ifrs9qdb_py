"""The stress config: how a macro scenario reaches the provision.

``config/stress.yml`` in the project when there is one -- frozen with the
config like any other file -- else the packaged default
(``stresstest/data/stress.yml``). The booked provision never reads it.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

__all__ = ["DEFAULT_PATH", "read_stress_config", "stress_config_path", "config_hash"]

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
            if isinstance(v, dict) and isinstance(out.get(k), dict):
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
