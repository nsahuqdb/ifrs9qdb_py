"""Which PD model a run uses, resolved the way R's load_model_config() does.

config.yml names the model (``run.internal_model``, ``run.model_id`` as a
fallback); config/model.yml holds the registry of models and the variable
dictionary. R resolves the name into one record -- the components, each with
its variable's ``stress_unit_multiplier``, and a weight derived from the
p-values wherever the registry leaves it null -- and model_inputs.yml's
``mev_model_weights.mode`` then decides whether those weights or weights
computed from the p-values combine the MEVs.

The port used to read ``models["internal_v4_production"]`` and the component
weights directly. That gives the same numbers only while config.yml names that
model and the mode is ``from_model_config``; a config version that switched
either was silently ignored.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import yaml

__all__ = ["DEFAULT_MODEL", "model_id_from_run_config", "resolve_model",
           "mev_model_weights", "run_model_id"]

# What config.yml ships with. Used only when there is no config.yml at all
# (the package called from a notebook); a config.yml that leaves the model
# unset is refused by CONFIG_run_block_complete before a run starts.
DEFAULT_MODEL = "internal_v4_production"


def model_id_from_run_config(run_config) -> str | None:
    """``run.internal_model`` (or ``run.model_id``) from a parsed config.yml."""
    if not isinstance(run_config, dict):
        return None
    run = run_config.get("run")
    if not isinstance(run, dict):
        return None
    mid = run.get("internal_model")
    if mid in (None, ""):
        mid = run.get("model_id")
    return None if mid in (None, "") else str(mid)


def _weights_from_p_values(p_values) -> list[float]:
    """R's .weights_from_p_values(): inverse p-values, normalised. A missing
    or non-positive p-value weighs nothing."""
    inv = []
    for p in p_values:
        try:
            v = float(p)
        except (TypeError, ValueError):
            v = float("nan")
        inv.append(1.0 / v if v == v and v > 0 else 0.0)
    total = sum(inv)
    if total == 0:
        return [0.0] * len(inv)
    return [x / total for x in inv]


def resolve_model(model_cfg: dict, model_id: str | None = None) -> dict:
    """R's resolve_model(): the named model as one record, in R's shape.

    Returns ``{"model": {name, id, portfolios, rating_type, ttc_anchor_pd,
    max_maturity, n_forecasts, mevs, calibration}}`` -- what R writes to the
    run manifest's ``config.model_config``. Raises, as R stops, when the model
    is not in the registry or a component names a variable the dictionary
    does not define.
    """
    cfg = model_cfg or {}
    models = cfg.get("models") or {}
    mid = model_id or DEFAULT_MODEL
    if mid not in models:
        raise ValueError(f"model_id '{mid}' not found in models.yml. "
                         f"Available: {', '.join(models)}")
    m = models[mid] or {}
    dictionary = cfg.get("variables") or {}
    mevs = []
    for i, c in enumerate(m.get("mev_components") or [], start=1):
        var = c.get("variable")
        if var is None:
            raise ValueError(f"model '{mid}' component {i} has no `variable:` field")
        if var not in dictionary:
            raise ValueError(f"model '{mid}' references unknown variable '{var}'. "
                             "Add it to variable_dictionary.yml")
        spec = dictionary[var] or {}
        if spec.get("deprecated"):
            warnings.warn(f"model '{mid}' uses deprecated variable '{var}'",
                          stacklevel=2)
        mult = spec.get("stress_unit_multiplier")
        mevs.append({
            "name": spec.get("display_name"),
            "variable_id": var,
            "intercept": c.get("intercept"),
            "coefficient": c.get("coefficient"),
            "p_value": c.get("p_value"),
            "weight": c.get("weight"),
            "standard_deviation": c.get("standard_deviation"),
            "units": spec.get("units"),
            "scale": spec.get("scale"),
            "stress_unit_multiplier": 1 if mult is None else mult,
        })
    if any(x["weight"] is None for x in mevs):
        derived = _weights_from_p_values([x["p_value"] for x in mevs])
        for x, w in zip(mevs, derived):
            if x["weight"] is None:
                x["weight"] = w
    horizons = cfg.get("horizons") or {}
    return {"model": {
        "name": f"{mid} — {m.get('description') or ''}",
        "id": mid,
        "portfolios": m.get("portfolios"),
        "rating_type": m.get("rating_type"),
        "ttc_anchor_pd": cfg.get("ttc_anchor_pd"),
        "max_maturity": horizons.get("max_maturity"),
        "n_forecasts": horizons.get("n_forecasts"),
        "mevs": mevs,
        "calibration": m.get("calibration"),
    }}


def mev_model_weights(resolved: dict, model_inputs: dict | None) -> list[float]:
    """R's load_model_inputs() step 4: the weights that combine the MEVs.

    ``from_model_config`` (the default) takes the resolved components'
    weights; ``auto_p_value`` weighs each MEV by max(p) / p, normalised.
    """
    block = (model_inputs or {}).get("mev_model_weights") or {}
    mode = block.get("mode") or "from_model_config"
    mevs = resolved["model"]["mevs"]
    if mode == "from_model_config":
        return [float(x["weight"]) for x in mevs]
    if mode == "auto_p_value":
        p = [float(x["p_value"]) for x in mevs]
        if any(v <= 0 for v in p):
            raise ValueError("MEV p-values must all be > 0 to compute auto "
                             "p-value weights")
        base = max(p)
        raw = [base / v for v in p]
        total = sum(raw)
        return [r / total for r in raw]
    raise ValueError(f"Unknown mev_model_weights mode: {mode} (expected "
                     "'from_model_config' or 'auto_p_value')")


def run_model_id(run_dir) -> str | None:
    """The model a finished run priced on.

    Its frozen config.yml names it; the manifest's resolved model is the
    fallback for a run that froze no config.yml. None when neither says --
    the caller then uses DEFAULT_MODEL, which is what every run before the
    model became selectable used.
    """
    run_dir = Path(run_dir)
    for base in (run_dir, run_dir.parent):
        p = base / "config_used" / "config" / "config.yml"
        if p.is_file():
            try:
                rc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                rc = {}
            mid = model_id_from_run_config(rc)
            if mid:
                return mid
    for p in (run_dir / "reports" / "manifest.json", run_dir / "manifest.json",
              run_dir.parent / "reports" / "manifest.json"):
        if not p.is_file():
            continue
        try:
            man = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        mc = ((man.get("config") or {}).get("model_config") or {})
        model = mc.get("model") if isinstance(mc, dict) else None
        if isinstance(model, dict) and model.get("id"):
            return str(model["id"])
    return None
