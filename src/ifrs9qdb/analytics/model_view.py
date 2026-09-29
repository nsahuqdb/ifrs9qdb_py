"""The model a run actually used, read back from what it froze.

A run copies its config and its static reference into ``config_used/`` beside
its outputs. That copy is the only honest source for "what assumptions produced
this number" -- the files in the repository have moved on since, and reading
them to explain a past quarter is how a reconciliation ends up explaining the
wrong thing.

Everything here reads that frozen copy and nothing else. A run made before
``config_used/`` existed reports that it cannot be explained, rather than
answering from today's config.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

__all__ = ["config_used", "scenario_severity", "scenario_weights",
           "mev_forecast_table", "mev_weights_table", "read_scenario_stpd"]

_MODEL = "internal_v4_production"


def _read_yaml(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick(d: pd.DataFrame, name: str) -> pd.Series | None:
    """A column by name, ignoring case and punctuation.

    The per-scenario StPD files are written in the engine's own snake_case
    (``portfolio_code``) while the headline ``StPD.csv`` carries the LIC
    PascalCase (``PortfolioCode``). Matching on the squashed name reads both,
    which is why the scenario curves came back empty when they existed.
    """
    want = re.sub(r"[^a-z0-9]", "", name.lower())
    for c in d.columns:
        if re.sub(r"[^a-z0-9]", "", str(c).lower()) == want:
            return d[c]
    return None


def _read_csv(directory, name: str) -> pd.DataFrame | None:
    if directory is None:
        return None
    f = Path(directory) / name
    if not f.is_file():
        return None
    try:
        return pd.read_csv(f, low_memory=False)
    except Exception:
        return None


def config_used(run) -> dict | None:
    """The run's frozen ``config/`` and ``static/``, or None.

    The pipeline writes ``config_used`` at the RUN root, beside ``Output/``,
    not inside it. Callers hold either path depending on where they came from,
    so all the likely places are tried rather than one.
    """
    if run is None:
        return None
    p = Path(run)
    for base in (p / "config_used", p.parent / "config_used",
                 p.parent.parent / "config_used"):
        if base.is_dir():
            cfg, static = base / "config", base / "static"
            if cfg.is_dir() and static.is_dir():
                return {"config": cfg, "static": static, "root": base}
    return None


# --------------------------------------------------------- scenarios ------
def scenario_severity(run) -> pd.DataFrame:
    """The scenario set and its severity, ordered worst to best.

    ``scenario_severity.csv`` is static reference rather than a run output, so
    it lives in the frozen static folder. Ordering by ``severity_z`` is what
    makes a scenario chart readable: the question asked of it is always whether
    the provision rises monotonically as the world gets worse.
    """
    cu = config_used(run)
    d = _read_csv(cu["static"] if cu else None, "scenario_severity.csv")
    if d is None:
        d = _read_csv(run, "scenario_severity.csv")
    if d is None or "scenario" not in d.columns:
        return pd.DataFrame()
    d = d.copy()
    d["severity_z"] = pd.to_numeric(d["severity_z"], errors="coerce")
    return d.sort_values("severity_z").reset_index(drop=True)


def scenario_weights(config_path) -> pd.DataFrame:
    """The explicit scenario weights, both scales side by side.

    These are what the config FILE states. The engine may compute them instead
    -- the internal scale runs on ``auto_non_oil_gdp_cdf``, where the explicit
    block is a rounded snapshot of the calculation -- so this is what was
    written down, not necessarily what was applied.
    """
    if config_path is None:
        return pd.DataFrame()
    p = Path(config_path)
    if p.is_dir():
        p = p / "model_inputs.yml"
    if not p.is_file():
        return pd.DataFrame()
    y = _read_yaml(p)
    if not y:
        return pd.DataFrame()

    def grab(node) -> pd.DataFrame | None:
        w = (y.get(node) or {}).get("explicit_weights")
        if not w:
            return None
        return pd.DataFrame({"scenario": list(w),
                             "weight": [float(v) for v in w.values()]})

    a = grab("internal_scenario_weights")
    b = grab("external_scenario_weights")
    if a is None and b is None:
        return pd.DataFrame()
    if a is None:
        a = b.assign(weight=float("nan"))
    a = a.rename(columns={"weight": "internal"})
    if b is None:
        a["external"] = float("nan")
        return a
    return a.merge(b.rename(columns={"weight": "external"}), on="scenario",
                   how="outer")


def read_scenario_stpd(run, scenario: str) -> dict[str, list]:
    """The cumulative PD curves a run wrote for ONE scenario.

    Keyed ``"<portfolio>|<bucket>"`` and zero-prepended, which is the shape
    ``sum_marginal_ecl`` expects: ``curve[m]`` is the cumulative PD at month
    ``m`` and ``curve[0]`` is zero.
    """
    if run is None or not scenario:
        return {}
    safe = re.sub(r"[^A-Za-z0-9]+", "_", scenario)
    d = _read_csv(run, f"StPD_{safe}.csv")
    if d is None:
        d = _read_csv(Path(run) / "Output", f"StPD_{safe}.csv")
    if d is None:
        return {}
    pf = _pick(d, "PortfolioCode")
    mon = _pick(d, "MonthLifetime")
    val = _pick(d, "PDLifetime")
    if pf is None or mon is None or val is None:
        return {}
    bk = _pick(d, "PDBucketDim1")
    t = pd.DataFrame({
        "pf": pf.astype(str),
        "bk": (bk if bk is not None else pd.Series("", index=d.index)).astype(str),
        "m": pd.to_numeric(mon, errors="coerce"),
        "v": pd.to_numeric(val, errors="coerce"),
    }).dropna()
    return {f"{p}|{b}": [0.0] + list(g.sort_values("m")["v"])
            for (p, b), g in t.groupby(["pf", "bk"])}


# --------------------------------------------------------------- MEV ------
def _mev_components(model_cfg, run=None) -> list | None:
    """The components of the model the run priced on -- the one its frozen
    config.yml names -- with R's resolved weights (a null weight derived from
    the p-values). The shipped model when the run does not say."""
    from ..etl.model_registry import resolve_model, run_model_id
    mid = (run_model_id(run) if run is not None else None) or _MODEL
    try:
        comps = model_cfg["models"][mid]["mev_components"]
    except (KeyError, TypeError):
        return None
    try:
        mevs = resolve_model(model_cfg, mid)["model"]["mevs"]
    except Exception:
        return comps
    return [{**c, "weight": m["weight"]} for c, m in zip(comps, mevs)]


def mev_forecast_table(run) -> pd.DataFrame:
    """The macroeconomic forecast the run priced on, one row per MEV per year.

    Labelled from the model's own variable dictionary rather than by position,
    because "MEV 2" in a committee pack is not an answer to "which variable
    moved".
    """
    cu = config_used(run)
    if cu is None:
        return pd.DataFrame()
    mi = _read_yaml(cu["config"] / "model_inputs.yml")
    mc = _read_yaml(cu["config"] / "model.yml")
    fc = ((mi or {}).get("mev_forecasts") or {}).get("forecasts")
    if not fc:
        return pd.DataFrame()

    comp = _mev_components(mc, run) or []
    names = [c.get("variable") for c in comp]
    variables = (mc or {}).get("variables", {}) or {}

    rows = []
    for year in sorted(fc, key=lambda k: int(k)):
        values = list(fc[year])
        for i, v in enumerate(values):
            name = names[i] if i < len(names) else f"MEV {i + 1}"
            label = (variables.get(name, {}) or {}).get("display_name", name) \
                if name else f"MEV {i + 1}"
            rows.append({"year": int(year), "idx": i + 1, "mev": name,
                         "label": label, "value": float(v)})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).sort_values(["idx", "year"])
            .reset_index(drop=True))


def mev_weights_table(run) -> pd.DataFrame:
    """How much each MEV contributes to the combined shift factor.

    The weights and the coefficients together are the model: a variable with a
    large coefficient and a small weight moves the provision less than its
    coefficient suggests, and reading either alone misleads.
    """
    cu = config_used(run)
    if cu is None:
        return pd.DataFrame()
    comp = _mev_components(_read_yaml(cu["config"] / "model.yml"), run)
    if not comp:
        return pd.DataFrame()
    return pd.DataFrame([{
        "idx": i + 1,
        "mev": c.get("variable"),
        "weight": pd.to_numeric(c.get("weight"), errors="coerce"),
        "coefficient": pd.to_numeric(c.get("coefficient"), errors="coerce"),
        "intercept": pd.to_numeric(c.get("intercept"), errors="coerce"),
        "standard_deviation": pd.to_numeric(c.get("standard_deviation"),
                                            errors="coerce"),
    } for i, c in enumerate(comp)])
