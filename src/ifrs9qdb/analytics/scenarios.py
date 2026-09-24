"""Scenario analysis over a completed run.

A run prices the book five times -- once per macro scenario -- and then once
more on the probability-weighted curve. The weighted figure is the one that is
booked; the five behind it are what makes it defensible, because "the
provision is 1.36bn" means little next to "it is 1.36bn, and it would be
2.1bn in the severe downturn we assign 15% weight to".

Everything here reads the per-scenario reports a run already wrote. Nothing is
re-priced, so these are cheap and cannot disagree with the run.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .profile import normalise

__all__ = ["scenario_files", "scenario_ecl_from_outputs", "scenario_comparison",
           "scenario_reweight", "scenario_sensitivity", "scenario_stage_split"]

_PATTERN = re.compile(r"^FinalEclReport_scenario_(.+)\.csv$")


def _output_dir(run) -> Path:
    p = Path(run)
    out = p / "Output"
    return out if out.is_dir() else p


def scenario_files(run) -> dict[str, Path]:
    """The per-scenario reports a run wrote, keyed by scenario name."""
    d = _output_dir(run)
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.glob("FinalEclReport_scenario_*.csv")):
        m = _PATTERN.match(f.name)
        if m:
            out[m.group(1).replace("_", " ")] = f
    return out


def scenario_ecl_from_outputs(run) -> dict:
    """Total ECL under each scenario, and the weighted figure beside them.

    Reports a REASON rather than an empty result when there is nothing to
    show. The three cases are genuinely different -- no scenario reports at
    all, reports that will not parse, and reports that parse and price to zero
    -- and a blank chart says none of them.
    """
    d = _output_dir(run)
    if not d.is_dir():
        return {"ok": False,
                "reason": "The run's Output folder could not be read."}

    files = scenario_files(run)
    if not files:
        return {"ok": False,
                "reason": ("This run has no per-scenario ECL reports. Runs "
                           "produced before scenario outputs were added do "
                           "not carry them.")}

    ecl, unreadable = {}, []
    for name, f in files.items():
        try:
            rep = normalise(pd.read_csv(f, low_memory=False))
        except Exception:
            unreadable.append(name)
            continue
        if rep is None or len(rep) == 0:
            unreadable.append(name)
            continue
        ecl[name] = float(pd.to_numeric(rep["ecl"], errors="coerce")
                          .fillna(0).sum())

    if not ecl:
        return {"ok": False,
                "reason": "The per-scenario reports could not be read."}

    # A report that reads fine and prices to nothing is a failure, not a
    # result. Saying so beats drawing a flat chart.
    if all(v == 0 for v in ecl.values()):
        return {"ok": False,
                "reason": ("Every per-scenario report prices to zero. That is "
                           "a failed run rather than a result -- re-run the "
                           "pipeline to regenerate them.")}

    weighted = None
    wf = d / "FinalEclReport.csv"
    if wf.is_file():
        try:
            weighted = float(pd.to_numeric(
                normalise(pd.read_csv(wf, low_memory=False))["ecl"],
                errors="coerce").fillna(0).sum())
        except Exception:
            weighted = None

    return {"ok": True, "ecl": ecl, "weighted": weighted,
            "files": len(files), "unreadable": unreadable}


def scenario_comparison(ecl_by_scenario: dict,
                        severity: pd.DataFrame | None = None) -> pd.DataFrame:
    """The scenarios side by side, ordered by severity where it is known.

    Ordered by severity rather than alphabetically, because the shape of the
    answer -- does the provision rise monotonically as the world gets worse? --
    is the first thing anybody looks for, and alphabetical order hides it.
    """
    if not ecl_by_scenario:
        return pd.DataFrame()
    out = pd.DataFrame({"scenario": list(ecl_by_scenario),
                        "ecl": [float(v) for v in ecl_by_scenario.values()]})
    if severity is not None and len(severity) and "scenario" in severity.columns:
        keep = [c for c in ("scenario", "severity_z", "description")
                if c in severity.columns]
        out = out.merge(severity[keep], on="scenario", how="left")
        if "severity_z" in out.columns:
            out = out.sort_values("severity_z")

    base = out.loc[out["scenario"] == "Base Case", "ecl"]
    if len(base) == 1 and float(base.iloc[0]) > 0:
        b = float(base.iloc[0])
        out["vs_base"] = out["ecl"] - b
        out["vs_base_pct"] = 100 * (out["ecl"] - b) / b
    return out.reset_index(drop=True)


def scenario_reweight(ecl_by_scenario: dict, weights: dict) -> dict | None:
    """The weighted ECL under a different set of scenario weights.

    The weights are normalised over the scenarios actually present, so a
    weight naming a scenario the run did not price cannot quietly shrink the
    total. Which ones were dropped is reported rather than assumed harmless.
    """
    if not ecl_by_scenario or not weights:
        return None
    common = [s for s in weights if s in ecl_by_scenario]
    if not common:
        return None
    w = {s: float(weights[s] or 0) for s in common}
    total_w = sum(w.values())
    if total_w <= 0:
        return None
    norm = {s: v / total_w for s, v in w.items()}
    return {"total": sum(ecl_by_scenario[s] * norm[s] for s in common),
            "normalised_weights": norm,
            "missing": [s for s in weights if s not in ecl_by_scenario]}


def scenario_sensitivity(ecl_by_scenario: dict, weights: dict,
                         shift: float = 0.10) -> pd.DataFrame:
    """What moving weight ONTO each scenario does to the provision.

    Each row takes ``shift`` of probability mass from the other scenarios --
    pro rata, so their relative standing is unchanged -- and gives it to one.
    That is the question a committee actually asks: if we thought the downturn
    ten points more likely, what would we book?
    """
    base = scenario_reweight(ecl_by_scenario, weights)
    if base is None:
        return pd.DataFrame()
    w = base["normalised_weights"]

    rows = []
    for s in w:
        room = 1.0 - w[s]
        if room <= 0:
            continue
        take = min(shift, room)
        others = [o for o in w if o != s]
        denom = sum(w[o] for o in others)
        if denom <= 0:
            continue
        up = {o: w[o] * (1 - take / denom) for o in others}
        up[s] = w[s] + take
        moved = scenario_reweight(ecl_by_scenario, up)
        if moved is None:
            continue
        rows.append({
            "scenario": s,
            "weight_before": w[s], "weight_after": up[s],
            "ecl_scenario": float(ecl_by_scenario[s]),
            "ecl_base": base["total"], "ecl_shifted": moved["total"],
            "change": moved["total"] - base["total"],
            "pct": 100 * (moved["total"] - base["total"])
            / max(base["total"], 1.0),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out.attrs["base_total"] = base["total"]
    out.attrs["shift"] = shift
    return (out.reindex(out["change"].abs().sort_values(ascending=False).index)
            .reset_index(drop=True))


def scenario_stage_split(run) -> pd.DataFrame:
    """Exposure and ECL by stage, under each scenario.

    The headline moves for two reasons -- the curves get worse, and contracts
    move stage -- and the split is what separates them.
    """
    rows = []
    for name, f in scenario_files(run).items():
        try:
            rep = normalise(pd.read_csv(f, low_memory=False))
        except Exception:
            continue
        if rep is None or len(rep) == 0:
            continue
        g = (rep.assign(stage=pd.to_numeric(rep["stage"], errors="coerce"))
             .groupby("stage", dropna=True)
             .agg(contracts=("contract", "size"),
                  exposure=("exposure", "sum"),
                  ecl=("ecl", "sum")).reset_index())
        g.insert(0, "scenario", name)
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["coverage"] = np.where(out["exposure"] > 0,
                               out["ecl"] / out["exposure"], 0.0)
    return out
