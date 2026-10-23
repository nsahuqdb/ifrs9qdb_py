"""Stress scenarios: shocks on any macro variables, several at once.

A scenario is a set of shocks on top of the quarter's baseline. A shock names
a concept (a variable of the macro store's catalogue), a size -- in the
variable's own unit or in standard deviations of its history -- a shape over
the forecast years, and the year it starts:

    one_year   the move in its start year only
    held       the move in every year from the start
    build      reaching the full move over ``years`` years, then held
    recover    the full move in the start year, gone after ``years`` more
    custom     a path given year by year (``path``), in the variable's unit

Shocks on the same variable add up. With ``knock_on`` (the default) every
other variable moves as the macro model says it moves with them
(``macro.var.MacroModel.condition``); without it they stay at baseline. An
``override`` fixes a variable's path outright and conditions the rest on it
too. ``levers`` add a parameter stress (a ``stress.StressSpec``: a PD
multiplier, notches, collateral, the largest N defaulting) on top, and
``scenario_weights`` price the internal book on one IFRS 9 scenario alone.

Saved scenarios live in the project's ``config/stress_scenarios.yml``; the
library below ships with the package (and identically with the R one).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

__all__ = ["SHAPES", "Shock", "Scenario", "library", "scenarios_path",
           "read_scenarios", "write_scenarios", "scenario_paths", "shape_profile"]

SHAPES = ("one_year", "held", "build", "recover", "custom")


def shape_profile(shape: str, H: int, start: int = 1, years: int = 2) -> np.ndarray:
    """The multiplier on a shock's size in each forecast year 1..H."""
    p = np.zeros(H)
    s = max(1, int(start)) - 1
    n = max(1, int(years))
    if s >= H:
        return p
    if shape == "one_year":
        p[s] = 1.0
    elif shape == "held":
        p[s:] = 1.0
    elif shape == "build":
        for j in range(s, H):
            p[j] = min(1.0, (j - s + 1) / n)
    elif shape == "recover":
        for j in range(s, H):
            p[j] = max(0.0, 1.0 - (j - s) / n)
    return p


@dataclass
class Shock:
    concept: str
    size: float = 0.0
    unit: str = "native"          # native | sd
    shape: str = "one_year"
    start: int = 1
    years: int = 2
    path: list | None = None      # custom: the move year by year, native unit

    def native_path(self, H: int, sd: float) -> np.ndarray:
        if self.shape == "custom":
            p = np.zeros(H)
            vals = [float(v) for v in (self.path or [])][:H]
            p[:len(vals)] = vals
            return p * (sd if self.unit == "sd" else 1.0)
        scale = float(self.size) * (sd if self.unit == "sd" else 1.0)
        return shape_profile(self.shape, H, self.start, self.years) * scale

    @classmethod
    def from_dict(cls, d: dict) -> "Shock":
        d = dict(d or {})
        return cls(concept=str(d.get("concept", "")),
                   size=float(d.get("size") or 0.0),
                   unit=str(d.get("unit") or "native"),
                   shape=str(d.get("shape") or "one_year"),
                   start=int(float(d.get("start") or 1)),
                   years=int(float(d.get("years") or 2)),
                   path=[float(v) for v in d["path"]] if d.get("path") is not None else None)


@dataclass
class Scenario:
    name: str = "Scenario"
    description: str = ""
    shocks: list = field(default_factory=list)
    knock_on: bool = True
    overrides: dict = field(default_factory=dict)     # concept -> path (native)
    levers: dict = field(default_factory=dict)        # StressSpec fields
    weights: str = "hold"                             # hold | auto
    scenario_weights: dict | None = None              # IFRS 9 scenario weights
    kind: str = "custom"                              # custom | library

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d or {})
        return cls(
            name=str(d.get("name") or "Scenario"),
            description=str(d.get("description") or ""),
            shocks=[Shock.from_dict(s) for s in (d.get("shocks") or [])],
            knock_on=bool(d.get("knock_on", True)),
            overrides={str(k): [float(x) for x in v]
                       for k, v in (d.get("overrides") or {}).items()},
            levers=dict(d.get("levers") or {}),
            weights=str(d.get("weights") or "hold"),
            scenario_weights=({str(k): float(v) for k, v in d["scenario_weights"].items()}
                              if d.get("scenario_weights") else None),
            kind=str(d.get("kind") or "custom"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shocks"] = [{k: v for k, v in s.items() if v is not None} for s in d["shocks"]]
        if not d["scenario_weights"]:
            d.pop("scenario_weights")
        return d

    def concepts(self) -> list[str]:
        out = [s.concept for s in self.shocks] + list(self.overrides)
        return list(dict.fromkeys(out))

    def given(self, H: int, sd: dict) -> dict:
        """The paths the scenario sets itself: its shocks summed by
        variable, then its overrides."""
        g: dict = {}
        for s in self.shocks:
            p = s.native_path(H, float(sd.get(s.concept, 1.0)))
            g[s.concept] = g.get(s.concept, np.zeros(H)) + p
        for c, v in self.overrides.items():
            p = np.zeros(H)
            vals = [float(x) for x in v][:H]
            p[:len(vals)] = vals
            g[c] = p
        return g


def scenario_paths(scn: Scenario, model, H: int = 5) -> dict:
    """Every concept's deviation from baseline under a scenario, the paths
    it set itself, and its severity."""
    given = scn.given(H, model.sd)
    if scn.knock_on:
        paths = model.condition(given, H)
    else:
        paths = {c: np.zeros(H) for c in model.concepts}
        paths.update({c: np.asarray(v, float) for c, v in given.items()})
    for c, v in given.items():                 # a concept outside the model
        paths[c] = np.asarray(v, float)
    return {"paths": paths, "given": given, "severity": model.severity(given, H)}


def _s(concept, size=0.0, shape="one_year", unit="native", start=1, years=2, path=None):
    return Shock(concept, size, unit, shape, start, years, path)


def library() -> list[Scenario]:
    """The scenarios that ship with the package. Moves are those of the
    episodes named, or round severe-but-plausible numbers; each is a
    starting point to copy and edit."""
    L = "library"
    return [
        Scenario("Oil and LNG price collapse",
                 "The 2014-16 fall replayed: Brent down 47% then 17%, LNG with it.",
                 [_s("OIL_PRICE", shape="custom", path=[-47, -17, 0, 0, 0]),
                  _s("LNG_PRICE", shape="custom", path=[-40, -25, 0, 0, 0])], kind=L),
        Scenario("Regional conflict or blockade",
                 "Trade and visitors cut, non-oil activity and property down, "
                 "recovering over two years (2017, 2026).",
                 [_s("VISITORS_QAT", -30, "recover", years=2),
                  _s("NONOIL_GDP_QAT", -3, "recover", years=2),
                  _s("PROPERTY_QAT", -10, "recover", years=3),
                  _s("GDP_QAT", -2, "recover", years=2)], kind=L),
        Scenario("Fed tightening through the peg",
                 "US and Qatar policy rates 2 points higher from the first year.",
                 [_s("US_RATE", 2, "held"), _s("QA_RATE", 2, "held")], kind=L),
        Scenario("Property price correction",
                 "Real estate prices down 15% then 10%.",
                 [_s("PROPERTY_QAT", shape="custom", path=[-15, -10, 0, 0, 0])], kind=L),
        Scenario("Fiscal consolidation",
                 "Government spending cut 15%, then 5% more.",
                 [_s("GOV_SPEND_QAT", shape="custom", path=[-15, -5, 0, 0, 0])], kind=L),
        Scenario("Global recession",
                 "World and China growth 3 points lower, oil and LNG prices down.",
                 [_s("GDP_WLD", -3, "recover", years=2),
                  _s("GDP_CHN", -3, "recover", years=2),
                  _s("OIL_PRICE", -35, "one_year"),
                  _s("LNG_PRICE", -30, "one_year")], kind=L),
        Scenario("GCC-wide downturn",
                 "Every GCC economy 1.5 standard deviations weaker, recovering over two years.",
                 [_s(c, -1.5, "recover", unit="sd", years=2)
                  for c in ("GDP_SAU", "GDP_ARE", "GDP_KWT", "GDP_OMN", "GDP_BHR", "GDP_QAT")],
                 kind=L),
        Scenario("Combined severe",
                 "Oil collapse, property correction, Fed tightening and a spending cut together.",
                 [_s("OIL_PRICE", shape="custom", path=[-47, -17, 0, 0, 0]),
                  _s("PROPERTY_QAT", shape="custom", path=[-15, -10, 0, 0, 0]),
                  _s("US_RATE", 2, "held"), _s("QA_RATE", 2, "held"),
                  _s("GOV_SPEND_QAT", shape="custom", path=[-15, -5, 0, 0, 0])], kind=L),
    ]


def scenarios_path(project_root) -> Path:
    return Path(project_root) / "config" / "stress_scenarios.yml"


def read_scenarios(path) -> list[Scenario]:
    import yaml
    p = Path(path)
    if not p.is_file():
        return []
    try:
        y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return [Scenario.from_dict(d) for d in (y.get("scenarios") or []) if isinstance(d, dict)]


def write_scenarios(scenarios, path) -> Path:
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [(s if isinstance(s, Scenario) else Scenario.from_dict(s)).to_dict()
            for s in scenarios]
    for r in rows:
        r["kind"] = "custom"
    p.write_text(yaml.safe_dump({"scenarios": rows}, sort_keys=False,
                                allow_unicode=True), encoding="utf-8")
    return p
