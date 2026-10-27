"""Macro stress testing: scenarios, how they reach the provision, and how
big the risk is.

    config        the stress parameters (config/stress.yml)
    scenarios     a scenario's main variables, the builder's shapes, saving
    library       the pre-built scenarios and the episodes behind them
    transmission  a scenario priced: PD model, stage moves, collateral
    risk          contributions, sensitivity, ECL at risk, reverse stress

A scenario sets a few main variables; the linked effects (macro.system) move
the rest. Everything prices through ``stress.reprice`` -- the same
``compute_ecl`` the run used -- so an empty scenario is the run's own
provision, and nothing here changes how a run is calculated.
"""
from .config import read_stress_config
from .library import CATEGORIES, REGIONS, library
from .scenarios import SHAPES, Scenario, ScenarioStore, scenarios_db_path, shape_path

__all__ = ["read_stress_config", "Scenario", "ScenarioStore", "scenarios_db_path", "SHAPES",
           "shape_path", "library", "CATEGORIES", "REGIONS", "RunContext", "prepare", "transmit"]


def __getattr__(name):
    if name in ("RunContext", "prepare", "transmit"):
        from . import transmission
        return getattr(transmission, name)
    raise AttributeError(name)
