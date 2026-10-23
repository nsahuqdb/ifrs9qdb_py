"""Macro stress testing: scenarios on any variable of the macro store, how
they reach the provision, and how big the risk is.

    config        the stress parameters (config/stress.yml)
    scenarios     shocks, scenarios, the library, saving
    transmission  a scenario's layers into one repricing (``transmit``)
    risk          the split by shock, sensitivities, ECL at risk, reverse

Everything prices through ``stress.reprice`` -- the same ``compute_ecl`` the
run used -- so an empty scenario is the run's own provision, and nothing here
changes how a run is calculated.
"""
from .config import read_stress_config
from .scenarios import Scenario, Shock, library, read_scenarios, scenarios_path, write_scenarios
from .transmission import RunContext, prepare, transmit

__all__ = ["read_stress_config", "Scenario", "Shock", "library", "read_scenarios",
           "scenarios_path", "write_scenarios", "RunContext", "prepare", "transmit"]
