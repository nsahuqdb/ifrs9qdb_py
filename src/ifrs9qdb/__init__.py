"""IFRS 9 ECL engine and analytics for Qatar Development Bank.

Port of the R package of the same name. The calculation is validated against
the same golden fixtures, so the two implementations cannot drift apart
silently.
"""
__version__ = "0.26.0"

from .engine import (  # noqa: F401
    EclConfig,
    compute_ecl_one,
    compute_lgd,
    fallback_ead_curve,
    months_to_maturity,
    resolve_ead_curve,
    sum_marginal_ecl,
)
from .inputs import EngineInputs, load_engine_inputs  # noqa: F401,E402
from .stress import (  # noqa: F401,E402
    Rule, StressSpec, apply_stress, compare_packages, reprice,
    reverse_stress, reverse_stress_all, roll_forward, tornado,
)
from .overlays import Overlay, apply_overlays, read_overlays  # noqa: F401,E402
from .governance import (  # noqa: F401,E402
    AuditLog, approve_run, approval_status, compare_snapshots, read_snapshot,
    take_snapshot,
)
from .validation import validate_run  # noqa: F401,E402
from .reconcile import build_export, compare_runs, reconcile_report  # noqa: F401,E402
