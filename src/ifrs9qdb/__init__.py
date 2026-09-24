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
from .ids import as_id  # noqa: F401,E402
from .code_version import code_status, get_current_code_sha  # noqa: F401,E402
from .calculator_versions import (  # noqa: F401,E402
    calculator_version_for_run, compute_code_fingerprint,
    current_calculator_version, list_calculator_versions,
    register_calculator_version, set_active_calculator_version,
)
from .snapshots import (  # noqa: F401,E402
    clone_snapshot, create_snapshot, diff_snapshots, editable_snapshot_files,
    list_snapshots, promote_snapshot, read_snapshot_metadata,
    read_static_csv_with_header, save_snapshot_csv, save_snapshot_yaml,
    snapshot_paths, write_static_csv_with_header,
)
from .run_status import (  # noqa: F401,E402
    annotate_runs_with_status, approve_run_status, init_run_status,
    list_runs_decided, list_runs_pending_approval, read_run_status,
    reject_run_status, transition_run,
)
from .acquisition import (  # noqa: F401,E402
    acquire_inputs_from_zip, list_data_drops, record_input_source,
    validate_input_directory,
)
