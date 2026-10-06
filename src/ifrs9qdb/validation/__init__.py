"""Validation: whether a run is safe to sign, not merely whether it completed.

Two entry points, for two questions.

``validate_stages()`` is the full suite -- 114 checks with the same ids as the
R engine, grouped INPUT / TRANSFORM / DERIVED. It runs inside the ETL, where
the intermediates still exist, and catches the things that are invisible once
the run is written: a rating that failed to resolve, a collateral allocation
pointing at nothing, a scenario weighting that fell back to an equal split.

``validate_run()`` is what can still be checked afterwards, from the run
directory alone.

Ids match the R package exactly, so a suppression approved against one engine
applies to the other and a finding can be compared across the two.
"""
from .checks import (ALL_VALIDATORS, CROSS_FILE_VALIDATORS,  # noqa: F401
                     DERIVED_VALIDATORS, INPUT_VALIDATORS, TRANSFORM_VALIDATORS)
from .checks_cross_file import (CONFIG_COVERAGE_VALIDATORS,  # noqa: F401
                                CROSS_FILE_STAGE_VALIDATORS)
from .checks_derived import DERIVED_STAGE_VALIDATORS  # noqa: F401
from .checks_input import INPUT_STAGE_VALIDATORS  # noqa: F401
from .checks_static import (CONFIG_PATH_VALIDATORS,  # noqa: F401
                            PREFLIGHT_VALIDATORS, STATIC_VALIDATORS)
from .checks_transform import TRANSFORM_STAGE_VALIDATORS  # noqa: F401
from .framework import (Issue, Severity, ValidationResult,  # noqa: F401
                        Validator, combine, run_suite, suite_passed, validator)
from .runner import validate_run  # noqa: F401
from .stages import (DERIVED_STAGE, INPUT_STAGE, STAGE_SUITES,  # noqa: F401
                     TRANSFORM_STAGE, effective_severity, validate_stages,
                     validation_frame, write_validation_reports)
from .suppressions import (ACCEPTED_FIELDS, RECORD_FIELDS,  # noqa: F401
                           accepted_findings_markdown, accepted_findings_record,
                           accepted_reasons, active_suppression_ids,
                           add_suppression, load_suppressions,
                           normalise_accepted_findings, remove_suppression,
                           suppression_reasons)

STAGE_VALIDATORS = INPUT_STAGE + TRANSFORM_STAGE + DERIVED_STAGE


def validate_preflight(static=None, run_config=None, base_dir=None,
                       suppressions=None) -> ValidationResult:
    """Check the configuration BEFORE a run starts.

    Separate from the run suite because it answers an earlier question: is the
    reference data the run is about to use intact? A run started on a broken
    static table completes and prices against nothing.
    """
    return run_suite("PREFLIGHT", PREFLIGHT_VALIDATORS,
                     {"static": static, "run_config": run_config,
                      "base_dir": base_dir}, suppressions or {})

__all__ = [
    "Severity", "Validator", "Issue", "ValidationResult", "validator",
    "run_suite", "combine", "suite_passed",
    "validate_run", "validate_stages", "validation_frame",
    "write_validation_reports", "effective_severity",
    "STAGE_SUITES", "STAGE_VALIDATORS", "INPUT_STAGE", "TRANSFORM_STAGE",
    "DERIVED_STAGE", "PREFLIGHT_VALIDATORS", "STATIC_VALIDATORS",
    "CONFIG_PATH_VALIDATORS", "validate_preflight",
    "load_suppressions", "active_suppression_ids", "add_suppression",
    "remove_suppression", "suppression_reasons", "ACCEPTED_FIELDS",
    "RECORD_FIELDS", "normalise_accepted_findings", "accepted_reasons",
    "accepted_findings_record", "accepted_findings_markdown",
    "ALL_VALIDATORS", "INPUT_VALIDATORS", "TRANSFORM_VALIDATORS",
    "DERIVED_VALIDATORS", "CROSS_FILE_VALIDATORS",
]
