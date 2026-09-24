"""
Validation.

A run that completes is not the same as a run that is safe to sign. These
checks are what stands between the two, and they are deliberately separate from
the ETL: the transformation reports what it built, the validators say whether it
should be believed.

Three severities, and the distinction matters because it decides whether a run
can proceed:

    ERROR  the numbers are wrong or cannot be computed. A run does not pass.
    WARN   the numbers are computable but something is off, and a person should
           look before signing.
    INFO   worth knowing, not worth stopping for.

Every validator carries its RATIONALE and REMEDIATION, because a failure that
does not say what to do about it gets ignored on the second quarter.

A check can be SUPPRESSED by id, with a reason recorded. That is deliberate:
some findings are known and accepted, and the alternative -- people learning to
ignore a permanently red screen -- is worse than an explicit, auditable
exception.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

__all__ = ["Severity", "Validator", "Issue", "ValidationResult",
           "run_suite", "combine", "suite_passed", "validator"]

SEVERITIES = ("ERROR", "WARN", "INFO")


class Severity:
    ERROR = "ERROR"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class Validator:
    id: str
    severity: str
    description: str
    fn: Callable
    context: str = ""
    rationale: str = ""
    remediation: str = ""
    tags: tuple = ()
    suppressible: bool = True

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"{self.id}: severity must be one of {SEVERITIES}")


@dataclass
class Issue:
    """One validator's verdict on one run."""
    id: str
    severity: str
    description: str
    passed: bool
    context: str = ""
    rationale: str = ""
    remediation: str = ""
    count: int = 0
    detail: str = ""
    suppressed: bool = False
    suppression_reason: str = ""
    examples: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "severity": self.severity,
            "description": self.description, "passed": self.passed,
            "context": self.context, "count": self.count,
            "detail": self.detail, "rationale": self.rationale,
            "remediation": self.remediation, "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
            "examples": self.examples[:10],
        }


@dataclass
class ValidationResult:
    suite: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def failed(self) -> list[Issue]:
        return [i for i in self.issues if not i.passed and not i.suppressed]

    def by_severity(self, severity: str) -> list[Issue]:
        return [i for i in self.failed if i.severity == severity]

    @property
    def passed(self) -> bool:
        """A suite passes when nothing UNSUPPRESSED failed at ERROR."""
        return not self.by_severity(Severity.ERROR)

    def summary(self) -> dict:
        return {
            "suite": self.suite,
            "checks": len(self.issues),
            "passed": self.passed,
            "errors": len(self.by_severity(Severity.ERROR)),
            "warnings": len(self.by_severity(Severity.WARN)),
            "info": len(self.by_severity(Severity.INFO)),
            "suppressed": len([i for i in self.issues if i.suppressed]),
        }

    def to_frame(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame([i.as_dict() for i in self.issues])


def validator(id: str, severity: str, description: str, *, context: str = "",
              rationale: str = "", remediation: str = "", tags: tuple = (),
              suppressible: bool = True):
    """Decorator form, so a check reads as one thing rather than two."""
    def wrap(fn):
        return Validator(id=id, severity=severity, description=description,
                         fn=fn, context=context, rationale=rationale,
                         remediation=remediation, tags=tags,
                         suppressible=suppressible)
    return wrap


def _run_one(v: Validator, args: dict, suppressions: dict) -> Issue:
    """Run one check. A validator that THROWS is a failure, not a crash.

    A suite that stops at the first exception hides every later finding, which
    is the opposite of what a validation run is for.
    """
    try:
        res = v.fn(**{k: val for k, val in args.items()
                      if k in v.fn.__code__.co_varnames})
    except Exception as exc:
        res = {"passed": False,
               "detail": f"the check itself failed: {type(exc).__name__}: {exc}"}
    if not isinstance(res, dict) or "passed" not in res:
        res = {"passed": False, "detail": "validator returned a non-standard result"}

    suppressed = v.suppressible and v.id in suppressions and not res["passed"]
    return Issue(
        id=v.id, severity=v.severity, description=v.description,
        passed=bool(res["passed"]), context=v.context,
        rationale=v.rationale, remediation=v.remediation,
        count=int(res.get("count", 0)), detail=str(res.get("detail", "")),
        examples=list(res.get("examples", [])),
        suppressed=suppressed,
        suppression_reason=suppressions.get(v.id, "") if suppressed else "",
    )


def run_suite(name: str, validators: list[Validator], args: dict,
              suppressions: dict | None = None) -> ValidationResult:
    """Run every check and return all of them, passes included.

    Passes are kept because a report that shows only failures cannot be read as
    evidence that anything was checked.
    """
    supp = suppressions or {}
    return ValidationResult(
        suite=name,
        issues=[_run_one(v, args, supp) for v in validators])


def combine(*results: ValidationResult) -> ValidationResult:
    out = ValidationResult(suite=" + ".join(r.suite for r in results))
    for r in results:
        out.issues.extend(r.issues)
    return out


def suite_passed(result: ValidationResult,
                 max_severity: str = Severity.ERROR) -> bool:
    """Whether a run may proceed.

    ``max_severity`` sets the bar: ERROR lets warnings through, WARN does not.
    """
    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    limit = order[max_severity]
    return not any(order[i.severity] <= limit for i in result.failed)
