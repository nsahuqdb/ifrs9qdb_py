"""Validation tests.

The framework matters as much as the checks: a suite that stops at the first
exception, or that reports only failures, is worse than none.
"""
import pandas as pd
import pytest

from ifrs9qdb.validation import (ALL_VALIDATORS, Severity, Validator, combine,
                                 run_suite, suite_passed)


def ok_check():
    return {"passed": True, "detail": "fine"}


def bad_check():
    return {"passed": False, "count": 3, "detail": "three problems"}


def throwing_check():
    raise RuntimeError("boom")


def V(id, sev, fn, **kw):
    return Validator(id, sev, "test", fn, **kw)


class TestFramework:
    def test_a_throwing_check_fails_rather_than_stopping_the_suite(self):
        """One broken check must not hide every later finding."""
        r = run_suite("s", [V("A", Severity.ERROR, throwing_check),
                            V("B", Severity.ERROR, ok_check)], {})
        assert len(r.issues) == 2
        assert not r.issues[0].passed
        assert "boom" in r.issues[0].detail
        assert r.issues[1].passed

    def test_a_non_standard_return_is_a_failure(self):
        r = run_suite("s", [V("A", Severity.ERROR, lambda: "nonsense")], {})
        assert not r.issues[0].passed

    def test_passes_are_kept(self):
        """A report of failures alone is not evidence anything was checked."""
        r = run_suite("s", [V("A", Severity.INFO, ok_check)], {})
        assert len(r.issues) == 1 and r.issues[0].passed

    def test_only_unsuppressed_errors_fail_a_suite(self):
        r = run_suite("s", [V("A", Severity.WARN, bad_check)], {})
        assert r.passed
        r2 = run_suite("s", [V("B", Severity.ERROR, bad_check)], {})
        assert not r2.passed

    def test_a_suppression_needs_no_code_change(self):
        r = run_suite("s", [V("A", Severity.ERROR, bad_check)], {},
                      {"A": "accepted by Risk, 2026 Q2"})
        assert r.passed
        assert r.issues[0].suppressed
        assert "Risk" in r.issues[0].suppression_reason

    def test_a_non_suppressible_check_cannot_be_waved_through(self):
        r = run_suite("s", [V("A", Severity.ERROR, bad_check,
                              suppressible=False)], {}, {"A": "please"})
        assert not r.passed and not r.issues[0].suppressed

    def test_the_bar_can_be_raised_to_warnings(self):
        r = run_suite("s", [V("A", Severity.WARN, bad_check)], {})
        assert suite_passed(r, Severity.ERROR)
        assert not suite_passed(r, Severity.WARN)

    def test_combining_keeps_every_issue(self):
        a = run_suite("a", [V("A", Severity.INFO, ok_check)], {})
        b = run_suite("b", [V("B", Severity.ERROR, bad_check)], {})
        c = combine(a, b)
        assert len(c.issues) == 2 and not c.passed

    def test_the_summary_counts_by_severity(self):
        r = run_suite("s", [V("A", Severity.ERROR, bad_check),
                            V("B", Severity.WARN, bad_check),
                            V("C", Severity.INFO, ok_check)], {})
        s = r.summary()
        assert s["errors"] == 1 and s["warnings"] == 1 and s["checks"] == 3


class TestTheChecksThemselves:
    def test_every_validator_explains_itself(self):
        """A failure that does not say what to do about it gets ignored by the
        second quarter."""
        for v in ALL_VALIDATORS:
            assert v.rationale, f"{v.id} has no rationale"
            assert v.remediation, f"{v.id} has no remediation"
            assert v.description, f"{v.id} has no description"

    def test_ids_are_unique(self):
        ids = [v.id for v in ALL_VALIDATORS]
        assert len(ids) == len(set(ids))

    def test_every_severity_is_valid(self):
        for v in ALL_VALIDATORS:
            assert v.severity in ("ERROR", "WARN", "INFO")

    def test_the_orphan_collateral_check_is_an_error(self):
        """LIC treats missing collateral as NaN, so one orphan reference zeroes
        a contract's ECL. It cannot be a warning."""
        xf001 = next(v for v in ALL_VALIDATORS if v.id == "XF001")
        assert xf001.severity == "ERROR"
        assert "NaN" in xf001.rationale
