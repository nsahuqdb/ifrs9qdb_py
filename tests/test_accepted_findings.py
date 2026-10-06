"""Findings accepted for one run, and standing suppressions ended, not deleted.

A finding accepted on the pipeline page applies to the run being prepared and
to nothing after it: the next run must ask again. So it is passed to the
pre-run check, the readiness dry run and the run as an argument -- never
written to validation_suppressions.yml -- and the run records it: what was
accepted, by whom, when and why, beside every standing suppression that took
effect, in reports/accepted_findings.csv, validation.md, the manifest and the
audit log.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest
import yaml

from conftest import needs_src_inputs, src_inputs
from ifrs9qdb.validation import (accepted_findings_markdown,
                                 accepted_findings_record, accepted_reasons,
                                 active_suppression_ids, add_suppression,
                                 load_suppressions, normalise_accepted_findings,
                                 remove_suppression, run_suite, validator)

RS = "INPUT_RS_dates_plausible"


@pytest.fixture(autouse=True)
def _audit_to_a_temporary_log(tmp_path, monkeypatch):
    """The audit events these tests cause go to a temporary log."""
    monkeypatch.setenv("IFRS9_AUDIT_LOG", str(tmp_path / "audit" / "etl_audit.jsonl"))


class TestNormalising:
    def test_a_mapping_a_list_and_a_frame_read_the_same(self):
        rows = [{"validator_id": RS, "reason": "two-digit years",
                 "accepted_by": "maker1", "accepted_at": "2026-10-06T09:00:00+0300"}]
        a = normalise_accepted_findings(rows)
        b = normalise_accepted_findings(pd.DataFrame(rows))
        assert a.equals(b)
        c = normalise_accepted_findings({RS: "two-digit years"}, user="maker1")
        assert c.loc[0, "accepted_by"] == "maker1" and c.loc[0, "accepted_at"]

    def test_a_reason_is_required(self):
        with pytest.raises(ValueError, match="reason is required"):
            normalise_accepted_findings([{"validator_id": RS, "reason": "  "}])

    def test_a_repeated_id_keeps_its_first_entry_and_blanks_are_dropped(self):
        a = normalise_accepted_findings([
            {"validator_id": RS, "reason": "first"},
            {"validator_id": RS, "reason": "second"},
            {"validator_id": "", "reason": "no id"}])
        assert list(a["validator_id"]) == [RS] and a.loc[0, "reason"] == "first"

    def test_nothing_accepted_is_an_empty_frame(self):
        assert len(normalise_accepted_findings(None)) == 0
        assert accepted_reasons(None) == {}


class TestTheRunnerSuppressesThemForThatCallOnly:
    @staticmethod
    def _suite():
        @validator("TEST_fails", "ERROR", "always fails")
        def fails():
            return {"passed": False, "detail": "broken"}

        @validator("TEST_passes", "ERROR", "always passes")
        def passes():
            return {"passed": True}

        @validator("TEST_locked", "ERROR", "cannot be suppressed",
                   suppressible=False)
        def locked():
            return {"passed": False, "detail": "missing file"}
        return [fails, passes, locked]

    def test_an_accepted_failure_is_recorded_as_info(self):
        acc = [{"validator_id": i, "reason": "known source issue"}
               for i in ("TEST_fails", "TEST_passes", "TEST_locked")]
        res = run_suite("TEST", self._suite(), {}, accepted_reasons(acc))
        by = {i.id: i for i in res.issues}
        assert by["TEST_fails"].suppressed and not by["TEST_fails"].passed
        assert "Accepted for this run" in by["TEST_fails"].suppression_reason
        # a passing check is not "suppressed"; a locked one cannot be
        assert not by["TEST_passes"].suppressed
        assert not by["TEST_locked"].suppressed
        rec = accepted_findings_record(acc, res.issues)
        assert dict(zip(rec["validator_id"], rec["in_effect"])) == {
            "TEST_fails": True, "TEST_passes": False, "TEST_locked": False}
        assert set(rec["source"]) == {"run"}
        assert rec.set_index("validator_id").loc["TEST_fails", "severity"] == "ERROR"

    def test_the_next_call_without_them_fails_again(self):
        res = run_suite("TEST", self._suite(), {}, {})
        assert not next(i for i in res.issues if i.id == "TEST_fails").suppressed

    def test_the_record_lists_standing_suppressions_that_took_effect(self, tmp_path):
        p = tmp_path / "validation_suppressions.yml"
        add_suppression(p, "TEST_fails", "known, ticket #12", "checker1",
                        valid_until="2099-12-31")
        add_suppression(p, "TEST_passes", "fixed at source since", "checker1")
        standing = load_suppressions(p)
        from ifrs9qdb.validation import suppression_reasons
        res = run_suite("TEST", self._suite(), {}, suppression_reasons(standing))
        rec = accepted_findings_record(None, res.issues, standing)
        # the suppression of a check that passed changed nothing: not listed
        assert list(rec["validator_id"]) == ["TEST_fails"]
        r = rec.iloc[0]
        assert (r["source"], r["reason"], r["accepted_by"], r["valid_until"],
                bool(r["in_effect"])) == ("standing", "known, ticket #12",
                                          "checker1", "2099-12-31", True)
        md = accepted_findings_markdown(rec)
        assert "## Accepted findings (1)" in md
        assert "standing suppression approved by checker1" in md
        assert "known, ticket #12" in md


class TestRemovingAStandingSuppression:
    def test_it_is_ended_not_deleted(self, tmp_path):
        p = tmp_path / "validation_suppressions.yml"
        add_suppression(p, RS, "accepted from the pipeline page", "maker1")
        add_suppression(p, "INPUT_other", "unrelated", "maker1")
        assert RS in active_suppression_ids(load_suppressions(p))
        n = remove_suppression(p, RS, "per-run acceptance replaces it", "checker1")
        assert n == 1
        assert active_suppression_ids(load_suppressions(p)) == ["INPUT_other"]
        entry = yaml.safe_load(p.read_text())["suppressions"][0]
        assert entry["validator_id"] == RS
        assert entry["reason"] == "accepted from the pipeline page"
        assert entry["valid_until"] == (dt.date.today()
                                        - dt.timedelta(days=1)).isoformat()
        assert entry["removed_by"] == "checker1"
        assert entry["removal_reason"] == "per-run acceptance replaces it"

    def test_nothing_in_force_is_nothing_removed(self, tmp_path):
        p = tmp_path / "validation_suppressions.yml"
        assert remove_suppression(p, RS, "x", "y") == 0
        add_suppression(p, RS, "old", "maker1", valid_until="2020-01-01")
        assert remove_suppression(p, RS, "x", "y") == 0
        assert "removed_by" not in yaml.safe_load(p.read_text())["suppressions"][0]

    def test_a_reason_and_a_name_are_required(self, tmp_path):
        p = tmp_path / "validation_suppressions.yml"
        add_suppression(p, RS, "r", "maker1")
        with pytest.raises(ValueError):
            remove_suppression(p, RS, "", "checker1")
        with pytest.raises(ValueError):
            remove_suppression(p, RS, "why", "")


@needs_src_inputs
class TestOnTheRealExtract:
    """The schedule-date finding the extract carries, accepted for one run."""

    def test_the_pre_run_check_downgrades_it_and_leaves_the_file_alone(self, tmp_path):
        from ifrs9qdb.prerun import pre_run_check
        supp = tmp_path / "validation_suppressions.yml"
        before = pre_run_check(src_inputs(), suppressions_path=supp, record=False)
        row = before["results"].set_index("id").loc[RS]
        if bool(row["passed"]):
            pytest.skip("this extract's schedule dates are clean")
        after = pre_run_check(src_inputs(), suppressions_path=supp, record=False,
                              accepted_findings=[{"validator_id": RS,
                                                  "reason": "two-digit years"}])
        a = after["results"].set_index("id").loc[RS]
        assert bool(a["suppressed"]) and a["effective_severity"] == "INFO"
        assert after["summary"]["errors"] == before["summary"]["errors"] - 1
        assert not supp.exists()

    def test_the_run_records_what_was_accepted(self, tmp_path):
        from ifrs9qdb.etl.pipeline import run_etl
        acc = [{"validator_id": RS, "reason": "two-digit years",
                "accepted_by": "maker1", "accepted_at": "2026-10-06T09:00:00+0300"}]
        res = run_etl(src_inputs(), tmp_path, run_id="run_00001",
                      stop_before_pricing=True, accepted_findings=acc)
        reports = tmp_path / "run_00001" / "reports"
        assert res.ok, res.error
        rec = pd.read_csv(reports / "accepted_findings.csv", dtype=str,
                          keep_default_na=False)
        assert list(rec.columns) == ["validator_id", "severity", "source",
                                     "reason", "accepted_by", "accepted_at",
                                     "valid_until", "in_effect"]
        mine = rec[rec["source"] == "run"].iloc[0]
        assert (mine["validator_id"], mine["reason"], mine["accepted_by"]) == \
            (RS, "two-digit years", "maker1")
        v = pd.read_csv(reports / "validation.csv").set_index("id")
        failed = not bool(v.loc[RS, "passed"])
        assert mine["in_effect"] == ("TRUE" if failed else "FALSE")
        if failed:
            assert v.loc[RS, "effective_severity"] == "INFO"
            assert mine["severity"] == "ERROR"
        assert "## Accepted findings" in (reports / "validation.md").read_text()
        m = json.loads((reports / "manifest.json").read_text())
        assert RS in [a["validator_id"] for a in m["accepted_findings"]]

    def test_a_run_with_nothing_accepted_writes_no_record(self, tmp_path):
        from ifrs9qdb.etl.pipeline import run_etl
        cfg = tmp_path / "config"
        import shutil
        from conftest import packaged_config
        shutil.copytree(packaged_config(), cfg)
        (cfg / "validation_suppressions.yml").unlink(missing_ok=True)
        run_etl(src_inputs(), tmp_path / "runs", run_id="run_00001",
                config_dir=cfg, stop_before_pricing=True)
        reports = tmp_path / "runs" / "run_00001" / "reports"
        assert not (reports / "accepted_findings.csv").exists()
        assert "## Accepted findings" not in (reports / "validation.md").read_text()
        m = json.loads((reports / "manifest.json").read_text())
        assert m["accepted_findings"] == []

    def test_the_audit_log_has_each_acceptance_against_its_run(self, tmp_path,
                                                                 monkeypatch):
        from ifrs9qdb.etl.pipeline import run_etl_phase1
        log = tmp_path / "etl_audit.jsonl"
        monkeypatch.setenv("IFRS9_AUDIT_LOG", str(log))
        st = run_etl_phase1(src_inputs(), tmp_path / "runs", run_id="run_00007",
                            accepted_findings={RS: "two-digit years"},
                            user="maker1")
        assert not st.done, st.result.error
        ev = [json.loads(x) for x in log.read_text().splitlines()]
        acc = [e for e in ev if e.get("event") == "finding_accepted"]
        assert [(e["run_id"], e["validator_id"], e["source"], e["reason"],
                 e["accepted_by"]) for e in acc] == \
            [("run_00007", RS, "run", "two-digit years", "maker1")]


class TestReadingARunsRecord:
    def test_the_runs_own_record_is_read_as_written(self, tmp_path):
        from ifrs9qdb.runs import read_run_accepted_findings
        rep = tmp_path / "reports"
        rep.mkdir()
        (rep / "accepted_findings.csv").write_text(
            "validator_id,severity,source,reason,accepted_by,accepted_at,"
            "valid_until,in_effect\n"
            f"{RS},ERROR,run,two-digit years,maker1,2026-10-06T09:00:00+0300,,TRUE\n")
        r = read_run_accepted_findings(tmp_path)
        assert r.loc[0, "reason"] == "two-digit years"
        assert r.loc[0, "recorded"] == "TRUE"

    def test_an_older_run_is_rebuilt_from_what_it_froze(self, tmp_path):
        """Runs made before the record existed: the failed-and-suppressed
        checks, with the reasons from the suppressions file the run froze."""
        from ifrs9qdb.runs import read_run_accepted_findings
        rep = tmp_path / "reports"
        rep.mkdir()
        pd.DataFrame([
            {"stage": "INPUT", "id": RS, "severity": "ERROR",
             "effective_severity": "INFO", "suppressed": "TRUE", "passed": "FALSE"},
            {"stage": "INPUT", "id": "INPUT_ok", "severity": "ERROR",
             "effective_severity": "ERROR", "suppressed": "TRUE", "passed": "TRUE"},
        ]).to_csv(rep / "validation.csv", index=False)
        (rep / "manifest.json").write_text(json.dumps(
            {"run": {"started_at": "2026-10-01T10:00:00+0300"}}))
        frozen = tmp_path / "config_used" / "config"
        frozen.mkdir(parents=True)
        (frozen / "validation_suppressions.yml").write_text(yaml.safe_dump({
            "suppressions": [
                {"validator_id": RS, "reason": "lapsed one", "approved_by": "a",
                 "approved_at": "2026-01-01", "valid_until": "2026-03-31"},
                {"validator_id": RS, "reason": "the one in force",
                 "approved_by": "maker1", "approved_at": "2026-09-30"}]}))
        r = read_run_accepted_findings(tmp_path)
        assert list(r["validator_id"]) == [RS]          # a passing check is not
        row = r.iloc[0]
        assert (row["source"], row["reason"], row["accepted_by"], row["recorded"]) \
            == ("standing", "the one in force", "maker1", "FALSE")

    def test_a_run_with_nothing_accepted_reads_empty(self, tmp_path):
        from ifrs9qdb.runs import read_run_accepted_findings
        assert len(read_run_accepted_findings(tmp_path)) == 0
