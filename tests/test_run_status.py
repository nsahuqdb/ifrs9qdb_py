"""Maker-checker, in the format the R engine writes.

The two engines share a runs/ folder, so a run written by one has to be
readable and approvable by the other. These check the format and the control.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from conftest import ref_run
from ifrs9qdb.run_status import (annotate_runs_with_status, approval_config,
                                 approve_run_status, init_run_status,
                                 list_runs_decided, list_runs_pending_approval,
                                 maker_for_run, normalise_status,
                                 read_run_status, reject_run_status,
                                 transition_run)


@pytest.fixture
def run(tmp_path):
    d = tmp_path / "runs" / "run_00001"
    (d / "reports").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"user": "nsahu"}))
    return d


class TestTheFormat:
    def test_an_official_run_starts_awaiting_a_checker(self, run):
        m = init_run_status(run, "run_00001", "official", by="nsahu")
        assert m["status"] == "pending_checker"
        assert m["run_type"] == "official"
        assert len(m["transitions"]) == 1

    def test_an_unofficial_run_is_terminal(self, run):
        m = init_run_status(run, "run_00001", "unofficial", by="nsahu")
        assert m["status"] == "unofficial"
        assert "no approval required" in m["transitions"][0]["reason"]

    def test_it_is_written_where_the_r_engine_looks(self, run):
        init_run_status(run, "run_00001", "unofficial")
        p = run / "reports" / "run_status.yml"
        assert p.is_file()
        meta = yaml.safe_load(p.read_text())
        assert set(meta) >= {"schema_version", "run_id", "status", "run_type",
                             "ecl_scenario", "transitions",
                             "overrides_applied", "snapshot_label"}

    def test_an_official_run_must_be_the_weighted_ecl(self, run):
        """A single scenario is by definition not the reported figure."""
        with pytest.raises(ValueError, match="weighted"):
            init_run_status(run, "run_00001", "official",
                            ecl_scenario="Significant Downturn")

    def test_the_old_status_name_still_reads(self):
        assert normalise_status("pending_approval") == "pending_checker"
        assert normalise_status(None) == "unknown"

    def test_a_missing_file_is_none_not_a_crash(self, tmp_path):
        assert read_run_status(tmp_path) is None


class TestTheControl:
    def test_approving_records_who_and_why(self, run):
        init_run_status(run, "run_00001", "official", by="nsahu")
        m = approve_run_status(run, "priya", "checked against last quarter")
        assert m["status"] == "approved"
        last = m["transitions"][-1]
        assert last["by"] == "priya" and last["reason"]

    def test_a_reason_is_required(self, run):
        init_run_status(run, "run_00001", "official", by="nsahu")
        with pytest.raises(ValueError, match="reason"):
            approve_run_status(run, "priya", "")
        with pytest.raises(ValueError, match="reason"):
            approve_run_status(run, "priya", "   ")

    def test_an_approver_is_required(self, run):
        init_run_status(run, "run_00001", "official", by="nsahu")
        with pytest.raises(ValueError, match="approver"):
            approve_run_status(run, "", "fine")

    def test_only_a_pending_run_transitions(self, run):
        init_run_status(run, "run_00001", "official", by="nsahu")
        approve_run_status(run, "priya", "fine")
        with pytest.raises(ValueError, match="only a run awaiting a checker"):
            approve_run_status(run, "priya", "again")

    def test_a_rejected_run_does_not_go_back_in_the_queue(self, run):
        """A rejected run is re-run, not re-argued."""
        init_run_status(run, "run_00001", "official", by="nsahu")
        reject_run_status(run, "priya", "stage 2 split looks wrong")
        with pytest.raises(ValueError):
            approve_run_status(run, "priya", "changed my mind")
        assert len(list_runs_pending_approval(run.parent)) == 0

    def test_an_unofficial_run_cannot_be_approved(self, run):
        init_run_status(run, "run_00001", "unofficial")
        with pytest.raises(ValueError, match="only a run awaiting a checker"):
            approve_run_status(run, "priya", "fine")

    def test_a_run_with_no_status_file_cannot_be_approved(self, run):
        with pytest.raises(FileNotFoundError):
            approve_run_status(run, "priya", "fine")

    def test_an_invalid_target_is_refused(self, run):
        init_run_status(run, "run_00001", "official")
        with pytest.raises(ValueError, match="approved.*rejected"):
            transition_run(run, "maybe", "priya", "fine")


class TestSeparationOfDuties:
    @staticmethod
    def _cfg(tmp_path, enforce: bool) -> Path:
        p = tmp_path / "config.yml"
        p.write_text(yaml.safe_dump(
            {"approval": {"enforce_separation_of_duties": enforce}}))
        return p

    def test_the_maker_cannot_approve_their_own_run(self, run, tmp_path):
        init_run_status(run, "run_00001", "official", by="nsahu")
        cfg = self._cfg(tmp_path, True)
        with pytest.raises(PermissionError, match="separation of duties"):
            approve_run_status(run, "nsahu", "fine", config_path=cfg)

    def test_the_check_ignores_case(self, run, tmp_path):
        init_run_status(run, "run_00001", "official", by="nsahu")
        cfg = self._cfg(tmp_path, True)
        with pytest.raises(PermissionError):
            approve_run_status(run, "NSAHU", "fine", config_path=cfg)

    def test_somebody_else_may_approve(self, run, tmp_path):
        init_run_status(run, "run_00001", "official", by="nsahu")
        cfg = self._cfg(tmp_path, True)
        m = approve_run_status(run, "priya", "fine", config_path=cfg)
        assert m["status"] == "approved"

    def test_the_maker_may_always_reject_their_own_run(self, run, tmp_path):
        """Making somebody find another person to withdraw a run achieves
        nothing."""
        init_run_status(run, "run_00001", "official", by="nsahu")
        cfg = self._cfg(tmp_path, True)
        m = reject_run_status(run, "nsahu", "wrong input folder",
                              config_path=cfg)
        assert m["status"] == "rejected"

    def test_it_is_off_unless_the_config_asks_for_it(self, run, tmp_path):
        init_run_status(run, "run_00001", "official", by="nsahu")
        cfg = self._cfg(tmp_path, False)
        assert approve_run_status(run, "nsahu", "fine",
                                  config_path=cfg)["status"] == "approved"

    def test_a_missing_config_does_not_enforce(self, tmp_path):
        assert approval_config(tmp_path / "nope.yml") == \
            {"enforce_separation_of_duties": False}

    def test_the_maker_comes_from_the_manifest(self, run):
        assert maker_for_run(run) == "nsahu"
        assert maker_for_run(run.parent) is None


class TestTheQueue:
    def test_a_run_with_no_status_file_still_surfaces(self, tmp_path):
        """The case most worth seeing: something wrote a run and did not
        record that it needs approving."""
        runs = tmp_path / "runs"
        (runs / "run_00001" / "reports").mkdir(parents=True)
        out = list_runs_pending_approval(runs)
        assert len(out) == 1 and out.iloc[0]["status"] == "unknown"

    def test_pending_and_decided_do_not_overlap(self, tmp_path):
        runs = tmp_path / "runs"
        for i, (rt, decide) in enumerate(
                [("official", None), ("official", "approved"),
                 ("official", "rejected"), ("unofficial", None)], start=1):
            d = runs / f"run_{i:05d}"
            (d / "reports").mkdir(parents=True)
            init_run_status(d, d.name, rt, by="nsahu")
            if decide == "approved":
                approve_run_status(d, "priya", "fine")
            elif decide == "rejected":
                reject_run_status(d, "priya", "no")
        pending = list_runs_pending_approval(runs)
        decided = list_runs_decided(runs)
        assert len(pending) == 1 and len(decided) == 2
        assert not set(pending["run_id"]) & set(decided["run_id"])

    def test_decided_runs_carry_both_sides_of_the_story(self, tmp_path):
        runs = tmp_path / "runs"
        d = runs / "run_00001"
        (d / "reports").mkdir(parents=True)
        init_run_status(d, "run_00001", "official", by="nsahu")
        approve_run_status(d, "priya", "checked against last quarter")
        row = list_runs_decided(runs).iloc[0]
        assert row["requested_by"] == "nsahu"
        assert row["decided_by"] == "priya"
        assert "last quarter" in row["decision_comment"]

    def test_annotating_never_drops_a_run(self, tmp_path):
        runs = tmp_path / "runs"
        for i in (1, 2):
            (runs / f"run_{i:05d}" / "reports").mkdir(parents=True)
        init_run_status(runs / "run_00001", "run_00001", "official")
        df = pd.DataFrame([{"run_id": f"run_{i:05d}",
                            "path": str(runs / f"run_{i:05d}")} for i in (1, 2)])
        out = annotate_runs_with_status(df)
        assert len(out) == 2
        assert set(out["status"]) == {"pending_checker", "unknown"}

    def test_an_empty_listing_has_the_columns(self, tmp_path):
        out = list_runs_pending_approval(tmp_path / "nope")
        assert "run_id" in out.columns and "status" in out.columns


@pytest.mark.skipif(ref_run() is None,
                    reason="set IFRS9_REF_RUN to a run folder")
class TestAgainstTheRealRuns:
    def test_the_r_engines_status_file_reads(self):
        meta = read_run_status(ref_run())
        assert meta is not None, "the R engine's run_status.yml did not parse"
        assert normalise_status(meta["status"]) in (
            "pending_checker", "approved", "rejected", "unofficial")
        assert meta["transitions"] and meta["transitions"][0]["by"]


class TestManifestLocation:
    """The two engines share a runs/ folder, so both locations must read."""

    def test_the_r_engines_reports_manifest_is_found(self, tmp_path):
        from ifrs9qdb.run_status import manifest_path
        d = tmp_path / "run"
        (d / "reports").mkdir(parents=True)
        (d / "reports" / "manifest.json").write_text(
            json.dumps({"run": {"user": "nsahu"}}))
        assert manifest_path(d).parent.name == "reports"
        assert maker_for_run(d) == "nsahu"

    def test_a_root_manifest_still_reads(self, tmp_path):
        from ifrs9qdb.run_status import manifest_path
        d = tmp_path / "run"
        d.mkdir()
        (d / "manifest.json").write_text(json.dumps({"user": "priya"}))
        assert manifest_path(d).parent == d
        assert maker_for_run(d) == "priya"

    def test_no_manifest_is_none(self, tmp_path):
        from ifrs9qdb.run_status import manifest_path
        assert manifest_path(tmp_path) is None
        assert maker_for_run(tmp_path) is None
