"""Overlay bundles: authoring, approving and applying a management adjustment.

The engine itself is covered elsewhere. These cover the layer around it -- the
bundle as a person writes it, its approval trail, and applying it to a
completed run without touching what that run produced.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest
import yaml

import ifrs9qdb.overlays as O
from conftest import ref_output


@pytest.fixture
def bundle():
    return {"id": "ov1", "owner": "FRM", "status": "draft",
            "effective_date": "2026-09-03",
            "rules": [{"method": "uplift_pct", "level": "portfolio",
                       "target": "Business Finance", "value": 0.1,
                       "comment": "model understates this book"}]}


class TestSelectors:
    def test_a_typed_list_is_split(self):
        """People write "a, b" into a text box; refusing it teaches them to
        write two rules instead of one."""
        assert O.rule_selector("portfolio", "Off BS, Tasdeer") == \
            {"portfolio": ["Off BS", "Tasdeer"]}

    def test_stages_come_back_as_numbers(self):
        assert O.rule_selector("stage", "1, 2") == {"stage": [1, 2]}

    def test_whole_book_needs_no_target(self):
        assert O.rule_selector("whole_book", None) == {"whole_book": True}

    def test_contract_maps_to_the_engines_key(self):
        assert O.rule_selector("contract", "610056") == \
            {"contract_id": ["610056"]}


class TestValidation:
    def test_a_good_bundle_passes(self, bundle):
        assert O.validate_overlay_bundle(bundle) == []

    def test_every_problem_is_reported_at_once(self):
        """Fixing one error per round trip is how people give up and edit the
        YAML by hand."""
        bad = {"id": "x", "rules": [{"method": "nope", "level": "zzz",
                                     "value": "abc"}]}
        errors = O.validate_overlay_bundle(bad)
        assert len(errors) >= 4

    def test_a_comment_is_mandatory(self, bundle):
        bundle["rules"][0].pop("comment")
        errors = O.validate_overlay_bundle(bundle)
        assert any("comment is mandatory" in e for e in errors)

    def test_a_bundle_with_no_rules_is_refused(self, bundle):
        bundle["rules"] = []
        assert any("at least one rule" in e
                   for e in O.validate_overlay_bundle(bundle))

    def test_a_target_is_required_unless_whole_book(self, bundle):
        bundle["rules"][0]["target"] = ""
        assert any("target is required" in e
                   for e in O.validate_overlay_bundle(bundle))
        bundle["rules"][0]["level"] = "whole_book"
        assert O.validate_overlay_bundle(bundle) == []


class TestTheRegistry:
    def test_a_bundle_round_trips(self, tmp_path, bundle):
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        assert O.get_overlay("ov1", p)["owner"] == "FRM"
        assert len(O.read_overlay_bundles(p)) == 1

    def test_an_update_keeps_its_place(self, tmp_path, bundle):
        """A file that reorders itself on every save is unreviewable."""
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        second = dict(bundle, id="ov2")
        O.upsert_overlay(second, p)
        bundle["owner"] = "Risk"
        O.upsert_overlay(bundle, p)
        assert [b["id"] for b in O.read_overlay_bundles(p)] == ["ov1", "ov2"]
        assert O.get_overlay("ov1", p)["owner"] == "Risk"

    def test_an_invalid_bundle_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            O.upsert_overlay({"id": "x", "rules": []}, tmp_path / "o.yml")

    def test_removing_reports_whether_it_was_there(self, tmp_path, bundle):
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        assert O.remove_overlay("ov1", p) is True
        assert O.remove_overlay("ov1", p) is False

    def test_a_missing_file_reads_as_empty(self, tmp_path):
        assert O.read_overlay_bundles(tmp_path / "nope.yml") == []


class TestApprovalTrail:
    def test_the_trail_is_append_only(self, tmp_path, bundle):
        """An overlay rejected then approved should read as exactly that."""
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        O.set_overlay_status("ov1", "pending", "nsahu", "submitting", p)
        O.set_overlay_status("ov1", "rejected", "priya", "needs a basis", p)
        b = O.set_overlay_status("ov1", "pending", "nsahu", "basis attached", p)
        assert b["status"] == "pending"
        assert [t["to"] for t in b["transitions"]] == \
            ["pending", "rejected", "pending"]

    def test_a_reason_and_a_user_are_required(self, tmp_path, bundle):
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        with pytest.raises(ValueError, match="reason"):
            O.set_overlay_status("ov1", "approved", "priya", "", p)
        with pytest.raises(ValueError, match="user"):
            O.set_overlay_status("ov1", "approved", "", "fine", p)

    def test_an_invalid_status_is_refused(self, tmp_path, bundle):
        p = tmp_path / "overlays.yml"
        O.upsert_overlay(bundle, p)
        with pytest.raises(ValueError, match="invalid status"):
            O.set_overlay_status("ov1", "blessed", "priya", "fine", p)

    def test_an_unknown_overlay_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            O.set_overlay_status("nope", "approved", "priya", "fine",
                                 tmp_path / "o.yml")


@pytest.mark.skipif(ref_output() is None,
                    reason="set IFRS9_REF_RUN to a run folder holding Output/")
class TestApplyingToARealRun:
    @pytest.fixture
    def run(self, tmp_path):
        d = tmp_path / "run" / "Output"
        d.mkdir(parents=True)
        shutil.copy(ref_output() / "FinalEclReport.csv", d / "FinalEclReport.csv")
        return tmp_path / "run"

    def test_it_reproduces_the_r_engines_overlay_to_the_cent(self, run, bundle):
        """The R app's own ov1: a 10% uplift on Business Finance."""
        ref = ref_output() / "FinalEclReport_overlay_ov1.csv"
        if not ref.is_file():
            pytest.skip("the reference run has no applied overlay")
        r = O.apply_overlay_to_run(run, bundle)
        assert r["ok"]
        mine = pd.read_csv(r["out_report"], low_memory=False)
        theirs = pd.read_csv(ref, low_memory=False)
        assert mine.shape == theirs.shape
        for col in ("Ecl Model Onbal", "Overlay Amount", "Ecl Final Onbal"):
            a = pd.to_numeric(mine[col], errors="coerce").fillna(0).sum()
            b = pd.to_numeric(theirs[col], errors="coerce").fillna(0).sum()
            assert a == pytest.approx(b, abs=0.01), col

    def test_the_audit_has_the_r_columns(self, run, bundle):
        r = O.apply_overlay_to_run(run, bundle)
        assert list(r["audit"].columns) == O.AUDIT_COLUMNS

    def test_the_model_report_is_never_modified(self, run, bundle):
        before = (run / "Output" / "FinalEclReport.csv").read_bytes()
        O.apply_overlay_to_run(run, bundle)
        assert (run / "Output" / "FinalEclReport.csv").read_bytes() == before

    def test_whole_book_means_everything_except_stage_3(self, run):
        """Stage 3 is booked manually; an overlay on top would double-count."""
        from ifrs9qdb.analytics import normalise
        rep = normalise(pd.read_csv(run / "Output" / "FinalEclReport.csv",
                                    low_memory=False))
        expected = int((rep["stage"] != 3).sum())
        r = O.apply_overlay_to_run(run, {
            "id": "wb", "owner": "FRM",
            "rules": [{"method": "uplift_pct", "level": "whole_book",
                       "target": None, "value": 0.05, "comment": "across"}]})
        assert int(r["audit"].iloc[0]["contracts"]) == expected

    def test_an_applied_overlay_is_found_and_can_be_removed(self, run, bundle):
        O.apply_overlay_to_run(run, bundle)
        listed = O.list_applied_overlays(run)
        assert list(listed["overlay_id"]) == ["ov1"]
        assert Path(listed.iloc[0]["audit_path"]).is_file()

        out = O.remove_applied_overlay(run, "ov1")
        assert out["ok"] and len(out["removed"]) == 2
        assert len(O.list_applied_overlays(run)) == 0
        # and the model report survived
        assert (run / "Output" / "FinalEclReport.csv").is_file()

    def test_overlapping_overlays_are_refused_not_compounded(self, run):
        """Otherwise the ORDER of application decides the provision."""
        r = O.apply_overlay_to_run(run, {
            "id": "clash", "owner": "FRM",
            "rules": [
                {"method": "uplift_pct", "level": "whole_book", "target": None,
                 "value": 0.1, "comment": "one"},
                {"method": "uplift_pct", "level": "portfolio",
                 "target": "Business Finance", "value": 0.2, "comment": "two"},
            ]})
        assert not r["ok"]
        assert any("more than one overlay" in e for e in r["errors"])

    def test_a_preview_writes_nothing(self, run, bundle):
        from ifrs9qdb.analytics import normalise
        rep = normalise(pd.read_csv(run / "Output" / "FinalEclReport.csv",
                                    low_memory=False))
        before = sorted(p.name for p in (run / "Output").iterdir())
        out = O.preview_overlays(rep, bundle)
        assert out["ok"] and out["total"]["overlay"] > 0
        assert sorted(p.name for p in (run / "Output").iterdir()) == before
