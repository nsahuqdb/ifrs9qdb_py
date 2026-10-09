"""Config snapshots: the frozen copy of everything a run is told.

"What changed?" is the first question asked of a provision that moved, and the
answer has to be a file rather than a memory. These check that the file is
complete, that it cannot be edited once it is being tested against, and that
the lifecycle refuses the transitions it is supposed to refuse.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

import ifrs9qdb.snapshots as S

R_SNAPSHOTS = Path(os.environ.get(
    "IFRS9_REF_SNAPSHOTS",
    "/tmp/claude-0/-home-user/a65f1d3e-1c4e-563f-850d-b7f810e2b39f"
    "/scratchpad/zip/ifrs9_app_v1.0.1/config_snapshots"))


@pytest.fixture
def project(tmp_path):
    """A minimal live project: config/, data-raw/static/ and config.yml."""
    (tmp_path / "config").mkdir()
    (tmp_path / "data-raw" / "static").mkdir(parents=True)
    (tmp_path / "config" / "model.yml").write_text("ttc_anchor_pd: 0.146\n")
    (tmp_path / "config" / "model_inputs.yml").write_text(
        "internal_scenario_weights:\n  mode: explicit\n")
    (tmp_path / "config" / "validation_suppressions.yml").write_text(
        "suppressions: []\n")
    (tmp_path / "config.yml").write_text(yaml.safe_dump(
        {"paths": {"models": "config/model.yml",
                   "model_inputs": "config/model_inputs.yml",
                   "static_dir": "data-raw/static"}}))
    (tmp_path / "data-raw" / "static" / "ttc_pd_table.csv").write_text(
        "# variable_id: TTC\n# source: QDB\nrating,ttc_pd\nQDB 1,0.0005\n")
    return tmp_path


def _make(project, label="v1", **kw):
    return S.create_snapshot(
        label, kw.pop("description", "first cut"), kw.pop("created_by", "nsahu"),
        config_dir=project / "config",
        static_dir=project / "data-raw" / "static",
        run_config_path=project / "config.yml",
        snapshots_root=project / "config_snapshots", **kw)


class TestCreating:
    def test_it_freezes_config_and_static_together(self, project):
        out = _make(project)
        assert (out / "config" / "model.yml").is_file()
        assert (out / "static" / "ttc_pd_table.csv").is_file()
        assert (out / "snapshot.yml").is_file()

    def test_it_starts_as_a_draft(self, project):
        _make(project)
        meta = S.read_snapshot_metadata("v1", project / "config_snapshots")
        assert meta["status"] == "draft"
        assert meta["base_source"] == "live"
        assert meta["created_by"] == "nsahu"

    def test_the_copied_config_points_at_its_own_frozen_files(self, project):
        """Otherwise the snapshot reads the LIVE model files, which is the one
        thing a frozen copy must not do."""
        out = _make(project)
        cfg = yaml.safe_load((out / "config" / "config.yml").read_text())
        assert cfg["paths"]["models"] == "model.yml"
        assert cfg["paths"]["model_inputs"] == "model_inputs.yml"

    def test_a_duplicate_label_is_refused(self, project):
        _make(project)
        with pytest.raises(FileExistsError):
            _make(project)

    def test_a_bad_label_is_refused(self, project):
        with pytest.raises(ValueError, match="letters, digits"):
            _make(project, label="../escape")
        with pytest.raises(ValueError):
            _make(project, label="")

    def test_a_child_starts_from_its_parent_not_from_live(self, project):
        _make(project, label="v1")
        root = project / "config_snapshots"
        # change the live config AFTER v1 was cut
        (project / "config" / "model.yml").write_text("ttc_anchor_pd: 0.999\n")
        S.create_snapshot("v2", "based on v1", "nsahu", parent="v1",
                          snapshots_root=root)
        text = (root / "v2" / "config" / "model.yml").read_text()
        assert "0.146" in text, "v2 took the live file instead of v1's"
        meta = S.read_snapshot_metadata("v2", root)
        assert meta["parent"] == "v1" and meta["base_source"] == "snapshot:v1"

    def test_an_unknown_parent_is_refused(self, project):
        with pytest.raises(FileNotFoundError):
            S.create_snapshot("v2", "", "nsahu", parent="nope",
                              snapshots_root=project / "config_snapshots")


class TestTheLifecycle:
    @staticmethod
    def _cfg(tmp_path, enforce):
        p = tmp_path / "approval.yml"
        p.write_text(yaml.safe_dump(
            {"approval": {"enforce_separation_of_duties": enforce}}))
        return p

    def test_the_happy_path(self, project):
        _make(project)
        root = project / "config_snapshots"
        for status, who in (("tested", "nsahu"), ("pending_final", "nsahu"),
                            ("approved", "priya"), ("archived", "priya")):
            m = S.promote_snapshot("v1", status, who, "moving on",
                                   snapshots_root=root)
            assert m["status"] == status
        assert len(m["transitions"]) == 4

    def test_an_illegal_transition_is_refused(self, project):
        _make(project)
        root = project / "config_snapshots"
        with pytest.raises(ValueError, match="illegal transition"):
            S.promote_snapshot("v1", "approved", "priya", "skip ahead",
                               snapshots_root=root)

    def test_a_reason_is_always_required(self, project):
        _make(project)
        root = project / "config_snapshots"
        with pytest.raises(ValueError, match="reason"):
            S.promote_snapshot("v1", "tested", "nsahu", "",
                               snapshots_root=root)

    def test_a_user_is_always_required(self, project):
        _make(project)
        with pytest.raises(ValueError, match="user"):
            S.promote_snapshot("v1", "tested", "", "because",
                               snapshots_root=project / "config_snapshots")

    def test_tested_records_who_tested_it(self, project):
        _make(project)
        m = S.promote_snapshot("v1", "tested", "nsahu", "impact run done",
                               snapshots_root=project / "config_snapshots")
        assert m["tested_by"] == "nsahu" and m["tested_at"]

    def test_approval_records_the_approver_and_the_reason(self, project):
        _make(project)
        root = project / "config_snapshots"
        S.promote_snapshot("v1", "tested", "nsahu", "tested", snapshots_root=root)
        S.promote_snapshot("v1", "pending_final", "nsahu", "submitting",
                           snapshots_root=root)
        m = S.promote_snapshot("v1", "approved", "priya", "reviewed the diff",
                               snapshots_root=root)
        assert m["approved_by"] == "priya"
        assert m["approval_reason"] == "reviewed the diff"

    def test_a_rejected_snapshot_goes_back_to_draft(self, project):
        _make(project)
        root = project / "config_snapshots"
        S.promote_snapshot("v1", "tested", "nsahu", "t", snapshots_root=root)
        S.promote_snapshot("v1", "pending_final", "nsahu", "s", snapshots_root=root)
        S.promote_snapshot("v1", "rejected", "priya", "weights look wrong",
                           snapshots_root=root)
        m = S.promote_snapshot("v1", "draft", "nsahu", "fixing",
                               snapshots_root=root)
        assert m["status"] == "draft"

    def test_an_archived_snapshot_is_terminal(self, project):
        _make(project)
        root = project / "config_snapshots"
        for s, w in (("tested", "nsahu"), ("pending_final", "nsahu"),
                     ("approved", "priya"), ("archived", "priya")):
            S.promote_snapshot("v1", s, w, "x", snapshots_root=root)
        with pytest.raises(ValueError, match="illegal transition"):
            S.promote_snapshot("v1", "draft", "nsahu", "reopen",
                               snapshots_root=root)

    def test_the_legacy_pending_status_still_reads(self, project):
        _make(project)
        root = project / "config_snapshots"
        p = root / "v1" / "snapshot.yml"
        meta = yaml.safe_load(p.read_text())
        meta["status"] = "pending"          # as an older snapshot has it
        p.write_text(yaml.safe_dump(meta))
        m = S.promote_snapshot("v1", "approved", "priya", "fine",
                               snapshots_root=root)
        assert m["status"] == "approved"


class TestSeparationOfDuties:
    def _cfg(self, tmp_path, enforce=True):
        p = tmp_path / "approval.yml"
        p.write_text(yaml.safe_dump(
            {"approval": {"enforce_separation_of_duties": enforce}}))
        return p

    def _to_pending(self, project):
        root = project / "config_snapshots"
        _make(project)
        S.promote_snapshot("v1", "tested", "nsahu", "t", snapshots_root=root)
        S.promote_snapshot("v1", "pending_final", "nsahu", "s",
                           snapshots_root=root)
        return root

    def test_the_creator_cannot_give_the_final_approval(self, project):
        root = self._to_pending(project)
        with pytest.raises(PermissionError, match="separation of duties"):
            S.promote_snapshot("v1", "approved", "nsahu", "fine",
                               snapshots_root=root,
                               config_path=self._cfg(project))

    def test_the_creator_may_test_and_submit_their_own(self, project):
        """Those are their own work; only the final step is gated."""
        root = project / "config_snapshots"
        _make(project)
        cfg = self._cfg(project)
        S.promote_snapshot("v1", "tested", "nsahu", "t", snapshots_root=root,
                           config_path=cfg)
        m = S.promote_snapshot("v1", "pending_final", "nsahu", "s",
                               snapshots_root=root, config_path=cfg)
        assert m["status"] == "pending_final"

    def test_somebody_else_may_approve(self, project):
        root = self._to_pending(project)
        m = S.promote_snapshot("v1", "approved", "priya", "reviewed",
                               snapshots_root=root,
                               config_path=self._cfg(project))
        assert m["status"] == "approved"

    def test_it_is_off_unless_the_config_asks(self, project):
        root = self._to_pending(project)
        m = S.promote_snapshot("v1", "approved", "nsahu", "fine",
                               snapshots_root=root,
                               config_path=self._cfg(project, False))
        assert m["status"] == "approved"


class TestEditing:
    def test_a_draft_can_be_edited(self, project):
        _make(project)
        root = project / "config_snapshots"
        r = S.save_snapshot_yaml("v1", "config/model.yml",
                                 "ttc_anchor_pd: 0.15\n", snapshots_root=root)
        assert r["ok"]
        assert "0.15" in (root / "v1" / "config" / "model.yml").read_text()

    def test_a_tested_snapshot_cannot_be_edited(self, project):
        """That is what `tested` is FOR: the numbers being impact-tested must
        not move underneath the test."""
        _make(project)
        root = project / "config_snapshots"
        S.promote_snapshot("v1", "tested", "nsahu", "t", snapshots_root=root)
        r = S.save_snapshot_yaml("v1", "config/model.yml", "x: 1\n",
                                 snapshots_root=root)
        assert not r["ok"] and "only a draft" in r["message"].lower()

    def test_unparseable_yaml_is_refused_before_it_is_written(self, project):
        _make(project)
        root = project / "config_snapshots"
        original = (root / "v1" / "config" / "model.yml").read_text()
        r = S.save_snapshot_yaml("v1", "config/model.yml", "a:\n  - [unclosed\n",
                                 snapshots_root=root)
        assert not r["ok"] and "parse error" in r["message"]
        assert (root / "v1" / "config" / "model.yml").read_text() == original

    def test_a_file_outside_the_whitelist_is_refused(self, project):
        _make(project)
        r = S.save_snapshot_yaml("v1", "snapshot.yml", "status: approved\n",
                                 snapshots_root=project / "config_snapshots")
        assert not r["ok"] and "editable" in r["message"]

    def test_saving_a_csv_keeps_its_comment_header(self, project):
        """The static tables carry their provenance in leading # lines. A
        plain round trip deletes all of it."""
        _make(project)
        root = project / "config_snapshots"
        p = root / "v1" / "static" / "ttc_pd_table.csv"
        before = S.read_static_csv_with_header(p)
        assert len(before["comment_header"]) == 2

        df = before["data"].copy()
        df.loc[0, "ttc_pd"] = 0.0009
        r = S.save_snapshot_csv("v1", "static/ttc_pd_table.csv", df,
                                snapshots_root=root)
        assert r["ok"]
        after = S.read_static_csv_with_header(p)
        assert after["comment_header"] == before["comment_header"]
        assert float(after["data"].loc[0, "ttc_pd"]) == pytest.approx(0.0009)

    def test_the_editable_list_hides_nothing_and_invents_nothing(self, project):
        out = _make(project)
        e = S.editable_snapshot_files(out)
        present = set(e["relpath"])
        assert "config/model.yml" in present
        assert "static/ttc_pd_table.csv" in present
        # registered but absent from this snapshot
        assert "static/fx_rates.csv" not in present


class TestIndustryCodesStayCodes:
    """Industry codes are 4-digit strings with leading zeros. Read as numbers,
    "0113" became 113.0 after a grid edit, and the R engine -- which compares
    the text -- stopped finding every Agriculture / Fisheries / Livestock code
    (Q3 2026, snapshot 26Q3)."""

    def test_a_float_written_file_reads_back_padded(self, tmp_path):
        p = tmp_path / "industry_sector_mapping.csv"
        p.write_text("industry_code,industry_description,sector\n"
                     "2822.0,x,Industry\n113.0,y,Agriculture\n0.0,z,No Sector\n",
                     encoding="utf-8")
        got = S.read_static_csv_with_header(p)["data"]["industry_code"].tolist()
        assert got == ["2822", "0113", "0000"]

    def test_saving_numbers_writes_padded_strings(self, tmp_path):
        p = tmp_path / "industry_sector_mapping.csv"
        df = pd.DataFrame({"industry_code": [113.0, 311.0, 2822.0],
                           "industry_description": ["a", "b", "c"],
                           "sector": ["Agriculture", "Fisheries", "Industry"]})
        S.write_static_csv_with_header(p, df, ["# provenance"])
        lines = p.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "# provenance"
        assert [l.split(",")[0] for l in lines[2:]] == ["0113", "0311", "2822"]


class TestDiff:
    def test_an_unchanged_copy_shows_no_modifications(self, project):
        _make(project, label="v1")
        root = project / "config_snapshots"
        S.create_snapshot("v2", "", "nsahu", parent="v1", snapshots_root=root)
        d = S.diff_snapshots("v1", "v2", root)
        assert d["files_modified"] == []
        assert d["files_added"] == [] and d["files_removed"] == []

    def test_an_edit_shows_up(self, project):
        _make(project, label="v1")
        root = project / "config_snapshots"
        S.create_snapshot("v2", "", "nsahu", parent="v1", snapshots_root=root)
        S.save_snapshot_yaml("v2", "config/model.yml", "ttc_anchor_pd: 0.2\n",
                             snapshots_root=root)
        d = S.diff_snapshots("v1", "v2", root)
        assert [m["file"] for m in d["files_modified"]] == ["config/model.yml"]
        assert d["files_modified"][0]["a_sha"] != d["files_modified"][0]["b_sha"]

    def test_an_added_file_shows_up(self, project):
        _make(project, label="v1")
        root = project / "config_snapshots"
        S.create_snapshot("v2", "", "nsahu", parent="v1", snapshots_root=root)
        (root / "v2" / "static" / "fx_rates.csv").write_text("ccy,rate\nUSD,3.64\n")
        d = S.diff_snapshots("v1", "v2", root)
        assert d["files_added"] == ["static/fx_rates.csv"]


@pytest.mark.skipif(not (R_SNAPSHOTS / "ddd").is_dir(),
                    reason="the R app's config_snapshots are not available")
class TestAgainstTheRSnapshot:
    def test_the_r_engines_snapshot_reads(self):
        meta = S.read_snapshot_metadata("ddd", R_SNAPSHOTS)
        assert meta is not None
        assert meta["status"] in S.SNAPSHOT_STATUSES
        assert meta["created_by"] and meta["transitions"]

    def test_it_appears_in_the_listing(self):
        out = S.list_snapshots(R_SNAPSHOTS)
        assert "ddd" in set(out["label"])

    def test_its_static_headers_survive_a_round_trip(self, tmp_path):
        src = R_SNAPSHOTS / "ddd" / "static" / "non_oil_gdp_history.csv"
        r = S.read_static_csv_with_header(src)
        assert len(r["comment_header"]) >= 5
        dst = tmp_path / "copy.csv"
        S.write_static_csv_with_header(dst, r["data"], r["comment_header"])
        back = S.read_static_csv_with_header(dst)
        assert back["comment_header"] == r["comment_header"]
        pd.testing.assert_frame_equal(back["data"], r["data"])
