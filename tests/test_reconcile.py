"""Reconciliation and export."""
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from ifrs9qdb.reconcile import KEY_SPEC, build_export, compare_runs


def make_run(root: Path, name: str, rows: list[dict]) -> Path:
    d = root / name / "Output"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "Collateral.csv", index=False)
    return root / name


class TestComparison:
    def test_row_order_is_not_a_difference(self, tmp_path):
        """A file written in a different order is not a difference anyone
        cares about, and comparing positionally reports thousands of false
        ones."""
        rows = [{"CollateralId": "1", "CollateralValue": 100},
                {"CollateralId": "2", "CollateralValue": 200}]
        a = make_run(tmp_path, "a", rows)
        b = make_run(tmp_path, "b", list(reversed(rows)))
        d = compare_runs(a, b)
        assert d.loc[d.file == "Collateral.csv", "status"].iloc[0] == "match"

    def test_added_and_removed_rows_are_reported_separately(self, tmp_path):
        a = make_run(tmp_path, "a", [{"CollateralId": "1", "CollateralValue": 1},
                                     {"CollateralId": "2", "CollateralValue": 2}])
        b = make_run(tmp_path, "b", [{"CollateralId": "1", "CollateralValue": 1},
                                     {"CollateralId": "3", "CollateralValue": 3}])
        d = compare_runs(a, b).iloc[0]
        assert d["rows_added"] == 1 and d["rows_removed"] == 1
        assert "2" in d["examples_added"]

    def test_a_numeric_difference_is_reported_with_its_column(self, tmp_path):
        a = make_run(tmp_path, "a", [{"CollateralId": "1", "CollateralValue": 100}])
        b = make_run(tmp_path, "b", [{"CollateralId": "1", "CollateralValue": 105}])
        d = compare_runs(a, b).iloc[0]
        assert d["status"] == "differs"
        assert d["detail"][0]["column"] == "CollateralValue"

    def test_keys_are_read_as_text(self, tmp_path):
        """Numeric-looking ids become floats otherwise, and then every row
        compares unequal."""
        rows = [{"CollateralId": "1123000471", "CollateralValue": 1}]
        a = make_run(tmp_path, "a", rows)
        b = make_run(tmp_path, "b", rows)
        assert compare_runs(a, b).iloc[0]["status"] == "match"

    def test_a_file_present_in_only_one_run_is_named(self, tmp_path):
        a = make_run(tmp_path, "a", [{"CollateralId": "1", "CollateralValue": 1}])
        b = tmp_path / "b" / "Output"
        b.mkdir(parents=True)
        d = compare_runs(a, tmp_path / "b")
        assert d.iloc[0]["status"] == "only in this run"

    def test_every_output_has_a_declared_key(self):
        for name in ("AccountMaster_1.csv", "LifeTimeParameterOther.csv",
                     "StPD.csv", "AccountCollateralAllocation.csv"):
            assert name in KEY_SPEC


class TestExport:
    def test_the_package_holds_what_is_needed_to_defend_the_figure(self, tmp_path):
        run = tmp_path / "run_00001"
        (run / "Output").mkdir(parents=True)
        (run / "Output" / "FinalEclReport.csv").write_text("Contract Id\nC1\n")
        (run / "config_used").mkdir()
        (run / "config_used" / "model.yml").write_text("a: 1\n")
        (run / "manifest.json").write_text("{}")
        (run / "audit.jsonl").write_text('{"action":"run"}\n')

        res = build_export(run, tmp_path / "out.zip")
        with zipfile.ZipFile(tmp_path / "out.zip") as z:
            names = z.namelist()
        # Everything sits under the run's own folder, so unzipping gives one
        # self-contained tree rather than scattering CSVs into whatever
        # directory it was opened in.
        assert {n.split("/")[0] for n in names} == {"run_00001"}
        flat = {n.split("/", 1)[1] for n in names}
        assert "Output/FinalEclReport.csv" in flat
        assert "config_used/model.yml" in flat
        assert "manifest.json" in flat
        assert "audit.jsonl" in flat
        assert "README.txt" in flat
        assert res["files"] >= 4

    def test_missing_pieces_are_named_not_silently_omitted(self, tmp_path):
        """A missing approval means that step was never run, not that it
        passed."""
        run = tmp_path / "run_00001"
        (run / "Output").mkdir(parents=True)
        (run / "Output" / "x.csv").write_text("a\n1\n")
        res = build_export(run, tmp_path / "out.zip")
        assert "approval.json" in res["skipped"]
        with zipfile.ZipFile(tmp_path / "out.zip") as z:
            readme = z.read("run_00001/README.txt").decode()
        assert "NOT PRESENT" in readme
        assert "never run, not that it passed" in readme

    def test_the_raw_extracts_are_excluded_by_default(self, tmp_path):
        run = tmp_path / "run_00001"
        (run / "Output").mkdir(parents=True)
        (run / "Output" / "x.csv").write_text("a\n1\n")
        (run / "input").mkdir()
        (run / "input" / "AccountMaster.xlsx").write_bytes(b"PK\x03\x04junk")
        with_out = build_export(run, tmp_path / "a.zip")
        with_in = build_export(run, tmp_path / "b.zip", include_inputs=True)
        assert with_in["files"] > with_out["files"]

    def test_the_readme_states_the_stage_3_divergence(self, tmp_path):
        """Anyone reconciling against a LIC extract needs to know before they
        start, not after."""
        run = tmp_path / "run_00001"
        (run / "Output").mkdir(parents=True)
        (run / "Output" / "x.csv").write_text("a\n1\n")
        build_export(run, tmp_path / "out.zip")
        with zipfile.ZipFile(tmp_path / "out.zip") as z:
            readme = z.read("run_00001/README.txt").decode()
        assert "DIVERGES" in readme and "Stage 3" in readme
