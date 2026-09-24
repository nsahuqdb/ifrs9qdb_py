"""Investigating a reconciliation difference.

compare_outputs says WHICH files and columns differ. That is enough to know
there is a problem and never enough to fix one: the next question is always
"which rows, and what do they hold?".
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from ifrs9qdb.etl.reconcile import (KEYS, compare_outputs, dump_mismatches,
                                    write_reconciliation_markdown)

REF_RUNS = Path(os.environ.get(
    "IFRS9_REF_RUNS",
    "/tmp/claude-0/-home-user/a65f1d3e-1c4e-563f-850d-b7f810e2b39f"
    "/scratchpad/zip/ifrs9_app_v1.0.1/runs"))
two_runs = pytest.mark.skipif(
    not ((REF_RUNS / "run_00001" / "Output").is_dir()
         and (REF_RUNS / "run_00002" / "Output").is_dir()),
    reason="set IFRS9_REF_RUNS to a folder holding two runs")


@pytest.fixture
def pair(tmp_path):
    """Two output folders that differ in every way the dumper reports."""
    a, b = tmp_path / "produced", tmp_path / "reference"
    a.mkdir()
    b.mkdir()
    pd.DataFrame({"ContractId": ["1", "2", "3"],
                  "Rating": ["QDB 5", "QDB 6", "QDB 7"],
                  "Amount": [100.0, 200.0, 300.0]}).to_csv(
        a / "AccountMaster_1.csv", index=False)
    pd.DataFrame({"ContractId": ["1", "2", "4"],
                  "Rating": ["QDB 5", "QDB 9", "QDB 8"],
                  "Amount": [100.0, 200.0, 400.0]}).to_csv(
        b / "AccountMaster_1.csv", index=False)
    return a, b


class TestTheKeyRegistry:
    def test_every_keyed_file_the_r_engine_knows_is_here(self):
        """A file without its key compares positionally, which on a file
        written in a different order is noise."""
        for name in ("AccountMaster_1.csv", "AccountMaster_2.csv",
                     "CustomerMaster_1.csv", "CustomerMaster_2.csv",
                     "CustomerStagingFlag_1.csv", "CustomerStagingFlag_2.csv",
                     "Origination_1.csv", "Origination_2.csv",
                     "Collateral.csv", "AccountCollateralAllocation.csv",
                     "LifeTimeParameterOther.csv", "StPD.csv", "FxRate.csv",
                     "Portfolios.csv", "PortfolioRatingType.csv",
                     "Ratings.csv", "RatingTypes.csv", "CollateralType.csv"):
            assert name in KEYS, name

    def test_the_composite_keys_are_right(self):
        assert KEYS["StPD.csv"] == ["PortfolioCode", "PDBucketDim1",
                                    "MonthLifetime"]
        assert KEYS["LifeTimeParameterOther.csv"] == ["ContractId",
                                                      "MonthLifetime"]


class TestDumping:
    def test_it_separates_the_three_kinds_of_difference(self, pair, tmp_path):
        a, b = pair
        out = tmp_path / "diffs"
        w = dump_mismatches(a, b, out)
        kinds = dict(zip(w["kind"], w["rows"]))
        assert kinds["unmatched_actual"] == 1       # contract 3
        assert kinds["unmatched_reference"] == 1    # contract 4
        assert kinds["value_diffs"] == 1            # contract 2's rating

    def test_the_value_diffs_show_both_sides(self, pair, tmp_path):
        a, b = pair
        out = tmp_path / "diffs"
        dump_mismatches(a, b, out)
        d = pd.read_csv(out / "AccountMaster_1_value_diffs.csv")
        assert list(d["ContractId"]) == [2]
        assert d.loc[0, "Rating (python)"] == "QDB 6"
        assert d.loc[0, "Rating (reference)"] == "QDB 9"

    def test_a_matching_pair_writes_nothing(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        df = pd.DataFrame({"ContractId": ["1"], "Amount": [1.0]})
        df.to_csv(a / "AccountMaster_1.csv", index=False)
        df.to_csv(b / "AccountMaster_1.csv", index=False)
        w = dump_mismatches(a, b, tmp_path / "out")
        assert len(w) == 0

    def test_a_file_with_no_key_is_skipped(self, tmp_path):
        """Lining rows up positionally would report a re-ordered file as
        wholly different."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        pd.DataFrame({"x": [1, 2]}).to_csv(a / "Unknown.csv", index=False)
        pd.DataFrame({"x": [2, 1]}).to_csv(b / "Unknown.csv", index=False)
        assert len(dump_mismatches(a, b, tmp_path / "out")) == 0

    def test_a_numeric_difference_inside_tolerance_is_not_reported(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        pd.DataFrame({"ContractId": ["1"], "Amount": [100.0000001]}).to_csv(
            a / "AccountMaster_1.csv", index=False)
        pd.DataFrame({"ContractId": ["1"], "Amount": [100.0]}).to_csv(
            b / "AccountMaster_1.csv", index=False)
        assert len(dump_mismatches(a, b, tmp_path / "out")) == 0

    def test_ids_are_compared_as_text(self, tmp_path):
        """1123000471 read as a float compares unequal to itself."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        df = pd.DataFrame({"ContractId": ["1123000471", "1133000496"],
                           "Amount": [1.0, 2.0]})
        df.to_csv(a / "AccountMaster_1.csv", index=False)
        df.iloc[::-1].to_csv(b / "AccountMaster_1.csv", index=False)
        assert len(dump_mismatches(a, b, tmp_path / "out")) == 0


class TestTheMarkdown:
    def test_it_writes_a_readable_document(self, pair, tmp_path):
        a, b = pair
        p = write_reconciliation_markdown(a, b, tmp_path / "rec.md")
        text = p.read_text()
        assert "# Reconciliation" in text
        assert "AccountMaster_1.csv" in text

    def test_a_ragged_detail_column_does_not_break_it(self, tmp_path):
        """A file reporting no detail gives pandas a NaN, not an empty list."""
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        pd.DataFrame({"ContractId": ["1"], "Amount": [1.0]}).to_csv(
            a / "AccountMaster_1.csv", index=False)
        pd.DataFrame({"ContractId": ["1", "2"], "Amount": [1.0, 2.0]}).to_csv(
            b / "AccountMaster_1.csv", index=False)
        pd.DataFrame({"Rating": ["A"]}).to_csv(a / "Ratings.csv", index=False)
        pd.DataFrame({"Rating": ["B"]}).to_csv(b / "Ratings.csv", index=False)
        p = write_reconciliation_markdown(a, b, tmp_path / "rec.md")
        assert "row count differs" in p.read_text()


@two_runs
class TestOnTwoRealRuns:
    def test_two_different_quarters_differ_and_say_where(self, tmp_path):
        a = REF_RUNS / "run_00002" / "Output"
        b = REF_RUNS / "run_00001" / "Output"
        w = dump_mismatches(a, b, tmp_path / "diffs")
        assert len(w) > 0
        assert set(w["kind"]) <= {"unmatched_actual", "unmatched_reference",
                                  "value_diffs", "unreadable"}
        for p in w["path"]:
            if p:
                assert Path(p).is_file()

    def test_nothing_is_written_into_the_runs_themselves(self, tmp_path):
        a = REF_RUNS / "run_00002" / "Output"
        b = REF_RUNS / "run_00001" / "Output"
        before = sorted(p.name for p in a.iterdir())
        dump_mismatches(a, b, tmp_path / "diffs")
        assert sorted(p.name for p in a.iterdir()) == before

    def test_a_run_reconciles_against_itself(self, tmp_path):
        out = REF_RUNS / "run_00001" / "Output"
        df = compare_outputs(out, out)
        assert (df["status"] == "match").all(), \
            list(df.loc[df["status"] != "match", "file"])
        assert len(dump_mismatches(out, out, tmp_path / "diffs")) == 0
