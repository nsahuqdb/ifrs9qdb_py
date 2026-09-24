"""Bulk curve edits and the customer roll-up used by every stress screen.

The property that matters is that none of them touch the caller's curves. The
run's own EAD curves are passed straight in, and a mutation there is silent:
the first stress looks right and every one after it compounds.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.stress import (
    advance_curves, customer_rows, reprofile_curves, staging_threshold,
)


def curves():
    return {"a": np.array([100.0, 80, 60, 40, 20, 0]),
            "b": np.array([50.0, 25, 0])}


class TestBulkCurveEdits:
    def test_advancing_drops_elapsed_months_off_the_front(self):
        out = advance_curves(curves(), ["a"], 2)
        assert list(out["a"]) == [60, 40, 20, 0]
        assert list(out["b"]) == [50, 25, 0], "untouched contracts must not move"

    def test_advancing_past_the_end_leaves_the_final_balance(self):
        out = advance_curves(curves(), ["a"], 99)
        assert len(out["a"]) == 1

    def test_reprofiling_changes_the_term_not_the_shape(self):
        out = reprofile_curves(curves(), ["a"], 6)
        assert len(out["a"]) == 12
        assert out["a"][0] == pytest.approx(100.0)
        assert out["a"][-1] == pytest.approx(0.0)
        assert (np.diff(out["a"]) <= 1e-9).all()

    def test_reprofiling_never_produces_an_empty_curve(self):
        out = reprofile_curves(curves(), ["a"], -99)
        assert len(out["a"]) >= 1

    @pytest.mark.parametrize("fn,arg", [(advance_curves, 3),
                                        (reprofile_curves, 3)])
    def test_the_callers_curves_are_never_mutated(self, fn, arg):
        original = curves()
        snapshot = {k: v.copy() for k, v in original.items()}
        fn(original, ["a", "b"], arg)
        for k, v in snapshot.items():
            assert np.array_equal(original[k], v), f"{k} was mutated in place"

    def test_nothing_to_do_leaves_every_curve_as_it_was(self):
        c = curves()
        for got in (advance_curves(c, ["a"], 0), reprofile_curves(c, ["a"], 0),
                    advance_curves(c, [], 5), reprofile_curves(c, [], 5)):
            assert set(got) == set(c)
            for k in c:
                assert np.array_equal(got[k], c[k])


class TestCustomerRows:
    @staticmethod
    def detail():
        return pd.DataFrame({
            "contract": ["c1", "c2", "c3"],
            "customer": ["A", "A", "B"],
            "portfolio": ["Business Finance", "Off BS", "Tasdeer"],
            "rating": ["QDB 5", "QDB 5", "QDB 8"],
            "rating_after": ["QDB 6", "QDB 6", "QDB 8"],
            "stage": [1, 2, 2], "stage_after": [2, 2, 2],
            "exposure": [1000.0, 500.0, 200.0],
            "ecl_before": [10.0, 20.0, 5.0],
            "ecl_after": [30.0, 25.0, 5.0],
        })

    def test_an_attribute_that_varies_collapses_to_multiple(self):
        """A customer in two portfolios is not in the first one."""
        out = customer_rows(self.detail()).set_index("customer")
        assert out.loc["A", "portfolio"] == "multiple"
        assert out.loc["A", "rating"] == "QDB 5"
        assert out.loc["B", "portfolio"] == "Tasdeer"

    def test_stage_is_the_worst_facility(self):
        out = customer_rows(self.detail()).set_index("customer")
        assert out.loc["A", "stage"] == 2

    def test_the_biggest_mover_comes_first(self):
        out = customer_rows(self.detail())
        assert out["customer"].iloc[0] == "A"
        assert (out["change"].abs().diff().dropna() <= 1e-9).all()

    def test_coverage_is_a_percentage_of_exposure(self):
        out = customer_rows(self.detail()).set_index("customer")
        assert out.loc["A", "coverage_after"] == pytest.approx(
            100 * 55 / 1500)

    def test_the_cap_is_a_cap_not_a_filter(self):
        assert len(customer_rows(self.detail(), n=1)) == 1
        assert len(customer_rows(self.detail(), n=None)) == 2

    def test_no_detail_is_empty(self):
        assert len(customer_rows(pd.DataFrame())) == 0
        assert len(customer_rows(None)) == 0


class TestStagingThreshold:
    def test_it_reads_the_policy_the_run_used(self, tmp_path):
        (tmp_path / "staging_thresholds.csv").write_text(
            "key,value\ndpd_stage2_threshold_days,45\n", encoding="utf-8")
        assert staging_threshold(tmp_path) == 45

    def test_a_run_without_the_file_falls_back(self, tmp_path):
        assert staging_threshold(tmp_path) == 60
        assert staging_threshold(tmp_path, default=30) == 30
        assert staging_threshold(None) == 60

    def test_an_unreadable_value_falls_back_rather_than_crashing(self, tmp_path):
        (tmp_path / "staging_thresholds.csv").write_text(
            "key,value\ndpd_stage2_threshold_days,sixty\n", encoding="utf-8")
        assert staging_threshold(tmp_path) == 60
