"""The risk parameters: PD, LGD, collateral, EAD, and the segment matrix.

Conservation is the theme. A segment matrix that does not sum to the report is
not a view of the report, and a weighted average that quietly drops the rows
with no weight is not an average of the book.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import (
    collateral_analysis, ead_runoff, lgd_floor_stats, lgd_vs_collateral,
    normalise, pd_profile, report_total, segment_matrix,
)

from conftest import ref_file, ref_output

REPORT = ref_file("FinalEclReport.csv")
OUT = ref_output()
real_only = pytest.mark.skipif(
    REPORT is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")
needs_inputs = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


@pytest.fixture(scope="module")
def rep():
    return normalise(pd.read_csv(REPORT, low_memory=False)) \
        if REPORT is not None else None


@pytest.fixture(scope="module")
def inputs():
    if OUT is None:
        return None
    from ifrs9qdb.inputs import load_engine_inputs
    return load_engine_inputs(OUT)


def synthetic():
    return normalise(pd.DataFrame({
        "Contract Id": ["C1", "C2", "C3", "C4"],
        "Customer Id": ["A", "A", "B", "C"],
        "Portfolio Code": ["Business Finance", "Off BS",
                           "Business Finance", "Tasdeer"],
        "Rating": ["QDB 5", "QDB 5", "QDB 8", "QDB 3"],
        "Ifrs Stage": [1, 1, 2, 2],
        "Exposure On Bal": [1000.0, 500.0, 2000.0, 300.0],
        "Cla Amount Onbal": [10.0, 5.0, 200.0, 9.0],
        "Pd Lifetime Value": [0.01, 0.02, 0.30, 0.10],
        "Lgd Rate": [0.45, 0.225, 0.45, 0.225],
        "Coll Cov": [0.0, 1.2, 0.1, 3.0],
    }))


class TestReportTotal:
    def test_an_empty_report_is_unknown_not_zero(self):
        """Zero is a provision. "No report" is not."""
        assert np.isnan(report_total(pd.DataFrame()))
        assert np.isnan(report_total(None))

    @real_only
    def test_it_is_the_sum_of_the_ecl_column(self, rep):
        assert report_total(rep) == pytest.approx(rep["ecl"].sum())


class TestPdProfile:
    @real_only
    @pytest.mark.parametrize("by", ["stage", "portfolio", "rating"])
    def test_every_contract_lands_in_exactly_one_group(self, rep, by):
        p = pd_profile(rep, by)
        assert p["contracts"].sum() == len(rep)
        assert p["exposure"].sum() == pytest.approx(rep["exposure"].sum())
        assert p["ecl"].sum() == pytest.approx(rep["ecl"].sum())

    @real_only
    def test_pd_rises_with_stage(self, rep):
        p = pd_profile(rep, "stage").set_index("group")
        assert p.loc["1", "pd_w"] < p.loc["2", "pd_w"] < p.loc["3", "pd_w"]

    def test_the_weighted_pd_is_weighted_by_exposure(self):
        p = pd_profile(synthetic(), "portfolio").set_index("group")
        bf = p.loc["Business Finance"]
        assert bf["pd_w"] == pytest.approx((0.01 * 1000 + 0.30 * 2000) / 3000)
        assert bf["pd_mean"] == pytest.approx((0.01 + 0.30) / 2)

    def test_an_unknown_grouping_is_refused(self):
        with pytest.raises(ValueError, match="stage"):
            pd_profile(synthetic(), "protfolio")


class TestLgd:
    def test_the_floor_is_counted_where_it_binds(self):
        """0.45 x 0.5 = 0.225. Two of the four sit on it."""
        f = lgd_floor_stats(synthetic(), by="stage").set_index("group")
        assert f.loc["1", "on_floor"] == 1
        assert f.loc["2", "on_floor"] == 1
        assert f.loc["1", "pct_on_floor"] == pytest.approx(50.0)

    @real_only
    def test_groups_cover_the_rated_book(self, rep):
        f = lgd_floor_stats(rep)
        have = pd.to_numeric(rep["lgd"], errors="coerce").notna()
        assert f["contracts"].sum() == int(have.sum())
        assert (f["exposure"].diff().dropna() <= 1e-6).all()

    @real_only
    def test_the_scatter_is_capped_and_reproducible(self, rep):
        """An unseeded sample redraws differently every rerun of the same run."""
        a = lgd_vs_collateral(rep, n=300)
        b = lgd_vs_collateral(rep, n=300)
        assert len(a) == 300
        assert a["collcov"].max() <= 200 + 1e-9
        assert a.equals(b)

    def test_a_report_with_no_lgd_gives_nothing_rather_than_zeros(self):
        d = synthetic()
        d["lgd"] = np.nan
        assert len(lgd_floor_stats(d)) == 0
        assert len(lgd_vs_collateral(d)) == 0


class TestCollateral:
    @needs_inputs
    def test_every_allocation_finds_its_collateral_record(self, inputs):
        """An orphan prices as unsecured without erroring. That is the risk."""
        c = collateral_analysis(inputs)
        assert c["allocations"] > 0
        assert c["orphan_allocations"] == 0, \
            f"{c['orphan_contracts']} contracts point at missing collateral"

    @needs_inputs
    def test_the_type_breakdown_covers_every_record(self, inputs):
        c = collateral_analysis(inputs)
        assert c["by_type"]["records"].sum() == c["collateral_records"]

    def test_no_collateral_tables_is_empty_not_an_error(self):
        class Bare:
            alloc = None
        assert collateral_analysis(Bare()) == {}
        assert collateral_analysis(None) == {}


class TestEadRunoff:
    @needs_inputs
    def test_exposure_runs_down_and_is_indexed_to_today(self, inputs):
        r = ead_runoff(inputs)
        assert r["month"].iloc[0] == 1
        assert r["pct_of_today"].iloc[0] == pytest.approx(100.0)
        assert r["contracts"].iloc[-1] <= r["contracts"].iloc[0]

    @needs_inputs
    def test_the_horizon_is_respected(self, inputs):
        assert len(ead_runoff(inputs, max_month=24)) <= 24

    def test_no_curves_is_empty(self):
        class Bare:
            ead_curves = {}
        assert len(ead_runoff(Bare())) == 0


class TestSegmentMatrix:
    @real_only
    def test_the_cells_conserve_the_report(self, rep):
        m = segment_matrix(rep, value="ecl")
        assert m["value"].sum() == pytest.approx(rep["ecl"].sum())
        assert m["contracts"].sum() == len(rep)

    @real_only
    def test_coverage_cells_are_percentages_not_ratios(self, rep):
        m = segment_matrix(rep, value="coverage")
        s3 = m[m["col"] == "3"]
        assert (s3["value"].dropna() > 1).all()
        assert m["value"].dropna().max() <= 100 + 1e-9

    @real_only
    def test_stage_columns_are_whole_numbers(self, rep):
        assert set(segment_matrix(rep)["col"]) <= {"1", "2", "3", "(unassigned)"}

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError, match="coverage"):
            segment_matrix(synthetic(), value="provision")

    def test_a_missing_column_is_empty_rather_than_a_crash(self):
        assert len(segment_matrix(synthetic(), rows="sector")) == 0
