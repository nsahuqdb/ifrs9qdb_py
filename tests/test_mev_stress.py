"""Repricing the book on a different macroeconomic path.

The whole PD chain is rebuilt from the run's frozen config, so the first test
is the one that matters: rebuilding with NO edit must reproduce the run's own
provision exactly. If it does not, every number this produces is measuring the
rebuild rather than the macro path.
"""
import pandas as pd
import pytest

from ifrs9qdb.analytics import config_used, normalise
from ifrs9qdb.stress import mev_stress, stpd_to_curves

from conftest import ref_output

OUT = ref_output()
needs_frozen = pytest.mark.skipif(
    OUT is None or config_used(OUT) is None,
    reason="set IFRS9_REF_RUN to a run with config_used/")


@pytest.fixture(scope="module")
def book():
    if OUT is None:
        pytest.skip("set IFRS9_REF_RUN")
    from ifrs9qdb.inputs import load_engine_inputs
    rep = normalise(pd.read_csv(OUT / "FinalEclReport.csv", low_memory=False))
    return load_engine_inputs(OUT), rep


@pytest.fixture(scope="module")
def unchanged(book):
    return mev_stress(*book, OUT)


class TestTheRebuildIsFaithful:
    @needs_frozen
    def test_no_edit_reproduces_the_runs_own_provision(self, unchanged):
        """The control. A rebuild that drifts makes every other number here
        a measure of the rebuild, not of the macro path."""
        assert unchanged["ok"], unchanged.get("reason")
        assert unchanged["delta"] == pytest.approx(0.0, abs=1e-6)
        assert unchanged["before"] == pytest.approx(unchanged["after"])

    @needs_frozen
    def test_every_contract_is_priced(self, unchanged, book):
        _, rep = book
        assert unchanged["priced"] == len(rep)


class TestTheMacroPath:
    @needs_frozen
    def test_a_shock_moves_every_year_of_the_named_mev(self, book):
        base = mev_stress(*book, OUT)["path"]
        shocked = mev_stress(*book, OUT, shock={1: -2.0})["path"]
        assert (shocked["mev_1"] - base["mev_1"]).abs().round(9).eq(2.0).all()
        assert shocked["mev_2"].equals(base["mev_2"]), "only MEV 1 was shocked"

    @needs_frozen
    def test_a_cell_edit_moves_only_that_cell(self, book):
        base = mev_stress(*book, OUT)["path"]
        edit = pd.DataFrame([{"year": 1, "idx": 1, "value": -5.0}])
        got = mev_stress(*book, OUT, mev_new=edit)["path"]
        assert got.loc[got["year"] == 1, "mev_1"].iloc[0] == pytest.approx(-5.0)
        assert got.loc[got["year"] == 2, "mev_1"].iloc[0] == pytest.approx(
            base.loc[base["year"] == 2, "mev_1"].iloc[0])

    @needs_frozen
    def test_holding_the_weights_isolates_the_pd_effect(self, book):
        """With weights on auto they follow the forecast too, so the two modes
        answer different questions and must not agree."""
        auto = mev_stress(*book, OUT, shock={1: -2.0}, weight_mode="auto")
        hold = mev_stress(*book, OUT, shock={1: -2.0}, weight_mode="hold")
        assert auto["ok"] and hold["ok"]
        assert auto["delta"] != pytest.approx(hold["delta"], rel=1e-6)
        assert hold["delta"] > 0, "a worse path must raise the PD-only provision"

    @needs_frozen
    def test_the_externally_rated_book_ignores_a_domestic_shock(self, book):
        """Investments and Banks and FIs price off GCC growth, not the domestic
        MEVs. A domestic shock reaching them means the scales got crossed."""
        r = mev_stress(*book, OUT, shock={1: -2.0})
        ext = r["by_portfolio"]
        ext = ext[ext["portfolio"].isin(["Investments", "Banks and Fis"])]
        assert len(ext) > 0
        assert ext["change"].abs().max() < 1e-6

    @needs_frozen
    def test_the_portfolio_split_adds_up(self, book):
        r = mev_stress(*book, OUT, shock={1: -1.0})
        assert r["by_portfolio"]["change"].sum() == pytest.approx(
            r["delta"], abs=1e-6)
        assert r["by_portfolio"]["contracts"].sum() == r["priced"]


class TestItExplainsItself:
    def test_a_run_with_no_frozen_config_says_so(self, tmp_path, book):
        (tmp_path / "Output").mkdir()
        r = mev_stress(*book, tmp_path / "Output")
        assert not r["ok"]
        assert "config_used" in r["reason"]

    def test_an_unknown_weight_mode_is_refused(self, book):
        with pytest.raises(ValueError, match="weight_mode"):
            mev_stress(*book, OUT, weight_mode="whatever_sounds_right")

    def test_no_engine_inputs_is_a_reason_not_a_crash(self, book):
        _, rep = book
        from ifrs9qdb.inputs import EngineInputs
        from pathlib import Path
        bare = EngineInputs(ok=False, out_dir=Path("."), missing=["StPD.csv"])
        r = mev_stress(bare, rep, OUT)
        assert not r["ok"] and "engine inputs" in r["reason"]


class TestStpdToCurves:
    def test_curves_are_zero_prepended_and_keyed_for_the_engine(self):
        stpd = pd.DataFrame({
            "PortfolioCode": ["P", "P", "P"], "PDBucketDim1": [1, 1, 1],
            "MonthLifetime": [2, 1, 3], "PDLifetime": [0.02, 0.01, 0.03]})
        c = stpd_to_curves(stpd)
        assert list(c) == ["P|1"]
        assert list(c["P|1"]) == [0.0, 0.01, 0.02, 0.03], "must sort by month"

    def test_a_table_without_the_columns_is_empty(self):
        assert stpd_to_curves(pd.DataFrame({"a": [1]})) == {}
        assert stpd_to_curves(None) == {}
