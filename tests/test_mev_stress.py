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
        # a run priced by this version of the calculation: the rebuild IS its
        # own provision (one priced by an earlier version fails here)
        assert unchanged["note"] is None
        assert unchanged["before"] == pytest.approx(unchanged["provision"], abs=1e-3)

    @needs_frozen
    def test_a_run_from_an_earlier_calculation_is_measured_from_the_rebuild(
            self, tmp_path, book):
        """Its own PD curves are not what this engine builds from its config
        (here: built on other scenario weights than the config gives). The
        stress is measured from the rebuild, so no edit is still no change,
        and the note says how far the rebuild is from its own provision."""
        import shutil
        from ifrs9qdb.etl.macro import build_stpd_from_static
        from ifrs9qdb.etl.pipeline import load_model_config
        from ifrs9qdb.etl.report import build_final_ecl_report
        from ifrs9qdb.etl.static_ref import load_static_reference
        from ifrs9qdb.etl.transform import write_outputs
        from ifrs9qdb.inputs import load_engine_inputs
        run = OUT.parent
        d = tmp_path / "older"
        shutil.copytree(OUT, d / "Output",
                        ignore=shutil.ignore_patterns("FinalEclReport_*", "StPD_*"))
        shutil.copytree(run / "config_used", d / "config_used")
        cfg = d / "config_used" / "config"
        model_cfg, inputs = load_model_config(cfg)
        inputs = dict(inputs)
        inputs["internal_scenario_weights"] = {"mode": "explicit", "explicit_weights": {
            "Significant Downturn": 0.4, "Slight Downturn": 0.1, "Base Case": 0.3,
            "Slight Uptrend": 0.1, "Significant Uptrend": 0.1}}
        static = load_static_reference(d / "config_used" / "static")
        _, rep0 = book
        ext = str(rep0["extract_date"].dropna().iloc[0])
        write_outputs(d / "Output", {"StPD.csv": build_stpd_from_static(
            static, model_cfg, inputs, ext)})
        build_final_ecl_report(d, static=static, model_cfg=model_cfg)
        rep = normalise(pd.read_csv(d / "Output" / "FinalEclReport.csv", low_memory=False))
        older = (load_engine_inputs(d / "Output"), rep)
        r = mev_stress(*older, d / "Output")
        assert r["ok"], r.get("reason")
        assert r["delta"] == pytest.approx(0.0, abs=1e-6)
        assert r["note"] and "another version of the calculation" in r["note"]
        assert r["provision"] == pytest.approx(rep["ecl"].sum(), abs=1e-3)
        assert abs(r["before"] - r["provision"]) > 1_000
        # and the rebuild is what this engine prices the unedited book at
        assert r["before"] == pytest.approx(mev_stress(*book, OUT)["before"], abs=1e-3)

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


class TestScenarioWeights:
    """A different weighting is repriced, not interpolated: the PD chain is
    rebuilt with the weights and the book priced with compute_ecl."""

    @needs_frozen
    def test_the_runs_weights_are_what_its_curves_were_built_with(self):
        from ifrs9qdb.stress import run_scenario_weights
        rw = run_scenario_weights(OUT)
        assert rw["ok"], rw.get("reason")
        assert len(rw["internal"]) == 5
        assert sum(rw["internal"].values()) == pytest.approx(1.0)
        assert sum(rw["external"].values()) == pytest.approx(1.0, abs=1e-6)

    @needs_frozen
    def test_at_the_runs_own_weights_nothing_moves(self, book):
        """The control: the run's own weights reprice to its own provision.
        A weighted sum of the five scenario provisions did not -- the external
        scale is weighted year by year, and the exposure cap is not linear."""
        from ifrs9qdb.stress import run_scenario_weights
        r = mev_stress(*book, OUT, weight_mode="custom",
                       weights=run_scenario_weights(OUT)["internal"])
        assert r["ok"], r.get("reason")
        assert r["delta"] == pytest.approx(0.0, abs=1e-6)
        assert r["after"] == pytest.approx(r["provision"], abs=1e-3)

    @needs_frozen
    def test_more_weight_on_the_downturn_costs_more(self, book):
        from ifrs9qdb.stress import run_scenario_weights
        w = dict(run_scenario_weights(OUT)["internal"])
        w["Significant Downturn"] += 0.2
        r = mev_stress(*book, OUT, weight_mode="custom", weights=w)
        assert r["ok"] and r["delta"] > 0
        # weights are normalised: the rebuilt chain used them summing to one
        assert sum(r["weights_used"].values()) == pytest.approx(1.0)
        ext = r["by_portfolio"].set_index("portfolio")
        for pf in ("Investments", "Banks and Fis"):
            if pf in ext.index:   # the external scale keeps its own weights
                assert ext.loc[pf, "change"] == pytest.approx(0.0, abs=1e-6)

    @needs_frozen
    def test_an_unknown_scenario_is_refused(self, book):
        from ifrs9qdb.stress import run_scenario_weights
        w = dict(run_scenario_weights(OUT)["internal"], **{"Mild Panic": 0.5})
        r = mev_stress(*book, OUT, weight_mode="custom", weights=w)
        assert not r["ok"] and "Mild Panic" in r["reason"]

    @needs_frozen
    def test_the_sensitivity_moves_ten_points_onto_each_scenario(self, book):
        from ifrs9qdb.stress import scenario_weights_sensitivity
        sv = scenario_weights_sensitivity(*book, OUT)
        assert len(sv) == 5
        assert (sv["weight_after"] - sv["weight_before"]).round(9).eq(0.10).all()
        down = sv.set_index("scenario").loc["Significant Downturn", "change"]
        up = sv.set_index("scenario").loc["Significant Uptrend", "change"]
        assert down > 0 > up


class TestTheExternalBook:
    """GCC growth drives the externally rated book (Investments, Banks and
    FIs) only; holding the weights holds the external ones year by year."""

    @needs_frozen
    def test_a_gcc_shock_moves_only_the_external_book(self, book):
        inp, _ = book
        r = mev_stress(*book, OUT, gcc_shock=-2.0, weight_mode="hold")
        assert r["ok"], r.get("reason")
        bp = r["by_portfolio"].set_index("portfolio")["change"]
        ext = set(inp.external_portfolios())
        assert bp[[p for p in bp.index if p not in ext]].abs().max() < 1e-6
        assert bp[[p for p in bp.index if p in ext]].sum() > 0, \
            "slower GCC growth must raise the external provision"

    @needs_frozen
    def test_holding_with_no_edit_moves_nothing(self, book):
        r = mev_stress(*book, OUT, weight_mode="hold")
        assert r["delta"] == pytest.approx(0.0, abs=1e-4)

    @needs_frozen
    def test_external_weights_alone_leave_the_internal_book(self, book):
        from ifrs9qdb.stress import run_scenario_weights
        inp, _ = book
        we = dict(run_scenario_weights(OUT)["external"])
        we["Significant Downturn"] += 0.1
        r = mev_stress(*book, OUT, weight_mode="custom", external_weights=we)
        assert r["ok"], r.get("reason")
        bp = r["by_portfolio"].set_index("portfolio")["change"]
        ext = set(inp.external_portfolios())
        assert bp[[p for p in bp.index if p not in ext]].abs().max() < 1e-6
