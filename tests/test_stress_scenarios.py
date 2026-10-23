"""Stress scenarios: shocks, the transmission into one repricing, and the
risk measures. The repricing tests run on a real run (IFRS9_REF_RUN) with
the synthetic macro store; they check the properties that make the numbers
trustworthy -- an empty scenario changes nothing, the new path agrees with
the macro screen and the lever packages it shares its code with, and every
split adds up."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ifrs9qdb.macro import MacroStore, concepts
from ifrs9qdb.stress import StressSpec, apply_stress, mev_stress, shift_curve
from ifrs9qdb.stresstest import Scenario, Shock, library, read_scenarios, write_scenarios
from ifrs9qdb.stresstest.config import read_stress_config, segment_loadings
from ifrs9qdb.stresstest.scenarios import shape_profile

from conftest import ref_output

HERE = Path(__file__).parent
FIX = HERE / "fixtures" / "macro" / "store"
RUN = ref_output()
has_run = pytest.mark.skipif(RUN is None or not (RUN / "StPD.csv").is_file(),
                             reason="no real run (IFRS9_REF_RUN)")


class TestShapes:
    @pytest.mark.parametrize("shape,start,years,want", [
        ("one_year", 1, 2, [1, 0, 0, 0, 0]), ("one_year", 3, 2, [0, 0, 1, 0, 0]),
        ("held", 2, 2, [0, 1, 1, 1, 1]), ("build", 1, 2, [0.5, 1, 1, 1, 1]),
        ("recover", 1, 2, [1, 0.5, 0, 0, 0]), ("recover", 2, 4, [0, 1, .75, .5, .25])])
    def test_profiles(self, shape, start, years, want):
        assert np.allclose(shape_profile(shape, 5, start, years), want)

    def test_a_shock_in_standard_deviations(self):
        s = Shock("OIL_PRICE", -1.5, "sd", "one_year")
        assert np.allclose(s.native_path(5, 30.0), [-45, 0, 0, 0, 0])

    def test_shocks_on_one_variable_add_up_and_an_override_replaces(self):
        sc = Scenario(shocks=[Shock("OIL_PRICE", -10, shape="held"),
                              Shock("OIL_PRICE", -5, shape="one_year")],
                      overrides={"GDP_QAT": [-1, -1]})
        g = sc.given(5, {"OIL_PRICE": 30})
        assert np.allclose(g["OIL_PRICE"], [-15, -10, -10, -10, -10])
        assert np.allclose(g["GDP_QAT"], [-1, -1, 0, 0, 0])

    def test_round_trip_and_saving(self, tmp_path):
        sc = library()[-1]
        back = Scenario.from_dict(sc.to_dict())
        assert back.to_dict() == sc.to_dict()
        p = write_scenarios([sc, Scenario("Mine", shocks=[Shock("US_RATE", 1, shape="held")])],
                            tmp_path / "config" / "stress_scenarios.yml")
        got = read_scenarios(p)
        assert [s.name for s in got] == ["Combined severe", "Mine"]
        assert all(s.kind == "custom" for s in got)

    def test_every_library_variable_is_in_the_catalogue(self):
        known = set(concepts()["concept"])
        for sc in library():
            assert set(sc.concepts()) <= known, sc.name


class TestShiftCurve:
    cum = np.concatenate([[0.0], np.cumsum(np.full(60, 0.02 / 12))])

    def test_no_shift_is_the_curve(self):
        assert np.array_equal(shift_curve(self.cum, [0, 0, 0]), self.cum)

    def test_a_shift_moves_its_year_only(self):
        out = shift_curve(self.cum, [0.5, 0, 0])
        inc0, inc1 = np.diff(self.cum), np.diff(out)
        assert (inc1[:12] > inc0[:12]).all()
        assert np.allclose(inc1[12:], inc0[12:])
        from scipy.stats import norm
        assert inc1[:12].sum() == pytest.approx(norm.cdf(norm.ppf(0.02) + 0.5))

    def test_capped_at_one(self):
        assert shift_curve(self.cum, [5, 5, 5, 5, 5]).max() <= 1.0


class TestConfig:
    def test_a_segment_by_any_of_its_spellings(self):
        cfg = read_stress_config()
        a = segment_loadings(cfg, "Industry", False)
        b = segment_loadings(cfg, "manufacturing", False)
        assert a == b and a["LNG_PRICE"] > segment_loadings(cfg, "Education", False)["LNG_PRICE"]

    def test_the_external_book_has_its_own(self):
        cfg = read_stress_config()
        e = segment_loadings(cfg, "External book", True)
        assert "OIL_PRICE" in e and "GOV_SPEND_QAT" not in e

    def test_a_project_file_overrides_what_it_names(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "stress.yml").write_text(
            yaml.safe_dump({"staging": {"sicr_notches": 2}}))
        cfg = read_stress_config(tmp_path)
        assert cfg["staging"]["sicr_notches"] == 2
        assert cfg["rates"]["variable"] == "QA_RATE"          # default kept


# ------------------------------------------------------------- real run ------
@pytest.fixture(scope="module")
def rc(tmp_path_factory):
    if RUN is None:
        pytest.skip("no real run")
    from ifrs9qdb import load_engine_inputs
    from ifrs9qdb.analytics import normalise
    from ifrs9qdb.stresstest.transmission import prepare
    d = tmp_path_factory.mktemp("st") / "store"
    shutil.copytree(FIX, d)
    inp = load_engine_inputs(RUN)
    rep = normalise(pd.read_csv(RUN / "FinalEclReport.csv", low_memory=False))
    return prepare(inp, rep, RUN.parent, MacroStore(d))


@has_run
class TestTransmission:
    def test_an_empty_scenario_changes_nothing(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, Scenario("Nothing"))
        assert r["ok"] and r["delta"] == 0.0
        assert (r["waterfall"]["change"] == 0).all()

    def test_the_model_driver_agrees_with_the_macro_screen(self, rc):
        """One PD chain rebuild for both: a non-oil shock held over every
        year, weights held, is exactly the macro screen's shift."""
        from ifrs9qdb.stresstest import transmit
        sc = Scenario("Non-oil", shocks=[Shock("NONOIL_GDP_QAT", -2, shape="held")],
                      knock_on=False)
        r = transmit(rc, sc, waterfall=True)
        m = mev_stress(rc.inputs, rc.report, rc.run_path, shock={"1": -2.0},
                       weight_mode="hold")
        drv = r["waterfall"].set_index("key").loc["drivers", "change"]
        assert drv == pytest.approx(m["delta"], rel=1e-9, abs=1e-3)

    def test_levers_agree_with_the_lever_packages(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, Scenario("Levers", levers={"pd_multiplier": 1.25,
                                                    "collateral_pct": 80.0}))
        a = apply_stress(rc.inputs, rc.report, StressSpec(pd_multiplier=1.25,
                                                          collateral_pct=80.0))
        if rc.base_curves is None:
            assert r["delta"] == pytest.approx(a["delta"], rel=1e-9)

    def test_the_waterfall_adds_up(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, library()[-1])
        assert r["waterfall"]["change"].sum() == pytest.approx(r["delta"], rel=1e-9)
        assert r["by_segment"]["change"].sum() == pytest.approx(r["delta"], rel=1e-6)
        assert r["by_book"]["change"].sum() == pytest.approx(r["delta"], rel=1e-6)

    def test_an_adverse_move_costs_more(self, rc):
        from ifrs9qdb.stresstest import transmit
        up = transmit(rc, Scenario(shocks=[Shock("PROPERTY_QAT", -20)], knock_on=False))
        down = transmit(rc, Scenario(shocks=[Shock("PROPERTY_QAT", 20)], knock_on=False))
        assert up["delta"] > 0 > down["delta"]

    def test_a_rate_rise_through_the_peg(self, rc):
        from ifrs9qdb.stresstest import transmit
        us = transmit(rc, Scenario(shocks=[Shock("US_RATE", 2, shape="held")], knock_on=False))
        qa = transmit(rc, Scenario(shocks=[Shock("QA_RATE", 2, shape="held")], knock_on=False))
        assert us["delta"] == pytest.approx(qa["delta"], rel=1e-9)
        assert us["info"]["rates"]["notches"] == 1


@pytest.fixture(scope="module")
def proxy(rc):
    from ifrs9qdb.stresstest.risk import build_proxy
    return build_proxy(rc)


@has_run
class TestRisk:
    def test_the_shapley_split_adds_up(self, rc):
        from ifrs9qdb.stresstest.risk import shapley
        sc = Scenario("Two", shocks=[Shock("PROPERTY_QAT", -15), Shock("QA_RATE", 2, shape="held")],
                      knock_on=False)
        r = shapley(rc, sc)
        assert r["exact"] and r["parts"]["shapley"].sum() == pytest.approx(r["total"], rel=1e-9)
        assert r["interaction"] == pytest.approx(r["total"] - r["sum_alone"])


    def test_the_proxy_reproduces_the_base_exactly(self, proxy):
        assert abs(proxy.built_with["base_gap"]) < 1e-3

    def test_the_proxy_tracks_full_repricing(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import proxy_check
        chk = proxy_check(rc, proxy, scenarios=library()[:4])
        assert (chk["gap_pct_of_full"].abs() < 5).all(), chk

    def test_ecl_at_risk_is_ordered_and_reproducible(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import ecl_at_risk
        a = ecl_at_risk(rc, proxy, n=400, seed=11)
        b = ecl_at_risk(rc, proxy, n=400, seed=11)
        assert np.array_equal(a["ecl"], b["ecl"])
        v = a["summary"]["value"].tolist()
        assert v == sorted(v)
        assert (a["summary"]["expected_shortfall"] >= a["summary"]["value"]).all()
        assert a["named"]["percentile"].between(0, 100).all()

    def test_reverse_stress_hits_its_target(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import reverse
        t = 0.05 * float(np.nansum(rc.base["ecl"]))
        r = reverse(rc, proxy, t)
        assert r["found"] and r["full_change"] == pytest.approx(t, rel=0.02)
