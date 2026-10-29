"""Stress scenarios: the builder's shapes, saving, the library, the
transmission into one repricing, and the risk measures. The repricing tests
run on a real run (IFRS9_REF_RUN) with a synthetic macro store; they check
the properties that make the numbers trustworthy -- an empty scenario
changes nothing, the PD channel agrees with the macro screen it shares its
code with, every split adds up, and the Monte Carlo pricing tracks full
repricing."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ifrs9qdb.macro.system import fit_system, system_spec
from ifrs9qdb.stress import mev_stress, shift_curve
from ifrs9qdb.stresstest import Scenario, ScenarioStore, library, read_stress_config, shape_path
from ifrs9qdb.stresstest.library import (REGIONS, check_library, derive, derivation,
                                         episode, library_entry)
from ifrs9qdb.stresstest.transmission import property_trough

from conftest import ref_output
from system_store import build

RUN = ref_output()
has_run = pytest.mark.skipif(RUN is None or not (RUN / "StPD.csv").is_file(),
                             reason="no real run (IFRS9_REF_RUN)")


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    return fit_system(build(tmp_path_factory.mktemp("st") / "store"))


class TestShapes:
    @pytest.mark.parametrize("shape,years,want", [
        ("one_year", 2, [-10, None, None, None, None]),
        ("held", 2, [-10] * 5),
        ("recover", 2, [-10, -5, None, None, None]),
        ("recover", 4, [-10, -7.5, -5, -2.5, None])])
    def test_paths(self, shape, years, want):
        assert shape_path(shape, -10, 5, years) == want

    def test_round_trip(self):
        sc = library()[3]
        back = Scenario.from_dict(sc.to_dict())
        assert back.to_dict() == sc.to_dict()

    def test_blank_years_are_left_to_the_linked_effects(self):
        sc = Scenario(main={"OIL_PRICE": [-40, None], "GDP_WLD": [None, None]})
        assert sc.given(5) == {"OIL_PRICE": [-40.0, None, None, None, None]}

    def test_an_edit_is_noticed(self):
        a = library()[3]
        b = Scenario.from_dict(a.to_dict())
        assert not b.edited_from(a)
        b.main["OIL_PRICE"][0] = -60.0
        assert b.edited_from(a)
        c = Scenario.from_dict(a.to_dict())
        c.linked = ["NONOIL_GDP_QAT"]
        assert c.edited_from(a)


class TestStore:
    def test_save_update_delete_and_history(self, tmp_path):
        st = ScenarioStore(tmp_path / "scenarios.sqlite")
        sid = st.save(Scenario("Mine", main={"US_RATE": [2, 2, None, None, None]},
                               based_on="H7"), user="alice")
        assert [d["name"] for d in st.list()] == ["Mine"]
        assert st.get(sid)["based_on"] == "H7" and st.get(sid)["category"] == "mine"
        sc = st.scenarios()[0]
        sc.main["US_RATE"][0] = 3.0
        st.save(sc, user="bob", sid=sid)
        assert st.get(sid)["main"]["US_RATE"][0] == 3.0
        assert st.get(sid)["updated_by"] == "bob"
        st.delete(sid, user="alice")
        assert st.list() == []
        assert [h["action"] for h in st.history(sid)] == ["created", "updated", "deleted"]

    def test_names_are_unique(self, tmp_path):
        st = ScenarioStore(tmp_path / "s.sqlite")
        st.save(Scenario("Oil"))
        with pytest.raises(ValueError):
            st.save(Scenario("oil "))


class TestLibrary:
    def test_every_variable_is_a_system_variable(self):
        known = set(system_spec()["variables"])
        for sc in library():
            assert set(sc.given()) <= known, sc.id
            assert sc.description and sc.source, sc.id

    def test_ids_are_unique_and_every_history_has_its_episode(self):
        ids = [s.id for s in library()]
        assert len(ids) == len(set(ids))
        for s in library():
            if s.category == "history":
                assert s.episode and s.window, s.id

    def test_a_shock_is_derived_from_the_data(self, system):
        ref = dict(library_entry("H4")["main"]["OIL_PRICE"], var="OIL_PRICE")
        d = system.display("OIL_PRICE").set_index("year")["value"]
        assert derive(ref, system) == [round(d[y], 2) for y in (2015, 2016, 2017)]
        ref = dict(library_entry("H4")["main"]["NONOIL_GDP_QAT"], var="NONOIL_GDP_QAT")
        q = system.series("NONOIL_GDP_QAT")
        assert derive(ref, system) == [round(q[y] - q[2014], 2) for y in (2015, 2016)]

    def test_the_check_covers_every_historical_shock(self, system):
        chk = check_library(system)
        n = sum(1 for e in yaml.safe_load(
            (Path(__file__).parents[1] / "src/ifrs9qdb/stresstest/data/library.yml")
            .read_text(encoding="utf-8"))["scenarios"]
            for r in e["main"].values() if r.get("how"))
        assert len(chk) == n

    def test_every_scenario_carries_its_reasons(self):
        for s in library():
            assert s.region in REGIONS, s.id
            assert s.history and s.narrative and s.book, s.id
            assert s.references and all(r["title"] for r in s.references), s.id
            for v in s.given():
                assert s.why.get(v), (s.id, v)
            if s.restructured:
                assert s.why.get("restructured"), s.id

    def test_the_library_has_gcc_and_qatar_scenarios(self):
        lib = library()
        assert len(lib) == 21
        assert sum(s.region in ("gcc", "qatar") for s in lib) >= 10

    def test_then_holds_the_new_level(self, system):
        ref = dict(library_entry("G4")["main"]["OIL_PRICE"], var="OIL_PRICE")
        got = derive(ref, system)
        assert got is not None and got[-1] == 0.0 and len(got) == 2
        assert next(s for s in library() if s.id == "G4").refs["OIL_PRICE"].endswith("then held")

    def test_each_number_is_explained(self, system):
        # the recorded value less the reference is the shock the data give
        d = derivation("H1", system)
        assert set(d["variable"]) == {"OIL_PRICE", "GDP_WLD"}
        entry = library_entry("H1")["main"]
        for v, g in d.groupby("variable"):
            got = derive(dict(entry[v], var=v), system)
            for r in g.itertuples():
                if r.rule == "change":
                    assert r.reference_year == 1985
                    assert r.recorded - r.reference == pytest.approx(got[r.y - 1], abs=0.006)
                else:
                    assert r.rule == "actual" and r.reference == 0.0
                    assert r.recorded == pytest.approx(got[r.y - 1], abs=0.006)
                assert r.shock == entry[v]["values"][r.y - 1]
        assert derivation("nope", system).empty

    def test_an_episode_has_every_variable(self, system):
        e = episode("H4", system)
        assert e["ok"] and set(e["series"]["variable"]) <= set(system.order)
        assert e["series"]["year"].between(*e["window"]).all()


class TestChannels:
    def test_property_trough(self):
        assert property_trough([-15, -10, 0]) == pytest.approx(-23.5)
        assert property_trough([14.5, -4.1]) == 0.0
        assert property_trough([5, -10, -10]) == pytest.approx(100 * (1.05 * .9 * .9 - 1))

    def test_shift_curve_capped_at_one(self):
        cum = np.concatenate([[0.0], np.cumsum(np.full(60, 0.02 / 12))])
        assert shift_curve(cum, [5, 5, 5, 5, 5]).max() <= 1.0

    def test_a_project_file_overrides_what_it_names(self, tmp_path):
        (tmp_path / "config").mkdir()
        (tmp_path / "config" / "stress.yml").write_text(
            yaml.safe_dump({"staging": {"p12": 0.05}}))
        cfg = read_stress_config(tmp_path)
        assert cfg["staging"]["p12"] == 0.05
        assert cfg["staging"]["rho_stage"] == 0.033            # default kept


# ------------------------------------------------------------- real run ------
@pytest.fixture(scope="module")
def rc(system):
    if RUN is None:
        pytest.skip("no real run")
    from ifrs9qdb import load_engine_inputs
    from ifrs9qdb.analytics import normalise
    from ifrs9qdb.stresstest.transmission import prepare
    inp = load_engine_inputs(RUN)
    rep = normalise(pd.read_csv(RUN / "FinalEclReport.csv", low_memory=False))
    return prepare(inp, rep, RUN.parent, system)


@has_run
class TestTransmission:
    def test_an_empty_scenario_changes_nothing(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, Scenario("Nothing"))
        assert r["ok"] and r["delta"] == 0.0
        assert (r["waterfall"]["change"] == 0).all()

    def test_the_pd_channel_agrees_with_the_macro_screen(self, rc):
        """One PD chain rebuild for both: non-oil 2 points lower in every
        year, nothing linked, weights held, is the macro screen's shift."""
        from ifrs9qdb.stresstest import transmit
        sc = Scenario("Non-oil", main={"NONOIL_GDP_QAT": [-2.0] * 5}, linked=[])
        r = transmit(rc, sc)
        col = rc.drivers_internal["NONOIL_GDP_QAT"][0]
        m = mev_stress(rc.inputs, rc.report, rc.run_path, shock={str(col): -2.0},
                       weight_mode="hold")
        pdc = r["waterfall"].set_index("key").loc["pd", "change"]
        assert pdc == pytest.approx(m["delta"], rel=1e-9, abs=1e-3)

    def test_the_waterfall_and_splits_add_up(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, next(s for s in library() if s.id == "H3"))
        assert r["waterfall"]["change"].sum() == pytest.approx(r["delta"], rel=1e-9)
        assert r["by_segment"]["change"].sum() == pytest.approx(r["delta"], rel=1e-6)
        assert r["by_book"]["change"].sum() == pytest.approx(r["delta"], rel=1e-6)

    def test_a_property_fall_reaches_collateral_only(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, Scenario(main={"PROPERTY_QAT": [-20, None, None, None, None]},
                                  linked=[]))
        w = r["waterfall"].set_index("key")["change"]
        assert w["collateral"] > 0 and w["pd"] == 0 and w["staging"] == 0

    def test_a_deferral_programme_moves_stages(self, rc):
        from ifrs9qdb.stresstest import transmit
        r = transmit(rc, Scenario(restructured=0.5))
        w = r["waterfall"].set_index("key")["change"]
        assert w["staging"] > 0 and w["pd"] == 0
        assert r["staging"]["expected_moves"] == pytest.approx(0.5 * r["staging"]["eligible"])

    def test_adverse_costs_more_than_favourable(self, rc):
        from ifrs9qdb.stresstest import transmit
        bad = transmit(rc, Scenario(main={"GDP_WLD": [-3, None, None, None, None]}))
        good = transmit(rc, Scenario(main={"GDP_WLD": [3, None, None, None, None]}))
        assert bad["delta"] > 0 > good["delta"]


@pytest.fixture(scope="module")
def proxy(rc):
    from ifrs9qdb.stresstest.risk import build_proxy
    return build_proxy(rc)


@has_run
class TestRisk:
    def test_drivers_add_up_in_full(self, rc):
        from ifrs9qdb.stresstest.risk import drivers
        r = drivers(rc, next(s for s in library() if s.id == "H1"))
        assert r["exact"] and r["parts"]["change"].sum() == pytest.approx(r["total"])

    def test_drivers_add_up_from_the_proxy(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import drivers
        r = drivers(rc, next(s for s in library() if s.id == "H6"), P=proxy)
        assert not r["exact"] and r["parts"]["change"].sum() == pytest.approx(r["total"])
        assert "_deferral" in set(r["parts"]["variable"])

    def test_the_proxy_reproduces_the_base_exactly(self, proxy):
        assert abs(proxy.built_with["base_gap"]) < 1e-3

    def test_the_proxy_tracks_full_repricing(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import proxy_check
        chk = proxy_check(rc, proxy, scenarios=library()[:6])
        assert (chk["gap_pct_of_full"].abs() < 5).all(), chk

    def test_ecl_at_risk_is_ordered_and_reproducible(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import ecl_at_risk
        a = ecl_at_risk(rc, proxy, n=300, seed=11)
        b = ecl_at_risk(rc, proxy, n=300, seed=11)
        assert np.array_equal(a["ecl"], b["ecl"])
        v = a["summary"]["value"].tolist()
        assert v == sorted(v)
        assert (a["summary"]["expected_shortfall"] >= a["summary"]["value"]).all()
        assert a["named"]["percentile"].between(0, 100).all()

    def test_reverse_stress_hits_its_target(self, rc, proxy):
        from ifrs9qdb.stresstest.risk import reverse
        t = 0.03 * rc.base_total
        r = reverse(rc, proxy, t, scenarios=library()[:4], n=200, confirm=1)
        top = r["scenarios"].iloc[0]
        assert top["full_change"] == pytest.approx(t, rel=0.02)
        assert 0 <= r["probability"] <= 1


class TestThePdLinkIsTheModels:
    def test_the_driver_and_slope_come_from_model_yml(self):
        import yaml
        from ifrs9qdb.macro.catalogue import concepts
        from ifrs9qdb.stresstest.config import read_stress_config
        from ifrs9qdb.stresstest.transmission import pd_link
        pkg = Path(__import__("ifrs9qdb").__file__).parent
        model = yaml.safe_load((pkg / "config" / "model.yml").read_text())
        cat = concepts()
        linked = dict(zip(cat["concept"], cat["model_variable"]))
        cfg = read_stress_config()
        assert "slope" not in cfg["pd_link"] and "internal" not in cfg["pd_link"]
        pl = pd_link(cfg, model, linked)
        assert pl["internal"] == "NONOIL_GDP_QAT"
        assert pl["slope"] == -0.0414145067387472          # the model's, every digit
        assert pl["external"] == "GDP_GCC" and pl["range"] == [-0.019, -0.048]

    def test_a_recalibrated_model_moves_the_slope_with_it(self):
        import yaml
        from ifrs9qdb.stresstest.transmission import pd_link
        pkg = Path(__import__("ifrs9qdb").__file__).parent
        model = yaml.safe_load((pkg / "config" / "model.yml").read_text())
        model["models"]["internal_v4_production"]["mev_components"][0]["coefficient"] = -0.05
        pl = pd_link({"pd_link": {"slope": -0.0414}}, model,
                     {"NONOIL_GDP_QAT": "VAR_NON_OIL_GDP_GROWTH"})
        assert pl["slope"] == -0.05                        # a project copy cannot go stale
