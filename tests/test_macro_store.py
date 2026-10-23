"""The macro store: sources, vintages, concepts, the quarter fill and the
macro model. Everything here runs on the synthetic store in
``fixtures/macro/store`` (no real data) and on fake network answers."""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ifrs9qdb.macro import MacroStore, match_candidates, refresh
from ifrs9qdb.macro.prng import mulmod, normals, uniforms
from ifrs9qdb.macro.quarter import (first_forecast_year, propose, render_model_inputs,
                                    replace_top_level_block)
from ifrs9qdb.macro.sources import (DataMapper, ImfSdmx, norm_period, parse_datamapper,
                                    parse_sdmx_csv, read_file, static_seed)
from ifrs9qdb.macro.store import annualise, to_concept_unit
from ifrs9qdb.macro.var import em_fill, fit_macro_model

HERE = Path(__file__).parent
FIX = HERE / "fixtures" / "macro" / "store"
PKG = Path(__import__("ifrs9qdb").__file__).parent


@pytest.fixture()
def store(tmp_path):
    d = tmp_path / "store"
    shutil.copytree(FIX, d)
    return MacroStore(d)


class TestPeriods:
    @pytest.mark.parametrize("raw,freq,want", [
        ("2026", None, "2026"), ("2026-Q3", None, "2026-Q3"), ("2026Q3", None, "2026-Q3"),
        ("2026-M09", None, "2026-09"), ("2026-09", None, "2026-09"),
        ("2026-07-01", "Q", "2026-Q3"), ("2026-07-01", "A", "2026"),
        ("2026-07-01", "M", "2026-07"), ("FY2025/26", None, "2025"), (2026.0, None, "2026"),
        ("junk", None, None)])
    def test_every_source_spelling_reads_one_way(self, raw, freq, want):
        assert norm_period(raw, freq) == want


class TestSdmx:
    CSV = ("STRUCTURE,STRUCTURE_ID,ACTION,COUNTRY: Country,INDICATOR: Indicator,"
           "FREQUENCY: Frequency,TIME_PERIOD: Time period,OBS_VALUE: Observation value,"
           "LATEST_ACTUAL_ANNUAL_DATA: Latest actual\n"
           'dataflow,IMF.RES:WEO(9.0.0),I,QAT: Qatar,"NGDP_RPCH: GDP, constant prices",'
           "A: Annual,2025,2.4,2025\n"
           'dataflow,IMF.RES:WEO(9.0.0),I,QAT: Qatar,"NGDP_RPCH: GDP, constant prices",'
           "A: Annual,2026,6.1,2025\n")

    def test_labels_are_read_and_the_last_actual_year_kept(self):
        s, o = parse_sdmx_csv(self.CSV, "WEO")
        assert list(s["series_id"]) == ["IMF:WEO:QAT:NGDP_RPCH:A"]
        assert s["name"].iloc[0] == "GDP, constant prices"
        assert s["last_actual"].iloc[0] == "2025"
        assert list(o["period"]) == ["2025", "2026"] and o["value"].tolist() == [2.4, 6.1]

    def test_the_key_follows_the_dataflows_own_dimension_order(self):
        seen = []

        def fake(url, headers):
            seen.append(url)
            if "/structure/" in url:
                dims = ["FREQUENCY", "COUNTRY", "INDICATOR"]
                return 200, json.dumps({"data": {
                    "dataflows": [{"id": "WEO", "agencyID": "IMF.RES", "version": "9.0.0"}],
                    "dataStructures": [{"dataStructureComponents": {"dimensionList": {
                        "dimensions": [{"id": d, "position": i} for i, d in enumerate(dims)]}}}]
                }}).encode()
            return 200, self.CSV.encode()

        s, o, info = ImfSdmx(transport=fake).fetch(
            {"dataset": "WEO", "candidates": "WEO", "frequency": "A",
             "country_dim": "COUNTRY", "start": "2000"}, ["QAT", "SAU"])
        assert "/data/dataflow/IMF.RES/WEO/~/A.QAT+SAU.*" in seen[-1]
        assert "c%5BTIME_PERIOD%5D=ge%3A2000" in seen[-1] or "c[TIME_PERIOD]=ge:2000" in seen[-1]
        assert len(s) == 1 and len(o) == 2

    def test_a_renamed_dataflow_is_found_by_its_next_candidate(self):
        def fake(url, headers):
            if "/structure/dataflow/*/FSIC/" in url:
                return 404, b"no such dataflow"
            if "/structure/" in url:
                return 200, json.dumps({"data": {"dataflows": [
                    {"id": "FSI", "agencyID": "IMF.STA", "version": "1.0.0"}],
                    "dataStructures": []}}).encode()
            return 200, b"TIME_PERIOD,OBS_VALUE\n"
        meta = ImfSdmx(transport=fake).resolve("FSIC|FSI")
        assert meta["id"] == "FSI" and meta["agency"] == "IMF.STA"


class TestOtherSources:
    def test_datamapper(self):
        j = {"values": {"NGDP_RPCH": {"QAT": {"2025": 2.0, "2026": None, "2027": 5.5}}}}
        s, o = parse_datamapper(j, {"NGDP_RPCH": {"label": "Real GDP growth", "dataset": "WEO"}})
        assert s["series_id"].tolist() == ["DM:WEO:QAT:NGDP_RPCH:A"]
        assert o["period"].tolist() == ["2025", "2027"]

    def test_a_long_and_a_wide_file(self, tmp_path):
        long = tmp_path / "long.csv"
        long.write_text("indicator,period,value\nREPI,2024-Q1,101\nREPI,2024-Q2,103\n")
        s, o = read_file(long, source="QCB", dataset="REPI", country="QAT")
        assert s["series_id"].tolist() == ["QCB:REPI:QAT:REPI:Q"] and len(o) == 2
        wide = tmp_path / "wide.csv"
        wide.write_text("period,ARRIVALS: Visitor arrivals,HOTEL\n2024,4000,70\n2025,4500,72\n")
        s, o = read_file(wide, source="NPC", dataset="TOURISM", country="QAT")
        assert set(s["series_id"]) == {"NPC:TOURISM:QAT:ARRIVALS:A", "NPC:TOURISM:QAT:HOTEL:A"}

    def test_the_package_history_seeds_the_store(self):
        s, o = static_seed(PKG / "static")
        assert "STATIC:NONOIL:QAT:NGDP_NOIL_RPCH:A" in set(s["series_id"])
        assert "STATIC:GCC_GDP:SAU:NGDP_RPCH:A" in set(s["series_id"])

    def test_a_blocked_network_falls_back_and_keeps_the_last_vintage(self, store):
        before = len(store.series())
        r = refresh(store, transport=lambda u, h: (0, b"proxy refused"),
                    sources=("IMF", "DM", "STATIC"), datasets=["WEO", "PCPS"])
        log = r["log"]
        assert (log.loc[log["dataset"] != "Package history files", "status"] == "failed").all()
        # the failed datasets keep the fixture's copies, the history is added
        assert r["series"] >= before
        assert len(store.vintages()) == 2


class TestStore:
    def test_a_year_in_progress_is_not_a_year(self):
        o = pd.DataFrame({"period": [f"2025-{m:02d}" for m in range(1, 13)]
                          + ["2026-01", "2026-02"], "value": [1.0] * 12 + [5.0, 5.0]})
        a = annualise(o, "M")
        assert a.index.tolist() == [2025]

    def test_pct_change_uses_consecutive_years_only(self):
        a = pd.Series([100.0, 110.0, 121.0, 150.0], index=[2020, 2021, 2022, 2024])
        assert to_concept_unit(a, "pct_change").round(6).to_dict() == {2021: 10.0, 2022: 10.0}

    def test_concepts_resolve_to_the_fixture(self, store):
        t = store.resolve_concepts()
        got = dict(zip(t["concept"], t["series_id"]))
        assert got["GDP_QAT"] == "IMF:WEO:QAT:NGDP_RPCH:A"
        assert got["NONOIL_GDP_QAT"] == "IMF:REO_MCD:QAT:NGDP_NOIL_RPCH:A"
        assert got["OIL_PRICE"] == "IMF:PCPS:W00:POILBRE.USD:M"
        assert got["NPL_QAT"] == "IMF:FSI:QAT:FSANL_PT:Q"      # by its name
        assert got["US_RATE"] == "FRED:FEDFUNDS::FEDFUNDS:M"

    def test_the_weo_marks_what_is_forecast(self, store):
        a = store.concept_annual("GDP_QAT")
        assert a.loc[a["year"] <= 2025, "actual"].all()
        assert not a.loc[a["year"] > 2025, "actual"].any()

    def test_a_refresh_replaces_what_it_fetched_and_carries_the_rest(self, store):
        s = pd.DataFrame([{"series_id": "IMF:WEO:QAT:NGDP_RPCH:A", "source": "IMF",
                           "dataset": "WEO", "country": "QAT", "indicator": "NGDP_RPCH",
                           "freq": "A", "name": "x", "unit": "", "last_actual": "2025",
                           "release": "new"}])
        o = pd.DataFrame({"series_id": ["IMF:WEO:QAT:NGDP_RPCH:A"], "period": ["2026"],
                          "value": [9.9]})
        v = store.save(s, o, label="partial")
        ser = store.series(v)
        assert "IMF:WEO:SAU:NGDP_RPCH:A" not in set(ser["series_id"])   # WEO re-fetched
        assert "IMF:PCPS:W00:POILBRE.USD:M" in set(ser["series_id"])     # carried
        assert store.meta(v)["parent"] == "v-fixture"

    def test_search(self, store):
        hit = store.search("brent")
        assert hit["series_id"].tolist() == ["IMF:PCPS:W00:POILBRE.USD:M"]
        assert hit["first"].iloc[0] == "1990-01"

    def test_candidates(self):
        s = pd.DataFrame({"series_id": ["A:B:QAT:X1:A", "A:B:QAT:X2:A"],
                          "name": ["Loans to private sector", "Deposits"]})
        assert match_candidates("A:B:QAT:X9*|name~[Pp]rivate@A:B:*", s) == "A:B:QAT:X1:A"
        assert match_candidates("Z:*", s) is None


class TestQuarter:
    def test_the_five_year_rule(self):
        r = first_forecast_year("2026-09-30", 2025)
        assert r["first"] == 2026 and "2030" in r["rule"]
        r = first_forecast_year("2026-09-30", 2024)
        assert r["first"] == 2025 and "before the reporting year" in r["rule"]
        # never after the reporting year
        assert first_forecast_year("2026-09-30", 2026)["first"] == 2026

    def test_a_proposal_from_the_store(self, store):
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30")
        assert p["ok"] and p["years"] == [2026, 2027, 2028, 2029, 2030]
        a = store.concept_annual("NONOIL_GDP_QAT")
        want = a.set_index("year").loc[2026:2030, "value"].round(9).tolist()
        assert [round(v, 9) for v in p["nonoil"]["values"]] == want
        assert set(p["gcc_growth"]) == {"Bahrain", "Kuwait", "Oman", "Qatar",
                                        "Saudi Arabia", "United Arab Emirates"}
        assert p["checks"]["ok"].all()
        assert len(p["changes"]) == 5 * 13

    def test_the_new_config_keeps_every_comment(self, store):
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30")
        text = render_model_inputs(p)
        old = (PKG / "config" / "model_inputs.yml").read_text()
        comments = [ln for ln in old.splitlines() if ln.strip().startswith("#")]
        assert all(c in text for c in comments)
        y = yaml.safe_load(text)
        assert y["macro_source"]["first_forecast_year"] == 2026
        assert y["internal_scenario_weights"] == yaml.safe_load(old)["internal_scenario_weights"]

    def test_replace_a_block(self):
        t = "# head\na: 1\nb:\n  - 1\n  - 2\n\n# c comment\nc: 3\n"
        out = replace_top_level_block(t, "b", "b:\n- 9\n")
        assert out == "# head\na: 1\nb:\n- 9\n\n# c comment\nc: 3\n"
        assert replace_top_level_block("a: 1\n", "z", "z: 2\n") == "a: 1\nz: 2\n"


class TestPrng:
    def test_mulmod_is_exact(self):
        a, x = 48271.0, 2147483646.0
        assert mulmod(a, x) == float((48271 * 2147483646) % 2147483647)

    def test_blocks_match_the_sequential_stream(self):
        u = uniforms(10000, 7)
        x, seq = 7, []
        for _ in range(10000):
            x = (48271 * x) % 2147483647
            seq.append(x / 2147483647)
        assert np.array_equal(u, np.asarray(seq))

    def test_pinned_values(self):
        # the R engine's test pins the same numbers
        assert np.round(uniforms(3, 12345), 12).tolist() == [0.277490120045, 0.725584676361,
                                                            0.697912643523]
        assert np.isfinite(normals(5, 1)).all()


class TestMacroModel:
    @pytest.fixture()
    def model(self, store):
        return fit_macro_model(store, extra=["PROPERTY_QAT", "VISITORS_QAT", "QA_RATE"])

    def test_em_keeps_what_is_there(self):
        X = np.array([[0.0, 1.0], [1.0, np.nan], [-1.0, -1.0], [0.5, 0.2]])
        F = em_fill(X)
        assert F[0, 0] == 0.0 and F[2, 1] == -1.0 and np.isfinite(F[1, 1])

    def test_a_shocked_variable_keeps_its_path(self, model):
        p = {"OIL_PRICE": np.array([-40.0, -10, 0, 0, 0])}
        c = model.condition(p)
        assert np.allclose(c["OIL_PRICE"], p["OIL_PRICE"])
        assert set(c) == set(model.concepts)

    def test_no_shock_moves_nothing(self, model):
        c = model.condition({})
        assert all(np.allclose(v, 0) for v in c.values())
        assert model.severity({})["probability"] == 1.0

    def test_knock_on_is_linear_and_severity_symmetric(self, model):
        a = model.condition({"OIL_PRICE": [-20, 0, 0, 0, 0]})
        b = model.condition({"OIL_PRICE": [-40, 0, 0, 0, 0]})
        assert np.allclose(2 * a["GDP_QAT"], b["GDP_QAT"])
        s1 = model.severity({"OIL_PRICE": [-40, 0, 0, 0, 0]})
        s2 = model.severity({"OIL_PRICE": [40, 0, 0, 0, 0]})
        assert s1["distance"] == pytest.approx(s2["distance"])
        assert s1["probability"] < model.severity({"OIL_PRICE": [-20, 0, 0, 0, 0]})["probability"]

    def test_the_synthetic_oil_factor_comes_through(self, model):
        # the fixture builds GCC growth on the oil factor: an oil fall lowers it
        c = model.condition({"OIL_PRICE": [-40, 0, 0, 0, 0]})
        assert c["GDP_SAU"][0] < 0

    def test_short_histories_are_carried_alone(self, model):
        assert "VISITORS_QAT" in model.info["independent"]
        c = model.condition({"VISITORS_QAT": [-30, 0, 0, 0, 0]})
        assert all(np.allclose(v, 0) for k, v in c.items() if k != "VISITORS_QAT")

    def test_simulation_is_reproducible(self, model):
        a = model.simulate(50, seed=3)
        b = model.simulate(50, seed=3)
        assert a.shape == (50, 5, len(model.concepts)) and np.array_equal(a, b)
