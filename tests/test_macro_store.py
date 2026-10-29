"""The macro store: sources, vintages, concepts, the quarter fill and the
macro model. Everything here runs on the synthetic store in
``fixtures/macro/store`` (no real data) and on fake network answers."""
import gzip
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ifrs9qdb.macro import MacroStore, match_candidates, refresh
from ifrs9qdb.macro.prng import mulmod, normals, uniforms
from ifrs9qdb.macro.quarter import (first_forecast_year, model_variables, propose,
                                    render_model_inputs, replace_top_level_block)
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
        # the IMF answers an unencoded time filter with 400 and ignores an
        # encoded one: none is sent, the years are cut after reading
        assert "TIME_PERIOD" not in seen[-1]
        assert len(s) == 1 and len(o) == 2

    # the real API's answer (October 2026): labels with ": ", a STRUCTURE[;]
    # header, a quoted STRUCTURE_ID, frequency labels that are not the
    # frequency's, empty values, and the last actual year only when asked for
    REAL = ('STRUCTURE[;],STRUCTURE_ID,ACTION,COUNTRY: Country,INDICATOR: Indicator,'
            'FREQUENCY: Frequency,TIME_PERIOD: Time Period,OBS_VALUE: Observation value,'
            'SCALE: Scale,LATEST_ACTUAL_ANNUAL_DATA: Latest Actual Annual Data \n'
            'dataflow,IMF.RES:WEO(9.0.0): World Economic Outlook (WEO),R,QAT: Qatar,'
            '"NGDP_RPCH: Gross domestic product (GDP), Constant prices, Percent change",'
            'A: Annual,1980,-1.016417,0,2025\n'
            'dataflow,IMF.RES:WEO(9.0.0): World Economic Outlook (WEO),R,QAT: Qatar,'
            '"NGDP_RPCH: Gross domestic product (GDP), Constant prices, Percent change",'
            'A: Assets,2030,5.239668,0,2025\n'
            'dataflow,IMF.RES:WEO(9.0.0): World Economic Outlook (WEO),R,QAT: Qatar,'
            '"NGDP_RPCH: Gross domestic product (GDP), Constant prices, Percent change",'
            'A: Assets,2031,,0,2025\n')

    def _imf(self, dims, seen, answers=None):
        def fake(url, headers):
            seen.append(url)
            if "/structure/" in url:
                return 200, json.dumps({"data": {
                    "dataflows": [{"id": "WEO", "agencyID": "IMF.RES", "version": "9.0.0"}],
                    "dataStructures": [{"dataStructureComponents": {"dimensionList": {
                        "dimensions": [{"id": d, "position": i} for i, d in enumerate(dims)]}}}]
                }}).encode()
            for k, v in (answers or {}).items():
                if k in url:
                    return v
            return 200, self.REAL.encode()
        return ImfSdmx(transport=fake)

    def test_the_real_answer_reads_with_its_last_actual_year(self):
        seen = []
        s, o, _ = self._imf(["COUNTRY", "INDICATOR", "FREQUENCY"], seen).fetch(
            {"dataset": "WEO", "candidates": "WEO", "frequency": "A", "country_dim": "COUNTRY",
             "start": "1990", "forecast": "yes"}, ["QAT"])
        assert "attributes=LATEST_ACTUAL_ANNUAL_DATA" in seen[-1]
        assert s["series_id"].tolist() == ["IMF:WEO:QAT:NGDP_RPCH:A"]
        assert s["last_actual"].iloc[0] == "2025"
        assert o["period"].tolist() == ["2030"]      # 1980 is before the start, 2031 empty

    def test_a_dataset_without_the_attribute_is_asked_again_plainly(self):
        seen = []
        s, o, _ = self._imf(["COUNTRY", "INDICATOR", "FREQUENCY"], seen,
                            {"attributes=": (400, b"<html>Bad Request</html>")}).fetch(
            {"dataset": "FM", "candidates": "FM", "frequency": "A", "country_dim": "COUNTRY",
             "start": "1990", "forecast": "yes"}, ["QAT"])
        assert "attributes=" not in seen[-1] and len(s) == 1

    def test_a_fixed_area_lands_on_the_country_dimension(self):
        """PCPS: world prices, in US dollars and as indices."""
        seen = []
        self._imf(["COUNTRY", "INDICATOR", "DATA_TRANSFORMATION", "FREQUENCY"], seen).fetch(
            {"dataset": "PCPS", "candidates": "PCPS", "frequency": "M",
             "country_dim": "REF_AREA=G001;DATA_TRANSFORMATION=USD+INDEX", "start": ""},
            ["QAT", "SAU"])
        assert seen[-1].endswith("/~/G001.*.USD+INDEX.M")

    def test_key_indicators_for_every_country(self):
        seen = []
        self._imf(["COUNTRY", "INDICATOR", "FREQUENCY"], seen).fetch(
            {"dataset": "WEO", "candidates": "WEO", "frequency": "A", "country_dim": "COUNTRY",
             "start": "", "world_indicators": "NGDP_RPCH+PCPIPCH"}, ["QAT"])
        data = [u for u in seen if "/data/" in u]
        assert data[0].endswith("/~/QAT.*.A") and data[1].endswith("/~/*.NGDP_RPCH+PCPIPCH.A")

    def test_a_throttled_answer_is_asked_again(self, monkeypatch):
        import ifrs9qdb.macro.sources as src
        calls = []

        class R:
            status, headers = 200, {}
            _left = [b"ok"]
            def read(self, n=-1): return self._left.pop() if self._left else b""
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def urlopen(req, timeout, context):
            calls.append(req.full_url)
            if len(calls) == 1:
                raise src.urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b""))
            return R()
        monkeypatch.setattr(src.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(src, "_sleep", lambda s: None)
        assert src.default_transport("https://api.imf.org/x") == (200, b"ok")
        assert len(calls) == 2

    def test_bis_labels_without_a_space(self):
        csv = ("STRUCTURE,STRUCTURE_ID,ACTION,FREQ:Frequency,REF_AREA:Reference area,"
               "TIME_PERIOD:Time period or range,OBS_VALUE:Observation Value,"
               "UNIT_MEASURE:Unit of measure\n"
               "dataflow,BIS:WS_CBPOL(1.0): Central bank policy rates,I,M: Monthly,"
               "SA: Saudi Arabia,2026-09,4.5,368: Per cent per year\n")
        s, o = parse_sdmx_csv(csv, "CBPOL", source="BIS")
        assert s["series_id"].tolist() == ["BIS:CBPOL:SA:VALUE:M"]
        assert s["unit"].iloc[0] == "Per cent per year" and o["period"].tolist() == ["2026-09"]

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
        assert p["models"] == ["internal", "external"]
        a = store.concept_annual("NONOIL_GDP_QAT").set_index("year")["value"]
        f = p["forecast"]
        no = f[f["variable"] == "Non-Oil Real GDP Growth (%)"]
        assert np.allclose(no["proposed"], a.loc[2026:2030])
        # every variable comes from model.yml and the catalogue: the internal
        # model's three, of which the store holds non-oil GDP, and the
        # external model's growth and weights for six countries
        assert set(f.loc[f["model"] == "External rated", "variable"]) >= {
            "Qatar real GDP growth", "Qatar nominal GDP (USD bn)"}
        assert len(f) == 5 * (1 + 12)
        assert p["checks"]["ok"].all()

    def test_the_variables_are_the_models_own(self):
        m = yaml.safe_load((PKG / "config" / "model.yml").read_text())
        assert model_variables(m, "internal") == ["VAR_NON_OIL_GDP_GROWTH",
                                                  "VAR_QATAR_REAL_ESTATE_INDEX_GROWTH",
                                                  "VAR_QATAR_DOMESTIC_CREDIT_GROWTH"]
        assert model_variables(m, "external") == ["VAR_GCC_REAL_GDP_GROWTH"]

    def test_the_history_takes_every_revised_and_new_year(self, store):
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30", models=["internal"])
        a = store.concept_annual("NONOIL_GDP_QAT").set_index("year")["value"]
        h = p["_files"]["static/non_oil_gdp_history.csv"]["rows"].set_index("year")["value"]
        assert h.index.min() == 2015 and h.index.max() == 2025   # the file keeps its first year
        assert np.allclose(h.loc[2015:2025], a.loc[2015:2025])
        assert len(p["history"]) == 11 and np.isnan(p["history"]["current"].iloc[-1])
        assert set(p["history"]["model"]) == {"Internal rated"}

    def test_the_external_history_is_in_the_same_table(self, store):
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30", models=["external"])
        assert set(p["_files"]) == {"static/gcc_real_gdp_growth.csv",
                                    "static/gcc_gdp_current_prices.csv"}
        assert {"Qatar real GDP growth", "Qatar nominal GDP (USD bn)"} <= \
            set(p["history"]["variable"])

    def test_a_value_the_file_already_has_is_not_a_revision(self, store, tmp_path):
        static = tmp_path / "static"
        shutil.copytree(PKG / "static", static)
        a = store.concept_annual("NONOIL_GDP_QAT")
        a = a[a["actual"] & (a["year"] >= 2015)]
        # the history as the store has it, written with percent signs as a
        # spreadsheet saves it, and to ten digits
        body = "\n".join(f"{y},{v:.10g}%" for y, v in zip(a["year"], a["value"]))
        (static / "non_oil_gdp_history.csv").write_text(
            "# units: percentage_points\nyear,value\n" + body + "\n")
        p = propose(store, PKG / "config", static, "2026-09-30", models=["internal"])
        assert "static/non_oil_gdp_history.csv" not in p["_files"]

    def test_the_weights_go_to_auto_unless_kept(self, store, tmp_path):
        cfg = tmp_path / "config"
        shutil.copytree(PKG / "config", cfg)
        t = (cfg / "model_inputs.yml").read_text().replace("mode: auto\n", "mode: explicit\n", 1)
        (cfg / "model_inputs.yml").write_text(t)
        p = propose(store, cfg, PKG / "static", "2026-09-30")
        assert [w["model"] for w in p["weights"]] == ["Internal rated"]
        y = yaml.safe_load(render_model_inputs(p))
        assert y["internal_scenario_weights"]["mode"] == "auto"
        # the explicit weights typed by hand are still in the file
        assert y["internal_scenario_weights"]["explicit_weights"] == \
            yaml.safe_load(t)["internal_scenario_weights"]["explicit_weights"]
        p = propose(store, cfg, PKG / "static", "2026-09-30", keep_weights=["internal"])
        assert p["weights"] == []
        assert yaml.safe_load(render_model_inputs(p))["internal_scenario_weights"]["mode"] \
            == "explicit"

    def test_a_model_not_chosen_stays_as_it_is(self, store):
        old = yaml.safe_load((PKG / "config" / "model_inputs.yml").read_text())
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30", models=["external"])
        new = yaml.safe_load(render_model_inputs(p))
        assert new["mev_forecasts"] == old["mev_forecasts"]
        assert new["internal_scenario_weights"] == old["internal_scenario_weights"]
        assert new["external_gcc_country_growth"] != old["external_gcc_country_growth"]
        assert "static/non_oil_gdp_history.csv" not in p["_files"]

    def test_a_forecast_in_fractions_is_flagged(self, store, tmp_path):
        cfg = tmp_path / "config"
        shutil.copytree(PKG / "config", cfg)
        y = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        for k in y["mev_forecasts"]["forecasts"]:
            y["mev_forecasts"]["forecasts"][k][0] = 0.029
        (cfg / "model_inputs.yml").write_text(yaml.safe_dump(y, sort_keys=False))
        # the external model only: the internal forecast is left in fractions,
        # as entered for a test, and said so
        p = propose(store, cfg, PKG / "static", "2026-09-30", models=["external"])
        assert not any("fractions" in w for w in p["warnings"])
        p = propose(store, cfg, PKG / "static", "2026-09-30", models=["internal"])
        assert not any("fractions" in w for w in p["warnings"])     # replaced by the store's
        ser = store.series(store.latest())
        assert not ser.empty

    def test_the_new_config_keeps_every_comment(self, store):
        p = propose(store, PKG / "config", PKG / "static", "2026-09-30")
        text = render_model_inputs(p)
        old = (PKG / "config" / "model_inputs.yml").read_text()
        comments = [ln for ln in old.splitlines() if ln.strip().startswith("#")]
        assert all(c in text for c in comments)
        y = yaml.safe_load(text)
        assert y["macro_source"]["first_forecast_year"] == 2026
        assert y["macro_source"]["models"] == ["Internal rated", "External rated"]
        # written unrounded: the YAML reads back the very numbers proposed
        f = p["forecast"]
        assert y["mev_forecasts"]["forecasts"][1][0] == \
            f.loc[f["variable"] == "Non-Oil Real GDP Growth (%)", "proposed"].iloc[0]

    def test_the_draft_gets_the_history_with_where_it_came_from(self, store, tmp_path):
        from ifrs9qdb import snapshots as S
        from ifrs9qdb.macro.quarter import apply
        root = tmp_path / "snaps"
        S.create_snapshot("d1", "", "nsahu", config_dir=PKG / "config",
                          static_dir=PKG / "static", run_config_path=None,
                          snapshots_root=root)
        d = root / "d1"
        p = propose(store, d / "config", d / "static", "2026-09-30")
        r = apply(p, "d1", snapshots_root=root)
        assert r["ok"], r
        text = (d / "static" / "non_oil_gdp_history.csv").read_text()
        assert "# units: percentage_points" in text
        assert f"# extracted_from: macro store vintage {p['vintage']}" in text
        assert "# source: IMF:REO_MCD:QAT:NGDP_NOIL_RPCH:A" in text
        h = pd.read_csv(d / "static" / "non_oil_gdp_history.csv", comment="#")
        want = p["_files"]["static/non_oil_gdp_history.csv"]["rows"]
        assert np.allclose(h.set_index("year")["value"], want.set_index("year")["value"])

    def test_a_new_model_variable_needs_only_the_config(self, store, tmp_path):
        # a model change: a fourth variable in the internal model, its
        # forecast in column 4, linked in the project's own catalogue
        cfg = tmp_path / "config"
        shutil.copytree(PKG / "config", cfg)
        m = yaml.safe_load((cfg / "model.yml").read_text())
        m["models"]["internal_v4_production"]["mev_components"].append(
            {"variable": "VAR_TEST_GROWTH", "intercept": 0.0, "coefficient": -0.01,
             "weight": 0.5})
        m["variables"]["VAR_TEST_GROWTH"] = {
            "display_name": "Test growth (%)", "units": "percentage_points",
            "forecast_source": {"kind": "model_inputs_yaml", "path": "mev_forecasts.forecasts",
                                "column_index": 4}}
        (cfg / "model.yml").write_text(yaml.safe_dump(m, sort_keys=False))
        cat = pd.read_csv(PKG / "macro" / "data" / "concepts.csv", dtype=str,
                          keep_default_na=False)
        row = cat[cat["concept"] == "GDP_QAT"].assign(concept="TEST_QAT",
                                                     model_variable="VAR_TEST_GROWTH")
        (tmp_path / "config" / "macro").mkdir(parents=True)
        pd.concat([cat, row]).to_csv(tmp_path / "config" / "macro" / "concepts.csv",
                                     index=False)
        p = propose(store, cfg, PKG / "static", "2026-09-30", models=["internal"],
                    project_root=tmp_path)
        t = p["forecast"][p["forecast"]["variable"] == "Test growth (%)"]
        want = store.concept_annual("GDP_QAT").set_index("year")["value"].loc[2026:2030]
        assert np.allclose(t["proposed"], want)
        y = yaml.safe_load(render_model_inputs(p))
        assert [r[3] for _, r in sorted(y["mev_forecasts"]["forecasts"].items())] == \
            pytest.approx(list(want))

    def test_nothing_chosen_writes_nothing(self, store, tmp_path):
        from ifrs9qdb import snapshots as S
        from ifrs9qdb.macro.quarter import apply
        root = tmp_path / "snaps"
        S.create_snapshot("d1", "", "nsahu", config_dir=PKG / "config",
                          static_dir=PKG / "static", run_config_path=None,
                          snapshots_root=root)
        d = root / "d1"
        before = (d / "config" / "model_inputs.yml").read_bytes()
        p = propose(store, d / "config", d / "static", "2026-09-30", models=[])
        assert p["changed"] == 0
        assert apply(p, "d1", snapshots_root=root)["ok"]
        assert (d / "config" / "model_inputs.yml").read_bytes() == before

    def test_dollars_are_read_as_billions(self):
        from ifrs9qdb.macro.store import unit_divisor
        assert unit_divisor("USD", "USD bn", [2.15e11, 2.3e11]) == 1e9
        assert unit_divisor("U.S. dollars, billions", "USD bn", [215.6]) == 1.0
        assert unit_divisor("USD", "% change", [2.15e11]) == 1.0
        assert 55320437000 / unit_divisor("USD", "USD bn", [55320437000]) == 55.320437

    def test_the_store_keeps_every_digit(self, store):
        s = pd.DataFrame([{"series_id": "IMF:WEO:QAT:NGDP_RPCH:A", "source": "IMF",
                           "dataset": "WEO", "country": "QAT", "indicator": "NGDP_RPCH",
                           "freq": "A", "name": "x", "unit": "", "last_actual": "2025",
                           "release": "new"}])
        o = pd.DataFrame({"series_id": ["IMF:WEO:QAT:NGDP_RPCH:A"] * 2,
                          "period": ["2026", "2027"], "value": [2.9192865516638, 0.1 + 0.2]})
        v = store.save(s, o, label="digits")
        got = store.observations(v, ["IMF:WEO:QAT:NGDP_RPCH:A"])["value"].tolist()
        assert got == [2.9192865516638, 0.1 + 0.2]

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

    def test_linked_effects_are_linear_and_severity_symmetric(self, model):
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


class TestFredUnreachable:
    def test_two_silent_requests_and_the_rest_are_skipped(self):
        import pandas as pd
        from ifrs9qdb.macro.sources import Fred
        calls = []

        def fake(url, headers):
            calls.append(url)
            return 0, b"timed out"
        cat = pd.DataFrame({"series": ["A", "B", "C", "D"], "label": "x", "unit": "", "freq": "M"})
        s, o, info = Fred(transport=fake).fetch_csv(cat)
        assert len(calls) == 2 and info["missing"] == ["A", "B", "C", "D"] and s.empty
