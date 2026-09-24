"""The scenario weighting and the regional inputs behind StPD.

Three plumbing defects put the StPD curves out by a mean of 0.0058 while every
shape test passed: the curves stayed monotonic, bounded and correctly sized,
because a wrong weighting produces a perfectly well-formed curve set. Shape
tests cannot catch this class of thing, so each cause has its own test here.

The fixtures are the packaged config and static reference, which are
byte-identical to the R package's inst/, so these run on a clean clone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import packaged_config, ref_output
from ifrs9qdb.etl.macro import (build_stpd_from_static,
                                compute_external_scenario_weights_per_year,
                                external_gcc_forecast, gcc_weighted_history,
                                resolve_external_scenario_weights,
                                resolve_internal_scenario_weights)
from ifrs9qdb.etl.static_ref import load_static_reference

INTERNAL = ["Business Finance", "Al Dhameen", "Off BS", "Tasdeer"]
EXTERNAL = ["Investments", "Banks and Fis"]


@pytest.fixture(scope="module")
def cfg():
    c = packaged_config()
    return (yaml.safe_load((c / "model.yml").read_text(encoding="utf-8")),
            yaml.safe_load((c / "model_inputs.yml").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def static():
    return load_static_reference()


class TestTheWeightsAreResolvedNotInvented:
    """An equal split is a plausible number that is never the right answer."""

    def test_the_configured_weights_are_used_when_none_are_passed(self, cfg, static):
        model, inputs = cfg
        got = resolve_internal_scenario_weights(inputs, static)
        # model_inputs.yml asks for auto_non_oil_gdp_cdf, and the explicit
        # block in the file is a rounded snapshot of that calculation.
        expected = inputs["internal_scenario_weights"]["explicit_weights"]
        for scenario, w in expected.items():
            assert got[scenario] == pytest.approx(w, abs=5e-4)
        assert sum(got.values()) == pytest.approx(1.0)

    def test_an_equal_split_gives_a_different_answer(self, cfg, static):
        """The defect this replaced: 1/n per scenario, silently."""
        model, inputs = cfg
        scen = static["scenario_severity"]["scenario"]
        equal = {s: 1.0 / len(scen) for s in scen}
        proper = build_stpd_from_static(static, model, inputs, "12/31/2025")
        flat = build_stpd_from_static(static, model, inputs, "12/31/2025",
                                      scenario_weights=equal)
        j = proper.merge(flat, on=["PortfolioCode", "PDBucketDim1",
                                   "MonthLifetime"], suffixes=("_p", "_f"))
        diff = (j["PDLifetime_p"] - j["PDLifetime_f"]).abs()
        assert diff.mean() > 1e-3, "equal weights must not coincide with the real ones"

    def test_an_equal_split_still_produces_a_well_formed_curve(self, cfg, static):
        """Why the shape tests did not catch it."""
        model, inputs = cfg
        scen = static["scenario_severity"]["scenario"]
        equal = {s: 1.0 / len(scen) for s in scen}
        out = build_stpd_from_static(static, model, inputs, "12/31/2025",
                                     scenario_weights=equal)
        assert len(out) == 75_600
        for _, g in out.groupby(["PortfolioCode", "PDBucketDim1"]):
            v = g.sort_values("MonthLifetime")["PDLifetime"].to_numpy()
            assert (np.diff(v) >= -1e-12).all() and v.max() <= 1.0
            break

    def test_an_unknown_mode_raises_rather_than_falling_back(self, cfg, static):
        model, inputs = cfg
        bad = dict(inputs)
        bad["internal_scenario_weights"] = {"mode": "whatever_sounds_right"}
        with pytest.raises(ValueError, match="unknown internal_scenario_weights.mode"):
            resolve_internal_scenario_weights(bad, static)


class TestTheTwoScalesAreWeightedSeparately:
    def test_the_external_weights_differ_year_to_year(self, cfg, static):
        """The regional forecast moves, so year 1 and year 5 cannot share a weight."""
        model, inputs = cfg
        hist = gcc_weighted_history(static["gcc_real_gdp_growth"],
                                    static["gcc_gdp_current_prices"])
        fc = external_gcc_forecast(inputs, static, 5)
        w = compute_external_scenario_weights_per_year(fc, hist,
                                                       static["scenario_severity"])
        per_year = w["per_year"]
        assert len(per_year) == 5
        assert np.allclose(per_year.sum(axis=1), 1.0)
        assert per_year.std(axis=0).max() > 1e-3, "every year got the same weights"
        assert np.allclose(w["average"], per_year.mean(axis=0))

    def test_the_external_scale_does_not_take_the_internal_weights(self, cfg, static):
        model, inputs = cfg
        hist = gcc_weighted_history(static["gcc_real_gdp_growth"],
                                    static["gcc_gdp_current_prices"])
        fc = external_gcc_forecast(inputs, static, 5)
        internal = resolve_internal_scenario_weights(inputs, static)
        external = resolve_external_scenario_weights(inputs, static, hist, fc)

        proper = build_stpd_from_static(static, model, inputs, "12/31/2025")
        wrong = build_stpd_from_static(static, model, inputs, "12/31/2025",
                                       scenario_weights=internal)
        p = proper[proper.PortfolioCode.isin(EXTERNAL)]
        w = wrong[wrong.PortfolioCode.isin(EXTERNAL)]
        j = p.merge(w, on=["PortfolioCode", "PDBucketDim1", "MonthLifetime"],
                    suffixes=("_p", "_w"))
        assert (j["PDLifetime_p"] - j["PDLifetime_w"]).abs().mean() > 1e-4

        # and the internal book is untouched by the external weights
        pi = proper[proper.PortfolioCode.isin(INTERNAL)]
        wi = wrong[wrong.PortfolioCode.isin(INTERNAL)]
        ji = pi.merge(wi, on=["PortfolioCode", "PDBucketDim1", "MonthLifetime"],
                      suffixes=("_p", "_w"))
        assert np.allclose(ji["PDLifetime_p"], ji["PDLifetime_w"])
        assert set(external) == {"per_year", "average"}


class TestTheRegionalInputs:
    def test_the_history_starts_in_1982(self, static):
        """1980-81 are outside the modelled window.

        Including them takes the standard deviation from 4.4912 to 4.3950, and
        that SD is the unit every scenario shift is measured in.
        """
        h = gcc_weighted_history(static["gcc_real_gdp_growth"],
                                 static["gcc_gdp_current_prices"])
        assert h.size == 43
        assert h.std(ddof=1) == pytest.approx(4.49116, abs=1e-4)

        loose = gcc_weighted_history(static["gcc_real_gdp_growth"],
                                     static["gcc_gdp_current_prices"],
                                     first_year=None)
        assert loose.size == 45
        assert loose.std(ddof=1) == pytest.approx(4.39497, abs=1e-4)

    def test_one_country_is_not_a_regional_average(self):
        growth = pd.DataFrame({"year": [2000, 2000, 2001],
                               "country": ["Qatar", "Oman", "Qatar"],
                               "value": [5.0, 1.0, 9.0]})
        prices = pd.DataFrame({"year": [2000, 2000, 2001],
                               "country": ["Qatar", "Oman", "Qatar"],
                               "value": [100.0, 100.0, 100.0]})
        h = gcc_weighted_history(growth, prices, first_year=None)
        assert h.size == 1 and h[0] == pytest.approx(3.0)

    def test_the_forecast_is_weighted_per_year_by_the_config_prices(self, cfg, static):
        """Year 1 by hand, from the growth and price blocks in model_inputs.yml."""
        model, inputs = cfg
        g = inputs["external_gcc_country_growth"]
        p = inputs["external_gcc_country_prices"]
        countries = list(g)
        num = sum(g[c][0] * p[c][0] for c in countries)
        den = sum(p[c][0] for c in countries)
        by_hand = num / den

        fc = np.asarray(external_gcc_forecast(inputs, static, 5), dtype=float)
        assert fc.size == 5
        assert fc[0] == pytest.approx(by_hand)
        assert fc[0] == pytest.approx(4.25693, abs=1e-5)

    def test_the_static_table_is_not_used_when_the_config_carries_prices(self, cfg, static):
        """The defect this replaced: one fixed weight vector from the static
        table's latest year, applied to every forecast year."""
        model, inputs = cfg
        proper = np.asarray(external_gcc_forecast(inputs, static, 5), dtype=float)

        no_prices = {k: v for k, v in inputs.items()
                     if k != "external_gcc_country_prices"}
        fallback = np.asarray(external_gcc_forecast(no_prices, static, 5),
                              dtype=float)
        assert not np.allclose(proper, fallback)


class TestAgainstTheReferenceStPD:
    """The whole point: the file the R pipeline wrote, reproduced."""

    @pytest.mark.skipif(ref_output() is None,
                        reason="set IFRS9_REF_RUN to a run folder holding Output/")
    def test_stpd_reproduces_the_reference_exactly(self, cfg, static):
        model, inputs = cfg
        ref = pd.read_csv(ref_output() / "StPD.csv", low_memory=False)
        out = build_stpd_from_static(static, model, inputs,
                                     str(ref["ExtractDate"].iloc[0]))
        j = out.merge(ref, on=["PortfolioCode", "PDBucketDim1", "MonthLifetime"],
                      suffixes=("_got", "_ref"))
        assert len(j) == len(ref) == 75_600
        d = (j["PDLifetime_got"] - j["PDLifetime_ref"]).abs()
        # The reference CSV is written to eight decimals; anything at 1e-8 is
        # the file format. This sits at 1e-15, i.e. the same arithmetic.
        assert d.max() < 1e-12, f"max abs diff {d.max():.3e}"
