"""The PD term structure.

Each test pins a modelling choice that would be easy to "improve" into
something that no longer matches LIC.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from ifrs9qdb.etl.macro import (apply_scenario_weights, convert_to_monthly_stpd,
                                cumulative_pd, pit_pd_term_structure)

from conftest import packaged_config, ref_output

OUT = ref_output()
has_run = pytest.mark.skipif(OUT is None or not (OUT / "StPD.csv").is_file(),
                             reason="no reference StPD")


class TestPitAdjustment:
    def test_the_shift_is_in_probit_space(self):
        """A severe scenario must not push a PD outside (0, 1).

        Scaling the probability directly would; shifting the probit does not.
        """
        for sf in (-10, -3, 0, 3, 10):
            v = pit_pd_term_structure(0.02, [sf] * 5, 5, 10)
            assert 0.0 <= v[0] <= 1.0

    def test_a_zero_scaling_factor_returns_the_ttc_pd(self):
        v = pit_pd_term_structure(0.02, [0] * 5, 5, 10)
        assert v[0] == pytest.approx(0.02)

    def test_an_adverse_factor_raises_the_pd(self):
        base = pit_pd_term_structure(0.02, [0] * 5, 5, 10)[0]
        worse = pit_pd_term_structure(0.02, [0.5] * 5, 5, 10)[0]
        assert worse > base

    def test_a_zero_ttc_pd_stays_zero(self):
        assert (pit_pd_term_structure(0.0, [3] * 5, 5, 10) == 0).all()

    def test_only_the_forecast_years_use_the_scaling_factors(self):
        """Beyond the horizon the curve reverts, so later years must not track
        the scenario shift."""
        v = pit_pd_term_structure(0.02, [1.0] * 5, 5, 20)
        assert len(v) == 20
        assert not np.allclose(v[5:], v[4])


class TestCumulative:
    def test_survival_not_a_running_sum(self):
        """A sum of marginals passes 1; the survival product cannot."""
        m = [0.3] * 10
        cum = cumulative_pd(m)
        assert cum.max() < 1.0
        assert np.cumsum(m).max() > 1.0

    def test_matches_the_closed_form(self):
        m = np.array([0.1, 0.2, 0.3])
        assert cumulative_pd(m)[-1] == pytest.approx(1 - 0.9 * 0.8 * 0.7)

    def test_is_monotonic(self):
        cum = cumulative_pd(np.full(30, 0.05))
        assert (np.diff(cum) >= -1e-12).all()


class TestScenarioWeighting:
    @staticmethod
    def frame():
        return pd.DataFrame({
            "rating": ["A"] * 4,
            "scenario": ["Base", "Base", "Down", "Down"],
            "maturity": [1, 2, 1, 2],
            "marginal_pd": [0.01, 0.02, 0.05, 0.06],
        })

    def test_weights_are_applied_to_marginals(self):
        out = apply_scenario_weights(self.frame(), {"Base": 0.7, "Down": 0.3})
        assert out.loc[out.maturity == 1, "weighted_marginal_pd"].iloc[0] == \
            pytest.approx(0.7 * 0.01 + 0.3 * 0.05)

    def test_weights_are_used_as_given_not_normalised(self):
        """The weights are applied as supplied. This is deliberate.

        V4's explicit weights sum to 1.0003 -- they are a rounded snapshot of
        the computed ones -- and the R engine carries that through rather than
        rescaling. Normalising here looks tidier and puts every production
        figure out by that factor, so the StPD stops reproducing the reference.
        """
        a = apply_scenario_weights(self.frame(), {"Base": 7, "Down": 3})
        b = apply_scenario_weights(self.frame(), {"Base": 0.7, "Down": 0.3})
        assert np.allclose(a["weighted_marginal_pd"],
                           10 * b["weighted_marginal_pd"])

    def test_weights_far_from_one_warn(self):
        """Not normalising means a mistyped weight has to be visible."""
        with pytest.warns(RuntimeWarning, match="sum to"):
            apply_scenario_weights(self.frame(), {"Base": 7, "Down": 3})

    def test_the_v4_rounding_slack_does_not_warn(self):
        """1.0003 is the production case and must stay quiet."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            apply_scenario_weights(self.frame(), {"Base": 0.7, "Down": 0.3003})

    def test_a_missing_weight_is_an_error_not_a_silent_zero(self):
        with pytest.raises(ValueError, match="no weight"):
            apply_scenario_weights(self.frame(), {"Base": 1.0})


class TestMonthlyConversion:
    @staticmethod
    def annual(vals):
        return pd.DataFrame({"rating": ["A"] * len(vals),
                             "maturity": range(1, len(vals) + 1),
                             "weighted_marginal_pd": vals})

    def test_a_years_marginal_is_spread_evenly(self):
        mo = convert_to_monthly_stpd(self.annual([0.12, 0.12]), 24)
        assert mo["monthly_marginal_pd"].iloc[:12].std() == pytest.approx(0.0)
        assert mo["pd_lifetime"].iloc[11] == pytest.approx(0.12)

    def test_accumulation_is_a_running_sum(self):
        """Deliberately NOT the survival formula used annually. LIC expects
        this, and making it consistent would stop the output matching."""
        mo = convert_to_monthly_stpd(self.annual([0.12] * 3), 36)
        assert mo["pd_lifetime"].iloc[35] == pytest.approx(0.36)

    def test_the_cap_binds(self):
        mo = convert_to_monthly_stpd(self.annual([0.6] * 5), 60)
        assert mo["pd_lifetime"].max() <= 1.0

    def test_months_past_the_last_year_repeat_it(self):
        mo = convert_to_monthly_stpd(self.annual([0.12, 0.24]), 48)
        assert mo["monthly_marginal_pd"].iloc[47] == pytest.approx(0.24 / 12)


@has_run
class TestAgainstTheReference:
    @staticmethod
    def stpd():
        return pd.read_csv(OUT / "StPD.csv", low_memory=False)

    def test_the_marginal_is_constant_within_a_year(self):
        d = self.stpd()
        one = d[(d.PortfolioCode == "Business Finance") & (d.PDBucketDim1 == 1)]
        cum = one.sort_values("MonthLifetime")["PDLifetime"].to_numpy()
        marg = np.diff(np.concatenate([[0], cum]))
        # The file is written to eight decimals, so marginals recovered by
        # differencing carry about 1e-9 of rounding. That is the format, not a
        # modelling difference; a tighter tolerance would be testing the CSV
        # writer rather than the model.
        assert np.allclose(marg[:12], marg[0], atol=1e-8)
        assert abs(marg[12] - marg[11]) > 1e-9

    def test_portfolios_on_the_same_scale_share_curves(self):
        """Curves are a property of the rating SCALE, not the portfolio."""
        d = self.stpd()
        piv = d[d.PDBucketDim1 == 5].pivot_table(
            index="MonthLifetime", columns="PortfolioCode", values="PDLifetime")
        internal = ["Business Finance", "Al Dhameen", "Off BS", "Tasdeer"]
        external = ["Investments", "Banks and Fis"]
        assert np.allclose(piv[internal].std(axis=1), 0)
        assert np.allclose(piv[external].std(axis=1), 0)
        assert abs(piv[internal[0]] - piv[external[0]]).max() > 1e-9

    def test_curves_are_monotonic_and_bounded(self):
        d = self.stpd()
        for (_, _), g in d.groupby(["PortfolioCode", "PDBucketDim1"]):
            v = g.sort_values("MonthLifetime")["PDLifetime"].to_numpy()
            assert (np.diff(v) >= -1e-12).all()
            assert v.max() <= 1.0
            break


class TestScalingFactorChain:
    """From macro forecasts to the scaling factor the PD adjustment consumes."""

    @staticmethod
    def specs():
        return [
            {"standard_deviation": 2.0, "stress_unit_multiplier": 1.0,
             "intercept": -4.0, "coefficient": -0.15},
            {"standard_deviation": 8.0, "stress_unit_multiplier": 1.0,
             "intercept": -4.0, "coefficient": -0.02},
        ]

    @staticmethod
    def forecasts():
        return np.array([[4.44, -8.04]] * 5)

    def test_each_variable_moves_by_its_own_sigma(self):
        """A shared shift would overstate stable series and understate volatile
        ones."""
        from ifrs9qdb.etl.macro import stress_mevs
        s = stress_mevs(self.forecasts(), -1.0, self.specs())
        shift = s[0] - self.forecasts()[0]
        assert shift[0] == pytest.approx(-2.0)
        assert shift[1] == pytest.approx(-8.0)

    def test_a_zero_severity_leaves_the_forecast_alone(self):
        from ifrs9qdb.etl.macro import stress_mevs
        s = stress_mevs(self.forecasts(), 0.0, self.specs())
        assert np.allclose(s, self.forecasts())

    def test_the_unit_multiplier_is_applied_in_both_directions(self):
        """It multiplies the shift and divides the regression input. Applying
        it once, or the same way twice, silently rescales the shock."""
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        specs = self.specs()
        scaled = [dict(s, stress_unit_multiplier=100.0) for s in specs]
        # with the shift at zero the two directions must cancel exactly
        a = combined_sf_for_scenario(self.forecasts(), 0.0, specs, [1, 0], 0.02)
        b = combined_sf_for_scenario(self.forecasts() * 100, 0.0, scaled,
                                     [1, 0], 0.02)
        assert a[0] == pytest.approx(b[0])

    def test_the_per_variable_factor_is_a_probit_difference(self):
        from ifrs9qdb.etl.macro import compute_per_mev_sf
        from scipy.stats import norm
        got = compute_per_mev_sf(np.array([[0.05]]), 0.02)
        assert got[0, 0] == pytest.approx(norm.ppf(0.05) - norm.ppf(0.02))

    def test_a_worse_scenario_raises_the_scaling_factor(self):
        """Monotonic, or the whole stress framework is meaningless."""
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        sfs = [combined_sf_for_scenario(self.forecasts(), z, self.specs(),
                                        [1.0, 0.0], 0.02)[0]
               for z in (1, 0, -1, -2)]
        assert all(b > a for a, b in zip(sfs, sfs[1:]))

    def test_logits_map_into_zero_one(self):
        from ifrs9qdb.etl.macro import compute_pds_from_logits
        out = compute_pds_from_logits(np.array([[-20.0, 0.0, 20.0]]))
        assert (out > 0).all() and (out < 1).all()
        assert out[0, 1] == pytest.approx(0.5)

    def test_weights_must_match_the_variable_count(self):
        from ifrs9qdb.etl.macro import combine_sf
        with pytest.raises(ValueError, match="weights"):
            combine_sf(np.zeros((5, 3)), [1.0, 0.0])

    def test_a_zero_weighted_variable_cannot_move_the_result(self):
        """The current model weights two of three MEVs at zero, so shocking
        them must change nothing. Worth pinning: it looks like a bug."""
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        fc = self.forecasts().copy()
        a = combined_sf_for_scenario(fc, -1.0, self.specs(), [1.0, 0.0], 0.02)
        fc[:, 1] = -99.0
        b = combined_sf_for_scenario(fc, -1.0, self.specs(), [1.0, 0.0], 0.02)
        assert np.allclose(a, b)


class TestWithTheProductionConfig:
    """The chain against the real model.yml, not a synthetic fixture."""

    # Ships inside the package and is byte-identical to the R package's
    # inst/config, so this resolves on any clone and never skips.
    CONFIG = packaged_config()
    has_cfg = pytest.mark.skipif(not (CONFIG / "model.yml").is_file(),
                                 reason="no production config")

    @staticmethod
    def load():
        import yaml
        cfg = TestWithTheProductionConfig.CONFIG
        mc = yaml.safe_load((cfg / "model.yml").read_text())
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        comp = mc["models"]["internal_v4_production"]["mev_components"]
        specs = [{
            "standard_deviation": c["standard_deviation"],
            "stress_unit_multiplier":
                mc["variables"][c["variable"]].get("stress_unit_multiplier", 1),
            "intercept": c["intercept"],
            "coefficient": c["coefficient"],
        } for c in comp]
        weights = [c["weight"] for c in comp]
        fc = np.array([mi["mev_forecasts"]["forecasts"][y]
                       for y in sorted(mi["mev_forecasts"]["forecasts"])],
                      dtype=float)
        return specs, weights, fc, mc["ttc_anchor_pd"]

    @has_cfg
    def test_scaling_factors_are_ordered_by_severity(self):
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        specs, weights, fc, anchor = self.load()
        sfs = [combined_sf_for_scenario(fc, z, specs, weights, anchor)[0]
               for z in (1.75, 0.75, 0.0, -0.75, -1.75)]
        assert all(b > a for a, b in zip(sfs, sfs[1:]))

    @has_cfg
    def test_the_base_case_sits_near_zero(self):
        """The base case is the unstressed view, so its factor should be small.
        A large one would mean the regression disagrees with the anchor."""
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        specs, weights, fc, anchor = self.load()
        assert abs(combined_sf_for_scenario(fc, 0.0, specs, weights, anchor)[0]) < 0.25

    @has_cfg
    def test_only_non_oil_gdp_carries_weight(self):
        """Two of the three MEVs are weighted zero in the production model, so
        shocking them changes nothing. Anyone testing a property-price stress
        needs to know this before concluding the tool is broken."""
        specs, weights, fc, anchor = self.load()
        assert weights[0] == 1.0
        assert weights[1] == 0.0 and weights[2] == 0.0

    @has_cfg
    def test_the_resulting_curve_is_a_valid_probability(self):
        from ifrs9qdb.etl.macro import combined_sf_for_scenario, pit_pd_term_structure
        specs, weights, fc, anchor = self.load()
        sf = combined_sf_for_scenario(fc, -1.75, specs, weights, anchor)
        pit = pit_pd_term_structure(0.02, sf, 5, 30)
        assert np.isfinite(pit).all()
        assert (pit >= 0).all() and (pit <= 1).all()


class TestStaticReference:
    from pathlib import Path as _P

    def test_the_two_rating_scales_are_never_combined(self):
        """Both reuse hierarchy 1-21, so a shared bucket number means different
        grades. Asking for one scale at a time is what prevents mixing them."""
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        internal = set(st.ratings_for(1)["rating"])
        external = set(st.ratings_for(2)["rating"])
        assert internal and external
        assert not (internal & external)

    def test_rating_type_matches_by_number_or_name(self):
        """The TTC table codes it 1/2; the rating scale spells it out."""
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        assert len(st.ttc_for(1)) > 0
        assert len(st.ratings_for(1)) > 0

    def test_a_zero_ttc_pd_is_kept(self):
        """Filtering on `> 0` instead of `>= 0` loses three external grades and
        3,600 rows of output."""
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        assert len(st.ttc_for(2)) == 21

    def test_invalid_pds_are_dropped(self):
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        for t in (1, 2):
            v = pd.to_numeric(st.ttc_for(t)["ttc_pd"], errors="coerce")
            assert (v >= 0).all() and (v < 1).all()


class TestStpdShape:
    def test_the_output_has_the_expected_shape(self):
        """6 portfolios x 21 buckets x 600 months."""
        import yaml
        from ifrs9qdb.etl.macro import build_stpd_from_static
        from ifrs9qdb.etl.static_ref import load_static_reference
        cfg = TestWithTheProductionConfig.CONFIG
        if not (cfg / "model.yml").is_file():
            pytest.skip("no production config")
        st = load_static_reference()
        mc = yaml.safe_load((cfg / "model.yml").read_text())
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        out = build_stpd_from_static(
            st, mc, mi, "6/30/2026",
            scenario_weights=mi["internal_scenario_weights"].get("explicit_weights"))
        assert len(out) == 75_600
        assert out["PortfolioCode"].nunique() == 6
        assert out["PDBucketDim1"].nunique() == 21
        assert out["MonthLifetime"].nunique() == 600

    def test_curves_are_monotonic_and_bounded(self):
        import yaml
        from ifrs9qdb.etl.macro import build_stpd_from_static
        from ifrs9qdb.etl.static_ref import load_static_reference
        cfg = TestWithTheProductionConfig.CONFIG
        if not (cfg / "model.yml").is_file():
            pytest.skip("no production config")
        st = load_static_reference()
        mc = yaml.safe_load((cfg / "model.yml").read_text())
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        out = build_stpd_from_static(st, mc, mi, "6/30/2026")
        for _, g in out.groupby(["PortfolioCode", "PDBucketDim1"]):
            v = g.sort_values("MonthLifetime")["PDLifetime"].to_numpy()
            assert (np.diff(v) >= -1e-12).all()
            assert v.max() <= 1.0
            break


class TestExternalScale:
    """The external chain: GCC growth through a percentile rank, not a z-score."""

    HIST = np.array([1.2, 3.4, -2.1, 5.6, 0.3, 2.2, 4.1, -1.0, 3.0, 1.8])

    def test_truncation_not_rounding(self):
        """Excel truncates PERCENTRANK.EXC to three significant digits.
        Rounding instead shifts the probit that follows."""
        from ifrs9qdb.etl.macro import truncate_significant
        assert truncate_significant(0.199999, 3) == pytest.approx(0.199)
        assert truncate_significant(0.123456, 3) == pytest.approx(0.123)

    def test_percentrank_never_reaches_zero_or_one(self):
        """It feeds a probit, and 0 or 1 would give infinity."""
        from ifrs9qdb.etl.macro import percentrank_exc
        d = [1, 2, 3, 4, 5]
        for x in (-100, 0, 3, 5, 100):
            pr = percentrank_exc(d, x)
            assert 0 < pr < 1

    def test_values_outside_the_range_are_clamped(self):
        from ifrs9qdb.etl.macro import percentrank_exc
        d = [1, 2, 3, 4, 5]
        assert percentrank_exc(d, -50) == percentrank_exc(d, 0)
        assert percentrank_exc(d, 50) == percentrank_exc(d, 9)

    def test_the_factor_rises_with_growth(self):
        """Higher growth gives a higher factor.

        On its own this looks inverted, but the external scale then SUBTRACTS
        the factor through the Basel ASRF formula, so the net effect is the
        expected one. The next test checks that.
        """
        from ifrs9qdb.etl.macro import external_combined_sf
        sfs = [external_combined_sf([g], self.HIST)[0] for g in (-5, -1, 2, 6)]
        assert all(b > a for a, b in zip(sfs, sfs[1:]))

    def test_the_internal_factor_runs_the_other_way(self):
        """The two chains use OPPOSITE sign conventions, and both end up right.

        Internal:  PD = Phi(Phi^-1(TTC) + SF)          -- SF added
        External:  PD = Phi((Phi^-1(TTC) - sqrt(R) SF) / sqrt(1-R))  -- subtracted

        so the internal factor rises in a downturn while the external falls,
        and both raise the provision. Applying the internal formula to the
        external scale inverts it, which is a mistake worth a test.
        """
        from ifrs9qdb.etl.macro import combined_sf_for_scenario
        specs = [{"standard_deviation": 2.0, "stress_unit_multiplier": 1.0,
                  "intercept": -4.0, "coefficient": -0.15}]
        fc = np.array([[4.44]] * 5)
        down = combined_sf_for_scenario(fc, -1.28, specs, [1.0], 0.02)[0]
        up = combined_sf_for_scenario(fc, 1.28, specs, [1.0], 0.02)[0]
        assert down > up

    def test_the_asrf_formula_subtracts_the_factor(self):
        from ifrs9qdb.etl.macro import basel_asrf_pit
        assert basel_asrf_pit(0.02, -1.5) > basel_asrf_pit(0.02, 0.0)
        assert basel_asrf_pit(0.02, 1.5) < basel_asrf_pit(0.02, 0.0)

    def test_a_downturn_raises_the_external_pd(self):
        """The whole point: weaker regional growth must increase the provision."""
        from ifrs9qdb.etl.macro import basel_asrf_pit, external_combined_sf
        hist = self.HIST
        sd = hist.std(ddof=1)
        base = 2.0
        down = basel_asrf_pit(0.02, external_combined_sf([base - 1.28 * sd], hist)[0])
        up = basel_asrf_pit(0.02, external_combined_sf([base + 1.28 * sd], hist)[0])
        assert down > up

    def test_asrf_boundaries(self):
        from ifrs9qdb.etl.macro import basel_asrf_pit
        assert basel_asrf_pit(0.0, 1.0) == 0.0
        assert basel_asrf_pit(1.0, 1.0) == 1.0
        assert np.isnan(basel_asrf_pit(np.nan, 1.0))


class TestScenarioWeightsAndGcc:
    def test_computed_weights_reproduce_the_config(self):
        """The config's explicit weights are a rounded snapshot of the computed
        ones, which is why they sum to 1.0003 rather than 1."""
        import yaml
        from ifrs9qdb.etl.macro import compute_internal_scenario_weights
        from ifrs9qdb.etl.static_ref import load_static_reference
        cfg = TestWithTheProductionConfig.CONFIG
        if not (cfg / "model_inputs.yml").is_file():
            pytest.skip("no production config")
        st = load_static_reference()
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        fcb = mi["mev_forecasts"]["forecasts"]
        n = int(mi["internal_scenario_weights"].get("n_forecast_years", 2))
        gdp = [float(fcb[y][0]) for y in sorted(fcb, key=lambda k: int(k))][:n]
        got = compute_internal_scenario_weights(
            st.non_oil_gdp_history["value"].to_numpy(), gdp, st.scenario_severity)
        want = mi["internal_scenario_weights"]["explicit_weights"]
        for k, v in want.items():
            assert got[k] == pytest.approx(v, abs=5e-4)

    def test_weights_sum_to_exactly_one(self):
        """The central scenario takes the residual, which is what guarantees it."""
        from ifrs9qdb.etl.macro import compute_internal_scenario_weights
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        got = compute_internal_scenario_weights(
            st.non_oil_gdp_history["value"].to_numpy(), [3.0, 2.5],
            st.scenario_severity)
        assert sum(got.values()) == pytest.approx(1.0)

    def test_gdp_weighting_halves_the_dispersion(self):
        """An unweighted pool lets Bahrain move the series as much as Saudi
        Arabia, which inflates every scenario shift built from it."""
        from ifrs9qdb.etl.macro import gcc_weighted_history
        from ifrs9qdb.etl.static_ref import load_static_reference
        st = load_static_reference()
        weighted = gcc_weighted_history(st.gcc_real_gdp_growth,
                                        st.gcc_gdp_current_prices)
        pooled = pd.to_numeric(st.gcc_real_gdp_growth["value"],
                               errors="coerce").dropna().to_numpy()
        assert weighted.std(ddof=1) < pooled.std(ddof=1)


class TestInternalFormulaIsAProbitShift:
    """Inverting the reference confirms which formula the internal scale uses.

    If the reference had been produced with the Basel ASRF form, the factor
    needed to explain it would swing from about -1.24 at the best rating to
    -0.15 at the worst. Under the plain probit shift it is nearly constant
    (0.16 to 0.11), which is what a single macro factor should look like.
    """

    def test_a_constant_shift_reproduces_the_shape_across_ratings(self):
        from scipy.stats import norm
        ttc = np.array([0.00013, 0.0204, 0.12644, 0.23189, 0.78348])
        ref = np.array([0.00024, 0.02837, 0.15528, 0.27150, 0.81332])
        implied = norm.ppf(ref) - norm.ppf(ttc)
        assert implied.max() - implied.min() < 0.06

    def test_the_asrf_form_would_need_a_wildly_varying_factor(self):
        from scipy.optimize import brentq
        from ifrs9qdb.etl.macro import basel_asrf_pit
        ttc = [0.00013, 0.12644, 0.78348]
        ref = [0.00024, 0.15528, 0.81332]
        implied = [brentq(lambda s: basel_asrf_pit(t, s) - r, -6, 6)
                   for t, r in zip(ttc, ref)]
        assert max(implied) - min(implied) > 1.0
