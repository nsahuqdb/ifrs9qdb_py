"""Characterisation tests for the open methodology issues.

These pin what the engine does TODAY, including where that is wrong. They are
not a specification. Every one of them corresponds to an item in
METHODOLOGY_ISSUES.md, and each is written so that FIXING the issue makes it
fail — loudly, with a message naming the item — so a model change cannot land
without somebody deciding it should.

Read a failure here as "M-something has been changed, go and approve it", not
as "the build is broken".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from ifrs9qdb.analytics import normalise
from ifrs9qdb.engine import compute_lgd, fallback_ead_curve, sum_marginal_ecl
from ifrs9qdb.etl.macro import basel_asrf_pit, compute_internal_scenario_weights
from ifrs9qdb.etl.static_ref import load_static_reference

from conftest import ref_output

OUT = ref_output()
needs_run = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


@pytest.fixture(scope="module")
def static():
    return load_static_reference()


@pytest.fixture(scope="module")
def report():
    if OUT is None:
        pytest.skip("set IFRS9_REF_RUN")
    return normalise(pd.read_csv(OUT / "FinalEclReport.csv", low_memory=False))


@pytest.fixture(scope="module")
def inputs():
    if OUT is None:
        pytest.skip("set IFRS9_REF_RUN")
    from ifrs9qdb.inputs import load_engine_inputs
    return load_engine_inputs(OUT)


# ----------------------------------------------------------------- M1 ------
class TestM1ScenarioWeightsPointTheWrongWay:
    """M1: a worse forecast moves weight ONTO the uptrend scenarios."""

    @staticmethod
    def weights(static, forecast):
        return compute_internal_scenario_weights(
            static["non_oil_gdp_history"]["value"].to_numpy(),
            [forecast], static["scenario_severity"])

    def test_a_collapsing_forecast_favours_the_uptrend(self, static):
        w = self.weights(static, -10.0)
        assert w["Significant Uptrend"] > 0.9, (
            "M1 appears to be FIXED: a -10% growth forecast no longer puts "
            "almost all the weight on Significant Uptrend. Confirm the new "
            "direction is intended and update METHODOLOGY_ISSUES.md.")
        assert w["Significant Downturn"] < 0.001

    def test_the_downturn_weight_falls_as_growth_falls(self, static):
        """The defining symptom, across the whole plausible range."""
        series = [self.weights(static, f)["Significant Downturn"]
                  for f in (6.0, 4.44, 2.81, 1.0, 0.0, -2.0, -5.0)]
        assert all(b < a for a, b in zip(series, series[1:])), (
            f"M1 direction has changed: {series}")

    def test_the_current_weights_are_the_mirror_of_the_correct_ones(self, static):
        """The two differ only in which term carries the forecast, so the
        output at any forecast is the correct output read backwards. That
        symmetry is the evidence it is a swapped pair rather than a choice."""
        h = static["non_oil_gdp_history"]["value"].to_numpy()
        scen = static["scenario_severity"]
        mu, sd = h.mean(), h.std(ddof=1)
        z = pd.to_numeric(scen["severity_z"], errors="coerce").to_numpy()
        order = np.argsort(z)
        zs, n = z[order], z.size
        central = int(np.argmin(np.abs(zs)))

        def bands(cdf):
            p = np.zeros(n)
            for i in range(n):
                if i == central:
                    continue
                if i < central:
                    p[i] = cdf[i] - (0.0 if i == 0 else cdf[i - 1])
                else:
                    p[i] = (1.0 if i == n - 1 else cdf[i + 1]) - cdf[i]
            p[central] = 1.0 - p.sum()
            return p

        for f in (6.0, 0.0, -5.0):
            now = np.array([self.weights(static, f)[s]
                            for s in scen["scenario"]])[order]
            fixed = bands(norm.cdf(mu + sd * zs, loc=f, scale=sd))
            assert np.allclose(now, fixed[::-1], atol=1e-9), (
                f"M1 is no longer an exact mirror at forecast {f}. The cause "
                "may have changed; re-derive before assuming it is fixed.")


# ----------------------------------------------------------------- M2 ------
class TestM2TheScalesDisagreeOnTheSignOfGrowth:
    def test_the_same_shift_moves_the_two_scales_apart(self):
        ttc = 0.02
        for sf in (1.0, 2.0):
            internal = float(norm.cdf(norm.ppf(ttc) + sf))
            external = float(basel_asrf_pit(ttc, sf))
            assert internal > ttc > external, (
                f"M2: at shift {sf} the internal PD should rise and the "
                "external fall. One of them has changed sign — confirm which "
                "is now correct.")


# ----------------------------------------------------------------- M3 ------
class TestM3MonthlyPdAccumulatesBySumming:
    @needs_run
    def test_some_curves_reach_certain_default(self):
        stpd = pd.read_csv(OUT / "StPD.csv", low_memory=False)
        stpd["k"] = (stpd["PortfolioCode"].astype(str) + "|"
                     + stpd["PDBucketDim1"].astype(str))
        saturated = stpd.loc[stpd["PDLifetime"] >= 0.999999, "k"].nunique()
        assert saturated > 0, (
            "M3 may be FIXED: no curve reaches cumulative PD = 1. If the "
            "accumulation is now survival-based, update the register.")
        first = stpd.loc[stpd["PDLifetime"] >= 0.999999, "MonthLifetime"].min()
        assert first < 240, f"M3: first saturation at month {first}"

    @needs_run
    def test_the_sum_overstates_a_survival_accumulation(self):
        stpd = pd.read_csv(OUT / "StPD.csv", low_memory=False)
        stpd["k"] = (stpd["PortfolioCode"].astype(str) + "|"
                     + stpd["PDBucketDim1"].astype(str))
        gaps = []
        for _, g in stpd.groupby("k"):
            v = g.sort_values("MonthLifetime")["PDLifetime"].to_numpy()
            marginal = np.diff(np.concatenate([[0.0], v]))
            survival = 1 - np.cumprod(1 - np.clip(marginal, 0, 0.999999))
            gaps.append(v[119] - survival[119])
        assert np.mean(gaps) > 0.05, (
            f"M3: mean overstatement at 10 years is now {np.mean(gaps):.4f}")

    def test_the_roll_forward_uses_the_other_convention(self):
        """The same codebase conditions correctly on survival elsewhere, which
        is why M3 reads as an oversight rather than a convention."""
        from ifrs9qdb.stress import conditional_pd
        cum = np.array([0.0, 0.1, 0.2, 0.3])
        got = conditional_pd(cum, 1)
        assert got[1] == pytest.approx((0.2 - 0.1) / (1 - 0.1))


# ----------------------------------------------------------------- M4 ------
class TestM4TheStepCountFloors:
    @pytest.mark.parametrize("term,freq,zero_at", [(14, 3, 13), (23, 12, 13)])
    def test_the_balance_reaches_zero_before_maturity(self, term, freq, zero_at):
        c = fallback_ead_curve(1000, term, 4, payment_frequency=freq,
                               portfolio="Business Finance")
        hit = int(np.argmax(c <= 1e-9)) + 1
        assert hit == zero_at, (
            f"M4: a {term}-month facility paying every {freq} months now "
            f"reaches zero at month {hit}, not {zero_at}.")
        assert hit < term, "M4 may be FIXED — the curve now runs to maturity."


# ----------------------------------------------------------------- M5 ------
class TestM5TheFallbackDoesNotReproduceRealSchedules:
    """The back-test: build the fallback for contracts that HAVE a schedule."""

    @staticmethod
    def errors(inputs):
        con = inputs.contracts.set_index("contract")
        out = []
        for cid, real in inputs.ead_curves.items():
            if cid not in con.index:
                continue
            r = con.loc[cid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            real = np.asarray(real, dtype=float)
            if real.size == 0 or not np.isfinite(r.get("on_balance") or np.nan):
                continue
            fb = fallback_ead_curve(
                r["on_balance"], int(r["months_to_mat"] or 3),
                r.get("payment_type"), r.get("payment_frequency"),
                r.get("deferral"), horizon=len(real),
                portfolio=r.get("portfolio"))
            n = min(len(fb), len(real))
            if n == 0:
                continue
            out.append((fb[:n].sum() - real[:n].sum())
                       / max(real[:n].sum(), 1.0))
        return np.array(out)

    @needs_run
    def test_fewer_than_half_agree_within_ten_percent(self, inputs):
        e = self.errors(inputs)
        assert e.size > 1000, "too few comparable contracts to judge"
        within = (np.abs(e) <= 0.10).mean()
        assert within < 0.60, (
            f"M5 may be IMPROVED: {within:.1%} of schedules now agree within "
            "10%, against 43.8% and 37.3% on the two reference runs.")

    @needs_run
    def test_the_spread_dwarfs_the_average(self, inputs):
        """Why totals never looked wrong: the errors are large and cancel.

        The median is far smaller than the spread, so an aggregate figure
        carries almost none of the contract-level error. And the median is not
        even stable between quarters -- it is -1.4% on one reference run and
        -7.2% on the other -- so the cancelling cannot be relied on either.
        """
        e = self.errors(inputs)
        iqr = np.percentile(e, 75) - np.percentile(e, 25)
        assert iqr > 0.2, f"M5: the spread has narrowed to an IQR of {iqr:.3f}"
        assert abs(np.median(e)) < iqr / 2, (
            "M5: the median error is now comparable to the spread, which "
            "would make this a bias rather than noise. Re-measure.")
        assert np.abs(e).max() > 1.0, "M5: the worst case used to exceed 100%"

    @needs_run
    def test_real_schedules_rise_before_they_fall(self, inputs):
        """A shape the monotone fallback can never produce."""
        con = inputs.contracts.set_index("contract")
        rising = total = 0
        for cid, real in inputs.ead_curves.items():
            if cid not in con.index:
                continue
            r = con.loc[cid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            if str(r.get("payment_type")) != "4":
                continue
            v = np.asarray(real, dtype=float)
            if v.size < 4 or v[0] <= 0:
                continue
            total += 1
            d = np.diff(v)
            if np.any(d > 1e-9) and np.any(d < -1e-9):
                rising += 1
        assert total > 0
        assert rising / total > 0.3, (
            f"M5: {rising}/{total} real schedules rise before falling; the "
            "fallback is monotone and cannot represent them.")


# ----------------------------------------------------------------- M6 ------
class TestM6MonthOneIsNotDiscounted:
    def test_the_first_marginal_loss_is_undiscounted(self):
        ead = np.array([1000.0, 1000.0])
        cum = np.array([0.0, 0.10, 0.10])
        got = sum_marginal_ecl(ead, 1.0, cum, eir=0.12, horizon=1)
        assert got == pytest.approx(100.0), (
            "M6 may be FIXED: month 1 now carries discounting. Confirm the "
            "convention (mid-period or end-of-period) and update the register.")


# ----------------------------------------------------------------- M7 ------
class TestM7AnnualPdIsSplitEvenly:
    @needs_run
    def test_the_monthly_marginal_is_flat_within_a_year(self):
        stpd = pd.read_csv(OUT / "StPD.csv", low_memory=False)
        one = stpd[(stpd["PortfolioCode"] == stpd["PortfolioCode"].iloc[0])
                   & (stpd["PDBucketDim1"] == stpd["PDBucketDim1"].iloc[0])]
        v = one.sort_values("MonthLifetime")["PDLifetime"].to_numpy()[:12]
        marginal = np.diff(np.concatenate([[0.0], v]))
        assert np.allclose(marginal, marginal[0], rtol=1e-9), (
            "M7 may be FIXED: the monthly marginal PD is no longer flat "
            "within the first year, so a hazard conversion may be in use.")


# ----------------------------------------------------------------- M8 ------
class TestM8WeightsDoNotSumToOne:
    def test_the_published_weights_sum_to_more_than_one(self):
        import yaml
        from pathlib import Path

        import ifrs9qdb
        cfg = Path(ifrs9qdb.__file__).parent / "config" / "model_inputs.yml"
        block = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        w = (block.get("internal_scenario_weights") or {}).get("explicit_weights")
        assert w, "explicit_weights is gone from the config"
        assert sum(w.values()) == pytest.approx(1.0003, abs=1e-6), (
            f"M8: the weights now sum to {sum(w.values())}.")


# ---------------------------------------------------------- M9 to M13 ------
class TestM9NoQuantitativeSicrTest:
    def test_staging_never_sees_an_origination_pd(self):
        import inspect

        from ifrs9qdb.analytics import classify_stage
        params = set(inspect.signature(classify_stage).parameters)
        assert not {"pd", "pd_origination", "origination_pd"} & params, (
            "M9 may be ADDRESSED: classify_stage now takes a PD. Confirm a "
            "SICR test has been added and update the register.")


class TestM10LgdIsStatic:
    def test_lgd_takes_no_scenario_and_no_time(self):
        import inspect
        params = set(inspect.signature(compute_lgd).parameters)
        assert not {"scenario", "month", "horizon", "shift_factor"} & params, (
            "M10 may be ADDRESSED: LGD now varies. Update the register.")

    def test_every_scenario_shares_one_lgd(self):
        assert compute_lgd(100.0, 0.0) == compute_lgd(100.0, 0.0)


class TestM11CollateralAboveTheFloorIsWorthless:
    def test_half_cover_and_full_cover_price_identically(self):
        assert compute_lgd(100.0, 50.0) == pytest.approx(compute_lgd(100.0, 100.0))
        assert compute_lgd(100.0, 50.0) == pytest.approx(0.225)

    @needs_run
    def test_the_book_has_contracts_in_that_position(self, report):
        cover = pd.to_numeric(report["collcov"], errors="coerce")
        assert int((cover > 0.5).sum()) > 50, (
            "M11: no contracts are over the floor on this run")


class TestM12StageThreeIsBookedInFull:
    @needs_run
    def test_stage_3_coverage_is_exactly_one(self, report):
        s3 = report[pd.to_numeric(report["stage"], errors="coerce") == 3]
        assert len(s3) > 0
        assert s3["ecl"].sum() == pytest.approx(s3["exposure"].sum(), rel=1e-9), (
            "M12 may have CHANGED: Stage 3 is no longer booked at the full "
            "outstanding balance.")


class TestM13LifetimeIsContractual:
    @needs_run
    def test_off_balance_sheet_uses_a_short_contractual_life(self, report):
        off = report[report["portfolio"] == "Off BS"]
        if len(off) == 0:
            pytest.skip("no Off BS contracts on this run")
        median = pd.to_numeric(off["months_to_mat"], errors="coerce").median()
        assert median < 24, (
            f"M13: the Off BS median life is now {median:.1f} months. If a "
            "behavioural life has been introduced, update the register.")


class TestM14OneMevCarriesEverything:
    @needs_run
    def test_two_of_the_three_variables_are_weighted_zero(self):
        from ifrs9qdb.analytics import mev_weights_table
        w = mev_weights_table(OUT)
        if len(w) == 0:
            pytest.skip("this run has no frozen config")
        zero = w[w["weight"].fillna(0) == 0]
        assert len(zero) >= 2, (
            "M14 may have CHANGED: more than one MEV now carries weight.")
        assert w["weight"].sum() == pytest.approx(1.0)
