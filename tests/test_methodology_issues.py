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

from conftest import needs_src_inputs, ref_output, src_inputs

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
class TestM2TheTwoFactorConventions:
    """M2: a STRESS factor and a SHIFT factor, opposite by design.

    Not a defect. These pin the two conventions so that a future reader who
    notices they disagree finds the answer here instead of re-deriving it —
    which has now happened twice.
    """

    def test_a_stress_factor_raises_pd_when_it_is_positive(self):
        """Internal: probit gap between the fitted PD and the anchor."""
        ttc = 0.146
        worse = float(norm.cdf(norm.ppf(ttc) + 0.32))
        better = float(norm.cdf(norm.ppf(ttc) - 0.26))
        assert worse > ttc > better

    def test_a_shift_factor_lowers_pd_when_it_is_positive(self):
        """External: probit of where growth sits in its own history."""
        ttc = 0.02
        better = basel_asrf_pit(ttc, +1.77)
        worse = basel_asrf_pit(ttc, -1.15)
        assert worse > ttc > better

    def test_the_internal_chain_produces_a_stress_factor(self, static):
        """A downturn must come out POSITIVE on the internal scale."""
        import yaml
        from pathlib import Path

        import ifrs9qdb
        from ifrs9qdb.etl.macro import (combine_sf, compute_logit_pds,
                                        compute_pds_from_logits,
                                        compute_per_mev_sf, stress_mevs)

        cfg = Path(ifrs9qdb.__file__).parent / "config"
        mc = yaml.safe_load((cfg / "model.yml").read_text(encoding="utf-8"))
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text(encoding="utf-8"))
        comp = mc["models"]["internal_v4_production"]["mev_components"]
        variables = mc.get("variables", {})
        specs = [{"standard_deviation": c["standard_deviation"],
                  "stress_unit_multiplier": variables.get(
                      c["variable"], {}).get("stress_unit_multiplier", 1),
                  "intercept": c["intercept"],
                  "coefficient": c["coefficient"]} for c in comp]
        weights = np.array([c["weight"] for c in comp], dtype=float)
        anchor = float(mc["ttc_anchor_pd"])
        block = mi["mev_forecasts"]["forecasts"]
        forecasts = np.array([np.asarray(block[y], dtype=float)
                              for y in sorted(block, key=int)])

        def factor(z):
            stressed = stress_mevs(forecasts, z, specs)
            fitted = compute_pds_from_logits(compute_logit_pds(stressed, specs))
            return combine_sf(compute_per_mev_sf(fitted, anchor), weights)[0]

        down, up = factor(-1.281552), factor(1.281552)
        assert down > 0 > up, (
            f"M2: the internal factor is meant to be a STRESS factor — "
            f"positive in a downturn. Got {down:+.4f} down, {up:+.4f} up.")
        assert norm.cdf(norm.ppf(anchor) + down) > anchor

    def test_the_external_chain_produces_a_shift_factor(self, static):
        """A downturn must come out NEGATIVE on the external scale, and the
        Vasicek form must still raise the PD."""
        from ifrs9qdb.etl.macro import (external_combined_sf,
                                        gcc_weighted_history)

        history = gcc_weighted_history(static.get("gcc_real_gdp_growth"),
                                       static.get("gcc_gdp_current_prices"))
        sd = float(np.std(np.asarray(history, dtype=float), ddof=1))
        base = float(np.mean(np.asarray(history, dtype=float)))
        down = float(external_combined_sf([base - 1.281552 * sd], history)[0])
        up = float(external_combined_sf([base + 1.281552 * sd], history)[0])
        assert down < 0 < up, (
            f"M2: the external factor is meant to be a SHIFT factor — "
            f"negative in a downturn. Got {down:+.4f} down, {up:+.4f} up.")
        assert basel_asrf_pit(0.02, down) > 0.02 > basel_asrf_pit(0.02, up)

    def test_both_scales_raise_the_provision_in_a_downturn(self, static):
        """The property that matters, and the one the old wording denied."""
        from ifrs9qdb.etl.macro import (external_combined_sf,
                                        gcc_weighted_history)

        history = gcc_weighted_history(static.get("gcc_real_gdp_growth"),
                                       static.get("gcc_gdp_current_prices"))
        arr = np.asarray(history, dtype=float)
        down_shift = float(external_combined_sf(
            [arr.mean() - 1.281552 * arr.std(ddof=1)], history)[0])
        assert basel_asrf_pit(0.02, down_shift) > 0.02
        assert float(norm.cdf(norm.ppf(0.146) + 0.3159)) > 0.146


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
    def test_month_zero_is_the_current_exposure(self, inputs):
        """The derivation rule, confirmed on every supplied curve.

        build_lifetime_parameter_other sets month 0 to AccountMaster.OnBalance
        and every later month to BALANCE + REPAYMENT at the first scheduled
        payment beyond it. That forward step lookup is what produces the
        staircase, and it is why a smooth parametric curve cannot match it.
        """
        con = inputs.contracts.set_index("contract")
        checked = mismatched = 0
        for cid, real in inputs.ead_curves.items():
            if cid not in con.index:
                continue
            r = con.loc[cid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            balance = r.get("on_balance")
            if not np.isfinite(balance or np.nan) or balance <= 0:
                continue
            checked += 1
            if abs(np.asarray(real, dtype=float)[0] - balance) > 0.01 * balance:
                mismatched += 1
        assert checked > 1000
        assert mismatched == 0, (
            f"M5: {mismatched} supplied curves no longer open at OnBalance. "
            "The derivation rule may have changed.")

    @needs_run
    def test_the_fallback_horizon_runs_past_the_schedule(self, inputs):
        """Duration is wrong before shape is even considered."""
        con = inputs.contracts.set_index("contract")
        longer = same = 0
        for cid, real in inputs.ead_curves.items():
            if cid not in con.index:
                continue
            r = con.loc[cid]
            if isinstance(r, pd.DataFrame):
                r = r.iloc[0]
            gap = int(r["months_to_mat"] or 3) - len(real)
            if gap > 0:
                longer += 1
            elif gap == 0:
                same += 1
        assert longer > 500, (
            "M5 may be IMPROVED: the fallback horizon no longer overruns the "
            f"schedule on many contracts ({longer}).")

    @needs_run
    def test_some_shapes_are_used_but_never_observed(self, inputs):
        """A shape nobody can check is an assumption, not a calibration."""
        from ifrs9qdb.engine import resolve_ead_shape

        con = inputs.contracts.copy()
        con["supplied"] = con["contract"].isin(inputs.ead_curves)
        groups = con.groupby([con["portfolio"].astype(str),
                              con["payment_type"].astype(str)])
        unchecked = 0
        for (portfolio, ptype), g in groups:
            if g["supplied"].sum() == 0:
                unchecked += int((~g["supplied"]).sum())
                resolve_ead_shape(ptype, portfolio)  # must still resolve
        assert unchecked > 500, (
            "M5 may be ADDRESSED: most shape choices now have at least one "
            "observed schedule behind them.")

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


# ---------------------------------------------------------------- M15-17 ----
# These three came from reading the raw Oracle extracts. The synthetic cases
# run everywhere; the ones that need real data skip with a reason.
def _schedule(rows):
    """A RepaymentSchedule frame in the source system's own column names."""
    return pd.DataFrame(rows, columns=["KEY_1", "POST_DATE", "START_DAT",
                                       "PRINCE_DUE", "PROJ_INT", "REPAYMENT",
                                       "BALANCE"])


def _accounts(rows):
    return pd.DataFrame(rows, columns=["ContractId", "OnBalance"])


def _curve(sched, accounts, ref="2026-06-09"):
    from ifrs9qdb.etl.lifetime import build_lifetime_parameter_other
    lp = build_lifetime_parameter_other(sched, accounts, pd.Timestamp(ref), ref)
    lp["MonthLifetime"] = pd.to_numeric(lp["MonthLifetime"])
    lp["EADLifetime"] = pd.to_numeric(lp["EADLifetime"])
    return lp.sort_values("MonthLifetime").reset_index(drop=True)


class TestM15EverythingAfter2029IsDiscarded:
    """M15: a two-digit-year pivot puts 2030+ payments in 1930+, and the
    derivation drops them as historical. The ECL horizon is the curve's own
    length, so the lifetime ends in December 2029."""

    # A loan maturing 2031-06, quarterly, whose last five payments come back
    # from the extract with their century lost.
    @staticmethod
    def wrapped_loan():
        rows, bal = [], 1000.0
        for k, d in enumerate(["2026-09-09", "2026-12-09", "2027-03-09",
                               "2030-03-09", "2030-06-09", "2031-06-09"]):
            bal -= 150.0
            y = int(d[:4])
            shown = f"{y - 100}{d[4:]}" if y >= 2030 else d
            rows.append(("C1", "2026-06-09", pd.Timestamp(shown),
                         150.0, 10.0, 160.0, max(bal, 0.0)))
        return _schedule(rows), _accounts([("C1", 1000.0)])

    def test_the_wrapped_payments_are_dropped(self):
        sched, acc = self.wrapped_loan()
        lp = _curve(sched, acc)
        assert lp["MonthLifetime"].max() == 8, (
            "M15 may be FIXED: the curve no longer stops at the last payment "
            f"before 2030 (it now reaches month {lp['MonthLifetime'].max()}). "
            "If the year pivot is handled, update METHODOLOGY_ISSUES.md.")

    def test_the_curve_ends_with_exposure_still_outstanding(self):
        """The defect that costs money: the curve does not run down to zero,
        it simply stops."""
        sched, acc = self.wrapped_loan()
        lp = _curve(sched, acc)
        assert lp["EADLifetime"].iloc[-1] > 0.0

    def test_repairing_the_year_restores_the_full_life(self):
        """The same loan with four-digit years prices over 60 months, not 9."""
        sched, acc = self.wrapped_loan()
        fixed = sched.copy()
        fixed["START_DAT"] = [d.replace(year=d.year + 100) if d.year < 1950 else d
                              for d in fixed["START_DAT"]]
        assert _curve(fixed, acc)["MonthLifetime"].max() == 59

    @needs_run
    def test_no_supplied_curve_in_the_run_passes_december_2029(self, inputs):
        """The live fingerprint. Nothing about the book makes December 2029
        special; the pivot does."""
        if not inputs.ead_curves:
            pytest.skip("this run supplies no EAD curves")
        ext = pd.to_datetime(inputs.contracts["extract_date"], errors="coerce").max()
        if pd.isna(ext):
            pytest.skip("no extract date on this run")
        cliff = (2029 - ext.year) * 12 + (12 - ext.month) - 1
        longest = max(len(c) for c in inputs.ead_curves.values()) - 1
        assert longest == cliff, (
            f"M15: the longest supplied curve reaches month {longest}; the "
            f"December-2029 cliff for this extract is month {cliff}. If they "
            "no longer coincide the pivot may be handled — re-check M15.")

    @needs_run
    def test_the_contracts_stopping_there_all_mature_later(self, inputs):
        if not inputs.ead_curves:
            pytest.skip("this run supplies no EAD curves")
        ext = pd.to_datetime(inputs.contracts["extract_date"], errors="coerce").max()
        cliff = (2029 - ext.year) * 12 + (12 - ext.month) - 1
        at = {c for c, v in inputs.ead_curves.items() if len(v) - 1 == cliff}
        if not at:
            pytest.skip("no curve ends at the cliff on this run")
        live = inputs.contracts[inputs.contracts["contract"].isin(at)]
        beyond = pd.to_numeric(live["months_to_mat"], errors="coerce") > cliff + 1
        assert beyond.mean() > 0.9, (
            f"M15: only {beyond.mean():.0%} of the contracts stopping at the "
            "cliff mature after it. Expected nearly all of them.")
        assert len(at) > 100


class TestM16InterestIsAddedToTheBalance:
    """M16: `BALANCE + REPAYMENT` is the outstanding balance plus that
    instalment's projected interest, not the balance before the payment."""

    # Monthly payments at months 1, 2 and 3. Month 1 of the curve looks up the
    # month-2 payment, whose balance-after is 700 and whose interest is 10 --
    # so the balance standing before it is 850 and the curve holds 860.
    @staticmethod
    def three_payments():
        return (_schedule([
            ("C1", "2026-06-09", pd.Timestamp("2026-07-09"), 150.0, 12.0, 162.0, 850.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-08-09"), 150.0, 10.0, 160.0, 700.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-09-09"), 150.0, 8.0, 158.0, 550.0)]),
            _accounts([("C1", 1000.0)]))

    def test_the_curve_carries_the_interest(self):
        sched, acc = self.three_payments()
        at1 = float(_curve(sched, acc).query("MonthLifetime == 1")["EADLifetime"].iloc[0])
        assert at1 == pytest.approx(860.0), (
            f"M16 may be FIXED: month 1 now holds {at1:.2f}. The balance "
            "standing before that payment is 850.00; the rule gives 860.00.")

    def test_the_overstatement_is_exactly_the_projected_interest(self):
        sched, acc = self.three_payments()
        at1 = float(_curve(sched, acc).query("MonthLifetime == 1")["EADLifetime"].iloc[0])
        true_prior = 700.0 + 160.0 - 10.0        # = 850, the actual outstanding
        assert at1 - true_prior == pytest.approx(10.0)

    def test_an_accrual_row_is_not_affected(self):
        """A grace-period row carries REPAYMENT = 0, so the rule reduces to
        the balance and is right. M17 explains why that matters."""
        sched = _schedule([("C1", "2026-06-09", pd.Timestamp(d), 0.0, 0.0, 0.0, b)
                           for d, b in [("2026-07-09", 1100.0),
                                        ("2026-08-09", 1200.0),
                                        ("2026-09-09", 1300.0)]])
        lp = _curve(sched, _accounts([("C1", 1000.0)]))
        assert float(lp.query("MonthLifetime == 1")["EADLifetime"].iloc[0]) \
            == pytest.approx(1200.0)

    @needs_src_inputs
    def test_the_raw_extract_satisfies_the_other_identity(self):
        """On the source data, subtracting the projected interest is what
        makes the roll-forward hold."""
        src = src_inputs()
        f = next((p for p in src.iterdir()
                  if p.stem.lower() == "repaymentschedule"), None)
        if f is None:
            pytest.skip("no RepaymentSchedule in IFRS9_SRC_INPUTS")
        rs = pd.read_excel(f) if f.suffix.startswith(".xls") else pd.read_csv(f)
        rs["cid"] = rs["KEY_1"].astype(str)
        rs = rs[rs["START_DAT"].dt.year >= 1950] \
            .sort_values(["cid", "START_DAT"]).reset_index(drop=True)
        prev = rs.groupby("cid", sort=False)["BALANCE"].shift(1)
        m = prev.notna()
        tol = np.maximum(1.0, prev.abs() * 1e-6)
        with_int = ((prev - (rs["BALANCE"] + rs["REPAYMENT"]
                             - rs["PROJ_INT"])).abs() <= tol) & m
        without = ((prev - (rs["BALANCE"] + rs["REPAYMENT"])).abs() <= tol) & m
        assert with_int.sum() > 3 * without.sum(), (
            f"M16: -PROJ_INT holds {with_int.sum()} times, the rule in use "
            f"{without.sum()}. The source convention may have changed.")


class TestM17TheMonthlyGridLosesPayments:
    """M17: one value per contract-month, and a quarter of the rows are not
    repayments at all."""

    def test_a_duplicated_row_is_correctly_absorbed(self):
        """The withdrawn half of M17. Two identical July rows, then August and
        September: holding one value per month is RIGHT here, because the
        second row is the same payment twice. All 933 real cases are like
        this, so the grid is not losing anything."""
        dupe = ("C1", "2026-06-09", pd.Timestamp("2026-07-05"),
                150.0, 0.0, 150.0, 850.0)
        sched = _schedule([
            dupe, dupe,
            ("C1", "2026-06-09", pd.Timestamp("2026-08-05"), 150.0, 0.0, 150.0, 700.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-09-05"), 150.0, 0.0, 150.0, 550.0)])
        lp = _curve(sched, _accounts([("C1", 1100.0)]))
        assert len(lp[lp["MonthLifetime"] == 1]) == 1, "one value per month"
        assert lp["EADLifetime"].round(2).tolist() == [1100.0, 850.0, 700.0]

    def test_a_grid_would_still_lose_a_genuinely_split_month(self):
        """The latent limitation, pinned so the withdrawal is not read as
        'the grid is fine'. This book has no such schedule; a fortnightly one
        would hit it."""
        sched = _schedule([
            ("C1", "2026-06-09", pd.Timestamp("2026-07-05"), 150.0, 0.0, 150.0, 850.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-07-20"), 150.0, 0.0, 150.0, 700.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-08-05"), 150.0, 0.0, 150.0, 550.0),
            ("C1", "2026-06-09", pd.Timestamp("2026-09-05"), 150.0, 0.0, 150.0, 400.0)])
        lp = _curve(sched, _accounts([("C1", 1100.0)]))
        held = lp["EADLifetime"].round(2).tolist()
        assert 850.0 not in held, (
            f"the curve {held} now carries the 20 July payment (700 + 150)")
        assert held == [1100.0, 700.0, 550.0]

    def test_a_rising_curve_can_be_correct(self):
        """The reason M16 must not be fixed by forcing the curve monotone."""
        sched = _schedule([("C1", "2026-06-09", pd.Timestamp(d),
                            0.0, 0.0, 0.0, b)
                           for d, b in [("2026-07-09", 1100.0),
                                        ("2026-08-09", 1200.0),
                                        ("2026-09-09", 1300.0)]])
        lp = _curve(sched, _accounts([("C1", 1000.0)]))
        e = lp["EADLifetime"].to_numpy(dtype=float)
        assert (np.diff(e) >= 0).all() and e[-1] > e[0], (
            "a grace-period facility should show exposure growing")
