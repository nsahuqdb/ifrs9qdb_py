"""Analytics tests, run against the real July report where it is available.

The properties asserted here are the ones that were wrong at some point in the
R app, kept as regressions:

  * the walk must reconcile exactly, not approximately
  * every drill-down must sum back to its step
  * staging must follow the real rule, Tasdeer and contagion included
  * customer aggregation must not claim a customer sits in one portfolio
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import (
    classify_stage, concentration, customer_view, data_quality, ecl_walk,
    ecl_walk_detail, flow_profile, hhi, hhi_band, hhi_equivalent_n, lorenz_curve,
    movement_by, normalise, run_profile, stage2_triggers, stage_transitions,
    staging_distribution,
)

from conftest import ref_file

FIX = Path(__file__).parent / "fixtures"
# From the configured reference run when there is one, else a bundled fixture.
REPORT = ref_file("FinalEclReport.csv")
real_only = pytest.mark.skipif(
    REPORT is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")


@pytest.fixture(scope="module")
def rep():
    if REPORT.is_file():
        return normalise(pd.read_csv(REPORT, low_memory=False))
    return None


def synthetic():
    return normalise(pd.DataFrame({
        "Contract Id": ["C1", "C2", "C3", "C4", "C5", "C6"],
        "Customer Id": ["A", "A", "B", "C", "D", "D"],
        "Portfolio Code": ["Business Finance", "Off BS", "Business Finance",
                           "Tasdeer", "Business Finance", "Business Finance"],
        "Rating": ["QDB 5", "QDB 5", "QDB 8", "QDB 3", "QDB 6", "QDB 6"],
        "Ifrs Stage": [1, 1, 2, 2, 3, 3],
        "Exposure On Bal": [1000, 500, 2000, 300, 800, 200],
        "Cla Amount Onbal": [10, 5, 200, 9, 800, 200],
        "Past Due Days": [0, 0, 65, 0, 200, 200],
        "Watchlist Flag": [0, 0, 1, 0, 0, 0],
        "Local Flag 1": [0, 0, 0, 0, 0, 0],
        "Default Flag": [0, 0, 0, 0, 1, 1],
    }))


class TestNormalise:
    def test_missing_columns_do_not_raise(self):
        d = normalise(pd.DataFrame({"Contract Id": ["A"], "Exposure On Bal": [10],
                                    "Cla Amount Onbal": [1]}))
        assert len(d) == 1
        assert "pd" in d.columns and d["pd"].isna().all()

    def test_coverage_is_derived(self):
        d = synthetic()
        assert d.loc[d.contract == "C3", "coverage"].iloc[0] == pytest.approx(0.1)

    def test_column_names_match_across_punctuation(self):
        a = normalise(pd.DataFrame({"Contract Id": ["X"], "Exposure On Bal": [5],
                                    "Cla Amount Onbal": [1]}))
        b = normalise(pd.DataFrame({"ContractId": ["X"], "ExposureOnBal": [5],
                                    "ClaAmountOnbal": [1]}))
        assert a["exposure"].iloc[0] == b["exposure"].iloc[0]


class TestCustomerView:
    def test_aggregates_and_conserves_ecl(self):
        d = synthetic()
        cv = customer_view(d)
        assert len(cv) == 4
        assert cv["ecl"].sum() == pytest.approx(d["ecl"].sum())

    def test_does_not_claim_one_portfolio_when_there_are_several(self):
        """Naming the first of several was misleading and is now explicit."""
        cv = customer_view(synthetic())
        a = cv[cv.customer == "A"].iloc[0]
        assert a["portfolio"] == "2 portfolios"
        assert "Business Finance" in a["portfolios"] and "Off BS" in a["portfolios"]

    def test_stage_is_the_worst_facility(self):
        cv = customer_view(synthetic())
        assert cv.loc[cv.customer == "D", "stage"].iloc[0] == 3


class TestStaging:
    def test_the_rule_including_tasdeer_and_contagion(self):
        d = synthetic()
        st = classify_stage(d["dpd"], d["default_flag"], d["watchlist"],
                            local_any=None, portfolio=d["portfolio"],
                            customer=d["customer"], dpd_threshold=60)
        by_contract = dict(zip(d["contract"], st))
        assert by_contract["C4"] == 2      # Tasdeer, collectively assessed
        assert by_contract["C5"] == 3      # default flag
        assert by_contract["C3"] == 2      # DPD 65

    def test_contagion_lifts_a_customers_other_facilities(self):
        d = synthetic()
        st = classify_stage(d["dpd"], d["default_flag"], d["watchlist"], None,
                            d["portfolio"], d["customer"], contagion=True)
        off = classify_stage(d["dpd"], d["default_flag"], d["watchlist"], None,
                             d["portfolio"], d["customer"], contagion=False)
        assert (st >= off).all()

    def test_survives_a_missing_dpd_column(self):
        """A zero-length stage vector blanked four screens in the R app."""
        d = synthetic()
        st = classify_stage(None, None, None, None, d["portfolio"], d["customer"])
        assert len(st) == len(d)

    def test_survives_an_unknown_customer(self):
        d = synthetic()
        cust = d["customer"].copy()
        cust.iloc[0] = None
        st = classify_stage(d["dpd"], d["default_flag"], d["watchlist"], None,
                            d["portfolio"], cust)
        assert len(st) == len(d)

    def test_distribution_reports_customers_and_contracts(self):
        sd = staging_distribution(synthetic())
        assert set(sd["stage"]) == {"Stage 1", "Stage 2", "Stage 3"}
        assert (sd["customers"] <= sd["contracts"]).all()

    def test_triggers_cover_tasdeer_and_watchlist(self):
        t = stage2_triggers(synthetic(), dpd_threshold=60)
        any_rows = t[t.basis == "any"].set_index("trigger")
        assert any_rows.loc["Tasdeer (collective)", "contracts"] == 1
        assert any_rows.loc["Watchlist", "contracts"] == 1


class TestWalk:
    @staticmethod
    def perturb(d, seed):
        rng = np.random.default_rng(seed)
        x = d.copy()
        x["exposure"] = x["exposure"] * rng.uniform(0.7, 1.4, len(x))
        x["ecl"] = x["ecl"] * rng.uniform(0.5, 1.6, len(x))
        zero = rng.choice(len(x), max(1, len(x) // 30), replace=False)
        x.iloc[zero, x.columns.get_loc("exposure")] = 0.0
        x["coverage"] = np.where(x["exposure"] > 0,
                                 x["ecl"] / x["exposure"].replace(0, np.nan), 0.0)
        x["coverage"] = x["coverage"].fillna(0.0)
        mig = rng.choice(len(x), max(1, len(x) // 12), replace=False)
        x.iloc[mig, x.columns.get_loc("stage")] = \
            x.iloc[mig, x.columns.get_loc("stage")].map({1: 2, 2: 1, 3: 2}).fillna(2)
        return x.iloc[: int(len(x) * 0.92)]

    @real_only
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_reconciles_exactly_on_the_real_book(self, rep, seed):
        w = ecl_walk(self.perturb(rep, seed), rep)
        assert abs(w["residual"]) < 1e-4, f"residual {w['residual']}"

    @real_only
    def test_every_drill_down_sums_back_to_its_step(self, rep):
        prev = self.perturb(rep, 9)
        w = ecl_walk(prev, rep)
        det = ecl_walk_detail(prev, rep, n=10**9)
        steps = dict(zip(w["steps"]["label"], w["steps"]["amount"]))
        for label, frame in det.items():
            if len(frame) == 0:
                continue
            assert frame["amount"].sum() == pytest.approx(steps[label], abs=1e-4), \
                f"{label} detail does not tie to its step"

    def test_synthetic_walk_reconciles(self):
        prev = synthetic()
        curr = prev.copy()
        curr["ecl"] = curr["ecl"] * 1.4
        curr["coverage"] = np.where(curr["exposure"] > 0,
                                    curr["ecl"] / curr["exposure"], 0.0)
        curr = curr[curr.contract != "C6"]
        w = ecl_walk(prev, curr)
        assert abs(w["residual"]) < 1e-9


    def test_a_contract_listed_twice_is_two_positions(self):
        """The walk keeps both, matched in order as the ECL bridge does, so
        its opening and closing are the reports' totals."""
        prev = synthetic()
        extra = prev[prev["contract"] == "C3"].assign(exposure=100.0, ecl=7.0,
                                                      coverage=0.07)
        prev = pd.concat([prev, extra], ignore_index=True)
        curr = prev.iloc[:-1].copy()            # the second C3 position left
        curr["ecl"] = curr["ecl"] * 1.2
        curr["coverage"] = np.where(curr["exposure"] > 0,
                                    curr["ecl"] / curr["exposure"], 0.0)
        w = ecl_walk(prev, curr)
        assert w["opening"] == pytest.approx(prev["ecl"].sum())
        assert w["closing"] == pytest.approx(curr["ecl"].sum())
        steps = dict(zip(w["steps"]["label"], w["steps"]["amount"]))
        assert steps["Derecognised"] == pytest.approx(-7.0)
        assert w["counts"]["left"] == 1
        det = ecl_walk_detail(prev, curr)
        assert list(det["Derecognised"]["contract"]) == ["C3"]
        fl = flow_profile(prev, curr).set_index("flow")
        assert fl.loc["Derecognised", "contracts"] == 1
        assert fl.loc["Derecognised", "ecl"] == pytest.approx(7.0)


class TestConcentration:
    def test_hhi_bounds_and_bands(self):
        h = hhi(synthetic(), "customer")
        assert 0 < h <= 10000
        assert hhi_equivalent_n(2500) == pytest.approx(4.0)
        assert hhi_band(500)["band"] == "Diversified"
        assert hhi_band(2000)["band"] == "Moderately concentrated"
        assert hhi_band(3000)["band"] == "Highly concentrated"

    @real_only
    def test_share_rises_with_n_and_lorenz_ends_at_100(self, rep):
        c = concentration(rep, level="customer")
        assert (c["share"].diff().dropna() >= -1e-9).all()
        lz = lorenz_curve(rep, level="customer")
        assert lz["pct_ecl"].iloc[-1] == pytest.approx(100.0, abs=1e-6)


class TestAgainstTheRealBook:
    @real_only
    def test_customer_level_differs_materially_from_contract_level(self, rep):
        """Stage 2 is a third of customers but two thirds of contracts."""
        sd = staging_distribution(rep)
        s2 = sd[sd.stage == "Stage 2"].iloc[0]
        assert s2["customers"] < s2["contracts"]

    KNOWN_CHECKS = {
        "Exposure but zero ECL", "Negative or zero exposure",
        "ECL exceeds exposure", "Missing rating", "Missing lifetime PD",
        "Missing LGD", "Collateral coverage missing", "Collateral over 100%",
        "Implausible months on book",
    }

    @real_only
    def test_data_quality_reports_only_checks_that_fired(self, rep):
        """A check with no rows is dropped, as in the R.

        Which checks fire is a property of the BOOK, so naming one here would
        make the test pass or fail on which quarter's run is configured. What
        must hold is that every row is a known check, carries a count, and is
        ordered worst first.
        """
        dq = data_quality(rep)
        assert len(dq) > 0
        assert set(dq["check"]) <= self.KNOWN_CHECKS
        assert (dq["contracts"] > 0).all()
        rank = {"error": 0, "warn": 1, "info": 2}
        assert list(dq["severity"].map(rank)) == sorted(dq["severity"].map(rank))
        assert (dq["exposure"] >= 0).all()

    def test_every_check_in_the_r_set_is_implemented(self):
        """The port dropped "Missing LGD" once; this is why it cannot again."""
        book = pd.DataFrame({
            "contract": ["a", "b"], "customer": ["c1", "c2"],
            "exposure": [0.0, 100.0], "ecl": [0.0, 0.0], "stage": [1, 1],
            "rating": [None, ""], "pd": [np.nan, np.nan], "lgd": [np.nan, np.nan],
            "collcov": [np.nan, 1.5], "mob": [-1.0, 5000.0],
        })
        fired = set(data_quality(book)["check"])
        assert fired == self.KNOWN_CHECKS - {"ECL exceeds exposure"}

    def test_a_blank_rating_counts_as_missing(self):
        """R treats "" and NA alike; matching only NA under-reports."""
        book = pd.DataFrame({
            "contract": ["a", "b"], "customer": ["c1", "c2"],
            "exposure": [100.0, 100.0], "ecl": [1.0, 1.0], "stage": [1, 1],
            "rating": [None, "   "], "pd": [0.1, 0.1], "lgd": [0.5, 0.5],
            "collcov": [0.5, 0.5], "mob": [10.0, 10.0],
        })
        dq = data_quality(book).set_index("check")
        assert dq.loc["Missing rating", "contracts"] == 2

    @real_only
    def test_profiles_conserve_the_total(self, rep):
        for by in ("portfolio", "stage"):
            p = run_profile(rep, by)
            assert p["ecl"].sum() == pytest.approx(rep["ecl"].sum(), rel=1e-9)

    @real_only
    def test_movement_by_is_zero_against_itself(self, rep):
        m = movement_by(rep, rep, "portfolio")
        assert m["change"].abs().max() == pytest.approx(0.0, abs=1e-6)

    @real_only
    def test_stage_transitions_are_customer_counts(self, rep):
        t = stage_transitions(rep, rep, by_customer=True)
        assert "customers" in t.columns
        assert (t["from"] == t["to"]).all()   # unchanged against itself
