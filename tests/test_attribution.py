"""Attribution: does the split add up, and does it say the right thing.

Two properties carry these. The factor attribution is rescaled, so its effects
must sum EXACTLY to the observed move whatever the raw effects were. The
coverage bridge is an algebraic identity, so mix plus rate must equal the real
change in coverage to floating point -- there is no rescaling to hide behind.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import (
    coverage_bridge, ecl_factor_attribution, normalise,
)

from conftest import prev_file, ref_file

REPORT = ref_file("FinalEclReport.csv")
PREV = prev_file("FinalEclReport.csv")
real_only = pytest.mark.skipif(
    REPORT is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")


def _load(p):
    return normalise(pd.read_csv(p, low_memory=False))


def _perturb(d, seed):
    """A plausible prior quarter: exposures and risk move, some names leave."""
    rng = np.random.default_rng(seed)
    x = d.copy()
    x["exposure"] = x["exposure"] * rng.uniform(0.7, 1.4, len(x))
    x["pd"] = x["pd"] * rng.uniform(0.6, 1.5, len(x))
    x["lgd"] = np.clip(x["lgd"] * rng.uniform(0.9, 1.1, len(x)), 0, 1)
    x["ecl"] = x["ecl"] * rng.uniform(0.5, 1.6, len(x))
    x["coverage"] = np.where(x["exposure"] > 0,
                             x["ecl"] / x["exposure"].replace(0, np.nan), 0.0)
    x["coverage"] = x["coverage"].fillna(0.0)
    return x.iloc[: int(len(x) * 0.94)]


@pytest.fixture(scope="module")
def pair():
    if REPORT is None:
        return None, None
    curr = _load(REPORT)
    return (_load(PREV) if PREV is not None else _perturb(curr, 7)), curr


class TestFactorAttribution:
    @real_only
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_effects_sum_to_the_actual_move(self, pair, seed):
        _, curr = pair
        prev = _perturb(curr, seed)
        fa = ecl_factor_attribution(prev, curr)
        assert set(fa["factor"]) == {"Exposure", "PD", "LGD"}
        assert fa["effect"].sum() == pytest.approx(
            fa["actual_change"].iloc[0], rel=1e-9, abs=1e-6)

    @real_only
    def test_it_says_it_is_indicative(self, pair):
        """The rescaling is the reason. Dropping the flag would hide it."""
        fa = ecl_factor_attribution(*pair)
        assert fa["indicative"].all()
        assert (fa["effect"] != fa["raw_effect"]).any()

    def test_a_pure_exposure_move_lands_on_exposure(self):
        prev = pd.DataFrame({"contract": ["a", "b"], "exposure": [100.0, 200.0],
                             "pd": [0.1, 0.2], "lgd": [0.45, 0.45],
                             "ecl": [4.5, 18.0]})
        curr = prev.assign(exposure=[200.0, 400.0], ecl=[9.0, 36.0])
        fa = ecl_factor_attribution(prev, curr).set_index("factor")
        assert fa.loc["Exposure", "effect"] == pytest.approx(22.5)
        assert fa.loc["PD", "effect"] == pytest.approx(0.0)
        assert fa.loc["LGD", "effect"] == pytest.approx(0.0)

    def test_contracts_that_start_at_zero_are_left_out(self):
        """No proportional story: every effect would be the whole move."""
        prev = pd.DataFrame({"contract": ["a", "b"], "exposure": [0.0, 200.0],
                             "pd": [0.1, 0.2], "lgd": [0.45, 0.45],
                             "ecl": [0.0, 18.0]})
        curr = prev.assign(exposure=[100.0, 200.0], ecl=[4.5, 18.0])
        assert ecl_factor_attribution(prev, curr)["contracts"].iloc[0] == 1

    def test_an_empty_side_returns_empty_not_an_error(self):
        d = pd.DataFrame(columns=["contract", "exposure", "pd", "lgd", "ecl"])
        assert len(ecl_factor_attribution(d, d)) == 0
        assert len(ecl_factor_attribution(None, d)) == 0


class TestCoverageBridge:
    @staticmethod
    def _coverage_pp(d):
        return 100 * d["ecl"].sum() / d["exposure"].sum()

    @real_only
    @pytest.mark.parametrize("by", ["portfolio", "stage", "rating"])
    def test_mix_plus_rate_is_the_whole_change(self, pair, by):
        prev, curr = pair
        cb = coverage_bridge(prev, curr, by=by)
        actual = self._coverage_pp(curr) - self._coverage_pp(prev)
        assert cb["total_effect"].sum() == pytest.approx(actual, abs=1e-9)
        assert (cb["mix_effect"] + cb["rate_effect"]).sum() == \
            pytest.approx(actual, abs=1e-9)

    @real_only
    def test_stage_groups_are_labelled_as_whole_numbers(self, pair):
        """Stage is numeric in the frame; "1.0" is not a stage anybody names."""
        cb = coverage_bridge(*pair, by="stage")
        assert set(cb["group"]) <= {"1", "2", "3", "(unassigned)"}

    @real_only
    def test_the_largest_mover_comes_first(self, pair):
        cb = coverage_bridge(*pair)
        assert (cb["total_effect"].abs().diff().dropna() <= 1e-12).all()

    def test_a_pure_mix_shift_has_no_rate_effect(self):
        """Nothing about the segments changed -- only how much sits in each."""
        prev = pd.DataFrame({"portfolio": ["A", "B"],
                             "exposure": [1000.0, 1000.0], "ecl": [10.0, 50.0]})
        curr = pd.DataFrame({"portfolio": ["A", "B"],
                             "exposure": [500.0, 1500.0], "ecl": [5.0, 75.0]})
        cb = coverage_bridge(prev, curr)
        assert cb["rate_effect"].abs().max() == pytest.approx(0.0, abs=1e-12)
        assert cb["mix_effect"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_a_pure_rate_shift_has_no_mix_effect(self):
        prev = pd.DataFrame({"portfolio": ["A", "B"],
                             "exposure": [1000.0, 1000.0], "ecl": [10.0, 50.0]})
        curr = prev.assign(ecl=[20.0, 50.0])
        cb = coverage_bridge(prev, curr)
        assert cb["mix_effect"].abs().max() == pytest.approx(0.0, abs=1e-12)
        assert cb["total_effect"].sum() == pytest.approx(0.5, abs=1e-9)

    def test_an_unknown_grouping_is_refused_not_silently_empty(self):
        """A typo must not look like "no movement"."""
        d = pd.DataFrame({"portfolio": ["A"], "exposure": [1.0], "ecl": [0.1]})
        with pytest.raises(ValueError, match="portfolio"):
            coverage_bridge(d, d, by="protfolio")


@pytest.fixture(scope="module")
def runs():
    """Two real runs, each with the engine inputs that priced it."""
    from conftest import prev_run, ref_output
    from ifrs9qdb.inputs import load_engine_inputs
    out_b = ref_output()
    run_a = prev_run()
    if out_b is None or run_a is None:
        pytest.skip("set IFRS9_REF_RUN and IFRS9_REF_RUN_PREV to two runs")
    out_a = run_a / "Output" if (run_a / "Output").is_dir() else run_a
    return (load_engine_inputs(out_a), _load(out_a / "FinalEclReport.csv"),
            load_engine_inputs(out_b), _load(out_b / "FinalEclReport.csv"))


class TestExactAttribution:
    """Through the engine, so the residual is zero rather than small."""

    def test_it_reconciles_exactly(self, runs):
        """Exact by construction: each effect is a difference between two
        repricings, so only the float accumulation over several thousand
        contracts separates the sum from the move. The tolerance is relative
        to the balance being decomposed, not a fixed number of currency units
        -- an absolute tolerance would pass or fail on the size of the book."""
        from ifrs9qdb.analytics import factor_attribution_exact
        r = factor_attribution_exact(*runs)
        assert r, "nothing was attributed"
        scale = max(abs(r["opening"]), abs(r["closing"]), 1.0)
        assert abs(r["residual"]) / scale < 1e-12, r["residual"]
        assert r["effects"]["effect"].sum() == pytest.approx(
            r["closing"] - r["opening"], rel=1e-12)

    def test_every_matched_contract_is_priced_or_declared(self, runs):
        """A contract the engine cannot price must not be folded into a factor."""
        from ifrs9qdb.analytics import factor_attribution_exact
        r = factor_attribution_exact(*runs)
        assert r["covered"] <= r["contracts"]
        if r["covered"] < r["contracts"]:
            assert r["uncovered"] != 0 or r["covered"] == r["contracts"]

    def test_the_factors_are_the_engines_own_ingredients(self, runs):
        from ifrs9qdb.analytics import factor_attribution_exact
        r = factor_attribution_exact(*runs)
        assert list(r["effects"]["factor"]) == ["Horizon", "EAD", "PD", "LGD"]

    def test_a_run_against_itself_moves_nothing(self, runs):
        from ifrs9qdb.analytics import factor_attribution_exact
        ia, ra, _, _ = runs
        r = factor_attribution_exact(ia, ra, ia, ra)
        assert r["effects"]["effect"].abs().max() == pytest.approx(0.0, abs=1e-6)
        assert r["opening"] == pytest.approx(r["closing"])

    def test_missing_inputs_are_explained_not_silently_empty(self):
        from ifrs9qdb.analytics import (factor_attribution_diagnosis,
                                        factor_attribution_exact)
        d = pd.DataFrame({"contract": ["a"], "exposure": [1.0], "ecl": [0.1]})
        assert factor_attribution_exact(None, d, None, d) == {}
        msg = factor_attribution_diagnosis(None, d, None, d)
        assert msg and "prior" in msg


class TestAnOverlayIsItsOwnLine:
    """A post-model overlay moves the provision without moving a factor. The
    split used to scale the factors to the whole move -- or, with nothing else
    changed, leave every factor at zero against it."""

    def _book(self, **kw):
        base = pd.DataFrame({"contract": ["1", "2"], "exposure": [100.0, 200.0],
                             "pd": [0.1, 0.2], "lgd": [0.5, 0.5],
                             "ecl": [5.0, 20.0], "overlay": [0.0, 0.0]})
        return base.assign(**kw)

    def test_an_overlay_alone_is_the_whole_move(self):
        from ifrs9qdb.analytics.attribution import ecl_factor_attribution
        out = ecl_factor_attribution(self._book(),
                                     self._book(ecl=[5.5, 22.0], overlay=[0.5, 2.0]))
        eff = out.set_index("factor")["effect"]
        assert list(out["factor"]) == ["Exposure", "PD", "LGD", "Overlay"]
        assert eff["Overlay"] == pytest.approx(2.5)
        assert eff.sum() == pytest.approx(out["actual_change"].iloc[0])

    def test_the_factors_explain_the_model_move_beside_it(self):
        from ifrs9qdb.analytics.attribution import ecl_factor_attribution
        curr = self._book(exposure=[110.0, 200.0], ecl=[6.5, 21.0],
                          overlay=[1.0, 1.0])
        out = ecl_factor_attribution(self._book(), curr)
        eff = out.set_index("factor")["effect"]
        assert eff["Overlay"] == pytest.approx(2.0)
        assert eff[["Exposure", "PD", "LGD"]].sum() == pytest.approx(0.5)
        assert eff.sum() == pytest.approx(2.5)

    def test_no_overlay_no_line(self):
        from ifrs9qdb.analytics.attribution import ecl_factor_attribution
        out = ecl_factor_attribution(self._book(), self._book(exposure=[110.0, 200.0],
                                                              ecl=[5.5, 20.0]))
        assert list(out["factor"]) == ["Exposure", "PD", "LGD"]
