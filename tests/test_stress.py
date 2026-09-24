"""Stress tests, run against a real run when one is available.

The properties here are the ones that were wrong in the R app at some point.
Each is a regression, not a smoke test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb import (
    StressSpec, apply_stress, load_engine_inputs, reverse_stress_all,
    roll_forward, tornado,
)
from ifrs9qdb.analytics import normalise
from ifrs9qdb.stress import advance_curve, conditional_pd, stretch_curve

from conftest import ref_output

RUN = ref_output()
has_run = pytest.mark.skipif(RUN is None or not (RUN / "StPD.csv").is_file(),
                             reason="no demo run available")


@pytest.fixture(scope="module")
def book():
    if RUN is None or not (RUN / "StPD.csv").is_file():
        return None, None, []
    inp = load_engine_inputs(RUN)
    rep = normalise(pd.read_csv(RUN / "FinalEclReport.csv", low_memory=False))
    return inp, rep, inp.internal_portfolios()


class TestCurveMaths:
    def test_stretch_preserves_the_endpoints(self):
        c = [1000, 800, 600, 400, 200, 0]
        for n in (12, 3, 6):
            out = stretch_curve(c, n)
            assert len(out) == n
            assert out[0] == pytest.approx(1000)
            assert out[-1] == pytest.approx(0)

    def test_advance_is_not_the_same_as_shortening(self):
        """Shortening keeps today's balance; advancing moves past it."""
        c = [1000, 800, 600, 400, 200, 0]
        assert list(advance_curve(c, 2)) == [600, 400, 200, 0]
        assert stretch_curve(c, 4)[0] == pytest.approx(1000)

    def test_conditional_pd_starts_at_zero_and_stays_a_probability(self):
        cum = np.array([0, .05, .10, .20, .35, .50])
        c = conditional_pd(cum, 2)
        assert c[0] == pytest.approx(0.0)
        assert ((c >= 0) & (c <= 1)).all()
        assert (np.diff(c) >= -1e-12).all()

    def test_conditional_pd_is_identity_at_zero(self):
        cum = np.array([0, .1, .2])
        assert np.allclose(conditional_pd(cum, 0), cum)


@has_run
class TestStressOnTheRealBook:
    def test_no_change_gives_exactly_zero(self, book):
        inp, rep, internal = book
        r = apply_stress(inp, rep, StressSpec(name="base", portfolios=internal))
        assert r["ok"]
        assert r["delta"] == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("kw,direction", [
        (dict(rating_notches=2), "up"),
        (dict(rating_notches=-2), "down"),
        (dict(pd_multiplier=1.5), "up"),
        (dict(collateral_pct=50), "up"),
        (dict(exposure_pct=150), "up"),
        (dict(maturity_years=2), "up"),
        (dict(lgd_base=0.55), "up"),
        (dict(default_top_n=3), "up"),
    ])
    def test_each_lever_moves_the_right_way(self, book, kw, direction):
        inp, rep, internal = book
        r = apply_stress(inp, rep, StressSpec(name="x", portfolios=internal, **kw))
        assert r["ok"]
        assert (r["delta"] > 1) if direction == "up" else (r["delta"] < -1)

    def test_asking_for_n_defaults_picks_n_customers(self, book):
        """The lever picks N CUSTOMERS by exposure, and keeps every one.

        Ranking by ECL put already-defaulted customers at the top -- they carry
        a 100% provision by construction -- so the lever delivered fewer
        defaults than asked for. Ranking by exposure fixed that.

        Fewer than N may actually MOVE, and that is correct: a customer already
        in Stage 3 is genuinely among the largest and stays in the list rather
        than being hidden, which is what the R does. What must not happen is
        the list itself being short.
        """
        inp, rep, internal = book
        r = apply_stress(inp, rep,
                         StressSpec(name="d", portfolios=internal, default_top_n=5))
        assert len(r["defaulted"]) == 5
        assert len(set(r["defaulted"])) == 5, "the same customer picked twice"

        scoped = rep[rep["portfolio"].isin(internal)]
        picked = scoped[scoped["customer"].isin(r["defaulted"])]
        already = picked.groupby("customer")["stage"].min().eq(3).sum()
        assert r["customers_moved"] == 5 - already

    def test_the_customers_picked_are_the_largest_by_exposure(self, book):
        inp, rep, internal = book
        r = apply_stress(inp, rep,
                         StressSpec(name="d", portfolios=internal, default_top_n=5))
        scoped = rep[rep["portfolio"].isin(internal) & (rep["exposure"] > 0)]
        biggest = (scoped.groupby("customer")["exposure"].sum()
                   .sort_values(ascending=False).head(5).index)
        assert set(r["defaulted"]) == set(biggest)

    def test_scope_confines_the_change(self, book):
        inp, rep, _ = book
        r = apply_stress(inp, rep, StressSpec(name="s", portfolios=["Off BS"],
                                              rating_notches=3))
        touched = r["by_portfolio"]
        moved = touched[touched["change"].abs() > 1]["portfolio"].tolist()
        assert moved == [] or moved == ["Off BS"]

    def test_maturity_reaches_contracts_with_a_supplied_curve(self, book):
        """The R lever only bit on the 24% without a supplied curve."""
        inp, rep, internal = book
        with_curve = [c for c in inp.contracts["contract"] if c in inp.ead_curves]
        assert len(with_curve) > 0
        r = apply_stress(inp, rep, StressSpec(name="m", portfolios=internal,
                                              maturity_years=2))
        changed = r["detail"][r["detail"]["contract"].isin(with_curve)]
        assert (changed["change"].abs() > 0).any()

    def test_roll_forward_reduces_exposure_and_provision(self, book):
        inp, rep, internal = book
        rf = roll_forward(inp, rep, months=12, portfolios=internal)
        assert rf["ok"]
        assert rf["exposure_after"] < rf["exposure_before"]
        assert rf["after"] < rf["before"]

    def test_tornado_ranks_by_absolute_effect(self, book):
        inp, rep, internal = book
        t = tornado(inp, rep, StressSpec(name="t", portfolios=internal))
        assert len(t) > 0
        assert (t["change"].abs().diff().dropna() <= 1e-6).all()

    def test_reverse_stress_finds_a_level_or_says_it_cannot(self, book):
        inp, rep, internal = book
        out = reverse_stress_all(inp, rep, 25.0,
                                 StressSpec(name="r", portfolios=internal))
        assert len(out) > 0
        assert set(out.columns) >= {"lever", "required", "found", "achieved_pct"}
        reached = out[out["found"]]
        if len(reached):
            assert reached["achieved_pct"].iloc[0] >= 24.0
