"""What-if rules: who each one claims, and what happens when two claim one.

Rules do not stack. A contract caught by two of them has no defined answer, so
it is excluded and listed rather than getting whichever rule happened to be
applied last. The tests that matter here are the ones about that boundary.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import normalise
from ifrs9qdb.stress import (
    Rule, filter_rating_type, match_rules, order_by_rating, reprice_rules,
)

from conftest import ref_file, ref_output

REPORT = ref_file("FinalEclReport.csv")
OUT = ref_output()
real_only = pytest.mark.skipif(
    REPORT is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")
needs_inputs = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


@pytest.fixture(scope="module")
def rep():
    return normalise(pd.read_csv(REPORT, low_memory=False)) \
        if REPORT is not None else None


@pytest.fixture(scope="module")
def inputs():
    if OUT is None:
        return None
    from ifrs9qdb.inputs import load_engine_inputs
    return load_engine_inputs(OUT)


def synthetic():
    return normalise(pd.DataFrame({
        "Contract Id": ["C1", "C2", "C3", "C4"],
        "Customer Id": ["A", "A", "B", "C"],
        "Portfolio Code": ["Business Finance", "Off BS",
                           "Business Finance", "Tasdeer"],
        "Rating": ["QDB 5", "QDB 5", "QDB 8", "QDB 3"],
        "Ifrs Stage": [1, 1, 2, 2],
        "Exposure On Bal": [1000.0, 500.0, 2000.0, 300.0],
        "Cla Amount Onbal": [10.0, 5.0, 200.0, 9.0],
    }))


class TestMatching:
    def test_a_contract_goes_to_the_one_rule_that_claims_it(self):
        rules = [Rule(label="BF", portfolios=["Business Finance"]),
                 Rule(label="Off", portfolios=["Off BS"])]
        m = match_rules(synthetic(), rules)
        assert list(m["assignment"].dropna()) == [0, 1, 0]
        assert len(m["conflicts"]) == 0

    def test_two_rules_claiming_one_contract_is_a_conflict(self):
        """Applying both would make the answer depend on the order; applying
        one silently would hide the other."""
        rules = [Rule(label="BF", portfolios=["Business Finance"]),
                 Rule(label="Stage 1", stages=[1])]
        m = match_rules(synthetic(), rules)
        assert len(m["conflicts"]) == 1
        assert m["conflicts"]["contract"].iloc[0] == "C1"
        assert m["conflicts"]["rules"].iloc[0] == "BF + Stage 1"

    def test_a_conflicted_contract_is_assigned_to_neither(self):
        rules = [Rule(label="BF", portfolios=["Business Finance"]),
                 Rule(label="Stage 1", stages=[1])]
        m = match_rules(synthetic(), rules)
        assert pd.isna(m["assignment"].iloc[0])
        assert m["assignment"].iloc[1] == 1

    def test_a_rule_matching_nothing_leaves_everything_unclaimed(self):
        m = match_rules(synthetic(), [Rule(label="None", portfolios=["Nope"])])
        assert m["assignment"].notna().sum() == 0
        assert len(m["conflicts"]) == 0

    def test_no_rules_is_nothing_claimed(self):
        assert len(match_rules(synthetic(), [])["assignment"]) == 0
        assert len(match_rules(None, [Rule()])["assignment"]) == 0


class TestRepricing:
    @needs_inputs
    def test_only_the_claimed_contracts_move(self, inputs, rep):
        r = reprice_rules(inputs, rep,
                          [Rule(label="Off BS to Stage 2", portfolios=["Off BS"],
                                stage_to=2)])
        assert r["ok"]
        assert set(r["by_rule"]["rule"]) == {"Off BS to Stage 2"}
        untouched = r["detail"]
        assert (untouched["portfolio"] == "Off BS").all()

    @needs_inputs
    def test_a_downgrade_raises_the_provision(self, inputs, rep):
        r = reprice_rules(inputs, rep,
                          [Rule(label="down", portfolios=["Business Finance"],
                                rating_notches=3)])
        assert r["delta"] > 0
        assert r["by_rule"]["change"].iloc[0] > 0

    @needs_inputs
    def test_the_baseline_is_priced_by_the_same_function(self, inputs, rep):
        """Taking it from the report's ECL column instead would show a
        difference whenever the report predates the current config -- and that
        difference would read as the what-if having done something."""
        nothing = Rule(label="no change", portfolios=["Off BS"])
        r = reprice_rules(inputs, rep, [nothing])
        assert r["delta"] == pytest.approx(0.0, abs=1e-6)

    @needs_inputs
    def test_conflicted_contracts_are_reported_and_not_repriced(self, inputs, rep):
        rules = [Rule(label="A", portfolios=["Off BS"], stage_to=2),
                 Rule(label="B", stages=[1], collateral_pct=50)]
        r = reprice_rules(inputs, rep, rules)
        assert len(r["conflicts"]) > 0
        claimed = set(r["detail"]["contract"])
        assert not (claimed & set(r["conflicts"]["contract"]))

    @needs_inputs
    def test_no_rule_matching_anything_says_so(self, inputs, rep):
        r = reprice_rules(inputs, rep, [Rule(label="x", portfolios=["Nope"])])
        assert not r["ok"]
        assert "matched" in r["reason"]

    def test_missing_inputs_are_a_reason_not_a_crash(self):
        from pathlib import Path
        from ifrs9qdb.inputs import EngineInputs
        bare = EngineInputs(ok=False, out_dir=Path("."), missing=["StPD.csv"])
        r = reprice_rules(bare, synthetic(), [Rule()])
        assert not r["ok"] and "StPD.csv" in r["reason"]


class TestReportHelpers:
    @needs_inputs
    def test_the_two_rating_scales_are_separable(self, inputs, rep):
        """An agency Aa2 beside a QDB 5 in one chart says they mean the same
        thing. They are different models on different grades."""
        internal = filter_rating_type(rep, inputs, 1)
        external = filter_rating_type(rep, inputs, 2)
        assert len(internal) and len(external)
        assert set(internal["contract"]) & set(external["contract"]) == set()
        assert len(internal) + len(external) <= len(rep)

    @needs_inputs
    def test_ordering_follows_the_scale_not_the_alphabet(self, inputs, rep):
        levels = inputs.scale_for(1).ratings
        prof = rep.groupby("rating", as_index=False)["ecl"].sum()
        out = order_by_rating(prof, "rating", levels)
        assert list(out["rating"]) == [r for r in levels
                                       if r in set(out["rating"])]

    @needs_inputs
    def test_a_grade_not_on_the_scale_is_dropped_not_placed(self, inputs, rep):
        """It has no position on that scale; first or last would invent one."""
        levels = inputs.scale_for(1).ratings
        prof = rep.groupby("rating", as_index=False)["ecl"].sum()
        out = order_by_rating(prof, "rating", levels)
        assert len(out) < len(prof)
        assert set(out["rating"]) <= set(levels)

    def test_nothing_to_order_is_returned_unchanged(self):
        d = pd.DataFrame({"rating": ["X"], "ecl": [1.0]})
        assert order_by_rating(d, "rating", None) is d
        assert len(order_by_rating(d, "rating", ["QDB 1"])) == 0


class TestDuplicateContractIds:
    """A contract id is not unique in a LIC book.

    An investment security held in two positions appears twice in the report
    AND twice in the account master. Joining them on the id squares that: two
    rows against two rows is four, and the security gets priced twice in every
    repriced total. The July book has one.
    """

    @needs_inputs
    def test_the_ingredients_have_one_row_per_report_row(self, inputs, rep):
        from ifrs9qdb.stress import _ingredients
        assert len(_ingredients(inputs, rep)) == len(rep)

    @needs_inputs
    def test_a_duplicated_id_does_not_get_priced_twice(self, inputs, rep):
        from ifrs9qdb.stress import _ingredients, reprice

        doubled = pd.concat([rep, rep.iloc[[0]]], ignore_index=True)
        assert len(_ingredients(inputs, doubled)) == len(doubled)
        assert len(reprice(inputs, doubled)) == len(doubled)
