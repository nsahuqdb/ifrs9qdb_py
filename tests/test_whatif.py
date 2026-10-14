"""What-ifs: a stress whose scope names customers, stages or portfolios.

There is one stress function (``apply_stress``); a what-if is a stress scoped
to the exposures it names. What matters here is that boundary: every lever
reaches the scope and nothing else, the PD curves and the LGD parameters
included, which live on the pricing context and once reached the whole book.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import normalise
from ifrs9qdb.stress import (
    StressSpec, apply_stress, filter_rating_type, order_by_rating,
    read_packages, standard_packages, stress_scope, write_packages,
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


class TestScope:
    def test_portfolios_stages_and_customers_narrow_together(self):
        d = synthetic()
        both = stress_scope(d, StressSpec(portfolios=["Business Finance"], stages=[1]))
        assert list(d.loc[both, "contract"]) == ["C1"]
        cust = stress_scope(d, StressSpec(customers=["A"]))
        assert list(d.loc[cust, "contract"]) == ["C1", "C2"]
        assert stress_scope(d, StressSpec()).all()      # nothing named: the book

    def test_a_customer_id_with_spaces_is_one_id(self):
        """R's what-if split ids on spaces, so a customer named with spaces
        never matched; a list is taken as it is."""
        d = synthetic().assign(customer=["A B", "A B", "B", "C"])
        assert stress_scope(d, StressSpec(customers=["A B"])).sum() == 2

    def test_a_saved_package_reads_back_as_written(self, tmp_path):
        pkgs = standard_packages(["Business Finance"])
        pkgs.append(StressSpec(name="named", customers=["A B", "C"], stages=[2],
                               stage_to=3, pd_scenario="Significant Downturn",
                               maturity_years=1.5))
        p = write_packages(pkgs, tmp_path / "config" / "stress_packages.yml")
        back = read_packages(p)
        assert [x.to_dict() for x in back] == [x.to_dict() for x in pkgs]

    def test_a_package_r_saved_reads_back(self, tmp_path):
        """R writes a vector of one as a scalar and a NULL as ~."""
        f = tmp_path / "stress_packages.yml"
        f.write_text("packages:\n- name: R one\n  portfolios: Business Finance\n"
                     "  stages: 2\n  customers: A B\n  stage_to: ~\n"
                     "  dpd_threshold: 30.0\n  rating_notches: 1\n")
        (sp,) = read_packages(f)
        assert sp.portfolios == ["Business Finance"] and sp.stages == [2]
        assert sp.customers == ["A B"] and sp.stage_to is None
        assert sp.dpd_threshold == 30.0 and sp.rating_notches == 1

    def test_a_package_from_a_request_ignores_what_it_does_not_know(self):
        sp = StressSpec.from_dict({"name": "x", "customers": "A, B;C",
                                   "stage_to": "", "dpd_threshold": "",
                                   "lgd_override": 0.6, "notches": 3})
        assert sp.customers == ["A", "B", "C"]
        assert sp.stage_to is None and sp.dpd_threshold is None
        assert sp.is_empty()


class TestRepricing:
    @needs_inputs
    def test_only_the_scope_moves(self, inputs, rep):
        r = apply_stress(inputs, rep, StressSpec(name="Off BS to Stage 2",
                                                 portfolios=["Off BS"], stage_to=2))
        assert r["ok"] and r["delta"] > 0
        moved = r["detail"][r["detail"]["change"] != 0]
        assert (moved["portfolio"] == "Off BS").all()

    @needs_inputs
    def test_named_customers_are_a_scope(self, inputs, rep):
        top = (rep.groupby("customer")["exposure"].sum()
               .sort_values(ascending=False).index[:2].astype(str))
        r = apply_stress(inputs, rep, StressSpec(customers=list(top), stage_to=3))
        moved = r["detail"][r["detail"]["change"] != 0]
        assert r["delta"] > 0 and set(moved["customer"].astype(str)) <= set(top)

    @needs_inputs
    def test_a_downgrade_raises_the_provision(self, inputs, rep):
        r = apply_stress(inputs, rep, StressSpec(portfolios=["Business Finance"],
                                                 rating_notches=3))
        assert r["delta"] > 0

    @needs_inputs
    def test_an_empty_stress_is_the_report(self, inputs, rep):
        """Priced by the same function on both sides, and the staging policy
        not re-run when it is the run's own."""
        r = apply_stress(inputs, rep, StressSpec(portfolios=["Off BS"]))
        assert r["delta"] == pytest.approx(0.0, abs=1e-6) and r["moved"] == 0

    @needs_inputs
    def test_the_pd_multiplier_and_lgd_stay_in_scope(self, inputs, rep):
        """They are the context's, so they once reached every contract priced
        on it: a Business Finance stress moved Al Dhameen, Off BS, Tasdeer and
        Investments too."""
        for kw in ({"pd_multiplier": 1.5}, {"lgd_base": 0.6, "lgd_floor": 0.7}):
            r = apply_stress(inputs, rep, StressSpec(portfolios=["Business Finance"], **kw))
            bp = r["by_portfolio"].set_index("portfolio")["change"]
            assert bp["Business Finance"] > 0
            assert bp.drop("Business Finance").abs().max() == pytest.approx(0.0, abs=1e-6)

    @needs_inputs
    def test_a_scenario_prices_the_scope_on_its_curves(self, inputs, rep):
        r = apply_stress(inputs, rep, StressSpec(portfolios=["Business Finance"],
                                                 pd_scenario="Significant Downturn"))
        assert r["ok"] and r["delta"] > 0
        bad = apply_stress(inputs, rep, StressSpec(pd_scenario="Mild Panic"))
        assert not bad["ok"] and "Mild Panic" in bad["reason"]

    @needs_inputs
    def test_a_scope_matching_nothing_says_so(self, inputs, rep):
        r = apply_stress(inputs, rep, StressSpec(portfolios=["Nope"]))
        assert not r["ok"] and "scope" in r["reason"]

    def test_missing_inputs_are_a_reason_not_a_crash(self):
        from pathlib import Path
        from ifrs9qdb.inputs import EngineInputs
        bare = EngineInputs(ok=False, out_dir=Path("."), missing=["StPD.csv"])
        r = apply_stress(bare, synthetic(), StressSpec())
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
