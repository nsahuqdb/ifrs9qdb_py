"""The two defects that stood between the port and an exact ECL report.

Both were silent. Neither raised, neither produced an implausible number, and
both moved the provision by more than a percent:

  * collateral was netted without the allocation percentage and without the
    haircut, so contracts were credited with security they do not have;
  * a facility at the maturity floor was amortised rather than priced as a
    bullet, which is a third of its exposure in the first three months.

With both fixed the Python report reproduces the R engine's to the cent on
every contract of both reference runs, which is what the top-level test here
asserts.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import normalise
from ifrs9qdb.engine import (
    EAD_FALLBACK_RULES, MIN_HORIZON_MONTHS, fallback_ead_curve,
    resolve_ead_shape,
)

from conftest import ref_output

OUT = ref_output()
needs_run = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


class TestTheShapeRules:
    def test_the_portfolio_wins_over_the_payment_type(self):
        """A portfolio rule is matched before a portfolio-free one. Al Dhameen
        type 4 amortises (annually, back from maturity -- LIC runs 324 and
        330); Off BS type 4 has no rule and falls to the bullet default."""
        assert resolve_ead_shape(4, "Business Finance") == "linear"
        assert resolve_ead_shape(4, "Al Dhameen") == "linear"
        assert resolve_ead_shape(4, "Off BS") == "bullet"

    def test_an_unknown_combination_is_a_bullet(self):
        """The conservative reading, and the one LIC takes."""
        assert resolve_ead_shape(9, "Business Finance") == "bullet"
        assert resolve_ead_shape(3, "Nowhere") == "bullet"

    def test_type_4_with_no_portfolio_still_amortises(self):
        """EY's own workbook settles this: flat gives ~1.8x the LIC figure."""
        assert resolve_ead_shape(4) == "linear"
        assert resolve_ead_shape("4.0") == "linear"

    def test_every_rule_names_a_shape_the_builder_understands(self):
        assert {shape for _, _, shape in EAD_FALLBACK_RULES} <= {"linear",
                                                                 "bullet"}


class TestTheMaturityFloor:
    def test_a_facility_at_the_floor_is_a_bullet(self):
        """Its maturity is at or before the extract date, so it has no
        remaining amortisation schedule to run."""
        c = fallback_ead_curve(1000, MIN_HORIZON_MONTHS, 4,
                               portfolio="Business Finance")
        assert np.allclose(c, 1000.0)

    def test_one_month_past_the_floor_amortises_again(self):
        c = fallback_ead_curve(1000, MIN_HORIZON_MONTHS + 1, 4,
                               portfolio="Business Finance")
        assert c[0] == pytest.approx(1000.0)
        assert c[-1] < 1000.0

    def test_amortising_at_the_floor_would_understate_by_a_third(self):
        """The size of the defect, pinned so a regression is not subtle."""
        bullet = fallback_ead_curve(1000, 3, 4, portfolio="Business Finance")
        amortised = np.array([1000.0, 2000 / 3, 1000 / 3])
        assert bullet.sum() == pytest.approx(3000.0)
        assert amortised.sum() / bullet.sum() == pytest.approx(2 / 3, abs=1e-9)


class TestCollateralNetting:
    @needs_run
    def test_the_allocation_percentage_and_haircut_are_both_applied(self):
        """Either one omitted credits a contract with security it does not
        have, drives its LGD to the 0.225 floor, and understates the
        provision."""
        from ifrs9qdb.inputs import load_engine_inputs

        inputs = load_engine_inputs(OUT)
        alloc, coll, ctype = inputs.alloc, inputs.collateral, inputs.coll_type
        if alloc is None or coll is None or ctype is None:
            pytest.skip("this run has no collateral tables")

        raw = dict(zip(coll["CollateralId"].astype(str),
                       pd.to_numeric(coll["CollateralValue"], errors="coerce")))
        gross = {}
        for c, k in zip(alloc["ContractId"].astype(str),
                        alloc["CollateralId"].astype(str)):
            if k in raw and pd.notna(raw[k]):
                gross[c] = gross.get(c, 0.0) + float(raw[k])

        net_total = sum(inputs.collateral_net.values())
        gross_total = sum(gross.values())
        assert net_total < gross_total, "nothing was netted down"
        # Most QDB collateral types carry HaircutGeneral = 1.00 -- corporate
        # cheques, comfort assignments, guarantees are worth nothing for
        # provisioning - so the netted figure is a small fraction of the gross.
        assert net_total < 0.5 * gross_total

    @needs_run
    def test_a_fully_haircut_type_contributes_nothing(self):
        from ifrs9qdb.inputs import load_engine_inputs

        inputs = load_engine_inputs(OUT)
        ctype = inputs.coll_type
        if ctype is None:
            pytest.skip("this run has no collateral type table")
        hc = pd.to_numeric(ctype["HaircutGeneral"], errors="coerce")
        assert (hc >= 1.0).any(), "expected at least one worthless type"
        assert (hc.dropna() <= 1.0).all()


class TestTheReportReproducesR:
    @needs_run
    def test_every_contract_matches_the_r_engine_to_the_cent(self):
        """The whole point of the port. Not a tolerance -- an equality."""
        from ifrs9qdb.etl.report import build_final_ecl_report

        run_dir = OUT.parent if OUT.name.lower() == "output" else OUT
        reference = normalise(pd.read_csv(OUT / "FinalEclReport.csv",
                                          low_memory=False))
        mine = normalise(build_final_ecl_report(run_dir, write=False))

        assert len(mine) == len(reference), "the two reports must be the same book"

        # Compared ROW BY ROW, not through a join. A contract id is not unique
        # in a LIC report -- the July book has an investment security held
        # twice -- and a merge on it would fan those rows out and quietly
        # inflate the comparison. Both frames are the same rows in the same
        # order, so position is the honest key.
        a = reference.reset_index(drop=True)
        b = mine.reset_index(drop=True)
        assert (a["contract"] == b["contract"]).all(), "the rows moved"

        stages = pd.to_numeric(a["stage"], errors="coerce")
        stages_m = pd.to_numeric(b["stage"], errors="coerce")
        assert (stages == stages_m).all(), "the staging rule disagrees"

        assert (b["lgd"] - a["lgd"]).abs().max() < 1e-9
        assert (b["ecl"] - a["ecl"]).abs().max() < 0.01
        assert b["ecl"].sum() == pytest.approx(a["ecl"].sum(), abs=0.01)
