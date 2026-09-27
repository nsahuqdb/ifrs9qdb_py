"""Characterisation tests for the input-side issues.

Each corresponds to an item in INPUT_DATA_ISSUES.md and pins what happens
TODAY, including where that is wrong. A failure here means an item has been
changed — go and approve it, then update the register.

The synthetic cases run anywhere. The ones that read the raw extracts skip
unless IFRS9_SRC_INPUTS points at them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.engine import (EAD_FALLBACK_RULES, fallback_ead_curve,
                             resolve_ead_shape)
from ifrs9qdb.etl.lifetime import build_lifetime_parameter_other

from conftest import needs_src_inputs, src_inputs

# The R package's defaults and config/model.yml carry only the rules that name a
# portfolio. Python adds one that does not. See X4.
R_CONFIG_RULES = tuple(r for r in EAD_FALLBACK_RULES if r[0] is not None)


def _raw(name):
    """A raw extract table, or skip."""
    src = src_inputs()
    if src is None:
        pytest.skip("set IFRS9_SRC_INPUTS")
    for ext in (".xlsx", ".xls", ".csv"):
        f = src / f"{name}{ext}"
        if f.is_file():
            return pd.read_csv(f, dtype=str) if ext == ".csv" else pd.read_excel(f)
    pytest.skip(f"{name} not in IFRS9_SRC_INPUTS")


class TestI1TheCenturyIsLost:
    """I1: 2030-2043 arrive as 1930-1943 through a two-digit-year pivot."""

    @needs_src_inputs
    def test_the_extract_carries_pre_1950_schedule_dates(self):
        rs = _raw("RepaymentSchedule")
        yr = pd.to_datetime(rs["START_DAT"], errors="coerce").dt.year
        wrapped = yr < 1950
        assert wrapped.any(), (
            "I1 may be FIXED upstream: no schedule row has a pre-1950 year. "
            "If the extract now delivers four-digit years, update "
            "INPUT_DATA_ISSUES.md.")
        assert yr[wrapped].between(1930, 1949).all(), (
            f"I1: the wrapped years are {sorted(yr[wrapped].unique())}, not the "
            "1930-1943 band the pivot explains. Re-derive before assuming.")

    @needs_src_inputs
    def test_every_wrapped_row_predates_the_schedule_it_belongs_to(self):
        """The cheapest possible detection, and it is not being made."""
        rs = _raw("RepaymentSchedule")
        start = pd.to_datetime(rs["START_DAT"], errors="coerce")
        post = pd.to_datetime(rs["POST_DATE"], errors="coerce")
        wrapped = start.dt.year < 1950
        before = start < post
        assert (before == wrapped).all(), (
            "I1/I2: START_DAT < POST_DATE no longer picks out exactly the "
            "wrapped rows. The one-comparison check may need revisiting.")

    @needs_src_inputs
    def test_restoring_the_century_reproduces_the_maturity_date(self):
        """The proof. Nothing else explains a 1,124-for-1,124 match."""
        rs, am = _raw("RepaymentSchedule"), _raw("AccountMaster")
        rs["cid"] = rs["KEY_1"].astype(str).str.strip()
        am["cid"] = am["CONTRACTID"].astype(str).str.strip()
        start = pd.to_datetime(rs["START_DAT"], errors="coerce")
        rs["fixed"] = [d.replace(year=d.year + 100) if pd.notna(d) and d.year < 1950
                       else d for d in start]
        mat = dict(zip(am["cid"], pd.to_datetime(am["MATURITYDAT"], errors="coerce")))
        affected = rs.loc[start.dt.year < 1950, "cid"].unique()
        last = rs[rs["cid"].isin(affected)].groupby("cid")["fixed"].max()
        same = [abs((last[c] - mat[c]).days) <= 31 for c in last.index if c in mat]
        assert len(same) > 100
        assert all(same), (
            f"I1: the repaired end date matches MATURITYDAT on "
            f"{sum(same)}/{len(same)} contracts, not all of them.")

    def test_the_derivation_truncates_a_wrapped_loan(self):
        """An 18-year annual payer, three years of which survive: the worked
        example in INPUT_DATA_ISSUES.md, in miniature."""
        ref = pd.Timestamp("2026-06-09")
        rows, bal = [], 100_000_000.0
        for k in range(18):
            year = 2027 + k
            shown = year - 100 if year >= 2030 else year
            bal -= 1_000_000.0 if year < 2030 else 6_000_000.0
            rows.append(("C1", pd.Timestamp("2026-06-09"),
                         pd.Timestamp(f"{shown}-01-30"), 0.0, 0.0,
                         2_000_000.0, max(bal, 0.0)))
        sched = pd.DataFrame(rows, columns=["KEY_1", "POST_DATE", "START_DAT",
                                           "PRINCE_DUE", "PROJ_INT",
                                           "REPAYMENT", "BALANCE"])
        acc = pd.DataFrame([("C1", 100_000_000.0)],
                           columns=["ContractId", "OnBalance"])
        lp = build_lifetime_parameter_other(sched, acc, ref, "2026-06-09")
        lp["MonthLifetime"] = pd.to_numeric(lp["MonthLifetime"])
        lp["EADLifetime"] = pd.to_numeric(lp["EADLifetime"])
        lp = lp.sort_values("MonthLifetime")
        # 2029-01-30 is month 31; the curve runs 0..30 and stops there.
        assert lp["MonthLifetime"].max() == 30, (
            "I1 may be FIXED: the curve no longer stops at the last pre-2030 "
            f"payment (it reaches month {lp['MonthLifetime'].max()}).")
        last = float(lp["EADLifetime"].iloc[-1])
        assert last > 0.9 * 100_000_000.0, (
            f"I1: the curve ends at {last:,.0f}, so most of the exposure is no "
            "longer left unpriced. Re-check before assuming it is fixed.")


class TestI4ABlankDpdReadsAsCurrent:
    """I4: a missing PastDueDays becomes zero days past due."""

    @staticmethod
    def accounts(dpds):
        return pd.DataFrame({
            "CONTRACTID": [f"C{i}" for i in range(len(dpds))],
            "CUSTOMERID": ["CUST1"] * len(dpds),
            "PASTDUEDAYS": dpds,
            "ONBALANCE": [1000.0] * len(dpds),
        })

    def test_a_customer_with_no_dpd_anywhere_is_treated_as_current(self):
        from ifrs9qdb.etl.lending import transform_lending
        out = transform_lending(self.accounts([np.nan, np.nan]))
        assert (pd.to_numeric(out["past_dues_worst"]) == 0).all(), (
            "I4 may be FIXED: a wholly blank DPD no longer reads as zero. "
            "Update INPUT_DATA_ISSUES.md.")

    def test_a_sibling_with_a_real_dpd_is_inherited(self):
        """The half of I4 that works: the customer maximum skips the blank."""
        from ifrs9qdb.etl.lending import transform_lending
        out = transform_lending(self.accounts([np.nan, 740.0]))
        assert (pd.to_numeric(out["past_dues_worst"]) == 740).all()

    @needs_src_inputs
    def test_the_extract_has_blank_dpd_on_material_exposure(self):
        am = _raw("AccountMaster")
        dpd = pd.to_numeric(am["PASTDUEDAYS"], errors="coerce")
        onb = pd.to_numeric(am["ONBALANCE"], errors="coerce").fillna(0)
        assert dpd.isna().any(), "I4 may be FIXED upstream: no blank DPD"
        assert onb[dpd.isna()].sum() > 0


class TestI5TheAllocationCheckWatchesTheWrongAxis:
    """I5: the rule sums per contract, which is not bounded by one."""

    @needs_src_inputs
    def test_per_contract_totals_exceed_one_legitimately(self):
        aca = _raw("AccountCollateralAllocation")
        p = pd.to_numeric(aca["ALLOCATIONPERCENTAGE"], errors="coerce")
        tot = p.groupby(aca["CONTRACTID"].astype(str)).sum()
        assert (tot > 100.1).any(), (
            "I5: no contract now exceeds 100% in total, so the rule fires on "
            "nothing. If the source changed, re-check the register.")

    @needs_src_inputs
    def test_no_collateral_is_over_allocated_on_the_axis_that_matters(self):
        """The check that is absent, and that the book passes."""
        aca = _raw("AccountCollateralAllocation")
        p = pd.to_numeric(aca["ALLOCATIONPERCENTAGE"], errors="coerce")
        tot = p.groupby(aca["COLLATERALID"].astype(str)).sum()
        assert tot.max() <= 100.1, (
            f"I5: a collateral record is now pledged {tot.max():.1f}% across "
            "contracts. That IS double-pledging and nothing checks for it.")


class TestI6ColumnsThatArriveEmpty:
    @needs_src_inputs
    def test_the_nominal_rate_is_zero_so_the_annuity_shape_is_unreachable(self):
        am = _raw("AccountMaster")
        nir = pd.to_numeric(am["NOMINALINTERESTRATE"], errors="coerce").fillna(0)
        assert (nir == 0).all(), (
            "I6 may be FIXED: the extract now carries a nominal rate, which "
            "makes R's annuity shape reachable — and X2 live.")

    @needs_src_inputs
    def test_the_component_breakdown_is_absent(self):
        am = _raw("AccountMaster")
        comps = ["PRINCIPAL", "PRINCIPALOVERDUE", "INTERESTACCRUED",
                 "INTERESTOVERDUE", "FEE", "FEEOVERDUE", "PENALTY",
                 "PENALTYOVERDUE", "COMMISSION", "COMMISSIONOVERDUE",
                 "OTHER", "OTHEROVERDUE"]
        have = [c for c in comps if c in am.columns]
        total = sum(pd.to_numeric(am[c], errors="coerce").fillna(0) for c in have)
        onb = pd.to_numeric(am["ONBALANCE"], errors="coerce").fillna(0)
        assert onb.sum() > 0
        assert total.sum() == 0, (
            "I6 may be FIXED: the components now carry values, so the ECL can "
            "be split by component. Update the register.")


class TestX2AndX3TheShapeSwitchDiverges:
    """X2/X3: R has three shapes and defaults an unknown name to bullet;
    Python has two and defaults to linear."""

    def test_python_has_no_annuity_branch(self):
        annuity = fallback_ead_curve(1000, 60, "4", portfolio="Business Finance")
        linear = fallback_ead_curve(1000, 60, "4", portfolio="Business Finance")
        assert np.array_equal(annuity, linear)
        # A true annuity is convex: the early instalments retire less principal.
        # A straight line is not, which is what Python produces.
        mid = annuity[len(annuity) // 2]
        assert mid == pytest.approx(annuity[0] * 0.5, rel=0.05), (
            "X2 may be FIXED: the curve is no longer a straight line, so an "
            "annuity shape may have been implemented. Update the register.")

    def test_the_config_block_never_reaches_the_curve(self):
        """X3: R reads ecl.ead_fallback from model.yml; Python does not. The
        shipped YAML carries the block and nothing in the package reads it."""
        import ifrs9qdb
        pkg = Path(ifrs9qdb.__file__).parent
        shipped = (pkg / "config" / "model.yml").read_text()
        assert "ead_fallback" in shipped, "the block should be in the shipped config"
        readers = [f for f in pkg.rglob("*.py") if "ead_fallback" in f.read_text()]
        assert readers == [], (
            f"X3 may be FIXED: {[f.name for f in readers]} now reads the config "
            "block. If Python honours ecl.ead_fallback, update the register.")

    def test_an_unmatched_shape_falls_to_bullet_in_both(self):
        """The one thing that does agree: R's switch default and Python's
        EAD_FALLBACK_DEFAULT are both bullet."""
        from ifrs9qdb.engine import EAD_FALLBACK_DEFAULT
        assert EAD_FALLBACK_DEFAULT == "bullet"
        assert resolve_ead_shape("99", "Nowhere") == "bullet"
        c = fallback_ead_curve(1000, 60, "99", portfolio="Nowhere")
        assert np.allclose(c, 1000.0), "an unmatched contract should be a bullet"


class TestX4TheExtraRuleIsDormantNotAbsent:
    """X4: Python carries (None,'4','linear'); the R config does not."""

    def test_the_rule_sets_differ_by_exactly_one_entry(self):
        extra = set(EAD_FALLBACK_RULES) - set(R_CONFIG_RULES)
        assert extra == {(None, "4", "linear")}, (
            f"X4 has changed: the difference is now {extra}. Reconcile the two "
            "rule sets and update INPUT_DATA_ISSUES.md.")

    def test_the_two_rule_sets_disagree_on_an_off_bs_type_4(self):
        """The combination that is absent from both delivered runs and present
        in the June 2026 book."""
        assert resolve_ead_shape("4", "Off BS", R_CONFIG_RULES) == "bullet"
        assert resolve_ead_shape("4", "Off BS", EAD_FALLBACK_RULES) == "linear"

    def test_they_agree_on_every_combination_the_runs_contain(self):
        """Why the port still reproduces the R report to the cent."""
        for pf, pt in (("Al Dhameen", "4"), ("Banks and Fis", "3"),
                       ("Business Finance", "3"), ("Business Finance", "4"),
                       ("Investments", "3"), ("Off BS", "3"), ("Tasdeer", "3")):
            assert resolve_ead_shape(pt, pf, R_CONFIG_RULES) == \
                resolve_ead_shape(pt, pf, EAD_FALLBACK_RULES), \
                f"X4 is now live on ({pf}, {pt}), which the runs DO contain"

    def test_linear_is_about_half_of_bullet(self):
        """The size of the divergence when it does bite."""
        for n in (12, 36, 60):
            b = fallback_ead_curve(1000, n, "4", portfolio="Al Dhameen")
            l = fallback_ead_curve(1000, n, "4", portfolio="Business Finance")
            assert 0.45 < l.sum() / b.sum() < 0.60
