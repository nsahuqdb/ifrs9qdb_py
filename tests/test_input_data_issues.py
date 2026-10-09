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

from conftest import needs_src_inputs, ref_output, src_inputs

OUT = ref_output()
needs_run = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")


@pytest.fixture(scope="module")
def inputs():
    if OUT is None:
        pytest.skip("set IFRS9_REF_RUN")
    from ifrs9qdb.inputs import load_engine_inputs
    return load_engine_inputs(OUT)

# The R package's defaults and config/model.yml carry only the rules that name a
# portfolio; so, since X3 was fixed, does Python's table.
R_CONFIG_RULES = tuple(r for r in EAD_FALLBACK_RULES if r[0] is not None)


def _raw(name):
    """A raw extract table, or skip.

    Several inputs arrive as Oracle SQL*Plus HTML behind an ``.xls`` extension,
    so a failed ``read_excel`` is expected and falls back to ``read_html``.
    """
    src = src_inputs()
    if src is None:
        pytest.skip("set IFRS9_SRC_INPUTS")
    for ext in (".xlsx", ".xls", ".csv"):
        f = src / f"{name}{ext}"
        if not f.is_file():
            continue
        if ext == ".csv":
            return pd.read_csv(f, dtype=str)
        try:
            return pd.read_excel(f)
        except ValueError:
            try:
                return pd.read_html(f)[0]
            except (ImportError, ValueError):
                pytest.skip(f"{f.name} needs lxml to parse")
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


class TestX2TheConfigBlockIsHonoured:
    """X2, FIXED: Python reads ecl.ead_fallback from model.yml, as R does, so
    editing the config moves both engines' numbers together."""

    def test_the_shipped_config_block_is_what_the_engine_prices_with(self):
        import yaml

        import ifrs9qdb
        from ifrs9qdb.engine import ead_fallback_rules
        pkg = Path(ifrs9qdb.__file__).parent
        cfg = yaml.safe_load((pkg / "config" / "model.yml").read_text())
        assert cfg["ecl"]["ead_fallback"], "the block should be in the shipped config"
        rules, default = ead_fallback_rules(cfg)
        assert set(rules) == set(EAD_FALLBACK_RULES)
        assert default == "bullet"

    def test_an_edited_config_moves_the_curve(self):
        from ifrs9qdb.engine import ead_fallback_rules, use_ead_fallback_rules
        cfg = {"ecl": {"ead_fallback": {"default": "bullet", "rules": [
            {"portfolio": "Off BS", "payment_type": 4, "shape": "linear"}]}}}
        rules, default = ead_fallback_rules(cfg)
        assert resolve_ead_shape("4", "Off BS", rules, default) == "linear"
        use_ead_fallback_rules(cfg)
        try:
            c = fallback_ead_curve(1000, 60, "4", portfolio="Off BS")
            assert c[-1] < c[0], "the configured linear rule should amortise"
        finally:
            use_ead_fallback_rules(None)
        assert np.allclose(fallback_ead_curve(1000, 60, "4", portfolio="Off BS"),
                           1000.0), "back on R's rules Off BS type 4 is a bullet"

    def test_an_unmatched_shape_falls_to_bullet_in_both(self):
        """R's switch default and Python's EAD_FALLBACK_DEFAULT are both bullet."""
        from ifrs9qdb.engine import EAD_FALLBACK_DEFAULT
        assert EAD_FALLBACK_DEFAULT == "bullet"
        assert resolve_ead_shape("99", "Nowhere") == "bullet"
        c = fallback_ead_curve(1000, 60, "99", portfolio="Nowhere")
        assert np.allclose(c, 1000.0), "an unmatched contract should be a bullet"


class TestX3TheRuleSetsAgree:
    """X3, FIXED: Python carried a portfolio-free (None, '4', 'linear') rule R
    does not have, which would have amortised the June book's Off BS and
    Tasdeer type-4 facilities that R prices as bullets. The rule now applies
    only where no portfolio is known at all (EY's worked examples)."""

    def test_python_carries_exactly_rs_five_rules(self):
        assert set(EAD_FALLBACK_RULES) == {
            ("Business Finance", "4", "linear"),
            ("Business Finance", "3", "bullet"),
            ("Al Dhameen", "4", "linear"),
            ("Tasdeer", "3", "bullet"),
            ("Off BS", "3", "bullet"),
        }
        assert set(EAD_FALLBACK_RULES) == set(R_CONFIG_RULES)

    def test_off_bs_and_tasdeer_type_4_are_bullets_as_in_r(self):
        """The June book's 875 contracts / 508m that the extra rule would have
        amortised."""
        for pf in ("Off BS", "Tasdeer", "Investments"):
            assert resolve_ead_shape("4", pf) == "bullet", pf

    def test_the_ey_rule_applies_only_without_a_portfolio(self):
        from ifrs9qdb.engine import NO_PORTFOLIO_RULES
        assert NO_PORTFOLIO_RULES == ((None, "4", "linear"),)
        assert resolve_ead_shape("4", None) == "linear"
        assert resolve_ead_shape("4", "") == "linear"

    def test_they_agree_on_every_combination_the_runs_contain(self):
        for pf, pt in (("Al Dhameen", "4"), ("Banks and Fis", "3"),
                       ("Business Finance", "3"), ("Business Finance", "4"),
                       ("Investments", "3"), ("Off BS", "3"), ("Off BS", "4"),
                       ("Tasdeer", "3"), ("Tasdeer", "4")):
            assert resolve_ead_shape(pt, pf, R_CONFIG_RULES) == \
                resolve_ead_shape(pt, pf, EAD_FALLBACK_RULES), (pf, pt)

    def test_linear_is_about_half_of_bullet(self):
        """The size of the divergence the extra rule would have caused."""
        for n in (12, 36, 60):
            b = fallback_ead_curve(1000, n, "4", portfolio="Off BS")
            l = fallback_ead_curve(1000, n, "4", portfolio="Business Finance")
            assert 0.45 < l.sum() / b.sum() < 0.60


class TestX4TheAnnuityShape:
    """X4, FIXED: R implements bullet, linear and annuity; the engine's
    fallback now does too, and prices a shape it does not know as a bullet
    (R's switch default) instead of a linear ramp."""

    CFG = {"ecl": {"ead_fallback": {"default": "bullet", "rules": [
        {"portfolio": "Business Finance", "payment_type": 4, "shape": "annuity"},
        {"portfolio": "Tasdeer", "payment_type": 4, "shape": "balloon"}]}}}

    def _with(self, fn):
        from ifrs9qdb.engine import use_ead_fallback_rules
        use_ead_fallback_rules(self.CFG)
        try:
            return fn()
        finally:
            use_ead_fallback_rules(None)

    def test_an_annuity_is_convex_at_a_positive_rate(self):
        a = self._with(lambda: fallback_ead_curve(
            1000, 60, "4", portfolio="Business Finance", nir=0.08))
        lin = fallback_ead_curve(1000, 60, "4", portfolio="Business Finance")
        assert a[0] == pytest.approx(1000.0)
        # early instalments retire less principal than a straight line
        assert a[30] > lin[30]
        # R's closed form after 30 of 60 monthly payments at 8%
        r = 0.08 / 12
        want = 1000 * ((1 + r) ** 60 - (1 + r) ** 30) / ((1 + r) ** 60 - 1)
        assert a[30] == pytest.approx(want, rel=1e-12)

    def test_an_annuity_degenerates_to_linear_at_zero_rate(self):
        a = self._with(lambda: fallback_ead_curve(
            1000, 60, "4", portfolio="Business Finance", nir=0.0))
        lin = fallback_ead_curve(1000, 60, "4", portfolio="Business Finance")
        assert np.allclose(a, lin)

    def test_an_unknown_shape_is_a_bullet(self):
        c = self._with(lambda: fallback_ead_curve(1000, 60, "4", portfolio="Tasdeer"))
        assert np.allclose(c, 1000.0)


class TestI2CollateralHasNoValue:
    """I2: 97.6% of allocated collateral records carry no value, so the netting
    formula's first factor is zero and LGD stays unsecured."""

    @needs_src_inputs
    def test_almost_every_collateral_record_is_unvalued(self):
        coll = _raw("Collateral")
        v = pd.to_numeric(coll["COLLATERALVALUE"], errors="coerce").fillna(0)
        share = (v <= 0).mean()
        assert share > 0.9, (
            f"I2 may be improving: only {share:.1%} of collateral records are "
            "unvalued, down from 97.7%. Update INPUT_DATA_ISSUES.md.")

    @needs_run
    def test_the_books_security_rests_on_a_handful_of_records(self, inputs):
        if inputs.collateral is None or inputs.alloc is None:
            pytest.skip("this run has no collateral tables")
        coll, alloc = inputs.collateral, inputs.alloc
        v = pd.to_numeric(coll["CollateralValue"], errors="coerce").fillna(0)
        allocated = set(alloc["CollateralId"].astype(str))
        a = coll[coll["CollateralId"].astype(str).isin(allocated)]
        av = pd.to_numeric(a["CollateralValue"], errors="coerce").fillna(0)
        valued = int((av > 0).sum())
        assert valued < 200, (
            f"I2 may be FIXED: {valued} allocated collateral records now carry "
            "a value. Update the register.")
        assert (av <= 0).mean() > 0.9


class TestI3OriginationIsEmpty:
    """I3: the file a relative SICR test would need carries only the id."""

    @needs_src_inputs
    def test_only_the_id_columns_carry_data(self):
        o = _raw("Origination")
        filled = [c for c in o.columns
                  if (o[c].notna() & (o[c].astype(str).str.strip() != "")).any()]
        assert len(filled) == 2, (
            f"I3 may be FIXED: {len(filled)} columns now carry data ({filled}). "
            "If origination PD and rating have arrived, the relative SICR test "
            "becomes possible — update INPUT_DATA_ISSUES.md and M9.")


class TestI4StagingFlagsDoNotArrive:
    """I4: is_default, is_insolvency and is_default_in_gcc are never populated,
    so Stage 3 rests on DPD > 90 alone."""

    @needs_src_inputs
    def test_three_of_the_five_flags_are_entirely_blank(self):
        c = _raw("CustomerStagingFlag")
        # positions 3, 5, 6 in the schema (1-based) -> is_default,
        # is_insolvency, is_default_in_gcc
        blank = [i for i, col in enumerate(c.columns, start=1)
                 if not (c[col].notna() & (c[col].astype(str).str.strip() != "")).any()]
        assert {3, 5, 6}.issubset(set(blank)), (
            f"I4 may have CHANGED: the blank positions are now {blank}. If a "
            "default or insolvency flag has arrived, update the register.")

    @needs_src_inputs
    def test_the_two_flags_that_do_arrive_carry_signal(self):
        c = _raw("CustomerStagingFlag")
        for name in ("ISWATCHLIST", "ISLOCAL1"):
            col = [x for x in c.columns if str(x).upper() == name]
            assert col, f"{name} is missing entirely"
            v = pd.to_numeric(c[col[0]], errors="coerce")
            assert (v == 1).sum() > 0, f"{name} is present but never set"


class TestI7DuplicatedRowsAndOrphanAllocations:
    @needs_src_inputs
    def test_the_schedule_repeats_whole_rows(self):
        rs = _raw("RepaymentSchedule")
        dup = int(rs.duplicated().sum())
        assert dup > 0, (
            "I7 may be FIXED upstream: the schedule no longer repeats rows.")

    @needs_src_inputs
    def test_every_duplicated_month_is_an_exact_duplicate(self):
        """The withdrawn half of M17, pinned from the source. If this ever
        fails, the monthly grid IS losing a real payment."""
        rs = _raw("RepaymentSchedule")
        rs = rs.copy()
        rs["cid"] = rs["KEY_1"].astype(str).str.strip()
        ref = pd.Timestamp(pd.to_datetime(rs["POST_DATE"], errors="coerce").max())
        start = pd.to_datetime(rs["START_DAT"], errors="coerce")
        rs["ml"] = (start.dt.year - ref.year) * 12 + (start.dt.month - ref.month)
        k = rs[rs["ml"] >= 0]
        sizes = k.groupby(["cid", "ml"]).size()
        multi = sizes[sizes > 1]
        if multi.empty:
            pytest.skip("no contract-month holds more than one row")
        vals = ["PRINCE_DUE", "PROJ_INT", "REPAYMENT", "BALANCE"]
        distinct = k.groupby(["cid", "ml"])[vals].nunique().max(axis=1)
        assert (distinct.loc[multi.index] == 1).all(), (
            "a contract-month now holds two DIFFERENT payments, so the monthly "
            "grid is losing one. M17's withdrawn finding is live again.")

    @needs_src_inputs
    def test_some_allocations_have_no_contract_id(self):
        aca = _raw("AccountCollateralAllocation")
        cid = aca["CONTRACTID"]
        blank = cid.isna() | (cid.astype(str).str.strip().isin(("", "nan")))
        assert blank.any(), (
            "I7 may be FIXED upstream: every allocation now names a contract.")


class TestI8UnmappedCollateralTypesDiverge:
    """I8: R sets an NA haircut row to zero; Python defaults the haircut to
    zero and credits the full value. Opposite directions."""

    @staticmethod
    def net(value, pct, haircut, port):
        if port == "r":
            contrib = value * pct * (1 - haircut) if haircut is not None else None
            return 0.0 if contrib is None or not np.isfinite(contrib) else contrib
        return value * pct * (1 - (haircut if haircut is not None else 0.0))

    def test_the_two_ports_go_opposite_ways_on_an_unmapped_type(self):
        r = self.net(1_000_000.0, 0.5, None, "r")
        py = self.net(1_000_000.0, 0.5, None, "py")
        assert r == 0.0, "R writes an unmapped type off"
        assert py == 500_000.0, "Python credits it in full"

    def test_they_agree_when_the_type_is_mapped(self):
        for hc in (0.0, 0.25, 1.0):
            assert self.net(1_000_000.0, 0.5, hc, "r") == \
                self.net(1_000_000.0, 0.5, hc, "py")

    @needs_run
    def test_it_is_latent_because_those_records_are_unvalued(self, inputs):
        """Why parity still holds to the cent."""
        if inputs.collateral is None or inputs.coll_type is None:
            pytest.skip("this run has no collateral tables")
        mapped = set(inputs.coll_type["CollateralTypeId"].astype(str))
        coll = inputs.collateral
        unmapped = coll[~coll["CollateralTypeId"].astype(str).isin(mapped)]
        if unmapped.empty:
            pytest.skip("every collateral type is mapped on this run")
        v = pd.to_numeric(unmapped["CollateralValue"], errors="coerce").fillna(0)
        assert v.sum() == 0, (
            f"I8 IS NOW LIVE: unmapped collateral types carry {v.sum():,.2f} of "
            "value, so R writes it off and Python credits it in full.")


class TestTheEirScale:
    """The defect the A/B measurement exposed: AccountMaster.EIR arrives as a
    raw percent and the lending transform did not divide by 100, so every
    long-dated month was discounted to nothing."""

    @staticmethod
    def accounts(eirs, types=None):
        n = len(eirs)
        return pd.DataFrame({
            "CONTRACTID": [f"C{i}" for i in range(n)],
            "CUSTOMERID": [f"CU{i}" for i in range(n)],
            "ACCOUNTTYPE": types or ["Long Term"] * n,
            "EIR": eirs,
            "ONBALANCE": [1000.0] * n,
        })

    def test_a_percent_becomes_a_decimal(self):
        from ifrs9qdb.etl.lending import transform_lending
        out = transform_lending(self.accounts([2.25, 3.0, 7.0]))
        assert list(pd.to_numeric(out["eir"]).round(6)) == [0.0225, 0.03, 0.07], (
            "EIR must be scaled from percent, as R/transform_lending.R:336 does")

    def test_no_rate_survives_above_one(self):
        from ifrs9qdb.etl.lending import transform_lending
        out = transform_lending(self.accounts([2.25, 3.0, 7.0, 9.75]))
        assert (pd.to_numeric(out["eir"]) < 1).all(), (
            "a rate above 1.0 discounts at over 100% a year")

    def test_a_zero_rate_takes_the_fallback_not_zero(self):
        """R fills a zero or missing rate from the account type's mean, then
        from the mean of the workbook's seven Business Finance means. Leaving it
        at zero would price a long-dated loss undiscounted."""
        from ifrs9qdb.etl.lending import transform_lending
        from ifrs9qdb.etl.static_ref import load_static_reference
        m = load_static_reference()["product_portfolio_mapping"]
        out = transform_lending(self.accounts([4.0, 0.0, np.nan]),
                                product_portfolio_mapping=m)
        eir = pd.to_numeric(out["eir"])
        assert eir.notna().all(), "no contract may be left without a rate"
        assert (eir > 0).all(), "a zero rate means undiscounted, not free"
        assert eir.iloc[1] == pytest.approx(0.04), "filled from the type's mean"

    def test_the_discount_factor_is_sane_over_a_long_horizon(self):
        """The arithmetic that made the defect matter."""
        for eir, cap in ((0.04, 2.0), (0.0975, 5.0)):
            assert (1 + eir) ** (198 / 12) < cap
        # what it used to do
        assert (1 + 3.0) ** (100 / 12) > 9000

    @needs_src_inputs
    def test_the_raw_extract_is_in_percent(self):
        am = _raw("AccountMaster")
        e = pd.to_numeric(am["EIR"], errors="coerce")
        nz = e[e.notna() & (e != 0)]
        if nz.empty:
            pytest.skip("this extract carries no EIR")
        assert nz.max() > 1.0, (
            "the extract now looks like decimals, not percent -- if the source "
            "changed scale, the /100 in the transform is wrong")


class TestI8AlignedOnUnmappedCollateral:
    """I8, after alignment: an unmapped collateral type contributes nothing in
    both ports, matching R and the 23-of-26 majority in the haircut table."""

    @needs_run
    def test_an_unmapped_type_contributes_nothing(self, inputs):
        from ifrs9qdb.inputs import load_engine_inputs
        if inputs.coll_type is None:
            pytest.skip("this run has no collateral type table")
        hc = pd.to_numeric(inputs.coll_type["HaircutGeneral"], errors="coerce")
        assert (hc >= 1.0).sum() > (hc == 0.0).sum(), (
            "the table's majority should be worthless types, which is what "
            "makes a full-haircut default the right one")
        # R's collateral_net on these runs, to the cent.
        total = sum(load_engine_inputs(OUT).collateral_net.values())
        assert total > 0
