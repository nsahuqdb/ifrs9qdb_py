"""
Golden tests: the Python engine must reproduce LIC, contract for contract.

These are the same fixtures the R package uses, so a divergence between the two
ports fails here rather than turning up in a quarterly report. The four EY
reconciliation contracts were agreed with EY and match LIC exactly; the twelve
run-324 contracts span both stages, both payment types, both payment
frequencies and the collateral variants.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.engine import (
    EclConfig,
    compute_ecl_one,
    compute_lgd,
    fallback_ead_curve,
    months_to_maturity,
    sum_marginal_ecl,
)

FIX = Path(__file__).parent / "fixtures"


def load_pd_curves(name):
    d = pd.read_csv(FIX / name)
    out = {}
    for key, g in d.groupby("curve_key"):
        g = g.sort_values("month")
        out[key] = np.concatenate([[0.0], g["cum_pd"].to_numpy(dtype=float)])
    return out


def load_ead_curves(name):
    d = pd.read_csv(FIX / name)
    return {
        str(cid): g.sort_values("month")["ead"].to_numpy(dtype=float)
        for cid, g in d.groupby("cid")
    }


# ------------------------------------------------------------- EY goldens --
class TestEyReconciliation:
    """The four contracts reconciled with EY. Each matches LIC exactly."""

    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        return pd.read_csv(FIX / "ey_contracts.csv"), load_pd_curves("ey_pd_curves.csv")

    def test_all_four_match_lic(self, data):
        contracts, curves = data
        assert len(contracts) == 4, "expected the four EY contracts"
        for row in contracts.itertuples():
            cum = curves[row.curve_key]
            lgd = compute_lgd(row.on_balance, row.coll_cov * row.on_balance)
            ead = fallback_ead_curve(
                row.on_balance, row.months_remaining, row.payment_type,
                row.payment_frequency, row.deferral, horizon=row.horizon,
            )
            got = sum_marginal_ecl(ead, lgd, cum, row.eir, horizon=row.horizon)
            ratio = got / row.lic_ecl
            assert ratio == pytest.approx(1.0, abs=1e-4), (
                f"contract {row.contract_id}: got {got:,.4f}, "
                f"LIC {row.lic_ecl:,.4f}, ratio {ratio:.6f}"
            )

    def test_payment_type_4_amortises(self, data):
        """Type 4 must amortise linearly, not run flat.

        This is the point EY's prose answer got wrong: the generic "flat when
        no schedule" fallback does not apply when PaymentTypeId is 4. Holding
        the balance flat gives roughly 1.8x the LIC figure on contract 11.
        """
        contracts, curves = data
        row = contracts[contracts.contract_id == 11].iloc[0]
        cum = curves[row.curve_key]
        lgd = compute_lgd(row.on_balance, row.coll_cov * row.on_balance)
        linear = fallback_ead_curve(row.on_balance, row.months_remaining, 4,
                                    row.payment_frequency, row.deferral,
                                    horizon=row.horizon)
        flat = np.full(int(row.horizon), row.on_balance, dtype=float)
        ecl_linear = sum_marginal_ecl(linear, lgd, cum, row.eir, row.horizon)
        ecl_flat = sum_marginal_ecl(flat, lgd, cum, row.eir, row.horizon)
        assert ecl_linear == pytest.approx(row.lic_ecl, rel=1e-4)
        assert ecl_flat > row.lic_ecl * 1.5

    def test_bullet_holds_the_balance(self, data):
        contracts, _ = data
        row = contracts[contracts.contract_id == 22].iloc[0]
        ead = fallback_ead_curve(row.on_balance, row.months_remaining, 3, 1, 0,
                                 horizon=row.horizon)
        assert np.allclose(ead, row.on_balance)


# ---------------------------------------------------------- run 324 golden --
class TestRun324:
    """Twelve real contracts spanning both stages and both payment types."""

    @staticmethod
    @pytest.fixture(scope="class")
    def data():
        return (
            pd.read_csv(FIX / "run324_contracts.csv"),
            load_pd_curves("run324_pd_curves.csv"),
            load_ead_curves("run324_ead_curves.csv"),
        )

    def test_every_contract_matches(self, data):
        contracts, pd_curves, ead_curves = data
        failures = []
        for row in contracts.itertuples():
            key = f"{row.portfolio}|{int(row.bucket)}"
            cum = pd_curves.get(key)
            if cum is None:
                failures.append(f"{row.cid}: no PD curve for {key}")
                continue
            # the fixture carries the LGD the engine computed, so the collateral
            # convention is pinned as well as the ECL
            lgd = compute_lgd(row.onbal, row.collcov * row.onbal)
            assert lgd == pytest.approx(row.lgd, abs=1e-6), (
                f"{row.cid}: LGD {lgd:.6f} vs expected {row.lgd:.6f}"
            )
            ead = ead_curves.get(str(row.cid))
            if ead is None:
                ead = fallback_ead_curve(row.onbal, int(row.H), row.pt, row.f, 0,
                                         horizon=int(row.H))
            got = sum_marginal_ecl(ead, lgd, cum, row.eir, horizon=int(row.H))
            if row.lic and abs(got / row.lic - 1) > 0.01:
                failures.append(
                    f"{row.cid} ({row.scenario}): got {got:,.2f} vs LIC "
                    f"{row.lic:,.2f} ({100 * (got / row.lic - 1):+.2f}%)"
                )
        assert not failures, "\n" + "\n".join(failures)

    def test_the_fixture_covers_both_stages_and_payment_types(self):
        c = pd.read_csv(FIX / "run324_contracts.csv")
        assert set(c["stage"]) >= {1, 2}
        assert set(c["pt"]) >= {3, 4}
        assert set(c["f"]) >= {1, 3}


# ------------------------------------------------------------- primitives --
class TestLgd:
    def test_unsecured_uses_the_base_rate(self):
        assert compute_lgd(100, 0) == pytest.approx(0.45)
        assert compute_lgd(100, 0, base=0.55) == pytest.approx(0.55)

    def test_the_floor_binds_on_secured_lending(self):
        assert compute_lgd(100, 100) == pytest.approx(0.225)
        assert compute_lgd(100, 75) == pytest.approx(0.225)
        assert compute_lgd(100, 50) == pytest.approx(0.225)
        assert compute_lgd(100, 25) == pytest.approx(0.3375)

    def test_raising_the_floor_hits_secured_not_unsecured(self):
        """Counter-intuitive, so pinned: only contracts already at the floor move."""
        assert compute_lgd(100, 0, unsecured_floor=0.7) == compute_lgd(100, 0)
        assert compute_lgd(100, 100, unsecured_floor=0.7) > compute_lgd(100, 100)

    def test_zero_or_missing_exposure(self):
        assert compute_lgd(0, 0) == 1.0
        assert compute_lgd(-5, 0) == 1.0
        assert compute_lgd(float("nan"), 0) == 1.0


class TestSumMarginalEcl:
    def test_zero_prepended_convention(self):
        """The first marginal PD is cumPD(1) - 0, and month 0 is undiscounted."""
        ead = [1000.0] * 3
        cum = [0.0, 0.1, 0.2, 0.3]
        got = sum_marginal_ecl(ead, 0.5, cum, 0.0)
        assert got == pytest.approx(1000 * 0.5 * 0.3)

    def test_discounting_starts_at_month_zero(self):
        ead = [1000.0, 1000.0]
        cum = [0.0, 0.1, 0.2]
        got = sum_marginal_ecl(ead, 1.0, cum, 0.12)
        expected = 1000 * 0.1 / (1.12 ** 0) + 1000 * 0.1 / (1.12 ** (1 / 12))
        assert got == pytest.approx(expected)

    def test_missing_curve_gives_nan(self):
        assert np.isnan(sum_marginal_ecl(None, 0.45, [0, 0.1], 0.05))
        assert np.isnan(sum_marginal_ecl([100], 0.45, None, 0.05))
        assert np.isnan(sum_marginal_ecl([], 0.45, [0, 0.1], 0.05))

    def test_horizon_never_runs_past_either_curve(self):
        got = sum_marginal_ecl([100.0] * 10, 0.5, [0.0, 0.1, 0.2], 0.0, horizon=10)
        assert got == pytest.approx(100 * 0.5 * 0.2)


class TestFallbackCurve:
    def test_bullet_is_flat(self):
        assert np.allclose(fallback_ead_curve(1000, 6, 3), 1000.0)

    def test_linear_runs_down_to_zero(self):
        c = fallback_ead_curve(1000, 5, 4, 1, 0)
        assert c[0] == pytest.approx(1000.0)
        assert c[-1] == pytest.approx(200.0)
        assert all(np.diff(c) <= 1e-9)

    def test_deferral_holds_the_balance_first(self):
        c = fallback_ead_curve(1000, 12, 4, 1, deferral=3)
        assert np.allclose(c[:4], 1000.0)
        assert c[-1] < 1000.0

    def test_frequency_steps_the_amortisation(self):
        """Floor convention: N // f steps, which matches actual LIC output."""
        c = fallback_ead_curve(1200, 12, 4, payment_frequency=3)
        assert c[0] == pytest.approx(1200.0)
        assert c[1] == pytest.approx(1200.0)
        assert c[2] == pytest.approx(1200.0)
        assert c[3] < 1200.0


class TestStageHandling:
    def test_stage_one_caps_at_twelve_months(self):
        r = compute_ecl_one(
            contract_id="X", stage=1, on_balance=1000, collateral_net=0,
            cum_pd=[0.0] + [0.01 * i for i in range(1, 60)], eir=0.05,
            months_remaining=48, payment_type=3,
        )
        assert r["horizon"] == 12

    def test_stage_three_books_full_outstanding_by_default(self):
        r = compute_ecl_one(
            contract_id="X", stage=3, on_balance=1000, collateral_net=500,
            cum_pd=[0.0, 0.5], eir=0.05, months_remaining=12, payment_type=3,
        )
        assert r["ecl"] == pytest.approx(1000.0)
        assert r["coverage"] == pytest.approx(1.0)

    def test_stage_three_zero_reproduces_lic(self):
        r = compute_ecl_one(
            contract_id="X", stage=3, on_balance=1000, collateral_net=0,
            cum_pd=[0.0, 0.5], eir=0.05, months_remaining=12, payment_type=3,
            cfg=EclConfig(stage3_method="zero"),
        )
        assert r["ecl"] == 0.0

    def test_the_cap_holds(self):
        """An EAD curve carrying undrawn commitments can exceed the balance."""
        big = [50_000.0] * 12
        r = compute_ecl_one(
            contract_id="X", stage=2, on_balance=1000, collateral_net=0,
            cum_pd=[0.0] + [0.1 * i for i in range(1, 13)], eir=0.0,
            months_remaining=12, schedule_curves={"X": big},
        )
        assert r["ecl"] == pytest.approx(1000.0)
        assert r["coverage"] <= 1.0

    def test_uncapped_when_switched_off(self):
        big = [50_000.0] * 12
        r = compute_ecl_one(
            contract_id="X", stage=2, on_balance=1000, collateral_net=0,
            cum_pd=[0.0] + [0.1 * i for i in range(1, 13)], eir=0.0,
            months_remaining=12, schedule_curves={"X": big},
            cfg=EclConfig(cap_ecl_at_exposure=False),
        )
        assert r["ecl"] > 1000.0


class TestMonthsToMaturity:
    def test_calendar_months(self):
        import datetime as dt
        assert months_to_maturity(dt.date(2029, 4, 30), dt.date(2026, 6, 30)) == 34

    def test_matured_gets_the_floor(self):
        import datetime as dt
        assert months_to_maturity(dt.date(2026, 6, 30), dt.date(2026, 6, 30)) == 3
        assert months_to_maturity(dt.date(2025, 1, 1), dt.date(2026, 6, 30)) == 3

    def test_counted_as_lic_counts(self):
        """LIC runs 324 and 330: a part month is a whole one, and only a
        facility at or past maturity gets the 3-month horizon."""
        import datetime as dt
        from ifrs9qdb.etl.report import _months_to_maturity
        ext = dt.date(2026, 9, 30)
        cases = {dt.date(2026, 10, 12): 1, dt.date(2026, 11, 30): 2,
                 dt.date(2026, 10, 31): 2, dt.date(2027, 3, 31): 7,
                 dt.date(2026, 12, 30): 3, dt.date(2027, 2, 28): 5,
                 dt.date(2026, 9, 30): 3}
        for mat, want in cases.items():
            assert months_to_maturity(mat, ext) == want, mat
            assert _months_to_maturity(pd.Timestamp(mat), pd.Timestamp(ext)) == want, mat
