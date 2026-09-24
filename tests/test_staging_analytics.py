"""Staging and migration analytics.

The property that matters most here is the last one: re-applying the engine's
own staging rule to a finished report must reproduce the stage the report
carries. On the real book it does, for every contract -- which is the strongest
statement available that this port's staging rule IS the R engine's.

The rest guard the counting. Every one of these was a real defect somewhere:
overlapping triggers double-counted, a customer counted once per facility, a
migration matrix that dropped the thin tails.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.analytics import (
    customer_stage_migration, customer_view, dpd_by_stage, migration_summary,
    normalise, rating_migration, stage2_trigger_overlap, stage3_drivers,
    stage_movers, staging_consistency,
)

from conftest import prev_file, ref_file

REPORT = ref_file("FinalEclReport.csv")
PREV = prev_file("FinalEclReport.csv")
real_only = pytest.mark.skipif(
    REPORT is None,
    reason="set IFRS9_REF_RUN, or add tests/fixtures/FinalEclReport.csv")
two_runs = pytest.mark.skipif(
    REPORT is None or PREV is None,
    reason="set IFRS9_REF_RUN and IFRS9_REF_RUN_PREV to two real runs")

QDB_SCALE = ["QDB 1", "QDB 2", "QDB 3", "QDB 4", "QDB 5", "QDB 6", "QDB 7",
             "QDB 8", "QDB 9", "QDB 10"]


def _load(p):
    return normalise(pd.read_csv(p, low_memory=False))


@pytest.fixture(scope="module")
def rep():
    return _load(REPORT) if REPORT is not None else None


@pytest.fixture(scope="module")
def pair():
    if REPORT is None or PREV is None:
        return None, None
    return _load(PREV), _load(REPORT)


def synthetic():
    """Two customers, five facilities, every trigger represented once."""
    return normalise(pd.DataFrame({
        "Contract Id": ["C1", "C2", "C3", "C4", "C5"],
        "Customer Id": ["A", "A", "B", "C", "D"],
        "Portfolio Code": ["Business Finance", "Business Finance",
                           "Business Finance", "Tasdeer", "Business Finance"],
        "Rating": ["QDB 5", "QDB 5", "QDB 8", "QDB 3", "QDB 9"],
        "Ifrs Stage": [2, 2, 2, 2, 3],
        "Exposure On Bal": [1000.0, 500.0, 2000.0, 300.0, 800.0],
        "Cla Amount Onbal": [100.0, 50.0, 400.0, 9.0, 800.0],
        "Past Due Days": [70, 0, 0, 0, 200],
        "Watchlist Flag": [0, 0, 1, 0, 1],
        "Local Flag 1": [0, 0, 1, 0, 0],
        "Default Flag": [0, 0, 0, 0, 1],
    }))


class TestDpdByStage:
    def test_every_stage_gets_every_band(self):
        """A missing row reads as zero customers; an absent row reads as a bug."""
        t = dpd_by_stage(synthetic())
        assert len(t) == t["stage"].nunique() * 5
        assert list(t["band"].unique()) == ["0", "1-30", "31-60", "61-90", "90+"]

    @real_only
    def test_counts_customers_not_contracts(self, rep):
        t = dpd_by_stage(rep)
        assert t["customers"].sum() == len(customer_view(rep))
        assert t["customers"].sum() < len(rep)

    @real_only
    def test_stage_3_sits_in_the_worst_band(self, rep):
        """Over 90 days is a Stage 3 trigger, so the converse should be rare."""
        s3 = t3 = dpd_by_stage(rep)
        s3 = t3[t3["stage"] == "Stage 3"]
        assert s3.loc[s3["band"] == "90+", "customers"].iloc[0] > 0


class TestStage2TriggerOverlap:
    def test_combinations_partition_the_stage(self):
        """Overlapping triggers counted once each, so the rows are additive."""
        d = synthetic()
        o = stage2_trigger_overlap(d)
        assert o["contracts"].sum() == int((d["stage"] == 2).sum())

    def test_a_multi_trigger_contract_appears_once(self):
        o = stage2_trigger_overlap(synthetic())
        combos = set(o["combination"])
        assert "Watchlist + Restructured" in combos
        assert o.loc[o["combination"] == "Watchlist + Restructured",
                     "contracts"].iloc[0] == 1

    def test_an_untriggered_stage_2_is_named_not_dropped(self):
        """Contagion is why C2 is there. Dropping it would break the total."""
        o = stage2_trigger_overlap(synthetic())
        assert "Contagion or override" in set(o["combination"])

    @real_only
    def test_the_real_book_adds_up_and_is_sorted(self, rep):
        o = stage2_trigger_overlap(rep)
        assert o["contracts"].sum() == \
            int((pd.to_numeric(rep["stage"], errors="coerce") == 2).sum())
        assert (o["contracts"].diff().dropna() <= 0).all()


class TestStage3Drivers:
    @real_only
    def test_shares_are_of_stage_3_customers(self, rep):
        """They do NOT add to 100: one customer trips several drivers."""
        t = stage3_drivers(rep)
        n3 = int((pd.to_numeric(customer_view(rep)["stage"],
                                errors="coerce") == 3).sum())
        assert (t["customers"] <= n3).all()
        for _, r in t.iterrows():
            assert r["pct_customers"] == pytest.approx(100 * r["customers"] / n3)

    def test_no_stage_3_returns_empty(self):
        d = synthetic()
        d = d[d["stage"] != 3]
        assert len(stage3_drivers(d)) == 0


class TestRatingMigration:
    @two_runs
    def test_the_diagonal_holds_most_of_the_book(self, pair):
        m = rating_migration(*pair)
        stable = m.loc[m["direction"] == "stable", "n"].sum()
        assert stable / m["n"].sum() > 0.8

    @two_runs
    def test_direction_follows_the_scale_not_the_alphabet(self, pair):
        """Without levels the axes are frequency-ordered and direction is noise."""
        m = rating_migration(*pair, levels=QDB_SCALE)
        for _, r in m.iterrows():
            a, b = QDB_SCALE.index(r["from"]), QDB_SCALE.index(r["to"])
            expect = "stable" if a == b else ("downgrade" if b > a else "upgrade")
            assert r["direction"] == expect

    @two_runs
    def test_by_customer_counts_fewer_than_by_contract(self, pair):
        """Rating is a customer attribute; a contract view inflates every cell."""
        cust = rating_migration(*pair, by_customer=True)["n"].sum()
        ctr = rating_migration(*pair, by_customer=False)["n"].sum()
        assert cust < ctr

    @two_runs
    def test_the_axis_order_travels_with_the_result(self, pair):
        m = rating_migration(*pair, levels=QDB_SCALE)
        assert m.attrs["levels"] == [r for r in QDB_SCALE
                                     if r in set(m["from"]) | set(m["to"])]

    @two_runs
    def test_the_summary_covers_the_whole_scale(self, pair):
        """top=12 would silently drop the thin grades the summary is asked about."""
        s = migration_summary(*pair)
        full = rating_migration(*pair, top=100)
        assert s["n"].sum() == full["n"].sum()
        assert s["n"].sum() >= rating_migration(*pair, top=12)["n"].sum()
        assert list(s["direction"]) == [d for d in
                                        ("upgrade", "stable", "downgrade")
                                        if d in set(s["direction"])]

    def test_an_unrated_pair_is_dropped_not_grouped_as_blank(self):
        a = normalise(pd.DataFrame({
            "Contract Id": ["C1", "C2"], "Customer Id": ["A", "B"],
            "Rating": ["QDB 5", ""], "Exposure On Bal": [10.0, 10.0],
            "Cla Amount Onbal": [1.0, 1.0], "Ifrs Stage": [1, 1]}))
        b = a.copy()
        m = rating_migration(a, b)
        assert set(m["from"]) == {"QDB 5"}


class TestStageMigration:
    @two_runs
    def test_the_matrix_counts_every_customer_in_both_runs(self, pair):
        prev, curr = pair
        m = customer_stage_migration(*pair)
        both = set(customer_view(prev)["customer"]) & \
            set(customer_view(curr)["customer"])
        assert m["customers"].sum() == len(both)

    @two_runs
    def test_the_movers_are_exactly_the_off_diagonal(self, pair):
        m = customer_stage_migration(*pair)
        moved = int(m.loc[m["from"] != m["to"], "customers"].sum())
        assert len(stage_movers(*pair)) == moved

    @two_runs
    def test_movers_are_ordered_by_what_they_cost(self, pair):
        mv = stage_movers(*pair)
        assert (mv["ecl_change"].abs().diff().dropna() <= 1e-9).all()
        up = mv[mv["direction"] == "deteriorated"]
        assert (up["stage_curr"] > up["stage_prev"]).all()

    def test_no_movement_returns_empty(self):
        d = synthetic()
        assert len(stage_movers(d, d)) == 0
        assert customer_stage_migration(d, d)["customers"].sum() == \
            len(customer_view(d))


class TestStagingConsistency:
    @real_only
    def test_the_engines_own_rule_reproduces_the_reported_stage(self, rep):
        """The port's staging rule against a finished R run, contract by
        contract. A single mismatch here means the two engines disagree."""
        c = staging_consistency(rep)
        assert c["rule_available"]
        assert c["checked"] == len(rep)
        assert c["mismatches"] == 0, c["findings"]

    def test_an_override_shows_up_as_staged_worse(self):
        """The report does not carry AccountMaster.Stage, so this is expected
        rather than an error -- but it must be visible and separable."""
        d = synthetic()
        d.loc[d["contract"] == "C2", "stage"] = 3
        d.loc[d["contract"] == "C2", "default_flag"] = 0
        c = staging_consistency(d)
        assert c["mismatches"] == 1
        f = c["findings"]
        assert list(f["check"]) == ["Staged worse than the rule alone implies"]
        assert f["severity"].iloc[0] == "info"

    def test_staged_better_than_the_rule_is_a_warning(self):
        d = synthetic()
        d.loc[d["contract"] == "C5", "stage"] = 1
        c = staging_consistency(d)
        f = c["findings"]
        assert list(f["check"]) == ["Staged better than the rule implies"]
        assert f["severity"].iloc[0] == "warn"
        assert f["pct"].iloc[0] == pytest.approx(100 / len(d))

    def test_agreement_reports_no_findings_rather_than_an_empty_row(self):
        c = staging_consistency(synthetic())
        assert c["mismatches"] == 0
        assert len(c["findings"]) == 0

    def test_a_report_with_no_stage_column_is_survivable(self):
        d = synthetic().drop(columns=["watchlist"])
        assert staging_consistency(d)["rule_available"]
