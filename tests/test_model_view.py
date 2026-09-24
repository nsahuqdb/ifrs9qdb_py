"""What a run froze: its config, its scenarios, its macro forecast.

A run explains itself from ``config_used/`` or not at all. Reading today's
config to explain last quarter's number is the failure these guard against,
so the tests assert the frozen copy is what gets read and that a run without
one says so rather than answering from the package.
"""
import pandas as pd
import pytest

from ifrs9qdb.analytics import (
    config_used, mev_forecast_table, mev_weights_table, read_scenario_stpd,
    scenario_severity, scenario_weights,
)

from conftest import ref_output

OUT = ref_output()
needs_run = pytest.mark.skipif(
    OUT is None, reason="set IFRS9_REF_RUN to a run folder holding Output/")
needs_frozen = pytest.mark.skipif(
    OUT is None or config_used(OUT) is None,
    reason="the reference run has no config_used/")


class TestConfigUsed:
    @needs_run
    def test_it_is_found_from_the_output_folder(self):
        """Callers hold Output/ or the run root depending on where they came
        from; config_used sits beside Output/, not inside it."""
        cu = config_used(OUT)
        assert cu is not None
        assert cu["config"].is_dir() and cu["static"].is_dir()
        assert config_used(OUT.parent) == cu

    def test_a_run_without_one_says_so(self, tmp_path):
        (tmp_path / "Output").mkdir()
        assert config_used(tmp_path / "Output") is None
        assert config_used(None) is None

    def test_a_half_frozen_copy_does_not_count(self, tmp_path):
        """config without static is not enough to rebuild anything."""
        (tmp_path / "config_used" / "config").mkdir(parents=True)
        assert config_used(tmp_path) is None


class TestScenarios:
    @needs_frozen
    def test_severity_runs_worst_to_best(self):
        s = scenario_severity(OUT)
        assert len(s) == 5
        assert (s["severity_z"].diff().dropna() > 0).all()
        assert s["scenario"].iloc[0] == "Significant Downturn"
        assert s["scenario"].iloc[-1] == "Significant Uptrend"

    @needs_frozen
    def test_both_scales_carry_their_own_weights(self):
        """One set for both is the defect this exists to make visible."""
        w = scenario_weights(config_used(OUT)["config"])
        assert set(w.columns) == {"scenario", "internal", "external"}
        assert len(w) == 5
        assert not w["internal"].equals(w["external"])

    @needs_frozen
    def test_the_weights_are_used_as_written_not_normalised(self):
        """V4 states weights summing to 1.0003. Normalising them here would
        silently disagree with the engine, which does not."""
        w = scenario_weights(config_used(OUT)["config"])
        assert w["internal"].sum() == pytest.approx(1.0, abs=5e-3)
        assert w["external"].sum() == pytest.approx(1.0, abs=5e-3)

    def test_a_missing_config_is_empty_not_an_error(self, tmp_path):
        assert len(scenario_weights(tmp_path)) == 0
        assert len(scenario_weights(None)) == 0
        assert len(scenario_severity(tmp_path)) == 0


class TestScenarioStpd:
    @needs_run
    def test_the_curves_are_zero_prepended_and_monotonic(self):
        """sum_marginal_ecl indexes cumPD[m] as month m, so curve[0] is zero."""
        c = read_scenario_stpd(OUT, "Base Case")
        if not c:
            pytest.skip("this run wrote no per-scenario StPD")
        curve = next(iter(c.values()))
        assert curve[0] == 0.0
        assert all(b >= a - 1e-12 for a, b in zip(curve, curve[1:]))
        assert curve[-1] <= 1.0

    @needs_run
    def test_the_downturn_prices_worse_than_the_base(self):
        base = read_scenario_stpd(OUT, "Base Case")
        down = read_scenario_stpd(OUT, "Significant Downturn")
        if not base or not down:
            pytest.skip("this run wrote no per-scenario StPD")
        shared = set(base) & set(down)
        assert shared
        worse = sum(down[k][-1] > base[k][-1] for k in shared)
        assert worse > len(shared) / 2

    @needs_run
    def test_an_unknown_scenario_is_empty(self):
        assert read_scenario_stpd(OUT, "Mild Panic") == {}
        assert read_scenario_stpd(OUT, "") == {}


class TestMev:
    @needs_frozen
    def test_the_forecast_is_labelled_from_the_model_not_by_position(self):
        """"MEV 2" is not an answer to "which variable moved"."""
        t = mev_forecast_table(OUT)
        assert len(t) > 0
        assert t["mev"].notna().all()
        assert (t["label"] != t["idx"].astype(str)).all()
        assert t["year"].min() == 1

    @needs_frozen
    def test_every_forecast_year_covers_every_mev(self):
        t = mev_forecast_table(OUT)
        counts = t.groupby("idx")["year"].nunique()
        assert counts.nunique() == 1

    @needs_frozen
    def test_the_weights_sum_to_one_over_the_components(self):
        w = mev_weights_table(OUT)
        assert len(w) == len(mev_forecast_table(OUT)["idx"].unique())
        assert w["weight"].sum() == pytest.approx(1.0)
        assert w["coefficient"].notna().all()

    def test_no_frozen_config_means_no_table(self, tmp_path):
        assert len(mev_forecast_table(tmp_path)) == 0
        assert len(mev_weights_table(tmp_path)) == 0
