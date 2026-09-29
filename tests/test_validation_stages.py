"""The staged validation suite, and its parity with the R engine.

The ids are the contract. A suppression approved against the R engine is
recorded by id, so if the Python invents its own the exception silently stops
applying and a finding the bank accepted comes back as a failure with no
explanation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ifrs9qdb.validation as V
from conftest import ref_output
from ifrs9qdb.etl.static_ref import load_static_reference
from ifrs9qdb.validation.framework import run_suite

R_IDS = Path(__file__).parent / "r_validator_ids.txt"


@pytest.fixture(scope="module")
def static():
    return load_static_reference()


class TestTheCatalogue:
    def test_the_run_suite_has_the_same_shape_as_the_r_report(self):
        """R's validation.csv on a phased run: 149 checks -- INPUT 70,
        TRANSFORM 28, DERIVED 29, READY 20, REPORT 2."""
        from ifrs9qdb.validation.readiness import (READY_STAGE_VALIDATORS,
                                                   REPORT_STAGE_VALIDATORS)
        assert len(V.INPUT_STAGE) == 70
        assert len(V.TRANSFORM_STAGE) == 28
        assert len(V.DERIVED_STAGE) == 29
        assert len(V.STAGE_VALIDATORS) == 127
        assert len(READY_STAGE_VALIDATORS) == 20
        assert len(REPORT_STAGE_VALIDATORS) == 2

    def test_ids_are_unique(self):
        ids = [v.id for v in V.STAGE_VALIDATORS + V.PREFLIGHT_VALIDATORS]
        assert len(set(ids)) == len(ids)

    @pytest.mark.skipif(not R_IDS.is_file(), reason="R id list not bundled")
    def test_every_r_validator_id_exists_here(self):
        """Parity, by id. Suppressions key on these."""
        from ifrs9qdb.validation.readiness import (READY_STAGE_VALIDATORS,
                                                   REPORT_STAGE_VALIDATORS)
        r = {line.strip() for line in R_IDS.read_text().split() if line.strip()}
        mine = {v.id for v in V.STAGE_VALIDATORS + V.PREFLIGHT_VALIDATORS
                + READY_STAGE_VALIDATORS + REPORT_STAGE_VALIDATORS}
        assert not (r - mine), f"missing from the port: {sorted(r - mine)}"
        assert not (mine - r), f"not in R: {sorted(mine - r)}"

    def test_every_check_says_what_to_do_about_it(self):
        """A finding with no remediation gets ignored on the second quarter."""
        thin = [v.id for v in V.STAGE_VALIDATORS
                if not v.rationale or not v.remediation]
        assert not thin, f"no rationale or remediation: {thin}"

    def test_severities_are_valid(self):
        for v in V.STAGE_VALIDATORS + V.PREFLIGHT_VALIDATORS:
            assert v.severity in ("ERROR", "WARN", "INFO")

    def test_a_missing_file_can_never_be_suppressed(self):
        """Some findings are not negotiable. (INPUT_<file>_present; a field
        check such as INPUT_AccountMaster_eir_present is a data finding.)"""
        import re
        files = [v for v in V.STAGE_VALIDATORS
                 if re.fullmatch(r"INPUT_[A-Za-z]+_present", v.id)]
        assert len(files) == 12
        for v in files:
            assert not v.suppressible, v.id


class TestTheChecksActuallyCatchThings:
    """A check that cannot fail is not a check."""

    @staticmethod
    def _run(validators, args):
        return run_suite("t", validators, args, {})

    def test_a_duplicate_contract_id_is_caught(self):
        class FakeInputs:
            tables = {}
            missing = []

            def __getitem__(self, k):
                return self.tables.get(k, pd.DataFrame())
        inp = FakeInputs()
        inp.tables["AccountMaster"] = pd.DataFrame({
            "CONTRACTID": [1, 2, 2], "CUSTOMERID": [10, 11, 12],
            "ONBALANCE": [1.0, 2.0, 3.0]})
        res = self._run(V.INPUT_STAGE, {"inputs": inp})
        bad = {i.id for i in res.issues if not i.passed}
        assert "INPUT_AccountMaster_contractid_unique" in bad

    def test_a_float_read_id_still_joins(self):
        """548840 and 548840.0 are the same contract."""
        class FakeInputs:
            tables = {}
            missing = []

            def __getitem__(self, k):
                return self.tables.get(k, pd.DataFrame())
        inp = FakeInputs()
        inp.tables["AccountMaster"] = pd.DataFrame({"CONTRACTID": [548840, 604177]})
        inp.tables["AccountCollateralAllocation"] = pd.DataFrame({
            "CONTRACTID": [548840.0, 604177.0], "COLLATERALID": [1, 2],
            "ALLOCATIONPERCENTAGE": [10.0, 20.0]})
        res = self._run(V.INPUT_STAGE, {"inputs": inp})
        bad = {i.id for i in res.issues if not i.passed}
        assert "INPUT_ACA_contract_fk" not in bad

    # The TRANSFORM checks read R's cm_view: one row per CustomerMaster
    # customer, with dpd_status, watchlist_status ("Watchlist" or "") and
    # restructuring_final ("Restructured" or "").
    def test_a_watchlisted_stage_1_customer_is_caught(self):
        cm_view = pd.DataFrame({
            "customer_id": ["a", "b"], "dpd_status": [10, 10],
            "watchlist_status": ["Watchlist", ""],
            "restructuring_final": ["", ""],
            "stage_final": ["Stage 1", "Stage 1"]})
        res = self._run(V.TRANSFORM_STAGE, {"cm_view": cm_view})
        bad = {i.id for i in res.issues if not i.passed}
        assert "TRANS_LENDPV_watchlist_implies_stage_2_or_3" in bad

    def test_dpd_over_90_not_stage_3_is_caught(self):
        cm_view = pd.DataFrame({
            "customer_id": ["a"], "dpd_status": [120],
            "watchlist_status": [""], "restructuring_final": [""],
            "stage_final": ["Stage 2"]})
        res = self._run(V.TRANSFORM_STAGE, {"cm_view": cm_view})
        bad = {i.id for i in res.issues if not i.passed}
        assert "TRANS_LENDPV_dpd_gt_90_implies_stage3" in bad

    def test_a_contract_whose_customer_is_not_in_customermaster_is_caught(self):
        """R failed three TRANSFORM checks here that the old port passed."""
        trans_l = pd.DataFrame({
            "contract_id": ["1", "2"], "customer_id": ["a", "z"],
            "exposure_amount": [100.0, 50.0],
            "rating_after_override": ["QDB 3", None],
            "is_default_final": [0, None]})
        cm_view = pd.DataFrame({
            "customer_id": ["a"], "exposure_total": [100.0],
            "dpd_status": [0], "watchlist_status": [""],
            "restructuring_final": [""], "stage_final": ["Stage 1"]})
        res = self._run(V.TRANSFORM_STAGE,
                        {"trans_l": trans_l, "cm_view": cm_view})
        bad = {i.id for i in res.issues if not i.passed}
        assert {"TRANS_LEND_pass6_overrides_filled",
                "TRANS_LENDPV_customer_count_matches_trans",
                "TRANS_LENDPV_exposure_reconciles"} <= bad

    def test_weights_that_do_not_sum_to_one_are_caught(self):
        res = self._run(V.DERIVED_STAGE,
                        {"internal_weights": {"a": 0.2, "b": 0.2}})
        bad = {i.id for i in res.issues if not i.passed}
        assert "DERIVED_SCEN_internal_weights_sum_to_one" in bad

    def test_a_falling_pd_curve_is_caught(self):
        stpd = pd.DataFrame({
            "ExtractDate": ["12/31/2025"] * 4,
            "PortfolioCode": ["Business Finance"] * 4,
            "PDBucketDim1": [1, 1, 1, 1], "PDBucketDim2": ["", "", "", ""],
            "MonthLifetime": [1, 2, 3, 4],
            "PDLifetime": [0.01, 0.02, 0.015, 0.03]})
        res = self._run(V.DERIVED_STAGE, {"stpd": stpd})
        bad = {i.id for i in res.issues if not i.passed}
        assert "DERIVED_STPD_pd_monotone_non_decreasing" in bad

    def test_a_short_stpd_is_caught(self):
        stpd = pd.DataFrame({
            "ExtractDate": ["12/31/2025"], "PortfolioCode": ["Business Finance"],
            "PDBucketDim1": [1], "PDBucketDim2": [""], "MonthLifetime": [1],
            "PDLifetime": [0.01]})
        res = self._run(V.DERIVED_STAGE, {"stpd": stpd})
        bad = {i.id for i in res.issues if not i.passed}
        assert "DERIVED_STPD_row_count" in bad
        assert "DERIVED_STPD_portfolio_set_complete" in bad

    def test_a_curve_with_a_gap_is_caught(self):
        ltpo = pd.DataFrame({
            "ExtractDate": ["12/31/2025"] * 3, "ContractId": [1, 1, 1],
            "MonthLifetime": [0, 1, 3], "EADLifetime": [100.0, 90.0, 80.0],
            "LGDLifetime": [None] * 3, "PaymentScheduleLifetime": [None] * 3,
            "TotalLimitLifetime": [None] * 3})
        res = self._run(V.DERIVED_STAGE, {"ltpo": ltpo})
        bad = {i.id for i in res.issues if not i.passed}
        assert "DERIVED_LTPO_months_contiguous" in bad

    def test_a_broken_static_table_is_caught(self):
        res = run_suite("p", V.PREFLIGHT_VALIDATORS,
                        {"static": {"ttc_pd_table": pd.DataFrame(
                            {"rating": ["x"], "ttc_pd": [1.5]})}}, {})
        bad = {i.id for i in res.issues if not i.passed}
        assert "STATIC_ttc_pd_in_unit_interval" in bad


class TestSuppressions:
    def test_a_suppressed_failure_is_recorded_and_stops_gating(self):
        stpd = pd.DataFrame({
            "ExtractDate": ["12/31/2025"], "PortfolioCode": ["Business Finance"],
            "PDBucketDim1": [1], "PDBucketDim2": [""], "MonthLifetime": [1],
            "PDLifetime": [0.01]})
        supp = {"DERIVED_STPD_row_count": "known short in the test fixture"}
        res = run_suite("t", V.DERIVED_STAGE, {"stpd": stpd}, supp)
        issue = next(i for i in res.issues if i.id == "DERIVED_STPD_row_count")
        assert not issue.passed          # still ran, still failed
        assert issue.suppressed          # and still recorded
        assert issue.suppression_reason
        assert V.effective_severity(issue) == "INFO"

    def test_an_expired_suppression_stops_applying(self, tmp_path):
        p = tmp_path / "s.yml"
        V.add_suppression(p, "INPUT_RS_coverage", "accepted for Q1", "priya",
                          valid_until="2020-01-01")
        assert V.active_suppression_ids(V.load_suppressions(p)) == []

    def test_a_suppression_needs_a_reason_and_an_approver(self, tmp_path):
        p = tmp_path / "s.yml"
        with pytest.raises(ValueError, match="reason"):
            V.add_suppression(p, "INPUT_RS_coverage", "", "priya")
        with pytest.raises(ValueError, match="approved_by"):
            V.add_suppression(p, "INPUT_RS_coverage", "because", "")

    def test_a_malformed_file_does_not_stop_the_run(self, tmp_path):
        """Better to validate with nothing suppressed than not to validate."""
        p = tmp_path / "s.yml"
        p.write_text("suppressions: [[[not yaml")
        assert len(V.load_suppressions(p)) == 0


class TestTheReport:
    def test_the_csv_has_the_r_columns(self, tmp_path, static):
        res = V.validate_stages(static=static, stages=["DERIVED"])
        out = V.write_validation_reports(res, tmp_path)
        frame = pd.read_csv(out["csv"])
        assert list(frame.columns) == [
            "stage", "id", "severity", "effective_severity", "context",
            "description", "rationale", "remediation", "suppressed", "passed",
            "message"]
        assert out["md"].is_file()
        assert "# Validation Report" in out["md"].read_text()

    def test_passes_are_kept_not_only_failures(self, static):
        """A report showing only failures is not evidence anything was checked."""
        res = V.validate_stages(static=static, stages=["DERIVED"])
        assert len(res.issues) == 29
        assert any(i.passed for i in res.issues)


@pytest.mark.skipif(ref_output() is None,
                    reason="set IFRS9_REF_RUN to a run folder holding Output/")
class TestAgainstTheReferenceRun:
    """The checks have to accept the R engine's own output."""

    def test_the_derived_suite_passes_on_the_reference_run(self, static):
        import yaml

        from ifrs9qdb.etl.macro import (external_gcc_forecast,
                                        gcc_weighted_history,
                                        resolve_external_scenario_weights,
                                        resolve_internal_scenario_weights)
        from conftest import packaged_config
        cfg = packaged_config()
        mc = yaml.safe_load((cfg / "model.yml").read_text())
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        out = ref_output()
        gh = gcc_weighted_history(static["gcc_real_gdp_growth"],
                                  static["gcc_gdp_current_prices"])
        res = V.validate_stages(
            static=static,
            ltpo=pd.read_csv(out / "LifeTimeParameterOther.csv", low_memory=False),
            stpd=pd.read_csv(out / "StPD.csv", low_memory=False),
            internal_weights=resolve_internal_scenario_weights(mi, static),
            external_weights=resolve_external_scenario_weights(
                mi, static, gh, external_gcc_forecast(mi, static, 5)),
            mev_weights=[c["weight"] for c in
                         mc["models"]["internal_v4_production"]["mev_components"]],
            stages=["DERIVED"])
        summary = res.summary()
        assert summary["errors"] == 0, [i.id for i in res.by_severity("ERROR")]
        assert summary["warnings"] == 0, [i.id for i in res.by_severity("WARN")]

    def test_the_preflight_suite_passes_on_the_packaged_static(self, static):
        res = V.validate_preflight(static=static)
        assert res.summary()["errors"] == 0
