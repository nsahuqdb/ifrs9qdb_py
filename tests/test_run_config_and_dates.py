"""The run config, the reporting date and the model, taken as R takes them.

Found by driving the app against the R engine: the port checked a different
set of config.yml paths than R (so the pre-run check refused runs R allows),
ignored run.internal_model and mev_model_weights.mode, dated a run from the
LATEST EXTRACTDA where R takes its own resolve_input_extract_date(), and
stamped the collateral files with their own dates rather than the run's.
Every expectation below was read off the R package, most by running it.

All synthetic: nothing here needs portfolio data.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from ifrs9qdb.etl.model_registry import (DEFAULT_MODEL, mev_model_weights,
                                         model_id_from_run_config,
                                         resolve_model, run_model_id)
from ifrs9qdb.etl.pipeline import load_model_config
from ifrs9qdb.etl.transform import transform_allocation, transform_collateral
from ifrs9qdb.prerun import apply_input_extract_date
from ifrs9qdb.validation import INPUT_STAGE
from ifrs9qdb.validation._helpers import (latest_extract_date, r_parse_date,
                                          resolve_input_extract_date)
from ifrs9qdb.validation.checks_static import (_v_optional_paths_resolve,
                                               _v_required_paths_exist,
                                               _v_run_block_complete)


@pytest.fixture(scope="module")
def model_cfg():
    mc, _ = load_model_config(None)
    assert mc is not None, "the package ships config/model.yml"
    return mc


# ------------------------------------------------------------------ model ----
class TestTheModelIsTheOneConfigNames:
    def test_config_yml_names_the_model(self):
        assert model_id_from_run_config(
            {"run": {"internal_model": "internal_alt"}}) == "internal_alt"
        # R: run$internal_model %||% run$model_id
        assert model_id_from_run_config({"run": {"model_id": "m2"}}) == "m2"
        assert model_id_from_run_config({"run": {}}) is None
        assert model_id_from_run_config(None) is None

    def test_the_shipped_model_resolves_in_rs_shape(self, model_cfg):
        m = resolve_model(model_cfg)["model"]
        assert m["id"] == DEFAULT_MODEL == "internal_v4_production"
        assert m["name"].startswith("internal_v4_production — ")
        assert [x["weight"] for x in m["mevs"]] == [1.0, 0.0, 0.0]
        assert m["max_maturity"] == 50 and m["n_forecasts"] == 5

    def test_an_unknown_model_is_refused_as_r_refuses_it(self, model_cfg):
        with pytest.raises(ValueError, match="not found in models.yml. Available:"):
            resolve_model(model_cfg, "no_such_model")

    def test_an_unknown_variable_is_refused(self, model_cfg):
        mc = copy.deepcopy(model_cfg)
        mc["models"]["bad"] = {"mev_components": [{"variable": "VAR_NOPE",
                                                   "weight": 1}]}
        with pytest.raises(ValueError, match="unknown variable 'VAR_NOPE'"):
            resolve_model(mc, "bad")

    def test_a_null_weight_is_derived_from_the_p_values(self, model_cfg):
        """R's .weights_from_p_values(): inverse p, normalised over ALL
        components, taken only where the registry leaves the weight null. The
        R run of this very model resolved the middle weight to 0.2669."""
        mc = copy.deepcopy(model_cfg)
        alt = copy.deepcopy(mc["models"]["internal_v4_production"])
        for c, w in zip(alt["mev_components"], [0.6, None, 0.4]):
            c["weight"] = w
        mc["models"]["internal_alt"] = alt
        w = [x["weight"] for x in resolve_model(mc, "internal_alt")["model"]["mevs"]]
        inv = [1 / 0.08, 1 / 0.165, 1 / 0.241]
        assert w[0] == 0.6 and w[2] == 0.4
        assert w[1] == pytest.approx(inv[1] / sum(inv))
        assert round(w[1], 4) == 0.2669

    def test_mev_model_weights_modes(self, model_cfg):
        r = resolve_model(model_cfg)
        assert mev_model_weights(r, {}) == [1.0, 0.0, 0.0]
        assert mev_model_weights(
            r, {"mev_model_weights": {"mode": "from_model_config"}}) == [1.0, 0.0, 0.0]
        auto = mev_model_weights(r, {"mev_model_weights": {"mode": "auto_p_value"}})
        raw = [0.241 / 0.08, 0.241 / 0.165, 1.0]      # R: max(p) / p
        assert auto == pytest.approx([x / sum(raw) for x in raw])
        with pytest.raises(ValueError, match="Unknown mev_model_weights mode"):
            mev_model_weights(r, {"mev_model_weights": {"mode": "equal"}})

    def test_the_stpd_build_refuses_a_model_the_registry_lacks(self, model_cfg):
        from ifrs9qdb.etl.macro import build_stpd_from_static
        with pytest.raises(ValueError, match="not found in models.yml"):
            build_stpd_from_static({}, model_cfg, {}, "6/9/2026", model_id="nope")

    def test_a_finished_run_names_its_model(self, tmp_path):
        run = tmp_path / "run_00001"
        cfg = run / "config_used" / "config"
        cfg.mkdir(parents=True)
        (cfg / "config.yml").write_text(yaml.safe_dump(
            {"run": {"internal_model": "internal_alt"}}))
        assert run_model_id(run) == "internal_alt"
        assert run_model_id(run / "Output") == "internal_alt"
        other = tmp_path / "run_00002" / "reports"
        other.mkdir(parents=True)
        (other / "manifest.json").write_text(json.dumps(
            {"config": {"model_config": {"model": {"id": "m9"}}}}))
        assert run_model_id(other.parent) == "m9"
        assert run_model_id(tmp_path / "nowhere") is None


# ------------------------------------------------------------------ dates ----
def _am(dates):
    return {"AccountMaster": pd.DataFrame({"EXTRACTDA": dates})}


class TestTheReportingDateIsRs:
    def test_a_clean_extract_has_one_date(self):
        assert resolve_input_extract_date(_am(["6/9/2026"] * 5)) == pd.Timestamp("2026-06-09")

    def test_the_date_most_rows_carry_wins(self):
        """R's resolve_input_extract_date() counts ROWS per date (each
        spelling parsed once): one stray row cannot redate the book. A tie
        goes to the earliest. INPUT_extract_date_matches_run_cfg reports the
        stray rows, INPUT_extract_date_plausible a wrong date."""
        assert resolve_input_extract_date(
            _am(["2026-06-09"] * 7000 + ["2026-07-01"])) == pd.Timestamp("2026-06-09")
        assert resolve_input_extract_date(
            _am(["2026-07-01"] * 7000 + ["2026-06-09"])) == pd.Timestamp("2026-07-01")
        # two spellings of one date are one date
        assert resolve_input_extract_date(
            _am(["2026-07-01", "7/1/2026", "2026-06-09"])) == pd.Timestamp("2026-07-01")
        assert resolve_input_extract_date(
            _am(["6/9/2026", "5/31/2026"])) == pd.Timestamp("2026-05-31")
        assert resolve_input_extract_date({}) is None

    def test_rs_date_parser_reads_each_shape_its_own_way(self):
        assert r_parse_date("09-JUN-26") == pd.Timestamp("2026-06-09")
        assert r_parse_date("45817") == pd.Timestamp("2025-06-09")      # Excel serial
        assert r_parse_date("20260609") == pd.Timestamp("2026-06-09")
        assert r_parse_date("2026-06-09 00:00:00") == pd.Timestamp("2026-06-09")
        # an unanchored %d-%b-%y once read this as 2020-12-31
        assert r_parse_date("31-DEC-2025") == pd.Timestamp("2025-12-31")
        assert r_parse_date("") is None and r_parse_date(None) is None

    def test_the_fallback_anchor_is_the_files_latest_date(self):
        """Only when the run has no reporting date (R's run_reporting_date)."""
        df = pd.DataFrame({"EXTRACTDA": ["6/9/2026", "7/1/2026", None]})
        assert latest_extract_date(df) == pd.Timestamp("2026-07-01")
        assert latest_extract_date(pd.DataFrame({"X": [1]})) is None

    def test_the_inputs_date_overlays_config(self):
        rc = {"run": {"extract_date": "", "internal_model": "m"}}
        out = apply_input_extract_date(rc, "2026-06-09")
        assert out["run"]["extract_date"] == "2026-06-09"
        assert rc["run"]["extract_date"] == ""          # a copy, not in place
        assert apply_input_extract_date(rc, None) is rc  # no date: config stands


# ---------------------------------------------------------- config checks ----
class TestTheConfigChecksAreRs:
    def test_only_static_and_model_files_are_required(self, tmp_path):
        (tmp_path / "static").mkdir()
        (tmp_path / "model.yml").write_text("x: 1")
        (tmp_path / "inputs.yml").write_text("x: 1")
        rc = {"paths": {"static_dir": "static", "variable_dictionary": "model.yml",
                        "models": "model.yml", "model_inputs": "inputs.yml",
                        # runtime locations: may not exist yet, not an error
                        "output_dir": "output", "runs_dir": "runs",
                        "model_config": "config/model_config.yml"}}
        assert _v_required_paths_exist(rc, tmp_path)["passed"]
        rc["paths"]["static_dir"] = "gone"
        r = _v_required_paths_exist(rc, tmp_path)
        assert not r["passed"]
        assert r["detail"] == ("Missing/unresolvable required paths: "
                               "paths$static_dir = 'gone'")
        assert _v_required_paths_exist({"run": {}}, tmp_path)["detail"] == \
            "config.yml has no `paths:` block"

    def test_optional_paths_warn_in_rs_words(self, tmp_path):
        r = _v_optional_paths_resolve({"paths": {"input_dir": "input",
                                                 "data_drop_root": ""}}, tmp_path)
        assert r["detail"] == "Optional paths set but not found: paths$input_dir = 'input'"

    def test_the_run_block_needs_the_model_and_a_date(self):
        r = _v_run_block_complete({"run": {"extract_date": ""}})
        assert r["detail"] == "Missing run.* keys: internal_model, extract_date"
        filled = apply_input_extract_date(
            {"run": {"internal_model": "m", "extract_date": ""}}, "2026-06-09")
        assert _v_run_block_complete(filled)["passed"]
        assert _v_run_block_complete({"paths": {}})["detail"] == \
            "config.yml has no `run:` block"

    def test_the_shipped_config_passes_once_the_inputs_date_is_applied(self):
        """The app's config.yml leaves extract_date empty on purpose; R fills
        it from the inputs before its config checks run, and so does this."""
        rc = {"run": {"internal_model": "internal_v4_production", "extract_date": ""}}
        assert not _v_run_block_complete(rc)["passed"]
        assert _v_run_block_complete(apply_input_extract_date(rc, "2026-06-09"))["passed"]


# ------------------------------------------------- reporting date sanity ----
def _plausible():
    v = next(v for v in INPUT_STAGE if v.id == "INPUT_extract_date_plausible")
    return v


class TestTheReportingDateIsPlausible:
    def test_a_clean_book_passes(self):
        inputs = {"AccountMaster": pd.DataFrame({
            "EXTRACTDA": ["6/9/2026"] * 3,
            "OPENDATE": ["1/1/2020", "5/5/2024", "6/9/2026"]})}
        assert _plausible().fn(inputs=inputs)["passed"]

    def test_a_file_split_between_two_dates_is_dated_by_the_earlier(self):
        inputs = {"AccountMaster": pd.DataFrame({
            "EXTRACTDA": ["6/9/2026", "5/31/2026"],
            "OPENDATE": ["1/1/2020", "6/5/2026"]})}
        r = _plausible().fn(inputs=inputs)
        assert not r["passed"]
        assert r["detail"] == ("reporting date 2026-05-31 is before the OPENDATE of "
                               "1 contract(s) (latest 2026-06-05) - EXTRACTDA is "
                               "stale, mistyped or mixed")
        # one stray row among many no longer redates the book
        inputs = {"AccountMaster": pd.DataFrame({
            "EXTRACTDA": ["6/9/2026", "6/9/2026", "5/31/2026"],
            "OPENDATE": ["1/1/2020", "6/5/2026", "6/9/2026"]})}
        assert _plausible().fn(inputs=inputs)["passed"]

    def test_a_future_date_fails(self):
        inputs = {"AccountMaster": pd.DataFrame({
            "EXTRACTDA": ["1/1/2099"], "OPENDATE": ["1/1/2020"]})}
        r = _plausible().fn(inputs=inputs)
        assert not r["passed"] and "after today" in r["detail"]

    def test_it_blocks_and_cannot_be_suppressed(self):
        assert _plausible().severity == "ERROR"
        assert _plausible().suppressible is False


# ------------------------------------------------------------- collateral ----
class TestCollateralFilesAsRWritesThem:
    def test_every_row_carries_the_runs_date(self):
        raw = pd.DataFrame({"EXTRACTDA": ["5/31/2026", "6/9/2026"],
                            "COLLATERALID": ["C1", "C2"], "P": ["X", ""],
                            "CO": ["31", "27"], "COL": ["QAR", "USD"],
                            "COLLATERALVALUE": ["100", "200"]}, index=[7, 9])
        out = transform_collateral(raw, "6/9/2026")
        assert list(out["ExtractDate"]) == ["6/9/2026", "6/9/2026"]
        # R's writer: ParentCollateralId and CollateralCode always empty
        assert list(out["ParentCollateralId"]) == ["", ""]
        assert list(out["CollateralCode"]) == ["", ""]
        assert len(out) == 2

    def test_the_allocation_share_is_always_a_percentage(self):
        """The extract writes 57 for 57%, always; R divides by 100 always."""
        pct = pd.DataFrame({"EXTRACTDA": ["6/9/2026"] * 2,
                            "COLLATERALID": ["C1", "C1"],
                            "CONTRACTID": ["1", "2"],
                            "ALLOCATIONPERCENTAGE": ["57", "43"]})
        small = pct.assign(ALLOCATIONPERCENTAGE=["0.57", "0.43"])
        share = lambda raw: list(transform_allocation(raw, "6/9/2026")
                                 ["AllocationPercentage"])
        assert share(pct) == pytest.approx([0.57, 0.43])
        # small shares are percentages too: the old guess read them as
        # fractions, 100 times too large
        assert share(small) == pytest.approx([0.0057, 0.0043])

# ------------------------------------------------------------- audit log ----
class TestTheAuditLog:
    def test_events_are_appended_and_read_back(self, tmp_path):
        from ifrs9qdb.audit_log import (audit_event, audit_suspended,
                                        read_audit_log, set_audit_log_path)
        log = tmp_path / "logs" / "etl_audit.jsonl"
        set_audit_log_path(log)
        try:
            audit_event({"event": "pre_run_readiness", "contracts": 3, "no_ecl": 0,
                         "blank_in_lic": 1, "with_gaps": 1})
            with audit_suspended():
                audit_event({"event": "run_start", "run_id": "hidden"})
            audit_event({"event": "run_cancelled", "run_id": "run_00008"})
            df = read_audit_log(log)
        finally:
            set_audit_log_path(None)
        assert list(df["event"]) == ["pre_run_readiness", "run_cancelled"]
        assert {"ts", "user"} <= set(df.columns)

    def test_every_event_both_apps_write_has_a_label_and_a_sentence(self):
        from ifrs9qdb.audit_log import event_label, event_summary
        cases = {
            "pre_run_readiness": ({"contracts": 7335, "no_ecl": 0, "blank_in_lic": 1,
                                   "with_gaps": 3810},
                                  "Pricing readiness: 7335 contracts, 0 with no ECL, "
                                  "1 blank in LIC, 3810 priced with a gap"),
            "run_cancelled": ({}, "Run cancelled at the review pause; its partial "
                                  "folder was removed"),
            "overlay_status": ({"overlay_id": "OV", "to_status": "approved"},
                               "Overlay 'OV' moved to approved"),
            "run_approved": ({"reason": "ok"}, "Run approved — ok"),
        }
        for ev, (fields, text) in cases.items():
            assert event_label(ev) != ev
            assert event_summary({"event": ev, **fields}) == text
