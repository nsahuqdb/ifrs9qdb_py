"""The ECL bridge: every contract's move split into its causes, at any level.

Run against two real consecutive runs (IFRS9_REF_RUN_PREV -> IFRS9_REF_RUN),
and against copies of the current one with the macro inputs, the model, or
both changed in their frozen config and the run repriced -- so the only thing
that moved is what the test changed.
"""
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import prev_run, ref_run
from ifrs9qdb.analytics.bridge import (BRIDGE_LEVELS, CONTRACT_COMPONENTS,
                                       bridge_by, bridge_members, bridge_view,
                                       ecl_bridge, load_bridge_run)

PREV, CURR = prev_run(), ref_run()
needs = pytest.mark.skipif(
    PREV is None or CURR is None or not (CURR / "config_used").is_dir()
    or not (PREV / "config_used").is_dir(),
    reason="set IFRS9_REF_RUN and IFRS9_REF_RUN_PREV to two runs with config_used")

LEVELS = [lv for lv in BRIDGE_LEVELS if lv != "book"]


@pytest.fixture(scope="module")
def quarters():
    return ecl_bridge(load_bridge_run(PREV), load_bridge_run(CURR))


@needs
class TestTwoQuarters:
    def test_every_contract_sums_to_its_change(self, quarters):
        rows = quarters["rows"]
        total = rows[list(CONTRACT_COMPONENTS)].sum(axis=1)
        np.testing.assert_allclose(total, rows["ecl_b"] - rows["ecl_a"],
                                   rtol=0, atol=1e-6)

    def test_the_engine_reproduces_both_reports(self, quarters):
        """Other holds what the report has that the repricing does not; on
        runs the engine priced, nothing."""
        c = quarters["rows"][quarters["rows"]["status"] == "continuing"]
        assert c["repriced"].all()
        assert c["other"].abs().max() < 1e-4

    def test_an_unchanged_config_moves_neither_macro_nor_model(self, quarters):
        ch = quarters["changes"]
        assert ch["known"] and not ch["model_changed"] and not ch["macro_changed"]
        assert quarters["rows"][["macro", "model", "pd_curves"]].abs().sum().sum() == 0

    def test_the_book_view_is_the_whole_move(self, quarters):
        rows = quarters["rows"]
        v = bridge_view(rows, "book")
        assert v["opening"] == pytest.approx(rows.loc[rows["status"] != "new", "ecl_a"].sum())
        assert v["closing"] == pytest.approx(
            rows.loc[rows["status"] != "derecognised", "ecl_b"].sum())
        assert abs(v["residual"]) < 1e-4
        steps = v["steps"].set_index("key")["amount"]
        assert steps["moved_in"] == 0 and steps["moved_out"] == 0

    @pytest.mark.parametrize("level", LEVELS)
    def test_every_group_closes_on_its_own_figures(self, quarters, level):
        rows = quarters["rows"]
        by = bridge_by(rows, level)
        steps = [c for c in by.columns
                 if c not in ("group", "label", "opening", "closing", "change")]
        np.testing.assert_allclose(by["opening"] + by[steps].sum(axis=1),
                                   by["closing"], rtol=0, atol=1e-4)
        assert by["opening"].sum() == pytest.approx(
            rows.loc[rows["status"] != "new", "ecl_a"].sum())
        assert by["closing"].sum() == pytest.approx(
            rows.loc[rows["status"] != "derecognised", "ecl_b"].sum())
        # what moves out of one group moves into another
        assert by["moved_out"].sum() == pytest.approx(-by["moved_in"].sum())

    @pytest.mark.parametrize("level", LEVELS)
    def test_every_group_is_its_own_view(self, quarters, level):
        """bridge_by() is bridge_view() for each group, in one pass."""
        rows = quarters["rows"]
        by = bridge_by(rows, level).set_index("group")
        for g in list(by.index[:3]) + list(by.index[-2:]):
            v = bridge_view(rows, level, [g], top=0)
            assert by.loc[g, "opening"] == pytest.approx(v["opening"], abs=1e-4)
            assert by.loc[g, "closing"] == pytest.approx(v["closing"], abs=1e-4)
            for s in v["steps"].to_dict("records"):
                if s["kind"] == "delta":
                    assert by.loc[g, s["key"]] == pytest.approx(s["amount"], abs=1e-4), \
                        (g, s["key"])

    def test_every_customer_and_facility_can_be_chosen(self, quarters):
        rows = quarters["rows"]
        customers = set(rows["customer_a"].dropna()) | set(rows["customer_b"].dropna())
        assert len(bridge_members(rows, "customer")) == len(customers)
        assert len(bridge_members(rows, "facility")) == rows["contract"].nunique()
        assert len(bridge_members(rows, "facility", limit=10)) == 10

    def test_a_customer_shows_before_after_and_its_contracts(self, quarters):
        rows = quarters["rows"]
        cust = str(bridge_members(rows, "customer")["value"].iloc[0])
        v = bridge_view(rows, "customer", [cust])
        mine_a = (rows["customer_a"] == cust) & (rows["status"] != "new")
        mine_b = (rows["customer_b"] == cust) & (rows["status"] != "derecognised")
        assert v["before"]["ecl"] == pytest.approx(rows.loc[mine_a, "ecl_a"].sum())
        assert v["after"]["ecl"] == pytest.approx(rows.loc[mine_b, "ecl_b"].sum())
        assert v["before"]["exposure"] == pytest.approx(rows.loc[mine_a, "exposure_a"].sum())
        assert set(v["contracts"]["contract"]) == set(rows.loc[mine_a | mine_b, "contract"])
        assert abs(v["residual"]) < 1e-4

    def test_a_stage_group_counts_who_moved_in_and_out(self, quarters):
        rows = quarters["rows"]
        v = bridge_view(rows, "stage", ["2"])
        cont = rows["status"] == "continuing"
        into = cont & (rows["stage_b"] == 2) & (rows["stage_a"] != 2)
        assert v["counts"]["moved_in"] == int(into.sum())
        assert v["steps"].set_index("key").loc["moved_in", "amount"] == \
            pytest.approx(rows.loc[into, "ecl_a"].sum())


# -------------------------------------------------- macro and model ------
def _variant(tmp_path, base, name, macro=False, model=False):
    """A copy of ``base`` with its frozen macro inputs and/or model changed,
    and its PD curves and report rebuilt from them."""
    from ifrs9qdb.etl.macro import build_stpd_from_static
    from ifrs9qdb.etl.pipeline import load_model_config
    from ifrs9qdb.etl.report import build_final_ecl_report
    from ifrs9qdb.etl.static_ref import load_static_reference
    from ifrs9qdb.etl.transform import write_outputs

    d = tmp_path / name
    shutil.copytree(base / "Output", d / "Output",
                    ignore=shutil.ignore_patterns("FinalEclReport_*", "StPD_*"))
    shutil.copytree(base / "config_used", d / "config_used")
    cfg = d / "config_used" / "config"
    if macro:
        mi = yaml.safe_load((cfg / "model_inputs.yml").read_text())
        for y in mi["mev_forecasts"]["forecasts"]:
            mi["mev_forecasts"]["forecasts"][y][0] -= 3.0
        (cfg / "model_inputs.yml").write_text(yaml.safe_dump(mi, sort_keys=False))
    if model:
        m = yaml.safe_load((cfg / "model.yml").read_text())
        m["models"]["internal_v4_production"]["mev_components"][0]["coefficient"] = -0.06
        (cfg / "model.yml").write_text(yaml.safe_dump(m, sort_keys=False))
    model_cfg, inputs = load_model_config(cfg)
    static = load_static_reference(d / "config_used" / "static")
    rc = yaml.safe_load((cfg / "config.yml").read_text()) or {}
    mid = (rc.get("run") or {}).get("internal_model")
    rep = pd.read_csv(base / "Output" / "FinalEclReport.csv", usecols=["Extract Date"])
    ext = str(pd.to_datetime(rep["Extract Date"], format="mixed").mode().iloc[0].date())
    write_outputs(d / "Output", {"StPD.csv": build_stpd_from_static(
        static, model_cfg, inputs, ext, model_id=mid)})
    build_final_ecl_report(d, static=static, model_cfg=model_cfg)
    return d


@pytest.fixture(scope="module")
def variants(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bridge")
    return {n: _variant(tmp, CURR, n, macro=n in ("macro", "both"),
                        model=n in ("model", "both"))
            for n in ("base", "macro", "model", "both")}


def _totals(a, b):
    br = ecl_bridge(load_bridge_run(a), load_bridge_run(b))
    rows = br["rows"]
    return br, {k: float(rows[k].sum()) for k in CONTRACT_COMPONENTS}, \
        float((rows["ecl_b"] - rows["ecl_a"]).sum())


@needs
class TestMacroAndModel:
    def test_a_macro_change_is_all_macro(self, variants):
        br, t, move = _totals(variants["base"], variants["macro"])
        assert br["changes"]["macro_changed"] and not br["changes"]["model_changed"]
        assert any("mev_forecasts" in c["item"] for c in br["changes"]["macro"])
        assert abs(move) > 1_000
        assert t["macro"] == pytest.approx(move, abs=1e-3)
        for k in ("exposure", "stage", "rating", "model", "lgd", "other"):
            assert abs(t[k]) < 1e-3, k

    def test_a_model_change_is_all_model(self, variants):
        br, t, move = _totals(variants["base"], variants["model"])
        assert br["changes"]["model_changed"] and not br["changes"]["macro_changed"]
        assert br["changes"]["model"][0]["after"] == -0.06
        assert abs(move) > 1_000
        assert t["model"] == pytest.approx(move, abs=1e-3)
        for k in ("exposure", "stage", "rating", "macro", "lgd", "other"):
            assert abs(t[k]) < 1e-3, k

    def test_both_split_with_the_macro_measured_on_the_old_model(self, variants):
        """The macro step prices the current macro inputs on the previous
        model, which is exactly what the macro-only copy is: so the two
        agree, and the model step takes the rest."""
        _, both, move = _totals(variants["base"], variants["both"])
        _, macro_only, _ = _totals(variants["base"], variants["macro"])
        assert both["macro"] == pytest.approx(macro_only["macro"], abs=1e-3)
        assert both["macro"] + both["model"] == pytest.approx(move, abs=1e-3)
        assert abs(both["other"]) < 1e-3


def test_a_customer_or_facility_is_labelled_with_its_name():
    """The id with the name from either run; a blank name leaves the id."""
    rows = pd.DataFrame({
        "contract": ["F1", "F2", "F3"],
        "customer_a": ["C1", "C1", None], "customer_b": ["C1", None, "C2"],
        "name": ["Acme", "Acme", ""],
        "status": ["continuing", "derecognised", "new"],
        "ecl_a": [10.0, 5.0, 0.0], "ecl_b": [12.0, 0.0, 3.0],
    })
    m = bridge_members(rows, "customer").set_index("value")["label"]
    assert m["C1"] == "C1 — Acme" and m["C2"] == "C2"
    f = bridge_members(rows, "facility").set_index("value")["label"]
    assert f["F1"] == "F1 — Acme" and f["F3"] == "F3"
