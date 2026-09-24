"""Overlays and governance.

The rules here exist to stop a provision being changed untraceably, so each is
pinned rather than assumed.
"""
import pandas as pd
import pytest

from ifrs9qdb.governance import (AuditLog, approval_status, approve_run,
                                 compare_snapshots, read_snapshot, take_snapshot)
from ifrs9qdb.overlays import Overlay, apply_overlays, validate_overlays


def book():
    from ifrs9qdb.analytics import normalise
    return normalise(pd.DataFrame({
        "Contract Id": ["C1", "C2", "C3", "C4"],
        "Customer Id": ["A", "A", "B", "C"],
        "Portfolio Code": ["Business Finance"] * 3 + ["Off BS"],
        "Ifrs Stage": [1, 2, 2, 3],
        "Exposure On Bal": [1000, 2000, 3000, 4000],
        "Cla Amount Onbal": [10, 100, 300, 4000],
        "Rating": ["QDB 5"] * 4,
    }))


def ov(**kw):
    base = dict(id="OV1", type="uplift_pct", value=0.10,
                rationale="test", whole_book=True)
    base.update(kw)
    return Overlay(**base)


class TestOverlayRules:
    def test_stage_3_is_never_touched(self):
        """Stage 3 is booked manually; an overlay would double-count."""
        r = apply_overlays(book(), [ov()])
        assert r["ok"]
        d = r["report"]
        assert d.loc[d["stage"] == 3, "overlay_amount"].abs().sum() == 0

    def test_overlapping_overlays_are_rejected(self):
        """Compounding would make the order of application decide the number."""
        r = apply_overlays(book(), [
            ov(id="A", stage=[2], whole_book=False),
            ov(id="B", portfolio=["Business Finance"], whole_book=False)])
        assert not r["ok"]
        assert "more than one overlay" in r["errors"][0]

    def test_an_overlay_needs_a_rationale(self):
        errs = validate_overlays([ov(rationale="")])
        assert any("rationale" in e for e in errs)

    def test_uplift_scales_the_model_figure(self):
        r = apply_overlays(book(), [ov(value=0.5)])
        d = r["report"]
        non3 = d[d["stage"] != 3]
        assert (non3["overlay_amount"] == non3["ecl_model"] * 0.5).all()

    def test_higher_of_is_a_floor_and_only_binds_below_it(self):
        r = apply_overlays(book(), [ov(type="higher_of", value=0.20)])
        d = r["report"]
        c1 = d[d["contract"] == "C1"].iloc[0]          # 10 on 1000 = 1%
        assert c1["ecl_final"] == pytest.approx(200.0)
        assert (d.loc[d["stage"] != 3, "overlay_amount"] >= 0).all()

    def test_absolute_add_is_spread_pro_rata_by_exposure(self):
        r = apply_overlays(book(), [ov(type="absolute_add", value=600)])
        d = r["report"]
        non3 = d[d["stage"] != 3]
        assert non3["overlay_amount"].sum() == pytest.approx(600.0)
        # C3 has three times C1's exposure, so takes three times the share
        got = dict(zip(non3["contract"], non3["overlay_amount"]))
        assert got["C3"] == pytest.approx(got["C1"] * 3)

    def test_a_customer_level_overlay_reaches_all_that_customers_facilities(self):
        r = apply_overlays(book(), [ov(level="customer", contract_id=["C1"],
                                       whole_book=False)])
        d = r["report"]
        touched = set(d.loc[d["overlay_amount"] != 0, "contract"])
        assert touched == {"C1", "C2"}

    def test_the_model_figure_is_kept_alongside_the_final_one(self):
        r = apply_overlays(book(), [ov(value=0.25)])
        d = r["report"]
        assert {"ecl_model", "overlay_amount", "ecl_final"} <= set(d.columns)
        assert (d["ecl_final"] == d["ecl_model"] + d["overlay_amount"]).all()

    def test_every_overlay_writes_an_audit_row(self):
        r = apply_overlays(book(), [ov(id="A"), ov(id="B", stage=[3],
                                                   whole_book=False)])
        assert r["ok"]
        assert set(r["audit"]["overlay_id"]) == {"A", "B"}
        assert "rationale" in r["audit"].columns


class TestGovernance:
    def test_a_snapshot_records_a_hash_per_file(self, tmp_path):
        """A copy can be edited afterwards and still look original."""
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        (cfg / "model.yml").write_text("a: 1\n")
        run = tmp_path / "run_00001"
        run.mkdir()
        snap = take_snapshot(run, cfg)
        assert snap["files"]["config/model.yml"]["sha256_16"]
        assert read_snapshot(run)["by"]

    def test_comparing_snapshots_names_what_changed(self, tmp_path):
        for name, body in (("run_a", "a: 1\n"), ("run_b", "a: 2\n")):
            cfg = tmp_path / f"cfg_{name}"
            cfg.mkdir()
            (cfg / "model.yml").write_text(body)
            r = tmp_path / name
            r.mkdir()
            take_snapshot(r, cfg)
        d = compare_snapshots(tmp_path / "run_a", tmp_path / "run_b")
        assert (d["status"] == "CHANGED").any()

    def test_the_audit_log_is_append_only(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("run", "first")
        log.record("approval", "second")
        e = log.entries()
        assert len(e) == 2 and e[0]["detail"] == "first"

    def test_a_corrupt_line_does_not_hide_the_rest(self, tmp_path):
        p = tmp_path / "audit.jsonl"
        p.write_text('{"action":"a"}\nnot json\n{"action":"b"}\n')
        e = AuditLog(p).entries()
        assert len(e) == 3
        assert e[1]["action"] == "unreadable entry"

    def test_approval_is_refused_while_validation_fails(self, tmp_path):
        """A gate that can be waved through on a bad run is decoration."""
        run = tmp_path / "run_00001"
        run.mkdir()
        with pytest.raises(ValueError, match="failing error-level"):
            approve_run(run, "risk", "Dmitry", validation_passed=False)

    def test_both_sign_offs_are_needed(self, tmp_path):
        run = tmp_path / "run_00001"
        run.mkdir()
        approve_run(run, "risk", "Dmitry", validation_passed=True)
        assert not approval_status(run)["complete"]
        assert approval_status(run)["outstanding"] == ["finance"]
        approve_run(run, "finance", "Adnan", validation_passed=True)
        assert approval_status(run)["complete"]

    def test_an_approval_needs_a_named_approver(self, tmp_path):
        run = tmp_path / "run_00001"
        run.mkdir()
        with pytest.raises(ValueError, match="named approver"):
            approve_run(run, "risk", "", validation_passed=True)
