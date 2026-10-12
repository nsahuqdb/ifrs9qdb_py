"""Rows an export adds that are not data are removed on reading, not reported
as findings.

SQL*Plus repeats its headings each page and ends with "N rows selected.";
an Excel copy can carry blank rows, a total line at the end, or the same row
twice. All of these are removed once, when the file is read, so the checks and
the ETL see the same rows -- and the report lists them as INFO. Two rows with
the same id that DIFFER are left in: which is right is a question for the
source, and the uniqueness checks still report it. Mirrors R's
tests/testthat/test-input-cleaning.R.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ifrs9qdb.etl.read_inputs import read_all_inputs
from ifrs9qdb.validation.checks_input import _v_duplicate_headers_stripped
from ifrs9qdb.validation.checks_transform import _pv_customer_count
from ifrs9qdb.validation.schema import canonicalise, clean_rows, strip_junk_rows


def _frame(rows):
    return pd.DataFrame(rows, columns=["EXTRACTDA", "CONTRACTID", "AMOUNT"])


class TestCleanRows:
    def test_each_kind_is_removed_and_counted(self):
        df = _frame([["30-SEP-26", "A", 1],
                     ["EXTRACTDA", "CONTRACTID", "AMOUNT"],     # header
                     ["30-SEP-26", "B", 2],
                     ["30-SEP-26", "B", 2],                     # duplicate
                     [None, None, None],                        # blank
                     ["Total", None, 5],                        # total
                     ["4 rows selected.", None, None]])         # trailer
        out, n = clean_rows(df, dedupe=True)
        assert list(out["CONTRACTID"]) == ["A", "B"]
        assert n == {"header": 1, "trailer": 1, "blank": 1, "total": 1,
                     "duplicate": 1}

    def test_a_total_label_above_the_last_data_row_is_kept(self):
        df = _frame([["Total", "A", 1], ["30-SEP-26", "B", 2]])
        out, n = clean_rows(df)
        assert len(out) == 2 and n["total"] == 0

    def test_total_labels(self):
        for lab in ["Total", "TOTAL:", "Grand Total", "Sub-total", "Totals", "Sum"]:
            out, n = clean_rows(_frame([["30-SEP-26", "A", 1], [lab, None, 9]]))
            assert n["total"] == 1, lab
        for lab in ["Total Energies", "Summit"]:
            out, n = clean_rows(_frame([["30-SEP-26", "A", 1], [lab, None, 9]]))
            assert n["total"] == 0, lab

    def test_rows_with_the_same_id_that_differ_are_kept(self):
        df = _frame([["30-SEP-26", "A", 1], ["30-SEP-26", "A", 2]])
        out, n = clean_rows(df, dedupe=True)
        assert len(out) == 2 and n["duplicate"] == 0

    def test_no_de_duplication_unless_asked(self):
        df = _frame([["30-SEP-26", "A", 1], ["30-SEP-26", "A", 1]])
        assert len(clean_rows(df)[0]) == 2
        assert strip_junk_rows(df)[1] == 0


class TestOnReading:
    def _write(self, d, name, df):
        df.to_csv(d / name, index=False)

    def test_read_all_inputs_cleans_and_logs_except_the_schedule(self, tmp_path):
        self._write(tmp_path, "OriginationInvestments.csv",
                    _frame([["30-SEP-26", "X1", 1], ["30-SEP-26", "X1", 1],
                            ["30-SEP-26", "X2", 1], ["Total", None, 3]]))
        sched = pd.DataFrame({"KEY_1": ["1", "1"], "START_DATE": ["2026-10-30"] * 2,
                              "PRINCE_DUE": [10, 10]})
        self._write(tmp_path, "RepaymentSchedule.csv", sched)
        src = read_all_inputs(tmp_path)
        assert len(src["OriginationInvestments"]) == 2
        assert len(src["RepaymentSchedule"]) == 2          # never de-duplicated
        assert src.clean_log == {"OriginationInvestments.csv":
                                 {"total": 1, "duplicate": 1}}
        canon = canonicalise(src)
        assert canon.clean_log == src.clean_log
        assert canon.strip_log == {"OriginationInvestments.csv": 2}

    def test_the_report_is_info_with_what_was_removed(self):
        r = _v_duplicate_headers_stripped(None, {"OriginationInvestments.xls":
                                                 {"duplicate": 1}})
        assert not r["passed"]
        assert "1 exact duplicate row" in r["detail"]
        from ifrs9qdb.validation.checks_input import INPUT_STAGE_VALIDATORS as INPUT_VALIDATORS
        sev = {v.id: v.severity for v in INPUT_VALIDATORS}
        assert str(sev["INPUT_duplicate_headers_stripped"]).endswith("INFO")


class TestCustomersInTheView:
    def test_a_customer_with_no_contract_is_not_a_finding(self):
        trans_l = pd.DataFrame({"customer_id": ["1", "1", "2"]})
        cm_view = pd.DataFrame({"customer_id": ["1", "2", "3"]})
        assert _pv_customer_count(trans_l, cm_view)["passed"]

    def test_a_contract_whose_customer_is_missing_is(self):
        trans_l = pd.DataFrame({"customer_id": ["1", "9", np.nan]})
        cm_view = pd.DataFrame({"customer_id": ["1", "2"]})
        r = _pv_customer_count(trans_l, cm_view)
        assert not r["passed"] and r["count"] == 2
        assert "9" in r["detail"]
