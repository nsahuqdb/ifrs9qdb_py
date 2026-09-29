"""Reading dates and numbers: each date format only on values of its shape, a
1900-2200 window, a record of what the typing could not read, one reporting
date per run. Mirrors the R engine's tests/testthat/test-dates-and-typing.R,
value for value."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from ifrs9qdb.dates import (normalise_extract_dates, r_parse_any_dates,
                            schema_parse_dates)
from ifrs9qdb.validation import INPUT_STAGE
from ifrs9qdb.validation._helpers import resolve_input_extract_date
from ifrs9qdb.validation.schema import canonicalise, schema_unread_values

X = ["2026-06-09", "2026-06-09 00:00:00", "09-JUN-26", "09-JUN-2026",
     "09-June-2026", "6/9/2026", "31/12/2025", "6/9/26", "20260609",
     "31-12-2025", "2025/12/31", "46182", "46182.75", "1780963200",
     "garbage", "2026-13-45", "0", "", None]
WANT = ["2026-06-09"] * 6 + ["2025-12-31", "2026-06-09", "2026-06-09",
                             "2025-12-31", "2025-12-31"] + ["2026-06-09"] * 3 + [None] * 5


def _iso(s):
    return [None if pd.isna(v) else v.strftime("%Y-%m-%d") for v in s]


def _check(id_):
    return next(v for v in INPUT_STAGE if v.id == id_)


class TestEachFormatReadsOnlyItsShape:
    def test_the_checks_parser(self):
        assert _iso(r_parse_any_dates(X)) == WANT

    def test_the_transforms_parser_has_no_numbers_and_no_compact_form(self):
        got = _iso(normalise_extract_dates(X))
        assert got[:8] == WANT[:8]
        assert all(got[i] is None for i in [8] + list(range(11, 19)))
        # the old unanchored list read these as 2020-12-31 and the year 31
        assert _iso(r_parse_any_dates(["31-DEC-2025", "31-12-2025"])) == \
            ["2025-12-31", "2025-12-31"]

    def test_outside_1900_to_2200_is_not_a_date(self):
        assert _iso(normalise_extract_dates(
            pd.Series([pd.Timestamp("2250-01-01")]))) == [None]


class TestTheSchemaTypesDatesAsR:
    def test_a_bad_value_does_not_fail_the_column(self):
        assert _iso(schema_parse_dates(["2026-13-45", "2026-06-09"])) == \
            [None, "2026-06-09"]
        assert _iso(schema_parse_dates(["46182.75", "09-JUN-26", "6/9/26",
                                        "20260609"])) == \
            ["2026-06-09", "2026-06-09", "2026-06-09", None]

    def test_only_rs_schema_shapes(self):
        # R's schema reads neither D/M/YYYY nor YYYY/MM/DD nor a month in full
        assert _iso(schema_parse_dates(["31/12/2025", "2025/12/31",
                                        "09-June-2026"])) == [None] * 3

    def test_excel_cells_as_the_reader_gives_them(self):
        s = pd.Series([dt.datetime(2026, 6, 9, 13, 45), 46182.0, 46182, True,
                       "2026-06-09"], dtype=object)
        assert _iso(schema_parse_dates(s)) == ["2026-06-09"] * 3 + [None, "2026-06-09"]


def _raw_accounts(**cols):
    base = {"EXTRACTDA": "6/9/2026", "CONTRACTID": ["1", "2", "3", "4"],
            "LIMID": "l", "CUSTOMERID": "c", "PORTFOLIOCODE": "p",
            "ACCOUNTTYPE": "a", "IMPAIRMENTAMOUNT": "0", "STAGE": "1",
            "OPENDATE": ["1/1/2020", "31/02/2020", "N/A", "NULL"], "X10": "",
            "PASTDUEDAYS": ["0", "1", "3", "x"], "OFFBALANCE": "0",
            "ONBALANCE": ["1,234.5", "100", "", "NULL"],
            "MATURITYDATE": ["1/1/2030", "2030-02-30", "20300101", "1/1/2031"]}
    base.update(cols)
    return pd.DataFrame(base)


class TestWhatTheTypingCannotRead:
    def test_it_is_recorded_per_column(self):
        u = schema_unread_values(canonicalise({"AccountMaster": _raw_accounts()}))
        assert list(u["column"]) == ["open_date", "past_due_days", "on_balance",
                                     "maturity_date"]
        assert list(u["n"]) == [1, 1, 1, 2]
        assert u.loc[u["column"] == "on_balance", "sample"].item() == "1,234.5"

    def test_it_is_reported(self):
        inputs = canonicalise({"AccountMaster": _raw_accounts()})
        r = _check("INPUT_values_typed").fn(inputs=inputs)
        assert not r["passed"]
        assert "5 value(s) in 4 column(s)" in r["detail"]
        assert "AccountMaster.ONBALANCE 1 (e.g. 1,234.5)" in r["detail"]
        # the OPENDATE check could never fire on a typed column before
        r = _check("INPUT_AccountMaster_opendate_parses").fn(inputs=inputs)
        assert not r["passed"]
        assert r["detail"] == "1 unparseable OpenDate values"
        assert r["examples"] == ["31/02/2020"]

    def test_a_clean_file_records_nothing(self):
        ok = _raw_accounts(OPENDATE="1/1/2020", PASTDUEDAYS="0", ONBALANCE="1",
                           MATURITYDATE="1/1/2030")
        assert schema_unread_values(canonicalise({"AccountMaster": ok})).empty

    def test_integers_truncate_as_rs_as_integer(self):
        typed = canonicalise({"AccountMaster": _raw_accounts(
            PASTDUEDAYS=["29.7", "-3.5", "3e9", "5"])})["AccountMaster"]
        assert typed["past_due_days"].tolist()[:2] == [29.0, -3.0]
        assert pd.isna(typed["past_due_days"].iloc[2])


class TestTheReportingDate:
    @staticmethod
    def _am(d):
        return canonicalise({"AccountMaster": pd.DataFrame({"EXTRACTDA": d})})

    def test_the_date_most_rows_carry(self):
        assert resolve_input_extract_date(self._am(
            ["6/9/2026", "6/9/2026", "5/31/2026"])) == pd.Timestamp("2026-06-09")
        assert resolve_input_extract_date(self._am(
            ["6/9/2026", "09-JUN-26", "5/31/2026", "5/31/2026", "6/9/2026"])) == \
            pd.Timestamp("2026-06-09")
        assert resolve_input_extract_date(self._am(
            ["6/9/2026", "5/31/2026"])) == pd.Timestamp("2026-05-31")
        assert resolve_input_extract_date(self._am(["garbage", None])) is None

    def test_every_row_of_every_file_must_carry_it(self):
        v = _check("INPUT_extract_date_matches_run_cfg")
        inputs = {
            "AccountMaster": pd.DataFrame({"EXTRACTDA": ["6/9/2026"] * 3 + ["5/31/2026"]}),
            "Collateral": pd.DataFrame({"EXTRACTDA": ["6/9/2026", "6/10/2026",
                                                     "6/10/2026", "1/1/2026"]}),
            "CustomerMaster": pd.DataFrame({"EXTRACTDA": ["09-JUN-26", "6/9/2026"]})}
        r = v.fn(inputs=canonicalise(inputs))
        assert not r["passed"]
        assert r["detail"] == (
            "input files disagree on EXTRACTDA (reporting date 2026-06-09): "
            "AccountMaster: 1 row(s) dated 2026-05-31; "
            "Collateral: 1 row(s) dated 2026-01-01, 2 row(s) dated 2026-06-10")
        inputs["AccountMaster"]["EXTRACTDA"] = "6/9/2026"
        inputs["Collateral"]["EXTRACTDA"] = "2026-06-09"
        assert v.fn(inputs=canonicalise(inputs))["passed"]

    def test_only_files_whose_schema_carries_the_date_are_read(self):
        """R's typed Origination has no extract_date column, so its EXTRACTDA
        is not compared; the raw column kept beside the typed ones here must
        not add it."""
        inputs = canonicalise({
            "AccountMaster": pd.DataFrame({"EXTRACTDA": ["6/9/2026"] * 2}),
            "Origination": pd.DataFrame({"EXTRACTDA": ["5/31/2026"],
                                         "CONTRACTID": ["1"]})})
        assert _check("INPUT_extract_date_matches_run_cfg").fn(inputs=inputs)["passed"]
        assert _check("INPUT_consistent_extract_date").fn(inputs=inputs)["passed"]

