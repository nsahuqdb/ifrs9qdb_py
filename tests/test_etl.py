"""ETL tests.

The reader tests run anywhere. The reconciliation tests need a real run, and
skip when one is not present -- the point of them is comparing the Python
output against the R output from the SAME inputs, which is the only test that
proves the port.
"""
from pathlib import Path

import pandas as pd
import pytest

from ifrs9qdb.etl import (INPUT_SPECS, compare_file, detect_format,
                          read_all_inputs, transform_allocation,
                          transform_collateral, transform_staging_flags)
from ifrs9qdb.etl.transform import at, pick

from conftest import ref_output, src_inputs

# The raw extracts and the reference output a past run produced. Both are real
# portfolio data, so they live outside the repository; see conftest.py.
IN = src_inputs()
OUT = ref_output()
has_run = pytest.mark.skipif(
    IN is None or OUT is None,
    reason="set IFRS9_SRC_INPUTS and IFRS9_REF_RUN")


class TestPositionalMapping:
    def test_at_takes_a_column_by_position(self):
        """The SQL*Plus exports truncate aliases, so only position identifies a
        column: I, ISWATCHLIST, I.1, I.2 ... are all different flags."""
        df = pd.DataFrame({"A": [1], "I": [2], "I.1": [3]})
        assert at(df, 0).iloc[0] == 1
        assert at(df, 2).iloc[0] == 3

    def test_at_falls_back_to_a_name_when_the_column_is_absent(self):
        df = pd.DataFrame({"CUSTOMERID": [7]})
        assert at(df, 9, "CustomerId").iloc[0] == 7

    def test_pick_ignores_punctuation_and_case(self):
        df = pd.DataFrame({"Past Due Days": [5]})
        assert pick(df, "PASTDUEDAYS").iloc[0] == 5
        assert pick(df, "past_due_days").iloc[0] == 5


class TestFormatDetection:
    def test_html_is_detected_regardless_of_extension(self, tmp_path):
        """The Oracle jobs deliver SQL*Plus HTML with an .xls extension."""
        p = tmp_path / "Origination.xls"
        p.write_text("<html><body><table><tr><td>1</td></tr></table></body></html>")
        assert detect_format(p) == "html"

    def test_csv_is_the_fallback(self, tmp_path):
        p = tmp_path / "x.xls"
        p.write_text("a,b\n1,2\n")
        assert detect_format(p) == "csv"

    def test_every_input_has_an_alternative_extension(self):
        for s in INPUT_SPECS:
            assert s.alternatives, f"{s.name} has no fallback filename"


@has_run
class TestAgainstTheRealExtracts:
    @staticmethod
    @pytest.fixture(scope="class")
    def inputs():
        return read_all_inputs(IN)

    def test_all_twelve_inputs_read(self, inputs):
        assert inputs.ok, f"missing: {inputs.missing}"
        assert len(inputs.tables) == len(INPUT_SPECS)

    def test_the_html_exports_parse(self, inputs):
        for name in ("Origination", "IndustryCode", "CustomerMasterInvestments"):
            assert len(inputs[name]) > 0, f"{name} came back empty"

    def test_collateral_matches_the_r_output(self, inputs, tmp_path):
        out = tmp_path / "Collateral.csv"
        transform_collateral(inputs["Collateral"]).to_csv(out, index=False, na_rep="")
        r = compare_file(out, OUT / "Collateral.csv", key=["CollateralId"])
        assert r["status"] == "match", r.get("detail")

    def test_allocation_matches_the_r_output(self, inputs, tmp_path):
        """Includes the percentage scale: the source writes 10.09 for ten per
        cent and LIC wants 0.1009. A hundredfold error here would show up as
        coverage above 100%, not as a failure."""
        out = tmp_path / "AccountCollateralAllocation.csv"
        transform_allocation(inputs["AccountCollateralAllocation"]).to_csv(
            out, index=False, na_rep="")
        r = compare_file(out, OUT / "AccountCollateralAllocation.csv",
                         key=["ContractId", "CollateralId"])
        assert r["status"] == "match", r.get("detail")

    def test_allocation_percentages_are_fractions(self, inputs):
        a = transform_allocation(inputs["AccountCollateralAllocation"])
        assert a["AllocationPercentage"].max() <= 1.5

    def test_staging_flags_keep_all_twelve_columns(self, inputs):
        f = transform_staging_flags(inputs["CustomerStagingFlag"])
        assert list(f.columns) == [
            "ExtractDate", "CustomerId", "IsDefault", "IsWatchlist",
            "IsInsolvency", "IsDefaultInGCC", "IsLocal1", "IsLocal2",
            "IsLocal3", "IsLocal4", "IsLocal5", "IsLocal6"]


class TestContractIdSubstitution:
    """The id rule is an Excel formula and is not guessable from the data."""

    def test_a_numeric_id_only_loses_its_leading_zeros(self):
        from ifrs9qdb.etl.lending import apply_id_substitutions
        assert list(apply_id_substitutions(["0000610057"])) == ["610057"]

    def test_a_product_id_substitutes_the_code_and_coerces(self):
        from ifrs9qdb.etl.lending import apply_id_substitutions
        # 0000121 + FGG->3 + 007659, then numeric: leading zeros go
        assert list(apply_id_substitutions(["0000121FGG007659"])) == ["1213007659"]

    def test_the_tail_is_six_characters_not_three(self):
        from ifrs9qdb.etl.lending import apply_id_substitutions
        got = apply_id_substitutions(["0000112FGG000471"])[0]
        assert got.endswith("000471")

    def test_every_product_code_maps(self):
        from ifrs9qdb.etl.lending import OFF_BALANCE_PRODUCTS, apply_id_substitutions
        for code, val in OFF_BALANCE_PRODUCTS.items():
            got = apply_id_substitutions([f"0000125{code}000123"])[0]
            assert got == str(int(f"0000125{val}000123"))

    def test_an_unknown_code_is_left_in_place(self):
        """A new product must not silently collide with an existing id."""
        from ifrs9qdb.etl.lending import apply_id_substitutions
        got = apply_id_substitutions(["0000125ZZZ000123"])[0]
        assert "ZZZ" in got


@has_run
class TestAccountMaster:
    def test_repeated_headers_are_removed(self):
        from ifrs9qdb.etl.lending import drop_repeated_headers
        raw = read_all_inputs(IN)["AccountMaster"]
        cleaned = drop_repeated_headers(raw)
        assert len(cleaned) < len(raw)
        assert not cleaned.iloc[:, 1].astype(str).str.upper().eq("CONTRACTID").any()

    def test_rating_and_dpd_are_the_customers_worst(self):
        """Every facility of a customer carries the same rating and DPD."""
        from ifrs9qdb.etl.lending import transform_lending
        v = transform_lending(read_all_inputs(IN)["AccountMaster"])
        per_customer = v.groupby("customer_id")[["rating_worst", "past_dues_worst"]].nunique()
        assert (per_customer["rating_worst"] <= 1).all()
        assert (per_customer["past_dues_worst"] <= 1).all()

    def test_payment_type_is_derived_from_frequency(self):
        from ifrs9qdb.etl.lending import transform_lending
        v = transform_lending(read_all_inputs(IN)["AccountMaster"])
        no_freq = v["payment_frequency"].isna() | (v["payment_frequency"] == 0)
        assert (v.loc[no_freq, "payment_type"] == 3).all()
        assert (v.loc[~no_freq, "payment_type"] == 4).all()


@has_run
class TestEadCurves:
    """The EAD curve is what the ECL sum prices against, so it is pinned hard."""

    @staticmethod
    def build():
        from ifrs9qdb.etl.lending import transform_lending
        from ifrs9qdb.etl.lifetime import build_lifetime_parameter_other
        s = read_all_inputs(IN)
        v = transform_lending(s["AccountMaster"])
        return build_lifetime_parameter_other(
            s["RepaymentSchedule"], s["AccountMaster"], "2026-06-30",
            "6/30/2026", dict(zip(v["contract_id_raw"], v["contract_id"])),
            contracts=set(v["contract_id_raw"]))

    def test_matches_the_r_output_exactly(self, tmp_path):
        out = tmp_path / "LifeTimeParameterOther.csv"
        self.build().to_csv(out, index=False, na_rep="")
        r = compare_file(out, OUT / "LifeTimeParameterOther.csv",
                         key=["ContractId", "MonthLifetime"])
        assert r["status"] == "match", r.get("detail")

    def test_a_schedule_ending_this_month_produces_no_curve(self):
        """The month bound is exclusive: no future months, no exposure to price."""
        lpo = self.build()
        counts = lpo.groupby("ContractId").size()
        assert counts.min() >= 1
        # every contract present has at least month 0 and stops before its end
        assert lpo["MonthLifetime"].min() == 0

    def test_most_curves_rise_above_the_current_balance(self):
        """Not a defect, and worth pinning because it looks like one.

        Month 0 is today's outstanding; from month 1 the curve follows the
        SCHEDULE, which carries committed but undrawn amounts. On this book
        3,324 of 5,520 curves rise at some point, and on many the jump at month
        1 is several times the current balance.

        This is exactly why the engine caps ECL at exposure: without the cap,
        a contract can be provisioned above its own balance. Anyone tempted to
        "fix" a rising curve should read this test first.
        """
        lpo = self.build().copy()
        lpo["MonthLifetime"] = lpo["MonthLifetime"].astype(int)
        lpo["EADLifetime"] = lpo["EADLifetime"].astype(float)
        lpo = lpo.sort_values(["ContractId", "MonthLifetime"])
        rises = lpo.groupby("ContractId")["EADLifetime"].apply(
            lambda s: bool((s.diff().dropna() > 1e-6).any()))
        assert rises.sum() > 0.4 * len(rises)

    def test_every_curve_ends_no_higher_than_it_starts_after_month_one(self):
        """A schedule draws down and then amortises; it does not end higher.

        Deliberately an END-TO-END check rather than a step-by-step one:
        drawdowns can arrive in more than one tranche, so a curve may rise
        several times before it runs off. What must hold is that it finishes
        below where the drawn period began.
        """
        lpo = self.build().copy()
        lpo["MonthLifetime"] = lpo["MonthLifetime"].astype(int)
        lpo["EADLifetime"] = lpo["EADLifetime"].astype(float)
        sample = lpo[lpo["ContractId"].isin(lpo["ContractId"].unique()[:300])]
        for _, g in sample.groupby("ContractId"):
            g = g[g["MonthLifetime"] >= 1].sort_values("MonthLifetime")
            if len(g) > 2:
                assert g["EADLifetime"].iloc[-1] <= g["EADLifetime"].max() + 1e-6

    def test_reserved_columns_stay_blank(self):
        """LGD, PaymentSchedule and TotalLimit belong to another engine."""
        lpo = self.build()
        for c in ("LGDLifetime", "PaymentScheduleLifetime", "TotalLimitLifetime"):
            assert (lpo[c].astype(str) == "").all()


@has_run
class TestPipeline:
    def test_a_full_run_writes_the_expected_files(self, tmp_path):
        from ifrs9qdb.etl.pipeline import run_etl
        from ifrs9qdb.etl.static_ref import PACKAGED_STATIC
        r = run_etl(IN, tmp_path, static_dir=PACKAGED_STATIC)
        assert r.ok, r.error
        assert r.run_id == "run_00001"
        for f in ("AccountMaster_1.csv", "Collateral.csv",
                  "LifeTimeParameterOther.csv", "CustomerMaster_1.csv"):
            assert f in r.written
        assert (r.output_dir / "AccountMaster_1.csv").is_file()

    def test_a_run_is_never_overwritten(self, tmp_path):
        """Reproducing a past quarter means it must still be there."""
        from ifrs9qdb.etl.pipeline import run_etl
        from ifrs9qdb.etl.static_ref import PACKAGED_STATIC
        run_etl(IN, tmp_path, static_dir=PACKAGED_STATIC)
        again = run_etl(IN, tmp_path, run_id="run_00001",
                        static_dir=PACKAGED_STATIC)
        assert not again.ok
        assert "already exists" in again.error

    def test_run_ids_increment(self, tmp_path):
        from ifrs9qdb.etl.pipeline import next_run_id, run_etl
        from ifrs9qdb.etl.static_ref import PACKAGED_STATIC
        assert next_run_id(tmp_path) == "run_00001"
        run_etl(IN, tmp_path, static_dir=PACKAGED_STATIC)
        assert next_run_id(tmp_path) == "run_00002"

    def test_the_manifest_records_what_produced_the_run(self, tmp_path):
        import json
        from ifrs9qdb.etl.pipeline import run_etl
        from ifrs9qdb.etl.static_ref import PACKAGED_STATIC
        r = run_etl(IN, tmp_path, static_dir=PACKAGED_STATIC)
        m = json.loads((r.run_dir / "reports" / "manifest.json").read_text())
        assert m["engine_version"] and m["reporting_date"]
        assert m["files_written"] and "files_pending" in m

    def test_the_reporting_date_comes_from_the_extract(self, tmp_path):
        """Not today's date, which would re-age every contract on a re-run."""
        import json
        from ifrs9qdb.etl.pipeline import run_etl
        from ifrs9qdb.etl.static_ref import PACKAGED_STATIC
        r = run_etl(IN, tmp_path, static_dir=PACKAGED_STATIC)
        m = json.loads((r.run_dir / "reports" / "manifest.json").read_text())
        assert m["reporting_date"] == "6/30/2026"

    def test_a_missing_input_folder_fails_with_a_reason(self, tmp_path):
        from ifrs9qdb.etl.pipeline import run_etl
        r = run_etl(tmp_path / "nope", tmp_path / "runs")
        assert not r.ok and "No input directory" in r.error


@has_run
class TestInvestments:
    @staticmethod
    def build():
        from ifrs9qdb.etl.lending import build_account_master, transform_investments
        s = read_all_inputs(IN)
        return build_account_master(
            transform_investments(s["AccountMasterInvestments"]),
            "6/30/2026", investments=True)

    def test_matches_the_r_output_exactly(self, tmp_path):
        out = tmp_path / "AccountMaster_2.csv"
        self.build().to_csv(out, index=False, na_rep="")
        r = compare_file(out, OUT / "AccountMaster_2.csv",
                         key=["ContractId"])
        assert r["status"] == "match", r.get("detail")

    def test_both_ids_are_sequential_surrogates(self):
        """The extract names the counterparty; LIC will not take that as a key."""
        am2 = self.build()
        assert list(am2["ContractId"]) == [str(i) for i in range(1, len(am2) + 1)]
        assert list(am2["CustomerId"]) == list(am2["ContractId"])

    def test_rates_stored_in_percent_are_detected(self):
        """3.8 must become 0.038, and 0.038 must stay put."""
        from ifrs9qdb.etl.lending import transform_investments
        pct = pd.DataFrame({"CONTRACTID": ["a", "b"], "EIR": [3.8, 4.2],
                            "ACCOUNTTYPE": ["x", "y"]})
        frac = pd.DataFrame({"CONTRACTID": ["a", "b"], "EIR": [0.038, 0.042],
                             "ACCOUNTTYPE": ["x", "y"]})
        assert transform_investments(pct)["eir"].iloc[0] == pytest.approx(0.038)
        assert transform_investments(frac)["eir"].iloc[0] == pytest.approx(0.038)

    def test_the_portfolio_comes_from_the_account_type(self):
        from ifrs9qdb.etl.lending import transform_investments
        d = pd.DataFrame({"CONTRACTID": ["a", "b"],
                          "ACCOUNTTYPE": ["Banks and Fis", "Sukuk"]})
        got = list(transform_investments(d)["portfolio_code"])
        assert got == ["Banks and Fis", "Investments"]

    def test_two_date_formats_in_one_file(self):
        """ExtractDate unpadded, OpenDate and MaturityDate zero-padded."""
        am2 = self.build()
        assert am2["ExtractDate"].iloc[0] == "6/30/2026"
        assert am2["OpenDate"].iloc[0].count("/") == 2
        assert len(am2["OpenDate"].iloc[0]) == 10


@has_run
class TestJunkRowStripping:
    def test_headers_footers_and_blanks_are_all_removed(self):
        """SQL*Plus leaves three kinds of junk: repeated headings every page, a
        'N rows selected.' footer, and blank lines."""
        from ifrs9qdb.etl.lending import drop_repeated_headers
        raw = read_all_inputs(IN)["AccountMaster"]
        cleaned = drop_repeated_headers(raw)
        assert len(cleaned) < len(raw)
        first = cleaned.iloc[:, 0].astype(str)
        assert not first.str.contains("rows selected", case=False).any()
        assert not cleaned.iloc[:, 1].astype(str).str.upper().eq("CONTRACTID").any()

    def test_the_row_count_matches_the_files_own_footer(self):
        """The footer states how many rows the query returned, which is the
        only authority on how many there should be."""
        import re
        import pandas as _pd
        from ifrs9qdb.etl.lending import drop_repeated_headers
        raw = _pd.read_excel(IN / "AccountMaster.xlsx")
        footer = [str(v) for v in raw.iloc[:, 0]
                  if re.match(r"^[0-9][0-9,]*\s+rows?\s+selected", str(v))]
        assert footer, "no footer row in the extract"
        stated = int(re.match(r"^([0-9,]+)", footer[0]).group(1).replace(",", ""))
        assert len(drop_repeated_headers(raw)) == stated

    def test_a_single_matching_cell_is_not_a_header(self):
        """A contract whose only populated field equals its heading must not be
        dropped; a header needs at least two matching cells."""
        import pandas as _pd
        from ifrs9qdb.etl.lending import drop_repeated_headers
        df = _pd.DataFrame({"A": ["A", "x"], "B": [None, "y"]})
        assert len(drop_repeated_headers(df)) == 2


class TestDateFormatting:
    """`%-m` is a glibc extension. On Windows it raises "Invalid format
    string" rather than falling back, which failed every run there."""

    def test_the_unpadded_form_works_on_any_platform(self):
        import pandas as _pd
        from ifrs9qdb.etl.transform import _fmt_date
        got = list(_fmt_date(_pd.Series(["2026-06-30", "2026-01-05"])))
        assert got == ["6/30/2026", "1/5/2026"]

    def test_the_padded_form_keeps_leading_zeros(self):
        import pandas as _pd
        from ifrs9qdb.etl.transform import _fmt_date
        got = list(_fmt_date(_pd.Series(["2026-01-05"]), pad=True))
        assert got == ["01/05/2026"]

    def test_no_strftime_extension_remains_in_the_package(self):
        """A grep, because one of these breaks every run on Windows and the
        failure is far from its cause."""
        import glob
        import re
        # Strip comments first: the explanation of why this is banned mentions
        # the very thing it bans.
        bad = []
        for f in glob.glob("src/ifrs9qdb/**/*.py", recursive=True):
            for line in open(f, encoding="utf-8").read().split("\n"):
                code = re.sub(r"#.*$", "", line)
                if "strftime" in code and re.search(r'%[-#][a-zA-Z]', code):
                    bad.append(f"{f}: {code.strip()}")
        assert not bad, "platform-specific strftime: " + "; ".join(bad)


class TestFailureReporting:
    def test_a_failed_run_names_the_step_it_reached(self):
        """"Invalid format string" tells nobody which step was running."""
        from ifrs9qdb.etl import run_etl
        if IN is None:
            pytest.skip("set IFRS9_SRC_INPUTS")
        r = run_etl(IN, "/tmp/failrun_test", reporting_date="not-a-date")
        assert not r.ok
        assert "Failed after" in r.error
        assert r.steps and r.steps[-1]["step"] == "FAILED"
        assert r.traceback
