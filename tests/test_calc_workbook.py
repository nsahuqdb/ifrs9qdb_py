"""The customer ECL calculator: one customer's provision as an Excel workbook
whose formulas reproduce the report.

A synthetic run is built from the package's own config and static tables, with
positions chosen to reach every rule the workbook writes as a formula: a
repayment schedule, the linear, annuity and bullet fallbacks, a deferral, a
payment every three months, Stage 3 by DPD and by flag, the watchlist, a
manual stage, Tasdeer, contagion, a missing EIR, an investment on the external
scale, collateral with a missing haircut. When LibreOffice is installed the
workbook is recalculated and every position compared with the engine;
otherwise only the structure is checked.

IFRS9_REF_RUN (conftest) adds the same comparison on a real run.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
import yaml

from conftest import ref_run
from ifrs9qdb.calc_workbook import (customer_ecl_facts, customer_ecl_workbook,
                                    run_customers, write_customer_workbook)
from ifrs9qdb.etl.macro import build_stpd_from_static
from ifrs9qdb.etl.pipeline import load_model_config
from ifrs9qdb.etl.reference import build_reference_files
from ifrs9qdb.etl.report import build_final_ecl_report
from ifrs9qdb.etl.static_ref import PACKAGED_STATIC, load_static_reference

PKG = Path(__file__).resolve().parents[1] / "src" / "ifrs9qdb"
EXT = "6/30/2026"
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")

AM_COLS = ["ExtractDate", "ContractId", "CustomerId", "AccountType", "Stage",
           "OpenDate", "Rating", "PastDueDays", "OffBalance", "OnBalance",
           "MaturityDate", "EIR", "PaymentFrequency", "CurrencyCode",
           "DeferralPeriod", "NominalInterestRate", "PaymentTypeId"]


# AccountMaster's whole LIC layout; the columns a test leaves out are blank
AM_LAYOUT = ["ExtractDate", "ContractId", "LimId", "CustomerId", "PortfolioCode",
             "AccountCode", "AccountType", "ImpairmentAmount", "OriginalECLOnbal",
             "OriginalECLOfbal", "Stage", "OpenDate", "Rating", "PastDueDays",
             "PD12M", "PDLifetimeValue", "IsPOCI", "LGDRate", "LoanToValue",
             "OffBalance", "OnBalance", "EAD", "CCF", "MaturityDate",
             "ExpectedMaturityDate", "EIR", "Principal", "PrincipalOverdue",
             "InterestAccrued", "InterestOverdue", "Fee", "FeeOverdue", "Penalty",
             "PenaltyOverdue", "Commission", "CommissionOverdue", "Other",
             "OtherOverdue", "PaymentFrequency", "CurrencyCode", "DeferralPeriod",
             "NominalInterestRate", "PaymentTypeId"]


def _am(rows):
    d = pd.DataFrame([dict(zip(AM_COLS, r)) for r in rows], columns=AM_COLS)
    return d.reindex(columns=AM_LAYOUT).fillna("")


def make_run(tmp: Path, *, annuity: bool = True, stage3: str = "full_outstanding",
             tamper_stpd: bool = False, overlay: bool = False,
             reference: bool = True) -> Path:
    """A run folder: config_used from the package, Output with a handful of
    positions, and the report priced on them."""
    root = tmp / "run_00001"
    out = root / "Output"
    cu = root / "config_used"
    out.mkdir(parents=True)
    shutil.copytree(PKG / "config", cu / "config")
    shutil.copytree(PACKAGED_STATIC, cu / "static")
    mpath = cu / "config" / "model.yml"
    model = yaml.safe_load(mpath.read_text(encoding="utf-8"))
    model.setdefault("ecl", {})["stage3_method"] = stage3
    if annuity:
        rules = model["ecl"].setdefault("ead_fallback", {}).setdefault("rules", [])
        rules.append({"portfolio": "Business Finance", "payment_type": "7",
                      "shape": "annuity"})
    mpath.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")

    static = load_static_reference(cu / "static")
    model_cfg, inputs = load_model_config(cu / "config")
    for name, df in build_reference_files(static, EXT).items():
        df.to_csv(out / name, index=False)
    stpd = build_stpd_from_static(static, model_cfg, inputs, EXT)
    if tamper_stpd:              # curves the macro model would not give
        stpd.loc[stpd["MonthLifetime"] > 6, "PDLifetime"] *= 1.01
    stpd.to_csv(out / "StPD.csv", index=False)

    lend = _am([
        # customer C1: schedule, linear, annuity, deferral + quarterly, bullet
        [EXT, "1001", "C1", "Long Term", "", "1/1/2020", "QDB 4", 0, 0, 1_000_000,
         "6/30/2029", 0.06, 1, "QAR", 0, 0.06, 4],
        [EXT, "1002", "C1", "Murabaha", "", "1/1/2021", "QDB 4", 0, 0, 500_000,
         "9/15/2028", 0.05, 1, "QAR", 0, 0.05, 4],
        [EXT, "1003", "C1", "Ijara", "", "1/1/2022", "QDB 4", 0, 0, 750_000,
         "12/31/2030", 0.07, 1, "QAR", 0, 0.07, 7],
        [EXT, "1004", "C1", "Short Term", "", "1/1/2023", "QDB 4", 0, 0, 300_000,
         "3/31/2031", 0.05, 3, "QAR", 9, 0.05, 4],
        [EXT, "1005", "C1", "LG", "", "1/1/2024", "QDB 4", 0, 200_000, 0,
         "12/31/2027", 0.05, 1, "QAR", 0, 0.05, 3],
        # PastDueDays is the customer's worst, as the ETL writes it (each
        # facility's own is in the reference below)
        # customer C2: 75 days past due (Stage 2) on one facility: both Stage 2
        [EXT, "2001", "C2", "Long Term", "", "1/1/2020", "QDB 5-", 75, 0, 400_000,
         "6/30/2028", 0.06, 1, "QAR", 0, 0.06, 3],
        [EXT, "2002", "C2", "Murabaha", "", "1/1/2020", "QDB 5-", 75, 0, 100_000,
         "6/30/2027", 0.06, 1, "QAR", 0, 0.06, 4],
        # customer C3: 120 days past due on one facility: both Stage 3
        [EXT, "3001", "C3", "Long Term", "", "1/1/2019", "QDB 7", 120, 0, 250_000,
         "6/30/2027", 0.08, 1, "QAR", 0, 0.08, 4],
        [EXT, "3002", "C3", "Long Term", "", "1/1/2019", "QDB 7", 120, 0, 90_000,
         "6/30/2027", 0.08, 1, "QAR", 0, 0.08, 4],
        # customer C4: Tasdeer (Stage 2 in the account master, as the ETL
        # writes it) lifts the other facility; missing EIR there, past maturity
        [EXT, "4001", "C4", "Tasdeer", "2", "1/1/2025", "QDB 4", 0, 0, 80_000,
         "12/31/2026", 0.05, 1, "QAR", 0, 0.05, 3],
        [EXT, "4002", "C4", "Long Term", "", "1/1/2018", "QDB 4", 0, 0, 60_000,
         "3/31/2026", "", 1, "QAR", 0, 0.05, 4],
        # customer C5: watchlist and default flags from the staging file
        [EXT, "5001", "C5", "Long Term", "", "1/1/2020", "QDB 6+", 0, 0, 120_000,
         "6/30/2030", 0.06, 1, "QAR", 0, 0.06, 4],
        [EXT, "6001", "C6", "Al Dhameen", "", "1/1/2020", "QDB 6", 0, 0, 700_000,
         "6/30/2031", 0.04, 12, "QAR", 0, 0.04, 4],
    ])
    lend.to_csv(out / "AccountMaster_1.csv", index=False)
    pd.DataFrame({"ExtractDate": EXT,
                  "CustomerId": ["C1", "C2", "C3", "C4", "C5", "C6"],
                  "IsDefault": ["FALSE"] * 6,
                  "IsWatchlist": ["FALSE"] * 4 + ["TRUE", "FALSE"],
                  "IsInsolvency": "", "IsDefaultInGCC": "",
                  "IsLocal1": "FALSE", "IsLocal2": "FALSE", "IsLocal3": "FALSE",
                  "IsLocal4": "", "IsLocal5": "", "IsLocal6": ""}
                 ).to_csv(out / "CustomerStagingFlag_1.csv", index=False)
    pd.DataFrame({"ExtractDate": EXT, "CustomerId": ["C1", "C2", "C3", "C4", "C5", "C6"],
                  "CustomerName": ["Alpha", "Beta", "Gamma", "Delta", "Eps", "Zeta"],
                  "IsIndividualAssessment": "FALSE"}
                 ).to_csv(out / "CustomerMaster_1.csv", index=False)
    inv = _am([
        [EXT, "9001", "ISSUER", "Bond", "", "1/1/2024", "A2", 0, 0, 5_000_000,
         "6/30/2029", 0.04, "", "QAR", 0, 0.04, 3],
        [EXT, "9002", "ISSUER", "Bond", "", "1/1/2024", "Baa3", 0, 0, 2_000_000,
         "6/30/2032", 0.05, "", "QAR", 0, 0.05, 3],
    ])
    inv.to_csv(out / "AccountMaster_2.csv", index=False)
    pd.DataFrame({"ExtractDate": EXT, "CustomerId": ["ISSUER"], "IsDefault": [0],
                  "IsWatchlist": [0], "IsInsolvency": [0], "IsDefaultInGCC": [0],
                  "IsLocal1": [0], "IsLocal2": [0], "IsLocal3": [0], "IsLocal4": [0],
                  "IsLocal5": [0], "IsLocal6": [0]}
                 ).to_csv(out / "CustomerStagingFlag_2.csv", index=False)
    pd.DataFrame({"ExtractDate": EXT, "CustomerId": ["ISSUER"],
                  "CustomerName": ["Issuer plc"]}).to_csv(out / "CustomerMaster_2.csv",
                                                          index=False)
    # a schedule for 1001 (descending months, as the ETL writes it)
    months = np.arange(36, -1, -1)
    ead = np.where(months == 0, 1_000_000, 1_000_000 * (1 - months / 40))
    pd.DataFrame({"ExtractDate": EXT, "ContractId": "1001", "MonthLifetime": months,
                  "EADLifetime": ead, "LGDLifetime": "", "PaymentScheduleLifetime": "",
                  "TotalLimitLifetime": ""}).to_csv(out / "LifeTimeParameterOther.csv",
                                                     index=False)
    pd.DataFrame({"ExtractDate": EXT, "CollateralId": ["K1", "K2", "K3", "K1"],
                  "ContractId": ["1001", "1001", "1002", "2001"],
                  "AllocationPercentage": [0.5, 1.0, 1.0, 0.25]}
                 ).to_csv(out / "AccountCollateralAllocation.csv", index=False)
    pd.DataFrame({"ExtractDate": EXT, "CollateralId": ["K1", "K2", "K3"],
                  "CollateralTypeId": ["4", "2", "999"],
                  "CollateralValue": [600_000, 100_000, 400_000]}
                 ).to_csv(out / "Collateral.csv", index=False)
    rep = build_final_ecl_report(root)
    if overlay:                  # an overlaid report, as the app writes one
        o = rep.copy()
        hit = o["Contract Id"].astype(str) == "1002"
        o.loc[hit, "Overlay Id"] = "OV1"
        o.loc[hit, "Overlay Amount"] = 1234.5
        o.loc[hit, "Ecl Final Onbal"] = o.loc[hit, "Ecl Model Onbal"] + 1234.5
        o.loc[hit, "Cla Amount Onbal"] = o.loc[hit, "Ecl Final Onbal"]
        o.to_csv(out / "FinalEclReport_overlay_OV1.csv", index=False)
    if reference:            # what the ETL keeps beside the run for the rating
        # C1 rated in the customer master; C2 too, overridden in the run; C3, C5
        # rated; C4 unrated, Agriculture: the program rating at 0 days past
        # due; C6 unrated, Al Dhameen: the fallback; ISSUER: one grade on the
        # external scale, one not (the investment fallback)
        given = {"C1": ("QDB 4", ""), "C2": ("QDB 4", ""), "C3": ("QDB 7", ""),
                 "C4": ("Unrated", "Agriculture"), "C5": ("QDB 6+", ""),
                 "C6": ("", "Others")}
        am1 = lend.reset_index(drop=True)
        own = {"2002": 0, "3002": 0}         # each facility's own days past due
        rows = [("AccountMaster_1", c, f"00{c}", k, "", own.get(c, d), *given[k])
                for c, k, d in zip(am1["ContractId"], am1["CustomerId"],
                                   am1["PastDueDays"])]
        rows += [("AccountMaster_2", "9001", "9001", "ISSUER", "Issuer plc", 0, "A2", ""),
                 ("AccountMaster_2", "9002", "9002", "ISSUER", "Issuer plc", 0, "NR", "")]
        (root / "reference").mkdir()
        pd.DataFrame(rows, columns=["File", "ContractId", "ContractIdOriginal",
                                    "CustomerId", "CustomerName", "PastDueDays",
                                    "RatingInput", "Sector"]).to_csv(
            root / "reference" / "contracts.csv", index=False)
    ov = root / "overrides"
    ov.mkdir()
    pd.DataFrame([{"customer_id": "C2", "value": "QDB 5-",
                   "reason": "test override", "created_by": "tester",
                   "created_at": "2026-07-01T10:00:00+0000",
                   "prior_value": "QDB 4", "source_run_id": "run_00001"}]
                 ).to_csv(ov / "rating_overrides.csv", index=False)
    return root


def recalc(path: Path, tmp: Path) -> openpyxl.Workbook:
    prof = tmp / "lo_profile"
    out = tmp / "recalc"
    subprocess.run([SOFFICE, f"-env:UserInstallation=file://{prof}", "--headless",
                    "--calc", "--convert-to", "xlsx:Calc MS Excel 2007 XML",
                    "--outdir", str(out), str(path)],
                   check=True, capture_output=True, timeout=600)
    return openpyxl.load_workbook(out / path.name, data_only=True)


def summary_rows(wb):
    ws = wb["Summary"]
    hdr = [c.value for c in ws[22]]
    rows = []
    r = 23
    while ws.cell(r, 1).value is not None:
        rows.append(dict(zip(hdr, [c.value for c in ws[r]])))
        r += 1
    return rows


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return make_run(tmp_path_factory.mktemp("calc"))


class TestFacts:
    def test_customers_list_both_books(self, run):
        c = run_customers(run)
        assert set(c["customer_id"]) == {"C1", "C2", "C3", "C4", "C5", "C6", "ISSUER"}
        assert c.loc[c["customer_id"] == "C1", "positions"].iloc[0] == 5

    def test_the_engine_figures_are_the_report(self, run):
        for cust in ("C1", "C2", "C3", "C4", "C5", "C6", "ISSUER"):
            f = customer_ecl_facts(run, cust)
            for l in f["lines"]:
                assert l["rep_ecl"] == pytest.approx(l["ecl_engine"], abs=1e-6,
                                                     nan_ok=True), (cust, l["cid"])
                assert l["rep_stage"] == l["stage_engine"]

    def test_the_pd_curves_are_the_runs_stpd(self, run):
        f = customer_ecl_facts(run, "ISSUER")
        st = pd.read_csv(run / "Output" / "StPD.csv")
        for c in f["curves"]:
            assert c["rt"] == 2
            ref = st[(st.PortfolioCode == c["portfolio"])
                     & (st.PDBucketDim1 == int(c["bucket"]))].sort_values("MonthLifetime")
            np.testing.assert_array_equal(c["stpd"][1:], ref["PDLifetime"].to_numpy())
            assert c["stpd"][0] == 0

    def test_overrides_are_listed(self, run):
        f = customer_ecl_facts(run, "C2")
        assert list(f["overrides"]["kind"]) == ["rating"]
        assert f["overrides"]["prior_value"].iloc[0] == "QDB 4"

    def test_an_unknown_customer_says_so(self, run):
        with pytest.raises(KeyError):
            customer_ecl_facts(run, "NOBODY")


class TestWorkbook:
    def test_sheets_and_formulas(self, run, tmp_path):
        p = tmp_path / "c1.xlsx"
        customer_ecl_workbook(run, "C1", p)
        wb = openpyxl.load_workbook(p)
        assert wb.sheetnames == ["Summary", "Facilities", "Rating", "Staging", "PD",
                                 "LGD", "EAD", "ECL", "Assumptions", "PD calculation"]
        assert [ws.title for ws in wb if ws.sheet_state == "hidden"] == [
            "Assumptions", "PD calculation"]
        # plain column names: no LIC flag names anywhere a reader looks
        for ws in wb:
            if ws.sheet_state == "visible":
                text = " ".join(str(c.value) for row in ws.iter_rows() for c in row
                                if isinstance(c.value, str) and not c.value.startswith("="))
                assert "IsLocal" not in text and "IsWatchlist" not in text, ws.title
        rows = summary_rows(wb)
        assert len(rows) == 5
        assert all(str(r["ECL (this workbook)"]).startswith("=") for r in rows)
        assert all(isinstance(r["ECL (report)"], (int, float)) for r in rows)

    def test_bytes_when_no_destination(self, run):
        b = customer_ecl_workbook(run, "C3")
        assert b[:2] == b"PK"


class TestThePdCalculation:
    """The PD tabs rebuild the curves from the run's model, forecasts and
    weights: one per rating scale and bucket, not one per portfolio."""

    @pytest.mark.parametrize("cust", ["C1", "ISSUER"])
    def test_the_calculated_curves_are_the_runs(self, run, cust):
        pm = customer_ecl_facts(run, cust)["pd_model"]
        assert pm["source"] == "Calculated" and pm["largest_difference"] < 1e-12

    def test_one_curve_per_scale_and_bucket(self, run, tmp_path):
        f = customer_ecl_facts(run, "C1")
        assert {l["portfolio"] for l in f["lines"]} >= {"Business Finance", "Off BS"}
        assert list(f["pd_model"]["calc_curves"]) == [(1, "11")]
        p = tmp_path / "c1.xlsx"
        write_customer_workbook(f, p)
        ws = openpyxl.load_workbook(p)["PD calculation"]
        text = [c.value for row in ws.iter_rows(max_col=1) for c in row]
        assert "Internal bucket 11 (QDB 4)" in text
        assert not any(str(t).startswith("External scale") for t in text)
        keys = [c.value for c in ws[_keys_row(ws)]][1:]
        assert keys[0] == "Internal | bucket 11"

    def test_a_run_whose_stpd_is_not_the_models_prices_on_its_own(self, tmp_path):
        r = make_run(tmp_path, tamper_stpd=True)
        pm = customer_ecl_facts(r, "C1")["pd_model"]
        assert pm["source"] == "Run's StPD" and "differs" in pm["reason"]

    @pytest.mark.skipif(SOFFICE is None, reason="LibreOffice is needed to recalculate")
    @pytest.mark.parametrize("cust", ["C1", "ISSUER"])
    def test_recalculated_the_curves_equal_the_runs_stpd(self, run, tmp_path, cust):
        f = customer_ecl_facts(run, cust)
        p = tmp_path / f"{cust}.xlsx"
        write_customer_workbook(f, p)
        ws = recalc(p, tmp_path)["PD calculation"]
        hr = _keys_row(ws)
        keys = [c.value for c in ws[hr]]
        st = pd.read_csv(run / "Output" / "StPD.csv")
        n = 0
        for col, k in enumerate(keys[1:], start=2):
            if not str(k).startswith(("Internal |", "External |")):
                continue
            rt = 1 if k.startswith("Internal") else 2
            pf = "Business Finance" if rt == 1 else "Investments"
            ref = st[(st.PortfolioCode == pf) & (st.PDBucketDim1 == int(k.split()[-1]))
                     ].sort_values("MonthLifetime")["PDLifetime"].to_numpy()
            got = np.array([ws.cell(hr + 2 + m, col).value for m in range(600)], float)
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)
            n += 1
        assert n == len(f["pd_model"]["calc_curves"]) > 0


@pytest.mark.skipif(SOFFICE is None, reason="LibreOffice is needed to recalculate")
class TestFromStartToEnd:
    """How each rating was set and why each stage, rebuilt by formula and
    equal to the run; the curve check reads no error."""

    SET_BY = {"C1": "Rating in the customer master", "C2": "Override in the run",
              "C4": "Program lending (Agriculture)", "C6": "Unrated: default rating"}

    @pytest.mark.parametrize("cust", ["C1", "C2", "C4", "C6", "ISSUER"])
    def test_the_rating_is_rebuilt(self, run, tmp_path, cust):
        f = customer_ecl_facts(run, cust)
        p = tmp_path / f"{cust}.xlsx"
        write_customer_workbook(f, p)
        wb = recalc(p, tmp_path)
        rows = summary_rows(wb)
        for row, l in zip(rows, f["lines"]):
            assert row["Rating"] == l["rating"], l["cid"]
        if cust in self.SET_BY:
            assert {r["How the rating was set"] for r in rows} == {self.SET_BY[cust]}
        else:
            assert [r["How the rating was set"] for r in rows] == [
                "Rating in the extract", "Unrated: default rating"]
        checks = [c.value for row in wb["Rating"].iter_rows() for c in row
                  if c.value in ("OK", "DIFFERS")]
        assert checks and set(checks) == {"OK"}

    def test_the_stage_says_why(self, run, tmp_path):
        f = customer_ecl_facts(run, "C2")
        p = tmp_path / "c2.xlsx"
        write_customer_workbook(f, p)
        wb = recalc(p, tmp_path)
        rows = summary_rows(wb)
        assert [r["Why"] for r in rows] == ["More than 60 days past due"] * 2
        assert [r["Stage"] for r in rows] == [2, 2]
        ws = wb["Staging"]
        assert [ws.cell(5 + k, 2).value for k in range(3)] == [
            2, "More than 60 days past due", "OK"]

    @pytest.mark.parametrize("cust,why", [("C3", "More than 90 days past due"),
                                          ("C4", "Has a Tasdeer facility"),
                                          ("C5", "On the watchlist"),
                                          ("C1", "Performing")])
    def test_one_stage_per_customer(self, run, tmp_path, cust, why):
        f = customer_ecl_facts(run, cust)
        p = tmp_path / f"{cust}.xlsx"
        write_customer_workbook(f, p)
        wb = recalc(p, tmp_path)
        ws = wb["Staging"]
        assert ws.cell(6, 2).value == why and ws.cell(7, 2).value == "OK"
        assert {r["Stage"] for r in summary_rows(wb)} == {ws.cell(5, 2).value}

    def test_the_curve_check_has_no_error(self, run, tmp_path):
        p = tmp_path / "is.xlsx"
        write_customer_workbook(customer_ecl_facts(run, "ISSUER"), p)
        ws = recalc(p, tmp_path)["PD calculation"]
        r = next(r for r in range(1, 400)
                 if ws.cell(r, 1).value == "The run's curve (StPD.csv)")
        got = [ws.cell(r + 1 + k, 3).value for k in range(2)]
        assert all(isinstance(v, float) and v < 1e-12 for v in got), got

    def test_a_run_that_kept_no_rating_inputs_shows_its_rating(self, tmp_path):
        r = make_run(tmp_path, reference=False)
        f = customer_ecl_facts(r, "C1")
        assert f["rating"]["recorded"] is False
        p = tmp_path / "c1.xlsx"
        write_customer_workbook(f, p)
        rows = summary_rows(recalc(p, tmp_path))
        assert all(x["Rating"] == "QDB 4" and x["Check"] == "OK" for x in rows)


def _keys_row(ws) -> int:
    """The row of the PD calculation tab holding the curve labels."""
    for r in range(1, ws.max_row + 1):
        if ws.cell(r, 1).value == "Month":
            return r
    raise AssertionError("no Month header")


@pytest.mark.skipif(SOFFICE is None, reason="LibreOffice is needed to recalculate")
class TestRecalculated:
    @pytest.mark.parametrize("cust", ["C1", "C2", "C3", "C4", "C5", "C6", "ISSUER"])
    def test_every_position_equals_the_engine(self, run, tmp_path, cust):
        f = customer_ecl_facts(run, cust)
        p = tmp_path / f"{cust}.xlsx"
        write_customer_workbook(f, p)
        rows = summary_rows(recalc(p, tmp_path))
        assert len(rows) == len(f["lines"])
        for row, l in zip(rows, f["lines"]):
            assert row["Stage"] == l["stage_engine"], l["cid"]
            if l["ecl_engine"] is None:
                assert not isinstance(row["ECL (this workbook)"], (int, float))
                continue
            assert row["ECL (this workbook)"] == pytest.approx(
                l["ecl_engine"], rel=1e-9, abs=1e-6), l["cid"]
            assert row["Check"] == "OK", l["cid"]

    def test_stage3_at_zero_and_any_stpd(self, tmp_path):
        r = make_run(tmp_path, stage3="zero", tamper_stpd=True)
        f = customer_ecl_facts(r, "C3")
        p = tmp_path / "c3.xlsx"
        write_customer_workbook(f, p)
        rows = summary_rows(recalc(p, tmp_path))
        for row, l in zip(rows, f["lines"]):
            assert row["ECL (this workbook)"] == pytest.approx(l["ecl_engine"],
                                                               rel=1e-9, abs=1e-6)
        assert [x["Stage"] for x in rows] == [3, 3]       # one stage per customer
        assert all(x["ECL (this workbook)"] == 0 for x in rows)
        assert rows[0]["Why"] == "More than 90 days past due"

    def test_an_overlay_is_added_to_the_model_figure(self, tmp_path):
        r = make_run(tmp_path, overlay=True)
        f = customer_ecl_facts(r, "C1")
        assert f["overlay_file"] == "FinalEclReport_overlay_OV1.csv"
        p = tmp_path / "c1.xlsx"
        write_customer_workbook(f, p)
        rows = {x["Contract"]: x for x in summary_rows(recalc(p, tmp_path))}
        o = rows["1002"]
        assert o["Overlay"] == 1234.5
        assert o["ECL final"] == pytest.approx(o["ECL (this workbook)"] + 1234.5)
        assert o["ECL final"] == pytest.approx(o["Report final"], abs=1e-6)
        assert all(x["Check"] == "OK" for x in rows.values())


REF = ref_run()


@pytest.mark.skipif(REF is None or SOFFICE is None,
                    reason="set IFRS9_REF_RUN and install LibreOffice")
def test_a_real_run(tmp_path):
    cust = run_customers(REF)
    for c in list(cust["customer_id"].head(2)) + list(cust["customer_id"].tail(2)):
        f = customer_ecl_facts(REF, c)
        p = tmp_path / "real.xlsx"
        write_customer_workbook(f, p)
        rows = summary_rows(recalc(p, tmp_path))
        for row, l in zip(rows, f["lines"]):
            if l["rep_ecl"] is not None:
                assert row["Check"] == "OK", (c, l["cid"])


def test_the_summary_names_the_config_version(run, tmp_path):
    import json
    from ifrs9qdb.calc_workbook import _versions
    assert _versions(run)["config_version"].startswith("not recorded")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "manifest.json").write_text(json.dumps({
        "run_metadata": {"config_version": "26Q3-PL-update",
                         "calculator_version": "v1.3", "calculator_label": "Q3"}}))
    v = _versions(tmp_path)
    assert v == {"config_version": "26Q3-PL-update", "calculator_version": "v1.3 (Q3)"}
    f = customer_ecl_facts(run, "C1")
    p = tmp_path / "c1.xlsx"
    write_customer_workbook(f, p)
    ws = openpyxl.load_workbook(p)["Summary"]
    labels = {ws.cell(r, 1).value: ws.cell(r, 3).value for r in range(3, 12)}
    assert labels["Config version"] == f["config_version"]
    assert all(w.freeze_panes is None for w in openpyxl.load_workbook(p).worksheets)


class TestContractWorkbook:
    """A contract's calculator shows the contract only, on every tab: what the
    customer's other facilities contribute is given as a value."""

    def test_the_contract_alone_from_lgd_to_the_summary(self, tmp_path):
        from ifrs9qdb.calc_workbook import (contract_ecl_workbook, run_contracts,
                                            customer_ecl_facts)
        run = make_run(tmp_path)
        rc = run_contracts(run)
        assert {"contract_id", "customer_id", "customer_name", "book"} <= set(rc.columns)
        assert rc.loc[rc["contract_id"] == "2002", "customer_id"].iloc[0] == "C2"
        p = tmp_path / "k.xlsx"
        p.write_bytes(contract_ecl_workbook(run, "2002"))
        wb = recalc(p, tmp_path)
        s = wb["Summary"]
        assert "contract 2002" in str(s["A1"].value)
        rows = summary_rows(wb)
        assert [str(r["Contract"]) for r in rows] == ["2002"]
        assert rows[0]["Check"] == "OK" and rows[0]["Stage"] == 2
        assert rows[0]["Contract (source id)"] == "002002"
        assert rows[0]["Customer name"] == "Beta"
        # only the contract on every tab
        for name in ("Facilities", "LGD", "EAD", "PD"):
            ws = wb[name]
            ids = {str(c.value) for row in ws.iter_rows(min_row=5) for c in row[:1]
                   if c.value is not None}
            assert "2001" not in ids, name
        assert "2002" in {str(c.value) for c in wb["Facilities"]["A"]}
        # the customer's worst days past due comes from its other facility
        ws = wb["Staging"]
        assert ws.cell(5, 2).value == 2 and ws.cell(6, 2).value == \
            "More than 60 days past due"

    def test_an_unknown_contract(self, tmp_path):
        import pytest
        from ifrs9qdb.calc_workbook import contract_ecl_workbook
        with pytest.raises(KeyError):
            contract_ecl_workbook(make_run(tmp_path), "nope")


def test_names_from_the_input_folder_for_an_older_run(tmp_path):
    """A run without the contract reference (before engine 0.45) takes the
    customer names from the input folder its manifest names."""
    import json
    import openpyxl
    from ifrs9qdb.calc_workbook import _input_names, _reference_names
    run = make_run(tmp_path / "r", reference=False)
    inp = tmp_path / "input"
    inp.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["CUSTOMERID", "CUSTOMERNAME"])
    ws.append(["C1", "Alpha Trading"])
    ws.append(["0C2", "Beta Foods"])
    wb.save(inp / "CustomerMaster.xlsx")
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "manifest.json").write_text(json.dumps({"input_dir": str(inp)}))
    _input_names.cache_clear()
    names = _reference_names(run / "Output")
    assert names.get("C1") == "Alpha Trading"
    assert len(names) == 2
