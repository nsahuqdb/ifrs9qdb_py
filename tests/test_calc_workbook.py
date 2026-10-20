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
             tamper_stpd: bool = False, overlay: bool = False) -> Path:
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
        # customer C2: DPD 75 (Stage 2) lifts its Stage 1 facility (contagion)
        [EXT, "2001", "C2", "Long Term", "", "1/1/2020", "QDB 5-", 75, 0, 400_000,
         "6/30/2028", 0.06, 1, "QAR", 0, 0.06, 3],
        [EXT, "2002", "C2", "Murabaha", "", "1/1/2020", "QDB 5-", 0, 0, 100_000,
         "6/30/2027", 0.06, 1, "QAR", 0, 0.06, 4],
        # customer C3: DPD 120 (Stage 3) and a manual Stage 2 on the other
        [EXT, "3001", "C3", "Long Term", "", "1/1/2019", "QDB 7", 120, 0, 250_000,
         "6/30/2027", 0.08, 1, "QAR", 0, 0.08, 4],
        [EXT, "3002", "C3", "Long Term", "Stage 2", "1/1/2019", "QDB 7", 0, 0, 90_000,
         "6/30/2027", 0.08, 1, "QAR", 0, 0.08, 4],
        # customer C4: Tasdeer, missing EIR on a second line, past maturity
        [EXT, "4001", "C4", "Tasdeer", "", "1/1/2025", "QDB 3", 0, 0, 80_000,
         "12/31/2026", 0.05, 1, "QAR", 0, 0.05, 3],
        [EXT, "4002", "C4", "Long Term", "", "1/1/2018", "QDB 3", 0, 0, 60_000,
         "3/31/2026", "", 1, "QAR", 0, 0.05, 4],
        # customer C5: watchlist and default flags from the staging file
        [EXT, "5001", "C5", "Long Term", "", "1/1/2020", "QDB 6+", 0, 0, 120_000,
         "6/30/2030", 0.06, 1, "QAR", 0, 0.06, 4],
        [EXT, "6001", "C6", "Al Dhameen", "", "1/1/2020", "QDB 5", 0, 0, 700_000,
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
        [EXT, "9002", "ISSUER", "Bond", "", "1/1/2024", "Baa1", 0, 0, 2_000_000,
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
        assert wb.sheetnames == ["Summary", "Settings", "Lines", "Overrides",
                                 "Staging", "LGD", "PD", "EAD", "ECL"]
        rows = summary_rows(wb)
        assert len(rows) == 5
        assert all(str(r["ECL (this workbook)"]).startswith("=") for r in rows)
        assert all(isinstance(r["ECL (report)"], (int, float)) for r in rows)

    def test_bytes_when_no_destination(self, run):
        b = customer_ecl_workbook(run, "C3")
        assert b[:2] == b"PK"


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
        assert rows[0]["Stage"] == 3 and rows[0]["ECL (this workbook)"] == 0
        assert rows[1]["Stage"] == 2             # the manual stage; contagion
        assert rows[1]["Why"] == "Manual stage 2"  # lifts Stage 1 only

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
