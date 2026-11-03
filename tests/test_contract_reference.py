"""The report shows the source contract id beside the transformed one, and the
customer's name.

The LIC files carry the transformed id (off-balance product codes substituted,
leading zeros dropped) and a blank customer name; the ETL keeps both, per
AccountMaster row, in reference/contracts.csv beside the run, and the report
reads them from there.
"""
from __future__ import annotations

import pandas as pd

from ifrs9qdb.etl.lending import apply_id_substitutions
from ifrs9qdb.etl.pipeline import _write_contract_reference
from ifrs9qdb.etl.report import REPORT_COLUMNS, build_final_ecl_report
from test_calc_workbook import make_run


def test_the_original_id_sits_beside_the_transformed_one():
    i = REPORT_COLUMNS.index("Contract Id")
    assert REPORT_COLUMNS[i + 1] == "Contract Id Original"


def test_the_etl_keeps_the_source_id_and_the_name(tmp_path):
    raw = pd.Series(["0000121FGG007659", "000548840"])
    view = pd.DataFrame({"contract_id_raw": raw,
                         "contract_id": apply_id_substitutions(raw),
                         "customer_id": ["C1", "C2"]})
    inv = pd.DataFrame({"contract_id": ["1028"], "customer_id": ["DUKHAN BANK"]})
    src = {"CustomerMaster": pd.DataFrame({"CUSTOMERID": ["C1", "C2"],
                                           "CUSTOMERNAME": ["Alpha LLC", "Beta"]}),
           "CustomerMasterInvestments": pd.DataFrame(),
           "AccountMasterInvestments": pd.DataFrame({"CONTRACTID": ["1028"]})}
    _write_contract_reference(tmp_path, view, inv, src)
    r = pd.read_csv(tmp_path / "reference" / "contracts.csv", dtype=str)
    assert list(r["ContractId"]) == ["1213007659", "548840", "1028"]
    assert list(r["ContractIdOriginal"]) == ["0000121FGG007659", "000548840", "1028"]
    assert list(r["CustomerName"]) == ["Alpha LLC", "Beta", "DUKHAN BANK"]
    assert list(r["File"]) == ["AccountMaster_1"] * 2 + ["AccountMaster_2"]


def test_the_report_reads_them(tmp_path):
    run = make_run(tmp_path, reference=False)
    cm = pd.read_csv(run / "Output" / "CustomerMaster_1.csv", dtype=str)
    cm["CustomerName"] = ""          # as the LIC file carries it
    cm.to_csv(run / "Output" / "CustomerMaster_1.csv", index=False)
    am1 = pd.read_csv(run / "Output" / "AccountMaster_1.csv", dtype=str)
    am2 = pd.read_csv(run / "Output" / "AccountMaster_2.csv", dtype=str)
    rows = ([("AccountMaster_1", c, f"SRC-{c}", k, f"Name {k}")
             for c, k in zip(am1["ContractId"], am1["CustomerId"])]
            + [("AccountMaster_2", c, f"SRC-{c}", k, k)
               for c, k in zip(am2["ContractId"], am2["CustomerId"])])
    (run / "reference").mkdir()
    pd.DataFrame(rows, columns=["File", "ContractId", "ContractIdOriginal",
                                "CustomerId", "CustomerName"]).to_csv(
        run / "reference" / "contracts.csv", index=False)
    rep = build_final_ecl_report(run, write=False)
    assert (rep["Contract Id Original"] == "SRC-" + rep["Contract Id"].astype(str)).all()
    lend = rep["Customer Id"].astype(str).str.startswith("C")
    assert (rep.loc[lend, "Customer Name"] == "Name " + rep.loc[lend, "Customer Id"]).all()
    # the customer master's own name wins where it has one
    assert set(rep.loc[~lend, "Customer Name"]) == {"Issuer plc"}


def test_a_run_without_the_reference_leaves_the_column_blank(tmp_path):
    rep = build_final_ecl_report(make_run(tmp_path, reference=False), write=False)
    assert "Contract Id Original" in rep.columns
    assert (rep["Contract Id Original"].fillna("") == "").all()


def test_the_etl_keeps_what_the_rating_is_built_from(tmp_path):
    """The customer calculator rebuilds each rating from these: the rating in
    the customer master, the sector (for program lending) and the facility's
    own days past due; for investments, the extract's rating."""
    view = pd.DataFrame({"contract_id_raw": ["001", "002", "003"],
                         "contract_id": ["1", "2", "3"],
                         "customer_id": ["C1", "C1", "C2"],
                         "past_dues_days": [0.0, 45.0, 120.0]})
    inv = pd.DataFrame({"contract_id": ["9"], "customer_id": ["BANK"]})
    src = {"CustomerMaster": pd.DataFrame({"CUSTOMERID": ["C1", "C2"],
                                           "CUSTOMERNAME": ["Alpha", "Beta"],
                                           "RATING": ["QDB 4", "Unrated"]}),
           "IndustryCode": pd.DataFrame({"CUSTOMERID": ["C1", "C2"],
                                         "DESCRIPTION": ["4100 Construction",
                                                         "0113 Growing of vegetables"]}),
           "CustomerMasterInvestments": pd.DataFrame(),
           "AccountMasterInvestments": pd.DataFrame({"CONTRACTID": ["9"],
                                                     "RATING": ["A2"]})}
    static = {"industry_sector_mapping": pd.DataFrame(
        {"industry_code": ["113", "4100"], "sector": ["Agriculture", "Construction"]})}
    _write_contract_reference(tmp_path, view, inv, src, static)
    r = pd.read_csv(tmp_path / "reference" / "contracts.csv", dtype=str,
                    keep_default_na=False)
    assert list(r["RatingInput"]) == ["QDB 4", "QDB 4", "Unrated", "A2"]
    assert list(r["Sector"]) == ["Construction", "Construction", "Agriculture", ""]
    assert [float(x) for x in r["PastDueDays"]] == [0, 45, 120, 0]
