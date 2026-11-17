"""The investment book keeps its two portfolios: "Banks and Fis" when the
account type says so, "Investments" otherwise -- in the engine frame the
report is priced from, and in the analytics frame of an older report that
had put the whole book in "Investments"."""
import pandas as pd

from ifrs9qdb.analytics import normalise
from ifrs9qdb.etl.report import engine_frame


def test_engine_frame_splits_the_investment_book(tmp_path):
    pd.DataFrame({
        "ExtractDate": ["2026-06-30"] * 3, "ContractId": ["1142", "XS1", "XS2"],
        "CustomerId": ["DUKHAN BANK", "ARAMCO SUKUK 34", "ALMARAI SUKUK"],
        "AccountType": ["Banks and Fis", "Investments", "Investments"],
        "Rating": ["A1", "A1", "Baa3"], "PastDueDays": ["0", "0", "0"],
        "OnBalance": ["100", "200", "300"], "MaturityDate": ["2028-06-30"] * 3,
    }).to_csv(tmp_path / "AccountMaster_2.csv", index=False)
    e = engine_frame(tmp_path, "_2")
    assert e["portfolio"].tolist() == ["Banks and Fis", "Investments", "Investments"]
    assert set(e["rating_type"]) == {"External"}


def test_old_report_banks_rows_read_as_banks_and_fis():
    rep = pd.DataFrame({
        "Contract Id": ["1142", "XS1", "500"],
        "Customer Id": ["DUKHAN BANK", "ARAMCO SUKUK 34", "123"],
        "Portfolio Code": ["Investments", "Investments", "Business Finance"],
        "Account Type": ["Banks and Fis", "Investments", "Murabaha"],
        "Exposure On Bal": [100, 200, 300], "Cla Amount Onbal": [1, 2, 3],
    })
    d = normalise(rep)
    assert d["portfolio"].tolist() == ["Banks and Fis", "Investments", "Business Finance"]
