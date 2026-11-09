"""Customer names for runs that predate them: taken from the other runs."""
import pandas as pd

from ifrs9qdb.names import customer_names, fill_names, name_book, run_names


def _run(root, name, rows, ref=None):
    out = root / name / "Output"
    out.mkdir(parents=True)
    pd.DataFrame(rows, columns=["Contract Id", "Customer Id", "Customer Name"]).to_csv(
        out / "FinalEclReport.csv", index=False)
    if ref is not None:
        (root / name / "reference").mkdir()
        pd.DataFrame(ref, columns=["ContractId", "CustomerId", "CustomerName"]).to_csv(
            root / name / "reference" / "contracts.csv", index=False)
    return root / name


def test_an_old_run_takes_names_from_a_newer_one(tmp_path):
    old = _run(tmp_path, "run_00001", [["1", "100", ""], ["2", "200", ""]])
    _run(tmp_path, "run_00002", [["1", "100", "Alpha"], ["3", "300", "Gamma"]])
    assert run_names(old) == {}
    names = customer_names(old)
    assert names["100"] == "Alpha" and names["300"] == "Gamma"
    assert "200" not in names


def test_a_run_keeps_its_own_names_first(tmp_path):
    old = _run(tmp_path, "run_00001", [["1", "100", "Alpha old"]])
    _run(tmp_path, "run_00002", [["1", "100", "Alpha new"]])
    assert customer_names(old)["100"] == "Alpha old"


def test_the_newest_run_wins_in_the_book(tmp_path):
    _run(tmp_path, "run_00001", [["1", "100", "Alpha 1"]])
    _run(tmp_path, "run_00002", [["1", "100", "Alpha 2"]])
    assert name_book(tmp_path)["100"] == "Alpha 2"


def test_the_reference_is_read_too_and_ids_join(tmp_path):
    run = _run(tmp_path, "run_00001", [["1", "100.0", ""]],
               ref=[["1", "100", "Alpha"]])
    assert customer_names(run) == {"100": "Alpha"}


def test_fill_names_only_fills_blanks():
    got = fill_names(pd.Series(["100", "200", "300"]),
                     pd.Series(["Own", None, ""]), {"100": "X", "200": "Beta"})
    assert got.tolist() == ["Own", "Beta", ""]
