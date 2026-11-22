"""The synthetic demonstration book: twelve readable extracts per date."""
import pandas as pd

from ifrs9qdb.demo import write_demo_inputs
from ifrs9qdb.etl.read_inputs import INPUT_SPECS, read_all_inputs


def test_demo_book_reads_back(tmp_path):
    folders = write_demo_inputs(tmp_path, dates=("2026-06-30", "2026-09-30"))
    assert [f.name for f in folders] == ["2026-06-30", "2026-09-30"]
    for f in folders:
        inputs = read_all_inputs(f)
        assert inputs.missing == []
        assert set(inputs.tables) == {s.name for s in INPUT_SPECS}
        assert all(len(df) > 0 for df in inputs.tables.values())


def test_demo_book_is_reproducible(tmp_path):
    a = read_all_inputs(write_demo_inputs(tmp_path / "a", dates=("2026-06-30",))[0])
    b = read_all_inputs(write_demo_inputs(tmp_path / "b", dates=("2026-06-30",))[0])
    for name, df in a.tables.items():
        pd.testing.assert_frame_equal(df, b.tables[name], obj=name)


def test_override_view_names_each_customers_portfolios(tmp_path):
    """The review step's customer list says which books each customer is in."""
    from ifrs9qdb.etl.pipeline import run_etl_phase1
    folder = write_demo_inputs(tmp_path / "in", dates=("2026-06-30",))[0]
    st = run_etl_phase1(folder, tmp_path / "runs", run_id="run_00001",
                        on_validation_error="warn")
    assert not st.done
    cv = st.customer_view()
    assert "portfolios" in cv.columns and len(cv) > 0
    books = {p for s in cv["portfolios"] for p in s.split(", ") if p}
    assert {"Business Finance", "Off BS"} <= books
    # lending only: the investment books are not offered for override
    assert not books & {"Investments", "Banks and Fis"}
    assert (cv["portfolios"] != "").all()
