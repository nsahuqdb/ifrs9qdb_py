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


def _phase1(tmp_path):
    from ifrs9qdb.etl.pipeline import run_etl_phase1
    folder = write_demo_inputs(tmp_path / "in", dates=("2026-06-30",))[0]
    return run_etl_phase1(folder, tmp_path / "runs", run_id="run_00001")


def test_review_view_holds_both_books(tmp_path):
    st = _phase1(tmp_path)
    rv = st.review_view()
    assert set(rv["book"]) == {"Lending", "Investments"}
    iv = rv[rv["book"] == "Investments"]
    assert len(iv) == iv["customer_id"].nunique() > 0
    assert set(iv["portfolios"]) <= {"Investments", "Banks and Fis"}
    assert (iv["stage_final"] == "Stage 1").all()     # nothing flags them as extracted
    # the investment staging file is unchanged when nothing is overridden
    csf = st.tables["CustomerStagingFlag_2.csv"]
    assert (csf["IsLocal3"] == "").all() and (csf["IsDefault"] == "FALSE").all()


def test_investment_overrides_reach_the_report(tmp_path):
    from ifrs9qdb.etl.pipeline import run_etl_phase2
    st = _phase1(tmp_path)
    iv = st.investment_view().sort_values("exposure_total", ascending=False)
    a, b, c = iv["customer_id"].iloc[:3]
    res = run_etl_phase2(st, overrides={
        "stage": [{"customer_id": a, "book": "Investments", "value": "Stage 2",
                   "reason": "sukuk issuer on negative watch"},
                  {"customer_id": b, "book": "Investments", "value": "Stage 3",
                   "reason": "coupon missed"}],
        "rating": [{"customer_id": c, "book": "Investments", "value": "Caa1",
                    "reason": "downgraded after the extract"}]})
    assert res.ok, res.error
    run = tmp_path / "runs" / "run_00001"
    import pandas as pd
    rep = pd.read_csv(next(run.rglob("FinalEclReport.csv")), dtype=str)
    cust, stg, rat = "Customer Id", "Ifrs Stage", "Rating"
    by = rep.groupby(cust)
    assert set(by.get_group(a)[stg].str.extract(r"(\d)")[0]) == {"2"}
    assert set(by.get_group(b)[stg].str.extract(r"(\d)")[0]) == {"3"}
    assert set(by.get_group(c)[rat]) == {"Caa1"}
    ov = pd.read_csv(run / "overrides" / "stage_overrides.csv", dtype=str)
    assert set(ov["book"]) == {"Investments"} and set(ov["prior_value"]) == {"Stage 1"}


def test_an_investment_override_never_touches_a_lending_customer(tmp_path):
    from ifrs9qdb.etl.pipeline import normalise_overrides
    st = _phase1(tmp_path)
    lend = st.customer_view()["customer_id"].iloc[0]
    ov = normalise_overrides({"stage": [{"customer_id": lend, "book": "Investments",
                                         "value": "Stage 2", "reason": "x"}]}, st)
    assert ov["stage"]["prior_value"].iloc[0] == ""      # no such counterparty
