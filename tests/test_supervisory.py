"""Supervisory scenarios from the Fed's published tables: read, translated
to world GDP and the US rate, and saved as supervisory."""
import numpy as np
import pytest

from ifrs9qdb.stresstest import Scenario, ScenarioStore
from ifrs9qdb.stresstest.supervisory import (FED_TABLES, fed_scenario, fed_urls,
                                             read_fed_table)

QUARTERS = [f"{y} Q{q}" for y in (2027, 2028, 2029) for q in (1, 2, 3, 4)] + ["2030 Q1"]


def _dom(gdp, rate):
    rows = ["Scenario Name,Date,Real GDP growth,Unemployment rate,3-month Treasury rate"]
    for d, g, r in zip(QUARTERS, gdp, rate):
        rows.append(f"Supervisory,{d},{g},5.0,{r}")
    return "\n".join(rows)


def _intl(gdp):
    rows = ["Scenario Name,Date,Euro area real GDP growth,Euro area inflation,"
            "Developing Asia real GDP growth,Japan real GDP growth,U.K. real GDP growth"]
    for d, g in zip(QUARTERS, gdp):
        rows.append(f"Supervisory,{d},{g},2.0,{g},{g},{g}")
    return "\n".join(rows)


N = len(QUARTERS)
BASE = {"baseline_domestic": _dom([2.0] * N, [4.0] * N),
        "baseline_international": _intl([1.5] * N)}
ADV = {"adverse_domestic": _dom([-8.0] * 4 + [2.0] * (N - 4), [0.1] * N),
       "adverse_international": _intl([-6.5] * 4 + [1.5] * (N - 4))}
W = {"US": 0.25, "Euro area": 0.2, "Developing Asia": 0.4, "Japan": 0.08, "UK": 0.07}


def test_the_table_is_read_by_quarter():
    d = read_fed_table(BASE["baseline_domestic"])
    assert list(d.columns[:2]) == ["scenario name", "date"]
    assert d["year"].tolist()[:4] == [2027] * 4 and d["q"].tolist()[:4] == [1, 2, 3, 4]
    assert "3-month treasury rate" in d.columns


def test_the_scenario_is_the_gap_to_the_fed_baseline():
    s = fed_scenario(2027, {**BASE, **ADV}, W)
    g = s["main"]["GDP_WLD"]
    assert len(g) == 3                          # three full calendar years
    # the US runs 10 points (8% fall against 2% growth) and the other regions
    # 8 points below their baselines through Y1, then at them
    assert g[0] == pytest.approx(0.25 * -10 + 0.75 * -8, abs=1e-6)
    assert g[1] == pytest.approx(0.0, abs=1e-9)
    r = s["main"]["US_RATE"]
    assert r[0] == pytest.approx(-3.9) and r[1] == pytest.approx(0.0)
    assert s["category"] == "supervisory" and s["region"] == "global"
    assert set(s["why"]) == {"GDP_WLD", "US_RATE"}
    assert s["references"][0]["url"].endswith("dfa-stress-tests-2027.htm")


def test_a_missing_table_is_named():
    with pytest.raises(ValueError, match="adverse_international"):
        fed_scenario(2027, {**BASE, "adverse_domestic": ADV["adverse_domestic"]}, W)


def test_the_addresses_follow_the_fed_pattern():
    u = fed_urls(2027)
    assert set(u) == set(FED_TABLES)
    assert u["adverse_domestic"].endswith("2027-Table_4A_Supervisory_Severely_Adverse_Domestic.csv")


def test_a_supervisory_save_stays_supervisory(tmp_path):
    st = ScenarioStore(tmp_path / "s.sqlite")
    s = fed_scenario(2027, {**BASE, **ADV}, W)
    sid = st.save(Scenario.from_dict(s), "me")
    got = st.get(sid)
    assert got["category"] == "supervisory" and got["saved"]
    mine = st.save(Scenario("Mine", main={"GDP_WLD": [-1.0]}), "me")
    assert st.get(mine)["category"] == "mine"
    assert np.isfinite(got["main"]["GDP_WLD"][0])
