"""Reference files written with percent signs.

A spreadsheet saving a percentage-formatted column writes "9.062050635890580%".
Read as text, the first sum failed -- "could not convert string to float" in
the PD curves step. The number is what the file's units say it is: percentage
points in the GDP growth series, a fraction everywhere else.
"""
import shutil
import time

import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.etl.macro import build_stpd_from_static
from ifrs9qdb.etl.pipeline import load_model_config
from ifrs9qdb.etl.static_ref import PACKAGED_STATIC, load_static_reference


def _rewrite(path, col, fn):
    lines = path.read_text().splitlines()
    head = [ln for ln in lines if ln.startswith("#")]
    df = pd.read_csv(path, comment="#", dtype=str, keep_default_na=False)
    df[col] = df[col].map(fn)
    path.write_text("\n".join(head) + ("\n" if head else "") + df.to_csv(index=False))


@pytest.fixture
def percent_static(tmp_path):
    d = tmp_path / "static"
    shutil.copytree(PACKAGED_STATIC, d)
    _rewrite(d / "non_oil_gdp_history.csv", "value", lambda v: f"{float(v):.15f}%")
    _rewrite(d / "gcc_real_gdp_growth.csv", "value",
             lambda v: f"{float(v):.15f}%" if v else v)
    _rewrite(d / "ttc_pd_table.csv", "ttc_pd", lambda v: f"{float(v) * 100:.15f}%")
    return d


def test_percentage_points_and_fractions_read_as_their_units(percent_static):
    assert "2015,9.062050635890580%" in \
        (percent_static / "non_oil_gdp_history.csv").read_text()
    want, got = load_static_reference(PACKAGED_STATIC), \
        load_static_reference(percent_static)
    for name, col in (("non_oil_gdp_history", "value"),
                      ("gcc_real_gdp_growth", "value"), ("ttc_pd_table", "ttc_pd")):
        assert pd.api.types.is_float_dtype(got[name][col])
        np.testing.assert_allclose(got[name][col].to_numpy(float),
                                   pd.to_numeric(want[name][col]).to_numpy(float),
                                   rtol=0, atol=1e-12)


def test_the_pd_curves_are_unchanged(percent_static):
    model, inputs = load_model_config(None)
    a = build_stpd_from_static(load_static_reference(PACKAGED_STATIC), model,
                               inputs, "2026-06-09")
    b = build_stpd_from_static(load_static_reference(percent_static), model,
                               inputs, "2026-06-09")
    num = a.select_dtypes("number").columns
    assert a.shape == b.shape
    np.testing.assert_allclose(b[num].to_numpy(float), a[num].to_numpy(float),
                               rtol=0, atol=1e-12)


def test_text_that_is_not_a_number_is_left_for_the_checks(tmp_path):
    d = tmp_path / "static"
    shutil.copytree(PACKAGED_STATIC, d)
    _rewrite(d / "non_oil_gdp_history.csv", "value",
             lambda v: "see note" if v.startswith("9.06") else f"{float(v):.4f}%")
    col = load_static_reference(d)["non_oil_gdp_history"]["value"]
    assert not pd.api.types.is_numeric_dtype(col)


def test_an_edited_file_is_read_again(percent_static):
    p = percent_static / "non_oil_gdp_history.csv"
    assert load_static_reference(percent_static)["non_oil_gdp_history"]["value"].iloc[0] \
        == pytest.approx(9.062050635890580)
    time.sleep(0.01)
    p.write_text(p.read_text().replace("9.062050635890580%", "9.5%"))
    assert load_static_reference(percent_static)["non_oil_gdp_history"]["value"].iloc[0] \
        == pytest.approx(9.5)


def test_a_header_saved_with_trailing_commas_still_declares_its_units(percent_static):
    """Excel saving the CSV writes "# units: percentage_points," -- the 26Q3
    snapshot. The comma was read as part of the unit, the file stopped being
    percentage points, and "9.06%" became 0.0906: against forecasts in points
    the scenario weights collapsed to 50% Significant Downturn / 50%
    Significant Uptrend."""
    p = percent_static / "non_oil_gdp_history.csv"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(ln + "," if ln.startswith("#") else ln
                           for ln in lines) + "\n")
    assert "# units: percentage_points," in p.read_text()
    v = load_static_reference(percent_static)["non_oil_gdp_history"]["value"]
    assert v.iloc[0] == pytest.approx(9.062050635890580)


def test_history_and_forecasts_in_different_units_stop_the_weights():
    from ifrs9qdb.etl.macro import compute_internal_scenario_weights
    st = load_static_reference(PACKAGED_STATIC)
    frac = st["non_oil_gdp_history"]["value"].to_numpy(float) / 100
    with pytest.raises(ValueError, match="different units"):
        compute_internal_scenario_weights(frac, [-0.19, 2.92],
                                          st["scenario_severity"])
    with pytest.warns(RuntimeWarning, match="look like fractions"):
        compute_internal_scenario_weights(
            st["non_oil_gdp_history"]["value"].to_numpy(float),
            [-0.0019, 0.0292], st["scenario_severity"])
