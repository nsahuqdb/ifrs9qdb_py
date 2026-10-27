"""The linked-effects system: estimated on outturns only, recovering the
coefficients the data were drawn from, and simulating a scenario the way
the screens describe it -- main variables as set, blank years and the other
variables from the equations, a variable left out passing nothing on."""
import numpy as np
import pytest

from ifrs9qdb.macro.system import fit_system, system_spec

from system_store import TRUE, build


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    return fit_system(build(tmp_path_factory.mktemp("sys") / "store"))


class TestFit:
    def test_the_order_is_the_spec_and_recursive(self, system):
        assert system.order == list(system_spec()["variables"])
        assert set(system.given_only()) == {"GDP_WLD", "US_RATE"}

    def test_only_outturns_are_used(self, system):
        for eq in system.equations.values():
            assert eq.years[1] <= 2025, eq.var

    @pytest.mark.parametrize("key", [k for k in TRUE if k[1] != k[0] and abs(TRUE[k]) >= 0.1])
    def test_the_drawn_coefficients_come_back(self, system, key):
        var, drv, lag = key
        got = system.equations[var].coef(drv, lag)
        assert got == pytest.approx(TRUE[key], rel=0.35), (key, got)

    def test_a_weak_link_is_kept_small_and_the_right_way_round(self, system):
        got = system.equations["NONOIL_GDP_QAT"].coef("CREDIT_QAT", 1)
        assert 0 <= got < 0.3

    def test_a_wrong_signed_driver_is_never_kept(self, system):
        for v, eq in system.equations.items():
            for t in eq.terms:
                s = system.signs.get(t["driver"])
                if t["driver"] != v and s:
                    assert np.sign(t["coef"]) in (0, s), (v, t)

    def test_the_table_has_a_row_per_term(self, system):
        tab = system.table()
        n = sum(len(e.terms) for e in system.equations.values()) + len(system.given_only())
        assert len(tab) == n


class TestSimulate:
    def test_main_years_are_held_and_blank_years_filled(self, system):
        r = system.simulate({"OIL_PRICE": [-40.0, None, None, None, None]})
        p = r["paths"]
        assert p["OIL_PRICE"][0] == -40.0
        assert r["role"]["OIL_PRICE"] == "main"
        assert abs(p["GOV_SPEND_QAT"][0]) > 0
        assert "GOV_SPEND_QAT" in r["recommended"]

    def test_the_parts_add_up_to_each_filled_value(self, system):
        given = {"OIL_PRICE": [-40.0, None, None, None, None], "GDP_WLD": [-2.0, 0, 0, 0, 0]}
        p = system.simulate(given)["paths"]
        ex = system.explain(given)
        assert len(ex) and not ((ex["variable"] == "GDP_WLD").any())
        tot = ex.groupby(["variable", "year"])["part"].sum()
        for (v, y), x in tot.items():
            assert x == pytest.approx(float(p[v][y - 1]), abs=1e-9), (v, y)
        assert not ((ex["variable"] == "OIL_PRICE") & (ex["year"] == 1)).any()

    def test_a_variable_left_out_stays_at_baseline_and_passes_nothing_on(self, system):
        on = system.simulate({"GDP_WLD": [-3.0, 0, 0, 0, 0]})
        off = system.simulate({"GDP_WLD": [-3.0, 0, 0, 0, 0]}, linked=["OIL_PRICE"])
        assert np.all(off["paths"]["NONOIL_GDP_QAT"] == 0)
        assert np.all(off["paths"]["CREDIT_QAT"] == 0)
        assert off["role"]["CREDIT_QAT"] == "off"
        assert np.any(on["paths"]["CREDIT_QAT"] != 0)

    def test_the_system_is_linear(self, system):
        a = system.simulate({"GDP_WLD": [-2.0, 1.0, 0, 0, 0], "US_RATE": [1, 1, 0, 0, 0]})
        b = system.simulate({"GDP_WLD": [-4.0, 2.0, 0, 0, 0], "US_RATE": [2, 2, 0, 0, 0]})
        for v in system.order:
            assert np.allclose(2 * a["paths"][v], b["paths"][v])

    def test_a_credit_shock_reaches_non_oil_a_year_later(self, system):
        p = system.simulate({"CREDIT_QAT": [-15.0, None, None, None, None]})["paths"]
        assert p["NONOIL_GDP_QAT"][0] == 0 and p["NONOIL_GDP_QAT"][1] < 0

    def test_a_shock_beyond_the_data_is_flagged(self, system):
        r = system.simulate({"GDP_WLD": [-40.0, None, None, None, None]})
        assert 0 in r["flags"]["OIL_PRICE"]

    def test_draws_are_reproducible(self, system):
        a, b = system.draw(50, 5, seed=3), system.draw(50, 5, seed=3)
        assert a.shape == (50, len(system.order), 5) and np.array_equal(a, b)

    def test_the_baseline_is_the_store_projection(self, system):
        b = system.baseline("GDP_WLD", [2026, 2027])
        d = system.display("GDP_WLD").set_index("year")
        assert b == pytest.approx([d.loc[2026, "value"], d.loc[2027, "value"]])
        assert not d.loc[2026, "actual"] and d.loc[2025, "actual"]
