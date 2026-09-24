"""Ids used as join keys.

A CSV column of whole numbers reads as int64 when it is complete and as float64
the moment one value is blank. ``astype(str)`` then renders the same id as
"548840" or "548840.0" depending on which, the two never match, and nothing
fails -- the lookup returns nothing and the number it fed goes quietly to zero.

That happened: AccountCollateralAllocation carries a blank ContractId, so all
5,932 allocations missed the account master. Collateral resolved to zero for
the entire book, LGD used none of it, and the collateral stress lever moved the
provision by exactly 0.00 at every setting including "remove all collateral".

The R is not exposed to it, because `as.character()` on an R integer has no
trailing ".0".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import ref_output
from ifrs9qdb.inputs import as_id


class TestAsId:
    def test_a_float_column_does_not_keep_the_decimal(self):
        assert list(as_id(pd.Series([548840.0, 604177.0]))) == ["548840", "604177"]

    def test_an_int_column_is_unchanged(self):
        assert list(as_id(pd.Series([548840, 604177]))) == ["548840", "604177"]

    def test_the_two_agree(self):
        """The whole point: the same id, whichever way the column was read."""
        complete = pd.Series([548840, 604177, 603041])
        with_a_blank = pd.Series([548840.0, 604177.0, np.nan])
        assert list(as_id(complete))[:2] == list(as_id(with_a_blank))[:2]

    def test_text_that_arrived_as_a_float_is_also_handled(self):
        assert list(as_id(pd.Series(["548840.0", " 604177 "]))) == ["548840", "604177"]

    def test_a_blank_becomes_empty_not_the_string_nan(self):
        """"nan" is a perfectly good dictionary key, which is the danger."""
        out = list(as_id(pd.Series([548840.0, np.nan])))
        assert out[1] == ""
        assert "nan" not in out

    def test_a_non_numeric_id_is_left_alone(self):
        """Investment contracts carry ISINs, not numbers."""
        assert list(as_id(pd.Series(["XS3233459968", "0000121FGG007659"]))) == \
            ["XS3233459968", "0000121FGG007659"]

    def test_a_genuine_decimal_is_not_truncated(self):
        assert list(as_id(pd.Series([12.5]))) == ["12.5"]


@pytest.mark.skipif(ref_output() is None,
                    reason="set IFRS9_REF_RUN to a run folder holding Output/")
class TestTheRealRunJoins:
    @staticmethod
    def inputs():
        from ifrs9qdb.inputs import load_engine_inputs
        return load_engine_inputs(ref_output())

    def test_collateral_resolves_to_real_contracts(self):
        inp = self.inputs()
        ids = set(inp.contracts["contract"])
        hit = set(inp.collateral_net) & ids
        assert len(hit) > 0.9 * len(inp.collateral_net), (
            f"only {len(hit)} of {len(inp.collateral_net)} allocations matched "
            "an account - the ids are not joining")

    def test_ead_curves_resolve_to_real_contracts(self):
        inp = self.inputs()
        ids = set(inp.contracts["contract"])
        assert set(inp.ead_curves) <= ids | {""}

    def test_the_collateral_lever_actually_moves_the_provision(self):
        """The symptom that exposed the join: 0.00 at every setting."""
        import pandas as pd

        from ifrs9qdb.analytics import normalise
        from ifrs9qdb.stress import StressSpec, apply_stress
        inp = self.inputs()
        rep = normalise(pd.read_csv(ref_output() / "FinalEclReport.csv",
                                    low_memory=False))
        internal = ["Al Dhameen", "Business Finance", "Off BS", "Tasdeer"]
        halved = apply_stress(inp, rep, StressSpec(name="h", portfolios=internal,
                                                   collateral_pct=50))
        none_left = apply_stress(inp, rep, StressSpec(name="n", portfolios=internal,
                                                      collateral_pct=0))
        assert halved["delta"] > 1
        assert none_left["delta"] > halved["delta"], (
            "removing all collateral must cost more than halving it")
