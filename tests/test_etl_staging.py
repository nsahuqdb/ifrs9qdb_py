"""The staging rule, ported from the engine rather than inferred.

Reverse-engineering this from output differences produced two plausible rules
that were both wrong -- on 55 and on hundreds of customers. Reading the rule
and porting it directly got it to one row. These pin the rule itself.
"""
import numpy as np
import pandas as pd
import pytest

from ifrs9qdb.etl.customer import apply_staging_rule


def rule(dpd, restructured=False, watchlist=False, threshold=60):
    n = len(dpd)
    return apply_staging_rule(
        dpd,
        [restructured] * n if isinstance(restructured, bool) else restructured,
        [watchlist] * n if isinstance(watchlist, bool) else watchlist,
        threshold)


def test_dpd_over_90_is_stage_3():
    assert list(rule([91, 200, 365])) == ["Stage 3"] * 3


def test_dpd_between_threshold_and_90_is_stage_2():
    assert list(rule([61, 75, 90])) == ["Stage 2"] * 3


def test_dpd_at_or_below_the_threshold_is_stage_1():
    assert list(rule([0, 30, 60])) == ["Stage 1"] * 3


def test_restructured_or_watchlisted_is_stage_2_at_zero_dpd():
    assert list(rule([0], restructured=True)) == ["Stage 2"]
    assert list(rule([0], watchlist=True)) == ["Stage 2"]


def test_stage_3_wins_over_a_stage_2_trigger():
    """A defaulted customer cannot be pulled back by also being watchlisted."""
    assert list(rule([120], watchlist=True, restructured=True)) == ["Stage 3"]


def test_the_threshold_is_configurable():
    assert list(rule([45], threshold=60)) == ["Stage 1"]
    assert list(rule([45], threshold=30)) == ["Stage 2"]


def test_missing_dpd_does_not_stage_a_customer():
    """A blank DPD must not be read as zero-and-fine or as a default."""
    assert list(rule([np.nan])) == ["Stage 1"]
    assert list(rule([np.nan], watchlist=True)) == ["Stage 2"]


def test_vectors_are_handled_elementwise():
    out = apply_staging_rule([0, 70, 120], [False, False, False],
                             [True, False, False], 60)
    assert list(out) == ["Stage 2", "Stage 2", "Stage 3"]
