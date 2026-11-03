"""The final report carries every stage's PD, LGD and collateral.

LIC leaves PD 12M blank, LGD Rate blank for Stage 3 (it fills Resid Lgd Rate
only) and Collateral Value blank outside Stage 3. The engine has all three
for every contract, so the report shows them."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ifrs9qdb.etl.report import build_final_ecl_report
from test_calc_workbook import make_run


def test_every_stage_has_its_pd_lgd_and_collateral(tmp_path):
    rep = build_final_ecl_report(make_run(tmp_path), write=False)
    num = lambda c: pd.to_numeric(rep[c], errors="coerce")
    priced = num("PD Lifetime Value").notna()
    assert priced.all()
    assert num("PD 12M")[priced].notna().all()
    assert num("LGD Rate").notna().all()
    s3 = num("Ifrs Stage") == 3
    assert (num("LGD Rate")[s3] == num("Resid Lgd Rate")[s3]).all()
    assert num("Resid Lgd Rate")[~s3].isna().all()
    assert num("Collateral Value").notna().all()
    # a secured Stage 1 facility shows its collateral too
    assert (num("Collateral Value")[num("Ifrs Stage") == 1] > 0).any()
    # Stage 1 lifetime PD is the twelve-month one, unless it matures sooner
    one = num("Ifrs Stage") == 1
    assert np.all(num("PD Lifetime Value")[one] <= num("PD 12M")[one] + 1e-12)
