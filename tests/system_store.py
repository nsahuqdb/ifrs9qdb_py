"""A synthetic macro store holding every series the linked-effects system
reads (macro/data/system.yml), drawn from a recursive system with known
coefficients, so a test can check the fit recovers them.

Years 1980-2031; 2026 on are projections (the WEO's ``last_actual`` 2025).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ifrs9qdb.macro import MacroStore

PANEL = ["QAT", "SAU", "ARE", "KWT", "OMN", "BHR"]
YEARS = list(range(1980, 2032))

# the coefficients the data are drawn from
TRUE = {
    ("OIL_PRICE", "GDP_WLD", 0): 10.0,
    ("NONOIL_GDP_QAT", "GDP_WLD", 0): 1.3,
    ("NONOIL_GDP_QAT", "NONOIL_GDP_QAT", 1): 0.3,
    ("NONOIL_GDP_QAT", "CREDIT_QAT", 1): 0.08,
    ("CREDIT_QAT", "NONOIL_GDP_QAT", 1): 0.8,
    ("CREDIT_QAT", "CREDIT_QAT", 1): 0.2,
    ("GOV_SPEND_QAT", "OIL_PRICE", 0): 0.1,
    ("GDP_GCC", "GDP_WLD", 0): 0.8,
    ("GDP_QAT", "NONOIL_GDP_QAT", 0): 0.7,     # Qatar's total GDP: its non-oil part
}


def _level(growth, start=100.0):
    lv = [start]
    for g in growth[1:]:
        lv.append(lv[-1] * (1 + g / 100.0))
    return np.array(lv)


def build(root, seed: int = 7) -> MacroStore:
    rng = np.random.default_rng(seed)
    T = len(YEARS)
    wld = 3.5 + rng.normal(0, 1.5, T)
    oil = TRUE[("OIL_PRICE", "GDP_WLD", 0)] * (wld - 3.5) + rng.normal(0, 20, T)
    us = np.cumsum(rng.normal(0, 1.2, T)).clip(-5, 10) + 5
    series, obs = [], []

    def add(sid, name, values, freq="A", last="2025"):
        src, ds, c, ind, _ = sid.split(":")
        series.append({"series_id": sid, "source": src, "dataset": ds, "country": c,
                       "indicator": ind, "freq": freq, "name": name, "unit": "",
                       "last_actual": last, "release": "synthetic"})
        for y, v in zip(YEARS, values):
            obs.append({"series_id": sid, "period": str(y), "value": float(v)})

    add("IMF:WEO:G001:NGDP_RPCH:A", "World real GDP growth", wld)
    add("IMF:WEO:G001:POILAPSP:A", "Oil price", _level(oil, 30.0))
    add("BIS:CBPOL:USA:VALUE:M", "US policy rate", us, last="")
    for c in PANEL:
        cpi = 2.0 + rng.normal(0, 1.0, T)
        nonoil, credit, gx, gdp = (np.zeros(T) for _ in range(4))
        for t in range(T):
            lag = (lambda a: a[t - 1] if t else 0.0)
            gx[t] = TRUE[("GOV_SPEND_QAT", "OIL_PRICE", 0)] * oil[t] + rng.normal(0, 5)
            nonoil[t] = (TRUE[("NONOIL_GDP_QAT", "GDP_WLD", 0)] * (wld[t] - 3.5)
                         + TRUE[("NONOIL_GDP_QAT", "NONOIL_GDP_QAT", 1)] * lag(nonoil)
                         + TRUE[("NONOIL_GDP_QAT", "CREDIT_QAT", 1)] * lag(credit)
                         + rng.normal(0, 1.5))
            credit[t] = (TRUE[("CREDIT_QAT", "NONOIL_GDP_QAT", 1)] * lag(nonoil)
                         + TRUE[("CREDIT_QAT", "CREDIT_QAT", 1)] * lag(credit)
                         + rng.normal(0, 3))
            if c == "QAT":
                gdp[t] = TRUE[("GDP_QAT", "NONOIL_GDP_QAT", 0)] * nonoil[t] + rng.normal(0, 1)
            else:
                gdp[t] = TRUE[("GDP_GCC", "GDP_WLD", 0)] * (wld[t] - 3.5) + rng.normal(0, 2)
        add(f"IMF:WEO:{c}:PCPIPCH:A", "CPI inflation", cpi)
        add(f"IMF:WEO:{c}:GGX:A", "Government expenditure", _level(gx + cpi))
        add(f"IMF:REO_MCD:{c}:NGDPXO_RPCH:A", "Non-oil real GDP growth", 4 + nonoil)
        add(f"IMF:MFS_ODC:{c}:ODCORP_A_ACO_PS_ODCS.XDC:M", "Private credit",
            _level(credit + cpi), last="")
        add(f"IMF:WEO:{c}:NGDP_RPCH:A", "Real GDP growth", 3 + gdp)
    # shares of world output at PPP, which weight a supervisor's regions
    for c, w in {"USA": 15.0, "DEU": 3.2, "FRA": 2.2, "ITA": 1.8, "CHN": 18.5, "IND": 7.5,
                 "IDN": 2.4, "THA": 0.9, "MYS": 0.7, "PAK": 0.9, "JPN": 3.6, "GBR": 2.2}.items():
        add(f"IMF:WEO:{c}:PPPSH:A", "Share of world GDP at PPP", [w] * T)
    st = MacroStore(root)
    st.save(pd.DataFrame(series), pd.DataFrame(obs), label="synthetic", carry_forward=False)
    return st
