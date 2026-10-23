"""Writes tests/fixtures/macro/store: a small SYNTHETIC macro store (no real
data) with the ids the catalogue's concepts resolve to, so both engines'
tests read the same store. Regenerate with: python make_store.py"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).parent
OUT = HERE / "store" / "vintages" / "v-fixture"
rng = np.random.default_rng(20261023)
years = np.arange(1990, 2032)
T = len(years)
g = np.zeros(T); o = np.zeros(T); d = np.zeros(T)
for t in range(1, T):
    g[t] = 0.5 * g[t - 1] + rng.normal()
    o[t] = 0.3 * o[t - 1] + 0.5 * g[t] + rng.normal()
    d[t] = 0.4 * d[t - 1] + 0.4 * o[t] + rng.normal()
srows, orows = [], []


def add(sid, name, unit, freq, periods, vals, last_actual=""):
    src, ds, ctry, ind, fq = sid.split(":")
    srows.append(dict(series_id=sid, source=src, dataset=ds, country=ctry, indicator=ind,
                      freq=fq, name=name, unit=unit, last_actual=last_actual,
                      release="fixture"))
    for p, v in zip(periods, vals):
        orows.append(dict(series_id=sid, period=p, value=round(float(v), 6)))


ys = [str(y) for y in years]
for c, mu, b in (("QAT", 4.0, 1.5), ("SAU", 3.0, 1.2), ("ARE", 3.5, 1.0),
                 ("KWT", 2.5, 1.4), ("OMN", 2.8, 0.9), ("BHR", 3.0, 0.8),
                 ("CHN", 7.0, 0.0), ("G001", 3.2, 0.0)):
    e = rng.normal(0, 0.8, T)
    gr = mu + 1.0 * g + b * o + e
    add(f"IMF:WEO:{c}:NGDP_RPCH:A", "Gross domestic product, constant prices",
        "Percent change", "A", ys, gr, "2025")
    lvl = 50 * np.cumprod(1 + (gr + 2) / 100)
    add(f"IMF:WEO:{c}:NGDPD:A", "Gross domestic product, current prices",
        "U.S. dollars, billions", "A", ys, lvl, "2025")
add("IMF:WEO:QAT:PCPIPCH:A", "Inflation, average consumer prices", "Percent change", "A",
    ys, 2 + 0.6 * o + rng.normal(0, 0.8, T), "2025")
spend = 100 * np.cumprod(1 + (5 + 4 * o + rng.normal(0, 3, T)) / 100)
add("IMF:WEO:QAT:GGX:A", "General government total expenditure", "National currency, billions",
    "A", ys, spend, "2025")
add("IMF:WEO:QAT:GGXCNL_NGDP:A", "General government net lending/borrowing",
    "Percent of GDP", "A", ys, 3 + 6 * o + rng.normal(0, 2, T), "2025")
ny = years >= 2000
add("IMF:REO_MCD:QAT:NGDP_NOIL_RPCH:A", "Non-oil real GDP growth", "Percent change", "A",
    [str(y) for y in years[ny]], (4 + 1.5 * d + rng.normal(0, 0.7, T))[ny], "2025")
# monthly prices to 2025, from the yearly factor
mper, oil, lng, ff = [], [], [], []
lv, lg = 30.0, 6.0
for t, y in enumerate(years[years <= 2025]):
    ga = 0.30 * o[t] + rng.normal(0, 0.05)
    for m in range(1, 13):
        mper.append(f"{y}-{m:02d}")
        lv *= (1 + ga) ** (1 / 12) * (1 + rng.normal(0, 0.02))
        lg *= (1 + 0.8 * ga) ** (1 / 12) * (1 + rng.normal(0, 0.02))
        oil.append(lv)
        lng.append(lg)
        ff.append(max(0.1, 3 + 1.2 * g[t] + rng.normal(0, 0.1)))
add("IMF:PCPS:W00:POILBRE.USD:M", "Crude Oil (petroleum), Brent · US dollars",
    "US dollars", "M", mper, oil)
add("IMF:PCPS:W00:PNGASJP.USD:M", "Natural gas, LNG, Japan · US dollars", "US dollars",
    "M", mper, lng)
add("FRED:FEDFUNDS::FEDFUNDS:M", "Federal Funds Effective Rate", "Percent", "M", mper, ff)
qy = years[(years >= 2008) & (years <= 2025)]
qper, repi = [], []
lv = 100.0
for y in qy:
    t = int(np.where(years == y)[0][0])
    for q in range(1, 5):
        qper.append(f"{y}-Q{q}")
        lv *= 1 + (0.02 * d[t] + rng.normal(0, 0.01))
        repi.append(lv)
add("QCB:REPI:QAT:INDEX:Q", "Real estate price index", "Index", "Q", qper, repi)
add("QCB:QMR:QAT:DEPOSIT:A", "QCB deposit rate (QMR)", "Percent", "A",
    [str(y) for y in qy], [max(0.1, 2.5 + 1.1 * g[int(np.where(years == y)[0][0])]) for y in qy])
cred = 100 * np.cumprod(1 + (8 + 3 * d + rng.normal(0, 2, T)) / 100)
add("QCB:CREDIT_PRIVATE:QAT:LEVEL:A", "Credit to the private sector", "QAR billions", "A",
    [str(y) for y in years[ny & (years <= 2025)]], cred[ny & (years <= 2025)])
vy = years[(years >= 2010) & (years <= 2025)]
add("NPC:VISITORS:QAT:ARRIVALS:A", "Visitor arrivals", "Thousands", "A", [str(y) for y in vy],
    [1000 * np.exp(0.05 * (y - 2010) + 0.1 * d[int(np.where(years == y)[0][0])]) for y in vy])
add("IMF:FSI:QAT:FSANL_PT:Q", "Nonperforming loans to total gross loans", "Percent", "Q",
    qper, [1.5 + 0.3 * q for q in rng.normal(0, 1, len(qper))])
if OUT.parent.parent.exists():
    shutil.rmtree(OUT.parent.parent)
OUT.mkdir(parents=True)
pd.DataFrame(srows).sort_values("series_id").to_csv(OUT / "series.csv", index=False)
pd.DataFrame(orows).sort_values(["series_id", "period"]).to_csv(
    OUT / "observations.csv.gz", index=False, compression="gzip")
(OUT / "meta.yml").write_text(yaml.safe_dump({
    "vintage": "v-fixture", "created": "2026-10-23T00:00:00", "by": "fixture",
    "label": "synthetic test store", "parent": None, "n_series": len(srows),
    "n_obs": len(orows), "log": []}, sort_keys=False))
print(len(srows), "series", len(orows), "observations")
