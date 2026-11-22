"""A synthetic demo book: the twelve source extracts for fictitious customers.

Used for training, screenshots in the user manual and trying the tool without
client data. Nothing here comes from QDB's portfolio: names are "Demo customer
NNN", amounts, ratings, days past due and collateral are drawn from fixed
random seeds, so the same call always writes the same files.

    python -m ifrs9qdb demo --out demo_inputs            # 30 Jun and 30 Sep 2026
    python -m ifrs9qdb demo --out demo_inputs --dates 2026-09-30

Each date gets its own folder holding the extracts with the headers and column
positions the Oracle jobs use, so the pipeline reads them exactly as it reads
a real drop. The same customers and contracts appear at every date: balances
amortise along their schedules, contracts mature or start, and customers move
between days-past-due buckets and grades, so a run on each date can be
compared (Movement, Migration).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["write_demo_inputs", "DEMO_DATES"]

DEMO_DATES = ("2026-06-30", "2026-09-30")
SEED = 20261122

# Internal grades in the master scale, best to worst (no default grade: a
# defaulted customer is found by its days past due).
GRADES = ["QDB 1+", "QDB 1", "QDB 1-", "QDB 2+", "QDB 2", "QDB 2-", "QDB 3+",
          "QDB 3", "QDB 3-", "QDB 4+", "QDB 4", "QDB 4-", "QDB 5+", "QDB 5",
          "QDB 5-", "QDB 6+", "QDB 6", "QDB 6-", "QDB 7", "QDB 8", "QDB 9"]
GRADE_P = np.array([0, 0, .005, .01, .015, .02, .04, .05, .06, .08, .1, .1,
                    .1, .09, .08, .07, .06, .05, .04, .02, .01])
BF_PRODUCTS = ["Long Term", "Murabaha", "Ijara", "Short Term", "Istinsa", "Tawaraq"]
BF_P = [.3, .25, .15, .12, .1, .08]
INDUSTRY = [  # (code, description) present in the packaged sector mapping
    ("2392", "2392 Manufacture of clay building"),
    ("3290", "3290 Other manufacturing n.e.c."),
    ("2790", "2790 Manufacture of other electrical"),
    ("4100", "4100 Construction of buildings"),
    ("4390", "4390 Other specialized construction"),
    ("6399", "6399 Other information service activities"),
    ("5210", "5210 Warehousing and storage"),
    ("4721", "4721 Retail sale of food in specialized"),
    ("5510", "5510 Short term accommodation"),
    ("5610", "5610 Restaurants and mobile food"),
    ("0113", "0113 Growing of vegetables and melons,"),
    ("0146", "0146 Raising of poultry"),
    ("0311", "0311 Marine fishing"),
    ("8620", "8620 Medical and dental practice"),
    ("8521", "8521 General secondary education"),
]
INVESTMENTS = [  # counterparty, account type, external rating, rate %
    ("Demo Sovereign Sukuk 2029", "Sukuk", "Aa3", 4.1),
    ("Demo Sovereign Sukuk 2031", "Sukuk", "Aa3", 4.4),
    ("Demo Gulf Sovereign Bond 2028", "Bonds", "A1", 4.6),
    ("Demo Telecom Sukuk 2030", "Sukuk", "A2", 5.0),
    ("Demo Utility Sukuk 2027", "Sukuk", "A3", 5.2),
    ("Demo Energy Bond 2032", "Bonds", "Baa1", 5.6),
    ("Demo Real Estate Sukuk 2028", "Sukuk", "Baa3", 6.1),
    ("Demo Islamic Bank A", "Banks and Fis", "A1", 4.8),
    ("Demo Islamic Bank B", "Banks and Fis", "A2", 4.9),
    ("Demo Commercial Bank C", "Banks and Fis", "A3", 5.0),
    ("Demo Investment Bank D", "Banks and Fis", "Baa2", 5.4),
    ("Demo Regional Bank E", "Banks and Fis", "Baa1", 5.3),
]


@dataclass
class _Contract:
    contract_id: str
    customer_id: str
    product: str
    open_date: pd.Timestamp
    maturity: pd.Timestamp
    amount: float
    rate: float           # percent a year
    freq: int             # months between instalments; 0 = bullet
    off_balance: bool


def _book(rng: np.random.Generator, n_customers: int = 420):
    customers, contracts = [], []
    for i in range(n_customers):
        cid = str(20001 + i)
        u = rng.random()
        if u < 0.07:
            kind = "Al Dhameen"
        elif u < 0.11:
            kind = "Tasdeer"
        else:
            kind = "BF"
        grade = rng.choice(GRADES, p=GRADE_P / GRADE_P.sum())
        unrated = kind == "BF" and rng.random() < 0.08
        ind = INDUSTRY[rng.integers(len(INDUSTRY))]
        customers.append(dict(customer_id=cid, name=f"Demo customer {i + 1:03d}",
                              kind=kind, grade=None if unrated else grade,
                              industry=ind, limit=0.0,
                              risk=rng.normal()))
        n_fac = 1 + rng.poisson(0.8)
        for k in range(n_fac):
            if kind == "Al Dhameen":
                product, off = "Al Dhameen", True
            elif kind == "Tasdeer":
                product, off = "Tasdeer", True
            elif rng.random() < 0.15:
                product, off = ("LG" if rng.random() < 0.7 else "LC"), True
            else:
                product, off = rng.choice(BF_PRODUCTS, p=BF_P), False
            opened = pd.Timestamp("2026-08-31") - pd.DateOffset(
                months=int(rng.integers(1, 72)))
            term = int(rng.choice([12, 24, 36, 48, 60, 84, 120],
                                  p=[.12, .14, .2, .16, .18, .12, .08]))
            if product in ("Short Term", "LC"):
                term = int(rng.choice([6, 9, 12]))
            maturity = opened + pd.DateOffset(months=term)
            amount = float(np.round(np.exp(rng.normal(14.6, 1.0)), -3))
            freq = 0 if product in ("Short Term", "LG", "LC", "Al Dhameen", "Tasdeer") \
                else int(rng.choice([1, 3, 6], p=[.6, .3, .1]))
            contracts.append(_Contract(
                contract_id=str(4000000001 + len(contracts)), customer_id=cid,
                product=product, open_date=opened, maturity=maturity,
                amount=amount, rate=float(np.round(rng.uniform(3.5, 7.5), 2)),
                freq=freq, off_balance=off))
    return customers, contracts


def _schedule(c: _Contract) -> pd.DataFrame:
    """Instalments from the first due date to maturity, equal principal."""
    if c.freq == 0:
        dates = [c.maturity]
    else:
        dates = list(pd.date_range(c.open_date + pd.DateOffset(months=c.freq),
                                   c.maturity, freq=pd.DateOffset(months=c.freq)))
        if not dates or dates[-1] < c.maturity:
            dates.append(c.maturity)
    n = len(dates)
    principal = np.full(n, c.amount / n)
    bal_before = c.amount - np.concatenate([[0], np.cumsum(principal)[:-1]])
    months = max(c.freq, 1) if c.freq else max(
        1, round((c.maturity - c.open_date).days / 30.4))
    interest = bal_before * c.rate / 100 * months / 12
    return pd.DataFrame({"date": dates, "principal": principal,
                         "interest": interest, "balance": bal_before - principal})


def _state(rng_c: np.random.Generator, risk: float, q: int):
    """Days past due and a grade move for one customer at quarter q (0, 1, ...).

    The customer's draws are fixed by its own seed, so the same customer gets
    the same path whatever dates are asked for."""
    draws = rng_c.normal(size=8)
    z = risk * 0.8 + draws[:q + 1].sum() * 0.35
    if z > 1.35:
        dpd = int(91 + abs(draws[q]) * 200)
    elif z > 1.15:
        dpd = 61 + int(abs(draws[q]) * 25) % 30
    elif z > 0.95:
        dpd = 31 + int(abs(draws[q]) * 25) % 30
    elif z > 0.6:
        dpd = 1 + int(abs(draws[q]) * 25) % 30
    else:
        dpd = 0
    notch = int(np.clip(np.round(draws[q] * 0.45 + (0.3 if z > 1.2 else 0)), -1, 2)) \
        if q > 0 else 0
    return dpd, notch


def _frames(customers, contracts, as_of: pd.Timestamp, q: int):
    rows_am, rows_rs, rows_or, rows_cm, rows_sf, rows_ic = [], [], [], [], [], []
    rows_col, rows_alloc = [], []
    state = {}
    for c in customers:
        rng_c = np.random.default_rng(SEED + int(c["customer_id"]))
        dpd, notch = _state(rng_c, c["risk"], q)
        g = c["grade"]
        if g is not None:
            gi = int(np.clip(GRADES.index(g) + notch, 0, len(GRADES) - 1))
            g = GRADES[gi]
        state[c["customer_id"]] = (dpd, g)
    live = []
    for k in contracts:
        if k.open_date > as_of:
            continue
        dpd, _ = state[k.customer_id]
        if k.maturity <= as_of and dpd <= 90:
            continue          # repaid; a defaulted contract stays past maturity
        sch = _schedule(k)
        paid = sch[sch["date"] <= as_of]
        bal = float(sch["balance"].iloc[len(paid) - 1]) if len(paid) else k.amount
        if dpd > 0 and len(paid):  # arrears: the missed instalments still owed
            missed = paid.tail(max(1, dpd // 30))
            bal += float(missed["principal"].sum())
        if bal <= 1:
            continue
        live.append((k, bal, sch))
    used = {k.customer_id for k, _, _ in live}
    rng = np.random.default_rng(SEED + q)
    cust = {c["customer_id"]: c for c in customers}
    col_id = 700001
    for k, bal, sch in live:
        dpd, _ = state[k.customer_id]
        restr = int(cust[k.customer_id]["risk"] > 1.6)
        # Off-balance products arrive as the real extract delivers them: the
        # exposure (guarantee or LC amount times its CCF) in ONBALANCE, which
        # is what the engine prices, and the full amount in OFFBALANCE.
        ccf = {"LG": 0.5, "LC": 0.2, "Al Dhameen": 1.0, "Tasdeer": 0.5}.get(k.product, 1.0)
        on = round(bal * ccf, 2) if k.off_balance else round(bal, 2)
        off = round(bal, 2) if k.off_balance else 0.0
        accr = 0.0 if k.off_balance else round(on * k.rate / 100 / 12, 2)
        row = {"EXTRACTDA": as_of, "CONTRACTID": k.contract_id,
               "LIMID": "L" + k.contract_id[-6:], "CUSTOMERID": k.customer_id,
               "PORTFOLIOCODE": "", "ACCOUNTTYPE": k.product,
               "IMPAIRMENTAMOUNT": "", "STAGE": "", "OPENDATE": k.open_date,
               "RATING": "", "PASTDUEDAYS": dpd, "PD12M": "",
               "ISRESTRUCTURED": restr, "LGDRATE": "", "LOANTOVALUE": "",
               "PRINCIPALOUTSTANDING": 0.0 if k.off_balance else on, "OFFBALANCE": off,
               "ONBALANCE": round(on + accr, 2), "EAD": "",
               "CCF": ccf if k.off_balance else "", "MATURITYDATE": k.maturity, "EXPECTEDMATURITYDATE": "",
               "EIR": k.rate}
        for j, nm in enumerate(["PRINCIPAL", "PRINCIPALOVERDUE", "INTERESTACCRUED",
                                "INTERESTOVERDUE", "FEE", "FEEOVERDUE", "PENALTY",
                                "PENALTYOVERDUE", "COMMISSION", "COMMISSIONOVERDUE",
                                "OTHER", "OTHEROVERDUE"]):
            row[nm] = accr if nm == "INTERESTACCRUED" else ""
        row.update({"PAYMENTFREQUENCY": k.freq, "CURRENCYCODE": "QAR",
                    "DEFERRALPERIOD": 0, "NOMINALINTERESTRATE": k.rate,
                    "PAYMENTTYPEID": 3 if k.freq == 0 else 4})
        rows_am.append(row)
        if not k.off_balance:
            fut = sch[sch["date"] > as_of].reset_index(drop=True)
            if len(fut):
                # arrears fall due with the next instalment, so the schedule
                # runs off to zero
                princ = fut["principal"].to_numpy().copy()
                princ[0] += on - princ.sum()
                prev_bal = on
                for j, r in fut.iterrows():
                    p_ = round(float(princ[j]), 2) if j < len(fut) - 1 else round(prev_bal, 2)
                    rows_rs.append({"KEY_1": k.contract_id, "POST_DATE": r["date"],
                                    "START_DATE": r["date"], "PRINCE_DUE": p_,
                                    "PROJ_INT": round(r["interest"], 2),
                                    "REPAYMENT": round(p_ + r["interest"], 2),
                                    "BALANCE": round(max(prev_bal - p_, 0), 2)})
                    prev_bal = max(prev_bal - p_, 0)
        orng = np.random.default_rng(SEED + int(k.contract_id) % 100000)
        og = cust[k.customer_id]["grade"] or "QDB 5"
        og = GRADES[int(np.clip(GRADES.index(og) - orng.integers(0, 3), 0, 20))]
        rows_or.append({"EXTRACTDA": as_of, "CONTRACTID": k.contract_id,
                        "PD12M": "", "RATING": og, "PASTDUEDAYS": 0,
                        "ISWATCHLIST": 0})
        # collateral on about half of the funded contracts
        if not k.off_balance and orng.random() < 0.5:
            ctype = int(orng.choice([4, 12, 2, 22, 7, 24, 10],
                                    p=[.3, .25, .1, .05, .1, .1, .1]))
            value = round(bal * orng.uniform(0.4, 1.6), -3)
            rows_col.append({"EXTRACTDA": as_of, "COLLATERALID": str(col_id),
                             "C": "", "CO": ctype, "COL": "QAR",
                             "COLLATERALVALUE": value})
            rows_alloc.append({"EXTRACTDA": as_of, "COLLATERALID": str(col_id),
                               "CONTRACTID": k.contract_id,
                               "ALLOCATIONPERCENTAGE": 100})
            col_id += 1
    for c in customers:
        if c["customer_id"] not in used:
            continue
        dpd, g = state[c["customer_id"]]
        rating = g if g is not None else "Unrated"
        rows_cm.append({"EXTRACTDA": as_of, "CUSTOMERID": c["customer_id"],
                        "PORTFOLIOCODE": "", "CUSTOMERNAME": c["name"],
                        "CUSTOMERLIMIT": "", "RATING": rating,
                        "PASTDUE_DAYS": dpd, "C8": "", "C9": "",
                        "ISINDIVIDUALASSESSMENT": 0})
        wl = int(c["risk"] > 1.3 or 30 < dpd <= 90)
        rows_sf.append({"EXTRACTDA": as_of, "CUSTOMERID": c["customer_id"],
                        "ISDEFAULT": "", "ISWATCHLIST": wl if wl else "",
                        "ISINSOLVENCY": "", "ISDEFAULTINGCC": "",
                        "ISLOCAL1": 1 if c["risk"] > 1.6 else ""})
        rows_ic.append({"CUSTOMERID": c["customer_id"],
                        "INDUSTRYCODE": c["industry"][0],
                        "INDUSTRYDESCRIPTION": c["industry"][1]})
    # investments
    inv_am, inv_cm, inv_sf, inv_or = [], [], [], []
    for j, (name, typ, rating, rate) in enumerate(INVESTMENTS):
        rng_i = np.random.default_rng(SEED + 900 + j)
        mat = pd.Timestamp(f"20{int(name[-2:])}-12-15") if name[-4:].isdigit() \
            else as_of + pd.DateOffset(months=int(rng_i.integers(3, 18)))
        amount = float(np.round(rng_i.uniform(20, 120), 0) * 1e6)
        inv_am.append({"EXTRACTDA": as_of, "CONTRACTID": name, "LIMID": "",
                       "CUSTOMERID": name, "PORTFOLIOCODE": "", "ACCOUNTTYPE": typ,
                       "C7": "", "C8": "", "OPENDATE": pd.Timestamp("2024-03-31"),
                       "RATING": rating, "PASTDUEDAYS": 0,
                       **{f"C{n}": "" for n in range(12, 18)},
                       "ONBALANCE": amount, "C19": "", "CCF": "",
                       "MATURITYDATE": mat, "C22": "", "EIR": rate,
                       **{f"C{n}": "" for n in range(24, 36)},
                       "PAYMENTFREQUENCY": 6, "CURRENCYCODE": "USD" if typ != "Banks and Fis" else "QAR",
                       "C38": "", "NOMINALINTERESTRATE": rate, "PAYMENTTYPEID": 3})
        inv_cm.append({"EXTRACTDA": as_of, "CUSTOMERID": name, "C3": "",
                       "CUSTOMERNAME": name, "C5": "", "RATING": rating})
        inv_sf.append({"EXTRACTDA": as_of, "CUSTOMERID": name, "ISDEFAULT": "",
                       "ISWATCHLIST": "", "C5": "", "C6": "", "ISLOCAL1": ""})
        inv_or.append({"EXTRACTDA": as_of, "CONTRACTID": name, "PD12M": "",
                       "RATING": rating})
    return {
        "AccountMaster.xlsx": pd.DataFrame(rows_am),
        "AccountMasterInvestments.xlsx": pd.DataFrame(inv_am),
        "CustomerMaster.xlsx": pd.DataFrame(rows_cm),
        "CustomerMasterInvestments.xlsx": pd.DataFrame(inv_cm),
        "CustomerStagingFlag.xlsx": pd.DataFrame(rows_sf),
        "CustomerStagingFlagInvestments.xlsx": pd.DataFrame(inv_sf),
        "Collateral.xlsx": pd.DataFrame(rows_col),
        "AccountCollateralAllocation.xlsx": pd.DataFrame(rows_alloc),
        "RepaymentSchedule.xlsx": pd.DataFrame(rows_rs),
        "Origination.xlsx": pd.DataFrame(rows_or),
        "OriginationInvestments.xlsx": pd.DataFrame(inv_or),
        "IndustryCode.xlsx": pd.DataFrame(rows_ic),
    }


def write_demo_inputs(out_dir, dates=DEMO_DATES) -> list[Path]:
    """Write one folder of the twelve extracts per date; return the folders."""
    out_dir = Path(out_dir)
    rng = np.random.default_rng(SEED)
    customers, contracts = _book(rng)
    base = pd.Timestamp(DEMO_DATES[0])
    folders = []
    for d in dates:
        as_of = pd.Timestamp(d)
        q = max(0, round((as_of - base).days / 91.3))
        folder = out_dir / as_of.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        for name, df in _frames(customers, contracts, as_of, q).items():
            # SQL*Plus headers for the filler columns are blank-ish letters;
            # the readers find every used column by name or position.
            df = df.rename(columns=lambda c: c if not (c.startswith("C") and c[1:].isdigit())
                           else f"X{c[1:]}")
            df.to_excel(folder / name, index=False)
        folders.append(folder)
    return folders
