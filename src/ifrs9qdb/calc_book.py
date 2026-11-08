"""The ECL calculator workbook: one customer's -- or one contract's --
provision, from the run's inputs to the figure in the report, in plain steps.

Visible tabs, in the order a reader follows the provision:

    Summary      the result per position, checked against the report
    Facilities   the inputs, one row per position
    Rating       how the customer's rating was set (one block per customer)
    Staging      the customer's stage and why (one block per customer)
    PD           each position's bucket, TTC PD, 12-month and lifetime PD
    LGD          collateral to LGD
    EAD          the exposure and the months counted
    ECL          month by month

Hidden (right-click a tab, Unhide): Assumptions -- every lookup table and
setting -- and PD calculation -- the forward-looking PD, condensed from the
Excel ETL's scenario sheets.

Every tab opens with its results; the working sits below. A contract's
workbook shows that contract only: what the customer's other facilities
contribute (its worst days past due, a Tasdeer facility, an Al Dhameen
facility) is given as a value, not as rows.

Written from :func:`ifrs9qdb.calc_workbook.customer_ecl_facts`.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

from .calc_workbook import (CALC, MAX_MONTH, MIN_MONTHS, RUN, S1_MONTHS,
                            SHAPE_LABEL, _Book, _a, _col_letter, _fallback_n,
                            _name, _num, _ov_text, _rng, _str, _trunc3)

FIRST_ROW = 23           # Summary: the per-position table starts here
YES, NO = "Yes", "No"
PROGRAM = ("Agriculture", "Fisheries", "Livestock")


def _yn(v) -> str:
    return YES if (_num(v) or 0) == 1 else NO


def _key_fill(B, ws, r, c):
    ws.cell(r, c).fill = B.fill_key


# ======================================================================
def write_book(facts: dict, dest=None):
    """Write the calculator for ``facts`` to ``dest`` (a path or a binary
    buffer); the bytes when ``dest`` is None."""
    B = _Book()
    focus = set(facts.get("focus") or [])
    L = [l for l in facts["lines"] if l["line"] in focus] if focus else facts["lines"]
    R: dict = {}

    ws = {n: B.sheet(n) for n in ("Summary", "Facilities", "Rating", "Staging", "PD",
                                  "LGD", "EAD", "ECL", "Assumptions", "PD calculation")}
    for n in ("Assumptions", "PD calculation"):
        ws[n].sheet_state = "hidden"

    S = _assumptions(B, ws["Assumptions"], facts, L)
    _facilities(B, ws["Facilities"], L, R)
    pm = _pd_calculation(B, ws["PD calculation"], facts, L, S)
    _rating(B, ws["Rating"], facts, L, R, S)
    _staging(B, ws["Staging"], facts, L, R, S)
    lg = _lgd(B, ws["LGD"], facts, L, R, S)
    ead = _ead(B, ws["EAD"], L, R, S)
    pdv = _pd(B, ws["PD"], facts, L, R, S, pm, ead)
    ecl = _ecl(B, ws["ECL"], L, R, S, lg, ead, pm, pdv)
    _summary(B, ws["Summary"], facts, L, R, S, lg, pdv, ecl)

    buf = io.BytesIO()
    B.wb.calculation.fullCalcOnLoad = True
    B.wb.save(buf)
    data = buf.getvalue()
    if dest is None:
        return data
    if hasattr(dest, "write"):
        dest.write(data)
        return None
    Path(dest).write_bytes(data)
    return None


# ================================================================ inputs
def _assumptions(B, ws, facts, L) -> dict:
    """Every setting and lookup table, named, one block under another."""
    st = facts["settings"]
    rt = facts.get("rating") or {}
    B.title(ws, "Assumptions", "The run's settings and lookup tables (config_used). "
            "Blue cells are inputs: change one and every formula follows.")
    S = {}
    r = 4
    ws.cell(r, 1, "Settings").font = B.f_sub
    B.header(ws, r + 1, 1, ["Setting", "Value", "Where it comes from"])
    rows = [
        ("thr", "Stage 2 from days past due over", st["threshold"], "int",
         "staging_thresholds.csv"),
        ("s3dpd", "Stage 3 from days past due over", 90, "int", "the staging rule"),
        ("lgd_base", "LGD base", st["lgd_base"], "x", "model.yml ecl.lgd_base"),
        ("lgd_floor", "Unsecured share at least", st["lgd_floor"], "x",
         "model.yml ecl.lgd_unsecured_floor"),
        ("zero_lgd", "LGD when there is no exposure", st["zero_lgd"], "x", ""),
        ("stage3", "Stage 3 ECL", st["stage3"], None,
         "full_outstanding: the on-balance exposure; zero: nil"),
        ("cap", "ECL at most the on-balance exposure", bool(st["cap"]), None,
         "model.yml ecl.cap_ecl_at_exposure"),
        ("s1m", "Months counted at Stage 1", S1_MONTHS, "int", "12-month ECL"),
        ("minm", "Months for a facility at or past maturity", MIN_MONTHS, "int", ""),
        ("maxm", "Last month of the PD curves", MAX_MONTH, "int", ""),
    ]
    r += 2
    for key, lab, val, kind, note in rows:
        B.put(ws, r, 1, lab)
        B.put(ws, r, 2, val, kind, inp=True)
        B.note(ws, r, 3, note)
        S[key] = _a(r, 2, "Assumptions")
        r += 1

    r += 1
    ws.cell(r, 1, "Repayment type when there is no repayment schedule").font = B.f_sub
    B.note(ws, r + 1, 1, "A rule for the position's portfolio and payment type, else "
           "one for its payment type, else the default. 3 months or less: bullet.")
    B.header(ws, r + 2, 1, ["Portfolio", "Payment type", "Repayment type", "Key"])
    r0 = r + 3
    for k, (pf, pt, sh) in enumerate(st["rules"]):
        B.put(ws, r0 + k, 1, pf or "", inp=True)
        B.put(ws, r0 + k, 2, pt, inp=True)
        B.put(ws, r0 + k, 3, SHAPE_LABEL.get(sh, SHAPE_LABEL["bullet"]), inp=True)
        B.put(ws, r0 + k, 4, f"={_a(r0 + k, 1, fixed=False)}&\"|\"&"
                             f"{_a(r0 + k, 2, fixed=False)}")
    r1 = r0 + max(len(st["rules"]), 1) - 1
    S["rule_key"] = _rng(r0, 4, r1, 4, "Assumptions")
    S["rule_shape"] = _rng(r0, 3, r1, 3, "Assumptions")
    B.put(ws, r1 + 1, 1, "Default")
    B.put(ws, r1 + 1, 3, SHAPE_LABEL.get(st["default_shape"], SHAPE_LABEL["bullet"]),
          inp=True)
    S["default_shape"] = _a(r1 + 1, 3, "Assumptions")
    r = r1 + 3

    # ---- rating scale
    scale = rt.get("scale") or []
    ws.cell(r, 1, "Rating scale").font = B.f_sub
    B.header(ws, r + 1, 1, ["Rating", "Scale", "Position (1 = best)"])
    for k, (g, t, h) in enumerate(scale):
        B.put(ws, r + 2 + k, 1, g, inp=True)
        B.put(ws, r + 2 + k, 2, t, inp=True)
        B.put(ws, r + 2 + k, 3, h, "int", inp=True)
    last = r + 1 + max(len(scale), 1)
    _name(B, "ScaleRating", _rng(r + 2, 1, last, 1, "Assumptions"))
    _name(B, "ScaleType", _rng(r + 2, 2, last, 2, "Assumptions"))
    _name(B, "ScaleHierarchy", _rng(r + 2, 3, last, 3, "Assumptions"))
    n_int = sum(1 for x in scale if x[1] == "Internal")
    _name(B, "InternalGrades", _rng(r + 2, 1, r + 1 + max(n_int, 1), 1, "Assumptions"))
    r = last + 2

    # ---- PD bucket of each rating, and the TTC PD of each bucket
    rm = facts["ratings_map"]
    ws.cell(r, 1, "PD bucket of each rating (Ratings.csv)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Scale", "Rating", "Bucket", "Key"])
    for k, m in enumerate(rm):
        rr = r + 2 + k
        B.put(ws, rr, 1, "External" if str(m["rt"]) == "2" else "Internal", inp=True)
        B.put(ws, rr, 2, m["rating"], inp=True)
        B.put(ws, rr, 3, m["bucket"], "int", inp=True)
        B.put(ws, rr, 4, f"={_a(rr, 1, fixed=False)}&\"|\"&{_a(rr, 2, fixed=False)}")
    last = r + 1 + max(len(rm), 1)
    _name(B, "BucketNumber", _rng(r + 2, 3, last, 3, "Assumptions"))
    _name(B, "BucketKey", _rng(r + 2, 4, last, 4, "Assumptions"))
    r = last + 2
    tt = facts.get("ttc_table") or []
    ws.cell(r, 1, "TTC PD of each bucket (ttc_pd_table.csv)").font = B.f_sub
    B.header(ws, r + 1, 1, ["Scale", "Bucket", "Rating", "TTC PD", "Key"])
    for k, (scl, b, g, p) in enumerate(tt):
        rr = r + 2 + k
        B.put(ws, rr, 1, scl, inp=True)
        B.put(ws, rr, 2, b, "int", inp=True)
        B.put(ws, rr, 3, g, inp=True)
        B.put(ws, rr, 4, p, "pd", inp=True)
        B.put(ws, rr, 5, f"={_a(rr, 1, fixed=False)}&\"|\"&{_a(rr, 2, fixed=False)}")
    last = r + 1 + max(len(tt), 1)
    _name(B, "TtcPd", _rng(r + 2, 4, last, 4, "Assumptions"))
    _name(B, "TtcKey", _rng(r + 2, 5, last, 5, "Assumptions"))
    r = last + 2

    # ---- program lending and default ratings
    rules, fb = rt.get("rules") or [], rt.get("fallback") or []
    ws.cell(r, 1, "Program lending ratings, by the customer's worst days past due"
            ).font = B.f_sub
    B.header(ws, r + 1, 1, ["Sector", "From days past due", "Rating"])
    S["ladders"] = {}
    rr = r + 2
    for sector in dict.fromkeys(x[0] for x in rules):
        first = rr
        for sec, d, g in [x for x in rules if x[0] == sector]:
            B.put(ws, rr, 1, sec, inp=True)
            B.put(ws, rr, 2, d, "int", inp=True)
            B.put(ws, rr, 3, g, inp=True)
            rr += 1
        short = re.sub(r"\s*Sector\s*$", "", sector, flags=re.I)
        nm = "Program_" + re.sub(r"[^A-Za-z0-9]", "", short)
        S["ladders"][sector] = (_name(B, nm, _rng(first, 2, rr - 1, 3, "Assumptions")),
                                sorted({short.lower(),
                                        re.sub("Lifestock", "Livestock", short,
                                               flags=re.I).lower()}))
    r = max(rr, r + 3) + 1
    ws.cell(r, 1, "Default ratings for an unrated customer").font = B.f_sub
    B.header(ws, r + 1, 1, ["Segment", "Rating"])
    for k, (seg, g) in enumerate(fb):
        B.put(ws, r + 2 + k, 1, seg, inp=True)
        B.put(ws, r + 2 + k, 2, g, inp=True)
    _name(B, "DefaultRatings", _rng(r + 2, 1, r + 1 + max(len(fb), 1), 2, "Assumptions"))
    B.widths(ws, {"A": 46, "B": 22, "C": 24, "D": 16, "E": 22})
    return S


FAC = [  # (key, shown, kind)
    ("Contract", "Contract", None),
    ("Original contract", "Contract (source id)", None),
    ("Account type", "Product", None),
    ("Portfolio", "Portfolio", None),
    ("Rating type", "Rating scale", None),
    ("Rating", "Rating in the run", None),
    ("Extract date", "Reporting date", "date"),
    ("Maturity date", "Maturity date", "date"),
    ("On balance", "On balance (QAR)", "amt"),
    ("Off balance", "Off balance (QAR)", "amt"),
    ("EIR", "Effective interest rate", "rate"),
    ("Past due days", "Days past due (customer's worst)", "int"),
    ("Facility DPD", "Days past due (this facility)", "int"),
    ("Payment type", "Payment type", None),
    ("Payment frequency (months)", "Months between payments", "int"),
    ("Deferral (months)", "Grace months", "int"),
    ("Nominal interest rate", "Nominal interest rate", "rate"),
    ("Repayment schedule", "Repayment schedule", None),
]


def _facilities(B, ws, L, R) -> None:
    B.title(ws, "Facilities: the inputs", "One row per position, as the run read it "
            "(the LIC account master, and the source id the ETL kept). Every value "
            "is an input.")
    B.header(ws, 4, 1, [s for _, s, _ in FAC])
    for k, l in enumerate(L):
        rr = 5 + k
        vals = {"Contract": l["cid"], "Original contract": l.get("contract_original") or "",
                "Account type": l["account_type"], "Portfolio": l["portfolio"],
                "Rating type": l["rating_type"], "Rating": l["rating"],
                "Extract date": l["d_extract"], "Maturity date": l["d_maturity"],
                "On balance": l["on_balance"], "Off balance": l["off_balance"],
                "EIR": l["eir"], "Past due days": l["dpd"],
                "Facility DPD": l.get("dpd_own"), "Payment type": l["payment_type"],
                "Payment frequency (months)": l["pay_freq"],
                "Deferral (months)": l["deferral"], "Nominal interest rate": l["nir"],
                "Repayment schedule": YES if l["schedule"] is not None else NO}
        for c, (key, _, kind) in enumerate(FAC, 1):
            B.put(ws, rr, c, vals[key], kind, inp=True)
            R[(l["line"], key)] = _a(rr, c, "Facilities")
    B.widths(ws, {_col_letter(c): w for c, w in enumerate(
        [14, 20, 14, 16, 12, 11, 12, 12, 15, 14, 11, 13, 13, 10, 11, 9, 11, 11], 1)})


# ================================================================ rating
def _rating(B, ws, facts, L, R, S) -> None:
    """The customer's rating in steps; the investment book position by
    position (each issue carries its own external rating)."""
    rt = facts.get("rating") or {}
    recorded = bool(rt.get("recorded"))
    books = facts.get("books") or {}
    ov = facts["overrides"]
    rov = ov[ov["kind"] == "rating"] if len(ov) else ov
    o = rov.iloc[-1] if len(rov) else None
    lend = [l for l in L if l["suffix"] == "_1"]
    inv = [l for l in L if l["suffix"] == "_2"]
    B.title(ws, "Rating: how the customer's rating was set",
            "Four ways, in this order: an override in the run; else the rating in the "
            "customer master; else, for program lending (Agriculture, Fisheries, "
            "Livestock), the program rating for the customer's worst days past due; "
            "else the default rating for an unrated customer.")
    r = 4
    if lend:
        b = books.get("_1", {})
        ws.cell(r, 1, "Result").font = B.f_sub
        top = r + 1                                # the result rows, filled below
        r = top + 4
        ws.cell(r, 1, "Steps").font = B.f_sub
        B.header(ws, r + 1, 1, ["Step", "Value", "Note"])
        r += 2
        cell = {}

        def row(key, label, value, kind=None, inp=False, note="", bold=False):
            nonlocal r
            B.put(ws, r, 1, label, bold=bold)
            B.put(ws, r, 2, value, kind, inp=inp, key=bold)
            if note:
                B.note(ws, r, 3, note)
            cell[key] = _a(r, 2)
            r += 1

        if recorded:
            lad = S.get("ladders") or {}
            row("cm", "1. Rating in the customer master", b.get("rating_input") or "",
                inp=True, note="Blank or not on the rating scale: unrated")
            row("on", "   On the rating scale?",
                f"=IF(AND({cell['cm']}<>\"\",COUNTIF(ScaleRating,{cell['cm']})>0),"
                f"\"{YES}\",\"{NO}\")")
            row("sec", "2. Sector", b.get("sector") or "", inp=True)
            names = [n for _, ns in lad.values() for n in ns]
            sec = f"LOWER(TRIM({cell['sec']}))"
            row("prog", "   Program lending?",
                ("=IF(OR(" + ",".join(f"{sec}=\"{n}\"" for n in names)
                 + f"),\"{YES}\",\"{NO}\")") if names else NO,
                note="Agriculture, Fisheries or Livestock")
            row("dpd", "   Customer's worst days past due", b.get("dpd_worst") or 0, "int",
                inp=True, note="Across all the customer's facilities")
            pr = "\"\""
            for nm, (rng, ns) in reversed(list(lad.items())):
                cond = "OR(" + ",".join(f"{sec}=\"{n}\"" for n in ns) + ")"
                pr = (f"IF({cond},IFERROR(VLOOKUP({cell['dpd']},{rng},2,TRUE),"
                      f"INDEX({rng},1,2)),{pr})")
            row("prr", "   Program rating", f"=IF({cell['prog']}=\"{YES}\",{pr},\"\")",
                note="Program lending ratings (Assumptions)")
            row("nad", "3. Al Dhameen facilities of the customer", b.get("n_aldhameen", 0),
                "int", inp=True)
            row("noth", "   Facilities that are not Al Dhameen", b.get("n_other", 0), "int",
                inp=True)
            h = lambda g: f"IFERROR(INDEX(ScaleHierarchy,MATCH({g},ScaleRating,0)),0)"
            ad = "VLOOKUP(\"Al Dhameen Customers\",DefaultRatings,2,FALSE)"
            un = "VLOOKUP(\"Unrated Customer (Internal Rating)\",DefaultRatings,2,FALSE)"
            row("dft", "   Default rating when unrated",
                f"=IFERROR(INDEX(InternalGrades,MAX(IF({cell['nad']}>0,{h(ad)},0),"
                f"IF({cell['noth']}>0,{h(un)},0))),\"\")",
                note="Al Dhameen facilities take the Al Dhameen rating, others the "
                     "unrated rating; the worse applies")
            row("pick", "4. Rating from the inputs",
                f"=IF({cell['on']}=\"{YES}\",{cell['cm']},IF({cell['prog']}=\"{YES}\","
                f"{cell['prr']},{cell['dft']}))")
            row("int", "   On the internal scale",
                f"=IFERROR(INDEX(InternalGrades,INDEX(ScaleHierarchy,MATCH({cell['pick']},"
                f"ScaleRating,0))),{cell['pick']})",
                note="An external grade becomes the internal grade at its position")
            row("ovr", "5. Override in the run", "" if o is None else _str(o.value),
                inp=True, note="" if o is None else _ov_text(o))
            row("fin", "Customer rating",
                f"=IF({cell['ovr']}<>\"\",{cell['ovr']},{cell['int']})", bold=True)
            row("how", "How the rating was set",
                f"=IF({cell['ovr']}<>\"\",\"Override in the run\",IF({cell['on']}="
                f"\"{YES}\",\"Rating in the customer master\",IF({cell['prog']}="
                f"\"{YES}\",\"Program lending (\"&{cell['sec']}&\")\","
                f"\"Unrated: default rating\")))")
        else:
            row("ovr", "Override in the run", "" if o is None else _str(o.value),
                inp=True, note="" if o is None else _ov_text(o))
            row("fin", "Customer rating", f"={R[(lend[0]['line'], 'Rating')]}", bold=True)
            row("how", "How the rating was set",
                f"=IF({cell['ovr']}<>\"\",\"Override in the run\",\"As in the run\")",
                note="This run did not keep how the rating was set (made before "
                     "engine 0.45): re-run to see the steps.")
        row("run", "The run's rating (account master)", f"={R[(lend[0]['line'], 'Rating')]}")
        row("chk", "Same as the run?",
            f"=IF({cell['fin']}={cell['run']},\"OK\",\"DIFFERS\")")
        # the result on top
        for k, (lab, key) in enumerate((("Customer rating", "fin"),
                                        ("How the rating was set", "how"),
                                        ("Same as the run?", "chk"))):
            B.put(ws, top + k, 1, lab, bold=True)
            B.put(ws, top + k, 2, f"={cell[key]}", key=True)
        for l in lend:
            R[(l["line"], "Final rating")] = _a(top, 2, "Rating")
            R[(l["line"], "Rating set by")] = _a(top + 1, 2, "Rating")
        r += 1
    if inv:
        ws.cell(r, 1, "Investments: each position's external rating").font = B.f_sub
        cols = ["Contract", "Rating in the extract", "On the external scale?",
                "Default rating", "Rating", "How the rating was set",
                "The run's rating", "Same?"]
        B.header(ws, r + 1, 1, cols)
        for k, l in enumerate(inv):
            rr = r + 2 + k
            x = l["line"]
            g = lambda c: _a(rr, c, fixed=False)
            B.put(ws, rr, 1, f"={R[(x, 'Contract')]}")
            B.put(ws, rr, 2, l.get("rating_input") or "", inp=True)
            B.put(ws, rr, 3, f"=IF(COUNTIFS(ScaleRating,{g(2)},ScaleType,\"External\")>0,"
                             f"\"{YES}\",\"{NO}\")")
            B.put(ws, rr, 4, "=IFERROR(VLOOKUP(\"Investment Portfolio\",DefaultRatings,2,"
                             "FALSE),\"\")")
            if recorded:
                B.put(ws, rr, 5, f"=IF({g(3)}=\"{YES}\",{g(2)},{g(4)})", key=True)
                B.put(ws, rr, 6, f"=IF({g(3)}=\"{YES}\",\"Rating in the extract\","
                                 f"\"Unrated: default rating\")")
            else:
                B.put(ws, rr, 5, f"={R[(x, 'Rating')]}", key=True)
                B.put(ws, rr, 6, "As in the run")
            B.put(ws, rr, 7, f"={R[(x, 'Rating')]}")
            B.put(ws, rr, 8, f"=IF({g(5)}={g(7)},\"OK\",\"DIFFERS\")")
            R[(x, "Final rating")] = _a(rr, 5, "Rating")
            R[(x, "Rating set by")] = _a(rr, 6, "Rating")
    B.widths(ws, {"A": 42, "B": 30, "C": 62, "D": 16, "E": 12, "F": 26, "G": 16, "H": 9})


# =============================================================== staging
def _staging(B, ws, facts, L, R, S) -> None:
    """The customer's stage from its criteria. Every facility of a customer
    takes the customer's stage: days past due and the flags are the
    customer's, and a Tasdeer facility (assessed collectively at Stage 2)
    lifts the others."""
    books = facts.get("books") or {}
    ov = facts["overrides"]
    sov = ov[ov["kind"] == "stage"] if len(ov) else ov
    rov = ov[ov["kind"] == "restructuring"] if len(ov) else ov
    B.title(ws, "Staging: the customer's stage and why",
            "Staging is done per customer. Stage 3: more than 90 days past due or the "
            "default flag. Stage 2: more than the threshold days past due, the "
            "watchlist, restructured, a Stage 2 flag in the LIC staging file, or a "
            "Tasdeer facility. Otherwise Stage 1. Every facility takes the customer's "
            "stage.")
    r = 4
    for suffix, label in (("_1", "Lending"), ("_2", "Investments")):
        ls = [l for l in L if l["suffix"] == suffix]
        if not ls:
            continue
        b = books.get(suffix, {})
        ws.cell(r, 1, f"{label}: result").font = B.f_sub
        top = r + 1
        r = top + 4
        ws.cell(r, 1, "Criteria").font = B.f_sub
        B.header(ws, r + 1, 1, ["Criterion", "Value", "Stage it gives", "Note"])
        r += 2
        rows = []                     # (label, value, kind, stage formula, note, reason)
        thr, s3 = S["thr"], S["s3dpd"]
        rows.append(("Days past due (customer's worst)", b.get("dpd_worst") or 0, "int",
                     lambda v: f"IF({v}>{s3},3,IF({v}>{thr},2,1))",
                     "Investments carry 0: they are staged on their flags" if suffix == "_2"
                     else "Across all the customer's facilities", "dpd"))
        rows.append(("Default flag (LIC staging file)", _yn(b.get("IsDefault")), None,
                     lambda v: f"IF({v}=\"{YES}\",3,1)",
                     "A Stage 3 override in the run sets this flag", "Default flag"))
        rows.append(("On the watchlist", _yn(b.get("IsWatchlist")), None,
                     lambda v: f"IF({v}=\"{YES}\",2,1)", "", "On the watchlist"))
        rn = "" if not len(rov) else "Override in the run: " + _ov_text(rov.iloc[-1])
        rows.append(("Restructured", _yn(b.get("IsLocal1")), None,
                     lambda v: f"IF({v}=\"{YES}\",2,1)", rn, "Restructured"))
        rows.append(("Stage 2 in the LIC staging file", _yn(b.get("IsLocal3")), None,
                     lambda v: f"IF({v}=\"{YES}\",2,1)",
                     "A Stage 2 override in the run sets this flag",
                     "Stage 2 in the LIC staging file"))
        other = max((_num(b.get(f"IsLocal{i}")) or 0) for i in (2, 4, 5, 6))
        rows.append(("Another Stage 2 flag in the LIC staging file", _yn(other), None,
                     lambda v: f"IF({v}=\"{YES}\",2,1)", "",
                     "Stage 2 flag in the LIC staging file"))
        rows.append(("Tasdeer facilities of the customer", b.get("n_tasdeer", 0), "int",
                     lambda v: f"IF({v}>0,2,1)",
                     "Tasdeer is assessed collectively at Stage 2", "Has a Tasdeer facility"))
        if b.get("manual_max"):
            rows.append(("Stage set in the account master", b["manual_max"], "int",
                         lambda v: f"IF(ISNUMBER({v}),{v},1)", "",
                         "Stage set in the account master"))
        crit = []
        for lab, v, kind, f, note, why in rows:
            B.put(ws, r, 1, lab)
            B.put(ws, r, 2, v, kind, inp=True)
            B.put(ws, r, 3, "=" + f(_a(r, 2)), "int")
            B.note(ws, r, 4, note)
            crit.append((_a(r, 3), why, _a(r, 2)))
            r += 1
        if len(sov):
            B.put(ws, r, 1, "Stage override in the run")
            B.put(ws, r, 2, _str(sov.iloc[-1].value), inp=True)
            B.note(ws, r, 4, "Applied through the flags above. " + _ov_text(sov.iloc[-1]))
            r += 1
        stage_rng = f"{crit[0][0]}:{crit[-1][0]}"
        B.put(ws, r, 1, "Customer stage", bold=True)
        B.put(ws, r, 3, f"=MAX({stage_rng})", "int", key=True)
        stage = _a(r, 3)
        # why: the first criterion that gives the customer's stage
        why = "\"Performing\""
        for c, w, v in reversed(crit):
            if w == "dpd":
                txt = (f"IF({v}>{s3},\"More than \"&{s3}&\" days past due\","
                       f"\"More than \"&{thr}&\" days past due\")")
            else:
                txt = f"\"{w}\""
            why = f"IF(AND({stage}>1,{c}={stage}),{txt},{why})"
        B.put(ws, r + 1, 1, "Why", bold=True)
        B.put(ws, r + 1, 2, "=" + why)
        whyc = _a(r + 1, 2)
        r += 3
        # the positions and the report
        ws.cell(r, 1, "Positions").font = B.f_sub
        B.header(ws, r + 1, 1, ["Contract", "Stage", "The report's stage", "Same?"])
        for k, l in enumerate(ls):
            rr = r + 2 + k
            x = l["line"]
            B.put(ws, rr, 1, f"={R[(x, 'Contract')]}")
            B.put(ws, rr, 2, f"={stage}", "int")
            B.put(ws, rr, 3, l["rep_stage"], "int", inp=True)
            B.put(ws, rr, 4, f"=IF({_a(rr, 3, fixed=False)}=\"\",\"not in report\","
                             f"IF({_a(rr, 2, fixed=False)}={_a(rr, 3, fixed=False)},"
                             f"\"OK\",\"DIFFERS\"))")
            R[(x, "Stage")] = _a(rr, 2, "Staging")
            R[(x, "Stage reason")] = _a(top + 1, 2, "Staging")
        chk = (f"IF(COUNTIF({_rng(r + 2, 4, r + 1 + len(ls), 4)},\"DIFFERS\")>0,"
               f"\"DIFFERS\",\"OK\")")
        for k, (lab, f) in enumerate((("Customer stage", f"={stage}"),
                                      ("Why", f"={whyc}"),
                                      ("Same as the report?", "=" + chk))):
            B.put(ws, top + k, 1, lab, bold=True)
            B.put(ws, top + k, 2, f, "int" if k == 0 else None, key=True)
        r += 3 + len(ls)
    B.widths(ws, {"A": 46, "B": 34, "C": 16, "D": 70})


# ================================================================ PD
def _pd_calculation(B, ws, facts, L, S) -> dict:
    """The forward-looking PD, condensed from the ETL's scenario sheets:
    scaling factor by scenario and year; for each bucket in use the PiT PD by
    year and scenario, weighted into one yearly PD; the monthly cumulative
    curve; beside it the run's own curve and the difference."""
    pdm = facts.get("pd_model") or {}
    ok = bool(pdm.get("ok"))
    used = {(l.get("rt"), l.get("bucket")) for l in L if l.get("bucket") is not None}
    B.title(ws, "PD calculation (the Excel ETL's scenario sheets, condensed)",
            "1) Each scenario moves the forecast by its severity; the model turns it "
            "into a default rate and a scaling factor. 2) A bucket's PiT PD in a "
            "forecast year is its TTC PD moved by the scaling factor; later years "
            "drift back to the TTC PD by the last year. 3) The scenarios are weighted. "
            "4) The yearly PD is spread evenly over its 12 months and summed into the "
            "cumulative curve.")
    B.put(ws, 4, 1, "PD curves used in the ECL", bold=True)
    sw = B.put(ws, 4, 2, pdm.get("source") or RUN, inp=True, key=True)
    from openpyxl.worksheet.datavalidation import DataValidation
    dv = DataValidation(type="list", formula1=f'"{CALC},{RUN}"', allow_blank=False)
    ws.add_data_validation(dv)
    dv.add(sw)
    B.note(ws, 4, 3, pdm.get("reason") or "The calculated curves equal the run's.")
    src = _a(4, 2, "PD calculation")
    r = 6
    wm = {}                                       # (rt, bucket key) -> weighted range
    names = [n for n, _ in pdm.get("scenarios") or []]
    if ok:
        for rt in (1, 2):
            buckets = [bk for bk in pdm["buckets"].get(rt, [])
                       if (str(rt), bk["bucket"]) in used and bk["ttc"] is not None]
            if not buckets or (rt == 2 and not pdm.get("ext_ok")):
                continue
            if rt == 1:
                sf, weight_of, r = _sf_internal(B, ws, r, pdm)
            else:
                sf, weight_of, r = _sf_external(B, ws, r, pdm)
            for bk in buckets:
                wm[(rt, bk["key"])], r = _bucket_block(B, ws, r, pdm, rt, bk, sf,
                                                       weight_of, names)
    # ---- monthly curves, the run's beside them, and the check
    calc = [((rt, k), f"{'Internal' if rt == 1 else 'External'} | bucket {k}")
            for (rt, k) in wm]
    keys_l = {(l["portfolio"], l.get("bucket")) for l in L}
    runs = [c for c in facts["curves"] if (c["portfolio"], c["bucket"]) in keys_l]
    where = {ck: (2 + k, lab) for k, (ck, lab) in enumerate(calc)}
    pairs = []                                    # (run curve, its column, calc col, label)
    for k, c in enumerate(runs):
        bk = str(int(c["bucket"])) if float(c["bucket"]).is_integer() else str(c["bucket"])
        hit = where.get((c["rt"], bk))
        if hit is not None and c["stpd"] is not None:
            pairs.append((c, 2 + len(calc) + k, hit[0], hit[1]))
    ws.cell(r, 1, "Check: the run's curves against the calculated ones").font = B.f_sub
    B.header(ws, r + 1, 1, ["The run's curve (StPD.csv)", "Calculated curve",
                            "Largest difference"])
    cmp_row = r + 2
    if not pairs:
        B.note(ws, cmp_row, 1, "No calculated curve to compare with: the ECL uses the "
               "run's curves.")
    r = cmp_row + max(len(pairs), 1) + 1
    ws.cell(r, 1, "Cumulative PD by month").font = B.f_sub
    hr = r + 2
    first = hr + 1
    n_c = len(calc) + len(runs)
    B.header(ws, hr, 1, ["Month"] + [c[1] for c in calc] + [c["label"] for c in runs]
             + [f"Difference: {p[0]['label']}" for p in pairs])
    if calc:
        B.put(ws, hr - 1, 2, "Calculated", bold=True)
    if runs:
        B.put(ws, hr - 1, 2 + len(calc), "The run's (StPD.csv)", bold=True)
    mm = pdm.get("max_mat", 50)
    for m in range(0, MAX_MONTH + 1):
        rr = first + m
        B.put(ws, rr, 1, m, "int")
        for k, (ck, _) in enumerate(calc):
            col = 2 + k
            if m == 0:
                B.put(ws, rr, col, 0, "pd")
                continue
            B.put(ws, rr, col, f"=MIN({_a(rr - 1, col, fixed=False)}+INDEX({wm[ck]},"
                               f"MIN(ROUNDUP({_a(rr, 1, fixed=False)}/12,0),{mm}))/12,1)",
                  "pd")
        for k, c in enumerate(runs):
            sp = c["stpd"]
            v = float(sp[m]) if sp is not None and m < len(sp) else None
            B.put(ws, rr, 2 + len(calc) + k, v, "pd", inp=True)
        for k, (_, rc, cc, _) in enumerate(pairs):
            B.put(ws, rr, 2 + n_c + k,
                  f"=ABS({_a(rr, rc, fixed=False)}-{_a(rr, cc, fixed=False)})", "pd")
    for k, (c, _, _, lab) in enumerate(pairs):
        dc = 2 + n_c + k
        B.put(ws, cmp_row + k, 1, c["label"])
        B.put(ws, cmp_row + k, 2, lab)
        B.put(ws, cmp_row + k, 3, f"=MAX({_rng(first, dc, first + MAX_MONTH, dc)})", "pd")
    keys = _rng(hr, 2, hr, 1 + max(n_c, 1), "PD calculation")
    curves = _rng(first, 2, first + MAX_MONTH, 1 + max(n_c, 1), "PD calculation")
    dc = 2 + n_c + len(pairs)
    B.widths(ws, {"A": 34, **{_col_letter(c): 18 for c in range(2, max(dc, 8))}})
    return {"source": src, "keys": keys, "used": curves, "ok": ok}


def _sf_internal(B, ws, r, pdm):
    """The internal model: variables, forecasts, scenarios and weights, the
    stressed forecast and the scaling factor by scenario and year."""
    mev, sc = pdm["mevs"], pdm["scenarios"]
    years = list(range(1, pdm["n_fc"] + 1))
    ws.cell(r, 1, f"Internal scale: model {pdm['model_id']}").font = B.f_sub
    B.header(ws, r + 1, 1, ["Variable", "Intercept", "Coefficient",
                            "Standard deviation", "Unit multiplier", "Weight"])
    M = []
    for k, m in enumerate(mev):
        rr = r + 2 + k
        B.put(ws, rr, 1, m["name"], inp=True)
        for c, key in enumerate(("intercept", "coefficient", "sd", "unit", "weight"), 2):
            B.put(ws, rr, c, m[key], "num", inp=True)
        M.append({key: _a(rr, c) for c, key in
                  enumerate(("intercept", "coefficient", "sd", "unit", "weight"), 2)})
    r += 2 + len(mev)
    B.put(ws, r, 1, "Long-run default rate (TTC anchor)")
    B.put(ws, r, 2, pdm["anchor"], "num", inp=True)
    anchor = _a(r, 2)
    r += 2
    B.header(ws, r, 1, ["Forecast year"] + [m["name"] for m in mev])
    F = {}
    for i, (y, row) in enumerate(zip(pdm["years"], pdm["forecasts"])):
        B.put(ws, r + 1 + i, 1, y, "int", inp=True)
        for k, v in enumerate(row):
            B.put(ws, r + 1 + i, 2 + k, v, "num", inp=True)
            F[(i + 1, k)] = _a(r + 1 + i, 2 + k)
    r += 2 + len(pdm["years"])
    B.header(ws, r, 1, ["Scenario", "Severity z", "Weight"])
    Z, W = [], []
    for k, ((nm, z), w) in enumerate(zip(sc, pdm["w_int"])):
        B.put(ws, r + 1 + k, 1, nm, inp=True)
        B.put(ws, r + 1 + k, 2, z, "num", inp=True)
        B.put(ws, r + 1 + k, 3, w, "num", inp=True)
        Z.append(_a(r + 1 + k, 2))
        W.append(_a(r + 1 + k, 3))
    B.note(ws, r + 1 + len(sc), 1, "Weights as the run used them (model_inputs.yml).")
    r += 3 + len(sc)
    # stressed forecast, one grid per variable, and the scaling factor
    ST = {}
    for j, m in enumerate(mev):
        B.header(ws, r, 1, [f"Stressed {m['name']}"] + [f"Year {y}" for y in years])
        for k, (nm, _) in enumerate(sc):
            B.put(ws, r + 1 + k, 1, nm)
            for y in years:
                B.put(ws, r + 1 + k, 1 + y,
                      f"={F[(y, j)]}+{Z[k]}*{M[j]['sd']}*{M[j]['unit']}", "num")
                ST[(j, k, y)] = _a(r + 1 + k, 1 + y)
        r += 2 + len(sc)
    B.header(ws, r, 1, ["Scaling factor"] + [f"Year {y}" for y in years])
    SF = {}
    for k, (nm, _) in enumerate(sc):
        B.put(ws, r + 1 + k, 1, nm)
        for y in years:
            parts = []
            for j, m in enumerate(M):
                x = f"({m['intercept']}+{m['coefficient']}*{ST[(j, k, y)]}/{m['unit']})"
                parts.append(f"{m['weight']}*(NORMSINV(EXP({x})/(EXP({x})+1))"
                             f"-NORMSINV({anchor}))")
            B.put(ws, r + 1 + k, 1 + y, "=" + "+".join(parts), "num", key=True)
            SF[(k, y)] = _a(r + 1 + k, 1 + y)
    B.note(ws, r + 1 + len(sc), 1, "Default rate = EXP(x)/(EXP(x)+1) with x = intercept "
           "+ coefficient x stressed / unit; scaling factor = NORMSINV(default rate) - "
           "NORMSINV(long-run rate), the variables combined by weight.")
    r += 3 + len(sc)
    return (lambda k, y: SF[(k, y)], lambda k, y: W[k], r)


def _sf_external(B, ws, r, pdm):
    """The external scale: GDP-weighted GCC growth; the scaling factor is
    NORMSINV of where the stressed forecast sits in the history."""
    sc = pdm["scenarios"]
    h, fc = pdm["gcc_hist"], pdm["ext_fc"]
    years = list(range(1, len(fc) + 1))
    ws.cell(r, 1, "External scale: GCC growth (GDP-weighted)").font = B.f_sub
    B.header(ws, r + 1, 1, ["History year", "Growth"])
    for k, (y, v) in enumerate(zip(pdm["gcc_years"], h)):
        B.put(ws, r + 2 + k, 1, y, "int", inp=True)
        B.put(ws, r + 2 + k, 2, v, "num", inp=True)
    H = _rng(r + 2, 2, r + 1 + len(h), 2)
    St = {}
    for k, (lab, f) in enumerate((("Years", f"=COUNT({H})"),
                                  ("Standard deviation", f"=STDEV({H})"),
                                  ("Lowest", f"=MIN({H})"), ("Highest", f"=MAX({H})"))):
        B.put(ws, r + 1 + k, 4, lab, bold=True)
        B.put(ws, r + 1 + k, 5, f, "num")
        St[lab] = _a(r + 1 + k, 5)
    N, SD = St["Years"], St["Standard deviation"]
    rr = r + 7
    B.header(ws, rr, 4, ["Forecast year", "Growth"] + [n for n, _ in sc])
    Fc, WR = [], []
    for i, v in enumerate(fc):
        B.put(ws, rr + 1 + i, 4, i + 1, "int")
        B.put(ws, rr + 1 + i, 5, v, "num", inp=True)
        Fc.append(_a(rr + 1 + i, 5))
        row = pdm["w_ext"][i] if i < len(pdm["w_ext"]) else pdm["w_ext"][-1]
        for k, w in enumerate(row):
            B.put(ws, rr + 1 + i, 6 + k, w, "num", inp=True)
        WR.append(rr + 1 + i)
    ra = rr + 1 + len(fc)
    B.put(ws, ra, 4, "Later years")
    for k in range(len(sc)):
        B.put(ws, ra, 6 + k, f"=AVERAGE({_rng(WR[0], 6 + k, WR[-1], 6 + k)})", "num")
    B.note(ws, ra + 1, 4, "Scenario weights by year, as the run used them; later years "
           "take the average.")
    Z = []
    B.header(ws, ra + 3, 4, ["Scenario", "Severity z"])
    for k, (nm, z) in enumerate(sc):
        B.put(ws, ra + 4 + k, 4, nm)
        B.put(ws, ra + 4 + k, 5, z, "num", inp=True)
        Z.append(_a(ra + 4 + k, 5))
    r = max(r + 2 + len(h), ra + 5 + len(sc)) + 1
    B.header(ws, r, 1, ["Stressed growth"] + [f"Year {y}" for y in years])
    X = {}
    for k, (nm, _) in enumerate(sc):
        B.put(ws, r + 1 + k, 1, nm)
        for y in years:
            B.put(ws, r + 1 + k, 1 + y, f"={Fc[y - 1]}+{Z[k]}*{SD}", "num")
            X[(k, y)] = _a(r + 1 + k, 1 + y)
    r += 2 + len(sc)
    B.header(ws, r, 1, ["Scaling factor"] + [f"Year {y}" for y in years])
    SF = {}
    for k, (nm, _) in enumerate(sc):
        B.put(ws, r + 1 + k, 1, nm)
        for y in years:
            x = X[(k, y)]
            j = f"COUNTIF({H},\"<=\"&{x})"
            inner = (f"(({j})+({x}-SMALL({H},{j}))/(SMALL({H},{j}+1)-SMALL({H},{j})))"
                     f"/({N}+1)")
            rank = (f"IF({x}<{St['Lowest']},{_trunc3(f'(1/({N}+1))')},IF({j}>={N},"
                    f"{_trunc3(f'({N}/({N}+1))')},{_trunc3(inner)}))")
            B.put(ws, r + 1 + k, 1 + y, f"=NORMSINV({rank})", "num", key=True)
            SF[(k, y)] = _a(r + 1 + k, 1 + y)
    B.note(ws, r + 1 + len(sc), 1, "Where the stressed growth sits in the history "
           "(percentile rank, truncated to 3 digits; 1/(n+1) below the lowest year, "
           "n/(n+1) above the highest), through NORMSINV.")
    r += 3 + len(sc)

    def weight_of(k, y):
        return _a(WR[y - 1] if y <= len(WR) else ra, 6 + k)
    return (lambda k, y: SF[(k, y)], weight_of, r)


def _bucket_block(B, ws, r, pdm, rt, bk, sf, weight_of, names):
    """One bucket: PiT PD and survival by scenario and year, and the
    weighted yearly PD = sum of weight x survival to the year before x PiT
    PD (the scenario's yearly default probability)."""
    mm = pdm["max_mat"]
    n_fy = pdm["n_fc"] if rt == 1 else len(pdm["ext_fc"])
    n_s = len(names)
    scale = "Internal" if rt == 1 else "External"
    ws.cell(r, 1, f"{scale} bucket {bk['key']} ({bk['rating']})").font = B.f_sub
    B.put(ws, r + 1, 1, "TTC PD")
    B.put(ws, r + 1, 2, f"=INDEX(TtcPd,MATCH(\"{scale}|{bk['key']}\",TtcKey,0))", "pd")
    T = _a(r + 1, 2)
    Rr = None
    if rt == 2:
        B.put(ws, r + 1, 3, "Asset correlation R")
        B.put(ws, r + 1, 4, f"=0.24-0.12*(1-EXP(-50*{T}))/(1-EXP(-50))", "x")
        Rr = _a(r + 1, 4)
    hr = r + 2
    cols = (["Year"] + [f"{n}: PiT PD" for n in names] + [f"{n}: survival" for n in names]
            + ["Weighted yearly PD"])
    B.header(ws, hr, 1, cols)
    first = hr + 1
    for y in range(1, mm + 1):
        rr = first + y - 1
        B.put(ws, rr, 1, y, "int")
        terms = []
        for k in range(n_s):
            cp, cs = 2 + k, 2 + n_s + k
            if y <= n_fy:
                if rt == 1:
                    f = f"=IF({T}=0,0,NORMSDIST(NORMSINV({T})+{sf(k, y)}))"
                else:
                    f = (f"=IF({T}<=0,0,IF({T}>=1,1,NORMSDIST((NORMSINV({T})"
                         f"-SQRT({Rr})*{sf(k, y)})/SQRT(1-{Rr}))))")
            else:
                anc = _a(first + n_fy - 1, cp)
                f = f"=LN(({y}-{n_fy})/({mm}-{n_fy})*(EXP({T})-EXP({anc}))+EXP({anc}))"
            B.put(ws, rr, cp, f, "pd")
            pit = _a(rr, cp, fixed=False)
            prev = "1" if y == 1 else _a(rr - 1, cs, fixed=False)
            B.put(ws, rr, cs, f"={prev}*(1-{pit})", "pd")
            terms.append(f"{weight_of(k, y)}*{prev}*{pit}")
        B.put(ws, rr, 2 + 2 * n_s, "=" + "+".join(terms), "pd", key=True)
    out = _rng(first, 2 + 2 * n_s, first + mm - 1, 2 + 2 * n_s, "PD calculation")
    return out, first + mm + 1


def _pd(B, ws, facts, L, R, S, pm, ead) -> dict:
    """Each position: rating, scale, bucket, TTC PD, 12-month and lifetime
    PD; the cumulative PD by year below."""
    B.title(ws, "PD: probability of default",
            "From the rating: its PD bucket and TTC PD (long-run average), made "
            "forward-looking with five weighted economic scenarios (hidden tab: PD "
            "calculation). 12-month PD: the cumulative PD after 12 months. Lifetime PD: "
            "after the months counted (12 at Stage 1, to maturity at Stages 2 and 3).")
    cols = ["Contract", "Rating", "Scale", "PD bucket", "TTC PD", "12-month PD",
            "Months to maturity", "Lifetime PD", "Curve"]
    B.header(ws, 4, 1, cols)
    out = {}
    for k, l in enumerate(L):
        rr = 5 + k
        x = l["line"]
        g = lambda c: _a(rr, c, fixed=False)
        B.put(ws, rr, 1, f"={R[(x, 'Contract')]}")
        B.put(ws, rr, 2, f"={R[(x, 'Final rating')]}")
        typ = R[(x, "Rating type")]
        B.put(ws, rr, 3, f"=IF(OR(LEFT({typ},8)=\"External\",{typ}=\"2\"),\"External\","
                         f"\"Internal\")")
        B.put(ws, rr, 4, f"=IFERROR(INDEX(BucketNumber,MATCH({g(3)}&\"|\"&{g(2)},"
                         f"BucketKey,0)),\"\")", "int")
        B.put(ws, rr, 5, f"=IFERROR(INDEX(TtcPd,MATCH({g(3)}&\"|\"&{g(4)},TtcKey,0)),\"\")",
              "pd")
        B.put(ws, rr, 9, f"=IF({g(4)}=\"\",\"\",IF({pm['source']}=\"{CALC}\",{g(3)},"
                         f"{R[(x, 'Portfolio')]})&\" | bucket \"&{g(4)})")
        cur = f"MATCH({g(9)},{pm['keys']},0)"
        B.put(ws, rr, 6, f"=IFERROR(INDEX({pm['used']},13,{cur}),\"\")", "pd", key=True)
        N = ead[x]["N"]
        B.put(ws, rr, 7, f"={N}", "int")
        h = (f"MIN(MAX(0,IF({R[(x, 'Stage')]}=1,MIN({S['s1m']},{N}),{N})),{S['maxm']})")
        B.put(ws, rr, 8, f"=IFERROR(INDEX({pm['used']},{h}+1,{cur}),\"\")", "pd", key=True)
        out[x] = {"curve": _a(rr, 9, "PD"), "pd12": _a(rr, 6, "PD"),
                  "life": _a(rr, 8, "PD")}
    # cumulative PD by year
    r = 6 + len(L)
    ws.cell(r, 1, "Cumulative PD by year").font = B.f_sub
    B.header(ws, r + 1, 1, ["Year"] + [f"={R[(l['line'], 'Contract')]}" for l in L])
    for y in range(1, 11):
        rr = r + 1 + y
        B.put(ws, rr, 1, y, "int")
        for k, l in enumerate(L):
            c = out[l["line"]]["curve"]
            B.put(ws, rr, 2 + k, f"=IFERROR(INDEX({pm['used']},{12 * y}+1,"
                                 f"MATCH({c},{pm['keys']},0)),\"\")", "pd")
    B.widths(ws, {"A": 16, "B": 12, "C": 11, "D": 10, "E": 14, "F": 14, "G": 12,
                  "H": 14, "I": 26})
    return out


# =========================================================== LGD, EAD, ECL
def _lgd(B, ws, facts, L, R, S) -> dict:
    co = facts["collateral"]
    shown = {l["cid"] for l in L}
    a = co["alloc"][co["alloc"]["contract"].isin(shown)].reset_index(drop=True)
    c = co["coll"][co["coll"]["collateral"].isin(set(a["collateral"]))]
    t = co["types"][co["types"]["type"].isin(set(c["type"]))]
    B.title(ws, "LGD: loss given default",
            "LGD = base x MAX(unsecured share at least, (exposure - net collateral) / "
            "exposure). Net collateral = value x allocation x (1 - haircut). No "
            "exposure: the setting's LGD.")
    B.header(ws, 4, 1, ["Contract", "On balance", "Net collateral", "Collateral cover",
                        "LGD", "The report's LGD"])
    top = 5
    r = top + len(L) + 2
    ws.cell(r, 1, "How the net collateral is worked out").font = B.f_sub
    B.header(ws, r + 1, 1, ["Contract", "Collateral id", "Allocation", "Type", "Value",
                            "Haircut", "Net collateral"])
    ar0 = r + 2
    cr0 = ar0 + max(len(a), 1) + 3            # the collateral and type tables below
    cr1 = cr0 + max(len(c), 1) - 1
    tr1 = cr0 + max(len(t), 1) - 1
    for k, row in enumerate(a.itertuples(index=False)):
        rr = ar0 + k
        B.put(ws, rr, 1, row.contract, inp=True)
        B.put(ws, rr, 2, row.collateral, inp=True)
        B.put(ws, rr, 3, _num(row.share), "pct", inp=True)
        idr = _a(rr, 2, fixed=False)
        B.put(ws, rr, 4, f"=IFERROR(INDEX({_rng(cr0, 2, cr1, 2)},MATCH({idr},"
                         f"{_rng(cr0, 1, cr1, 1)},0)),\"\")")
        B.put(ws, rr, 5, f"=IFERROR(INDEX({_rng(cr0, 3, cr1, 3)},MATCH({idr},"
                         f"{_rng(cr0, 1, cr1, 1)},0)),\"\")", "amt")
        B.put(ws, rr, 6, f"=IFERROR(INDEX({_rng(cr0, 7, tr1, 7)},MATCH("
                         f"{_a(rr, 4, fixed=False)},{_rng(cr0, 5, tr1, 5)},0)),\"\")", "pct")
        v, s_, h = (_a(rr, i, fixed=False) for i in (5, 3, 6))
        B.put(ws, rr, 7, f"=IF(COUNT({v},{s_},{h})=3,{v}*{s_}*(1-{h}),0)", "amt")
    ar1 = ar0 + max(len(a), 1) - 1
    if not len(a):
        B.note(ws, ar0, 1, "No collateral allocated.")
    ws.cell(cr0 - 2, 1, "Collateral and haircuts").font = B.f_sub
    B.header(ws, cr0 - 1, 1, ["Collateral id", "Type", "Value"])
    for k, row in enumerate(c.itertuples(index=False)):
        B.put(ws, cr0 + k, 1, row.collateral, inp=True)
        B.put(ws, cr0 + k, 2, row.type, inp=True)
        B.put(ws, cr0 + k, 3, _num(row.value), "amt", inp=True)
    B.header(ws, cr0 - 1, 5, ["Type", "Name", "Haircut"])
    for k, row in enumerate(t.itertuples(index=False)):
        B.put(ws, cr0 + k, 5, row.type, inp=True)
        B.put(ws, cr0 + k, 6, _str(row.name), inp=True)
        B.put(ws, cr0 + k, 7, _num(row.haircut), "pct", inp=True)
    out = {}
    for k, l in enumerate(L):
        rr = top + k
        x = l["line"]
        B.put(ws, rr, 1, f"={R[(x, 'Contract')]}")
        onb = R[(x, "On balance")]
        B.put(ws, rr, 2, f"=IF(ISNUMBER({onb}),{onb},0)", "amt")
        B.put(ws, rr, 3, f"=SUMIFS({_rng(ar0, 7, ar1, 7)},{_rng(ar0, 1, ar1, 1)},"
                         f"{_a(rr, 1, fixed=False)})", "amt")
        e, n = _a(rr, 2, fixed=False), _a(rr, 3, fixed=False)
        B.put(ws, rr, 4, f"=IF({e}>0,{n}/{e},\"\")", "pct")
        B.put(ws, rr, 5, f"=IF({e}>0,{S['lgd_base']}*MAX({S['lgd_floor']},({e}-{n})/{e}),"
                         f"{S['zero_lgd']})", "pct", key=True)
        B.put(ws, rr, 6, l["rep_lgd"], "pct", inp=True)
        out[x] = {"lgd": _a(rr, 5, "LGD"), "onb": _a(rr, 2, "LGD"),
                  "net": _a(rr, 3, "LGD")}
    B.widths(ws, {"A": 16, "B": 16, "C": 16, "D": 16, "E": 14, "F": 16, "G": 16})
    return out


def _ead(B, ws, L, R, S) -> dict:
    B.title(ws, "EAD: exposure and the months counted",
            "With a repayment schedule the exposure follows it; without one the "
            "repayment type sets how it runs down (Bullet: stays until maturity; Equal "
            "instalments: falls by the same amount each payment; Annuity: level "
            "payments with interest). Months counted: up to 12 at Stage 1, to maturity "
            "at Stages 2 and 3.")
    cols = ["Contract", "Exposure at the reporting date", "Months to maturity",
            "Repayment type", "Stage", "Months counted in ECL", "Exposure from",
            "Months between payments", "Grace months", "Number of payments",
            "First payment in month", "Monthly interest rate", "Months of exposure"]
    E = {n: i + 1 for i, n in enumerate(cols)}
    B.header(ws, 4, 1, cols)
    out = {}
    for k, l in enumerate(L):
        rr = 5 + k
        x = l["line"]
        g = lambda n: _a(rr, E[n], fixed=False)
        B.put(ws, rr, E["Contract"], f"={R[(x, 'Contract')]}")
        onb = R[(x, "On balance")]
        B.put(ws, rr, E["Exposure at the reporting date"], f"=IF(ISNUMBER({onb}),{onb},0)",
              "amt")
        sched = l["schedule"]
        B.put(ws, rr, E["Exposure from"], "Repayment schedule" if sched is not None
              else "Repayment type (no schedule)")
        mat, ext = R[(x, "Maturity date")], R[(x, "Extract date")]
        mm = (f"((YEAR({mat})-YEAR({ext}))*12+MONTH({mat})-MONTH({ext})+"
              f"IF(DAY({mat})>DAY({ext}),1,0))")
        B.put(ws, rr, E["Months to maturity"],
              f"=IF(OR({mat}=\"\",{ext}=\"\"),{S['minm']},IF({mm}>0,{mm},{S['minm']}))",
              "int")
        N = g("Months to maturity")
        pf, pt = R[(x, "Portfolio")], R[(x, "Payment type")]
        shape = (f"IFERROR(INDEX({S['rule_shape']},MATCH({pf}&\"|\"&{pt},"
                 f"{S['rule_key']},0)),IFERROR(INDEX({S['rule_shape']},MATCH("
                 f"\"|\"&{pt},{S['rule_key']},0)),{S['default_shape']}))")
        B.put(ws, rr, E["Repayment type"],
              f"=IF({g('Exposure from')}=\"Repayment schedule\",\"As scheduled\","
              f"IF({N}<={S['minm']},\"{SHAPE_LABEL['bullet']}\",{shape}))")
        pfq, dfr = R[(x, "Payment frequency (months)")], R[(x, "Deferral (months)")]
        B.put(ws, rr, E["Months between payments"],
              f"=MAX(1,IF(ISNUMBER({pfq}),TRUNC({pfq}),1))", "int")
        B.put(ws, rr, E["Grace months"], f"=MAX(0,MIN(IF(ISNUMBER({dfr}),TRUNC({dfr}),0),"
                                          f"MAX(1,{N})-1))", "int")
        B.put(ws, rr, E["Number of payments"],
              f"=MAX(1,ROUNDUP(MAX(1,MAX(1,{N})-{g('Grace months')})/"
              f"{g('Months between payments')},0))", "int")
        B.put(ws, rr, E["First payment in month"],
              f"=MAX(1,{N})-({g('Number of payments')}-1)*{g('Months between payments')}",
              "int")
        nir = R[(x, "Nominal interest rate")]
        B.put(ws, rr, E["Monthly interest rate"], f"=IF(ISNUMBER({nir}),{nir}/12,0)", "rate")
        if sched is not None:
            B.put(ws, rr, E["Months of exposure"], len(sched), "int")
        else:
            B.put(ws, rr, E["Months of exposure"], f"=MIN(MAX(1,{N}),{S['maxm']})", "int")
        B.put(ws, rr, E["Stage"], f"={R[(x, 'Stage')]}", "int")
        B.put(ws, rr, E["Months counted in ECL"],
              f"=MIN(IF({g('Stage')}=1,MIN({S['s1m']},{g('Months of exposure')}),"
              f"{g('Months of exposure')}),{S['maxm']})", "int", key=True)
        out[x] = {"N": _a(rr, E["Months to maturity"], "EAD"),
                  "shape": _a(rr, E["Repayment type"], "EAD"),
                  "f": _a(rr, E["Months between payments"], "EAD"),
                  "n": _a(rr, E["Number of payments"], "EAD"),
                  "p1": _a(rr, E["First payment in month"], "EAD"),
                  "r": _a(rr, E["Monthly interest rate"], "EAD"),
                  "h": _a(rr, E["Months counted in ECL"], "EAD")}
    B.widths(ws, {_col_letter(c): w for c, w in enumerate(
        [16, 16, 11, 18, 7, 12, 26, 11, 9, 10, 11, 12, 11], 1)})
    return out


def _ecl(B, ws, L, R, S, lg, ead, pm, pdv) -> dict:
    B.title(ws, "ECL: month by month",
            "ECL in a month = exposure x LGD x PD in the month / (1 + EIR)^(month / 12); "
            "the PD in the month is the cumulative PD at its end less at its start. "
            "A position's ECL is the sum over the months counted, at most the exposure; "
            "Stage 3 follows the Assumptions.")
    B.header(ws, 4, 1, ["Contract", "Stage", "Months counted", "Sum of the months",
                        "ECL"])
    top = 5
    r = top + len(L) + 2
    cols = ["Contract", "Month", "Exposure", "Counted", "Cumulative PD at start",
            "Cumulative PD at end", "PD in the month", "LGD", "EIR", "Discount factor",
            "ECL in the month"]
    B.header(ws, r, 1, cols)
    rr = r + 1
    out = {}
    for k, l in enumerate(L):
        x = l["line"]
        e = ead[x]
        sched = l["schedule"]
        n = (min(len(sched), MAX_MONTH) if sched is not None
             else min(max(1, _fallback_n(l)), MAX_MONTH))
        start = rr
        onb = R[(x, "On balance")]
        curve = f"MATCH({pdv[x]['curve']},{pm['keys']},0)"
        for t in range(n):
            B.put(ws, rr, 1, l["cid"])
            B.put(ws, rr, 2, t, "int")
            T = _a(rr, 2, fixed=False)
            if sched is not None:
                B.put(ws, rr, 3, float(sched[t]), "amt", inp=True)
            else:
                paid = (f"IF({T}<{e['p1']},0,MIN({e['n']},INT(({T}-{e['p1']})/{e['f']})+1))")
                rp = f"({e['r']}*{e['f']})"
                lin, ann = SHAPE_LABEL["linear"], SHAPE_LABEL["annuity"]
                coef = (f"IF({e['shape']}=\"{lin}\",MAX(0,1-{paid}/{e['n']}),"
                        f"IF(AND({e['shape']}=\"{ann}\",{e['r']}>1E-9),"
                        f"IF({paid}>={e['n']},0,MAX(0,((1+{rp})^{e['n']}"
                        f"-(1+{rp})^{paid})/((1+{rp})^{e['n']}-1))),"
                        f"IF({e['shape']}=\"{ann}\",MAX(0,1-{paid}/{e['n']}),1)))")
                B.put(ws, rr, 3, f"=IF(ISNUMBER({onb}),{onb},0)*{coef}", "amt")
            B.put(ws, rr, 4, f"={T}<{e['h']}")
            B.put(ws, rr, 5, f"=INDEX({pm['used']},{T}+1,{curve})", "pd")
            B.put(ws, rr, 6, f"=INDEX({pm['used']},{T}+2,{curve})", "pd")
            B.put(ws, rr, 7, f"={_a(rr, 6, fixed=False)}-{_a(rr, 5, fixed=False)}", "pd")
            B.put(ws, rr, 8, f"={lg[x]['lgd']}", "pct")
            eir = R[(x, "EIR")]
            B.put(ws, rr, 9, f"={eir}", "rate")
            B.put(ws, rr, 10, f"=IF(ISNUMBER({eir}),(1+{eir})^({T}/12),NA())", "rate")
            B.put(ws, rr, 11, f"=IF({_a(rr, 4, fixed=False)},{_a(rr, 3, fixed=False)}*"
                              f"{_a(rr, 8, fixed=False)}*{_a(rr, 7, fixed=False)}/"
                              f"{_a(rr, 10, fixed=False)},0)", "amt")
            rr += 1
        # the result on top
        t0 = top + k
        B.put(ws, t0, 1, f"={R[(x, 'Contract')]}")
        B.put(ws, t0, 2, f"={R[(x, 'Stage')]}", "int")
        B.put(ws, t0, 3, f"={e['h']}", "int")
        tot = f"SUM({_rng(start, 11, rr - 1, 11)})"
        B.put(ws, t0, 4, f"=IFERROR({tot},\"\")", "amt")
        o = lg[x]["onb"]
        # a position the model cannot price (no interest rate, say) reads
        # "no ECL" rather than an error, so the totals still add up
        B.put(ws, t0, 5, f"=IFERROR(IF({_a(t0, 2)}=3,IF({S['stage3']}=\"zero\",0,"
                         f"MAX(0,{o})),IF(AND({S['cap']},{o}>0),MIN({tot},{o}),{tot})),"
                         f"\"no ECL\")", "amt", key=True)
        out[x] = _a(t0, 5, "ECL")
    ws.auto_filter.ref = f"A{r}:K{max(rr - 1, r + 1)}"
    B.widths(ws, {"A": 16, "B": 8, "C": 16, "D": 16, "E": 16, "F": 16, "G": 14,
                  "H": 10, "I": 10, "J": 12, "K": 16})
    return out


# ================================================================ summary
def _summary(B, ws, facts, L, R, S, lg, pdv, ecl) -> None:
    who = facts["customer_id"] + (f" ({facts['customer_name']})"
                                  if facts["customer_name"] else "")
    if facts.get("focus"):
        cs = ", ".join(dict.fromkeys(l["cid"] for l in L))
        B.title(ws, f"ECL calculator: contract {cs}, customer {who}")
    else:
        B.title(ws, f"ECL calculator: customer {who}")
    info = [("Run", facts["run_id"]),
            ("Reporting date", L[0]["d_extract"] if L else None),
            ("Config version", facts["config_version"]),
            ("Calculator version", facts["calculator_version"]),
            ("Engine", facts["engine"]),
            ("Report compared", facts["report_file"] or "none in the run"),
            ("Overlay report", facts["overlay_file"] or "none"),
            ("PD curves used", "='PD calculation'!$B$4"),
            ("Generated", facts["generated_at"])]
    for k, (lab, v) in enumerate(info):
        B.put(ws, 3 + k, 1, lab, bold=True)
        c = B.put(ws, 3 + k, 3, v, "date" if lab == "Reporting date" else None)
        c.alignment = B.Align(horizontal="left")
    B.note(ws, 12, 1, "Read it left to right: Facilities (inputs) -> Rating -> Staging -> "
           "PD -> LGD -> EAD -> ECL. Each tab starts with its result. Blue cells are the "
           "run's inputs, black cells formulas. Hidden tabs (right-click a tab, Unhide): "
           "Assumptions and PD calculation.")
    cols = ["Contract", "Contract (source id)", "Customer", "Customer name", "Product",
            "Portfolio", "Rating", "How the rating was set", "Stage", "Why",
            "On balance", "LGD", "12-month PD", "Lifetime PD", "ECL (this workbook)",
            "ECL (report)", "Difference", "Overlay", "ECL final", "Report final", "Check"]
    U = {n: i + 1 for i, n in enumerate(cols)}
    ws.cell(FIRST_ROW - 2, 1, "Per position").font = B.f_sub
    B.header(ws, FIRST_ROW - 1, 1, cols)
    first = FIRST_ROW
    for k, l in enumerate(L):
        rr = first + k
        x = l["line"]
        g = lambda n: _a(rr, U[n], fixed=False)
        B.put(ws, rr, U["Contract"], f"={R[(x, 'Contract')]}")
        B.put(ws, rr, U["Contract (source id)"],
              f"=IF({R[(x, 'Original contract')]}=\"\",\"\",{R[(x, 'Original contract')]})")
        B.put(ws, rr, U["Customer"], facts["customer_id"])
        B.put(ws, rr, U["Customer name"], facts["customer_name"] or "")
        B.put(ws, rr, U["Product"], f"={R[(x, 'Account type')]}")
        B.put(ws, rr, U["Portfolio"], f"={R[(x, 'Portfolio')]}")
        B.put(ws, rr, U["Rating"], f"={R[(x, 'Final rating')]}")
        B.put(ws, rr, U["How the rating was set"], f"={R[(x, 'Rating set by')]}")
        B.put(ws, rr, U["Stage"], f"={R[(x, 'Stage')]}", "int")
        B.put(ws, rr, U["Why"], f"={R[(x, 'Stage reason')]}")
        B.put(ws, rr, U["On balance"], f"={lg[x]['onb']}", "amt")
        B.put(ws, rr, U["LGD"], f"={lg[x]['lgd']}", "pct")
        B.put(ws, rr, U["12-month PD"], f"={pdv[x]['pd12']}", "pd")
        B.put(ws, rr, U["Lifetime PD"], f"={pdv[x]['life']}", "pd")
        B.put(ws, rr, U["ECL (this workbook)"], f"={ecl[x]}", "amt", key=True)
        B.put(ws, rr, U["ECL (report)"], l["rep_ecl"], "amt", inp=True)
        B.put(ws, rr, U["Difference"],
              f"=IF(OR({g('ECL (report)')}=\"\",NOT(ISNUMBER({g('ECL (this workbook)')}))),"
              f"\"\",{g('ECL (this workbook)')}-{g('ECL (report)')})", "amt")
        B.put(ws, rr, U["Overlay"], l["overlay"] if l["overlay"] is not None else 0,
              "amt", inp=True)
        B.put(ws, rr, U["ECL final"],
              f"=IF(ISNUMBER({g('ECL (this workbook)')}),{g('ECL (this workbook)')}"
              f"+{g('Overlay')},\"\")", "amt")
        B.put(ws, rr, U["Report final"], l["rep_final"], "amt", inp=True)
        tol = lambda a, b: f"ABS({a}-{b})<=MAX(0.01,1E-9*ABS({b}))"
        B.put(ws, rr, U["Check"],
              f"=IF({g('ECL (report)')}=\"\",\"not in report\","
              f"IF(NOT(ISNUMBER({g('ECL (this workbook)')})),\"DIFFERS\",IF(AND("
              f"{tol(g('ECL (this workbook)'), g('ECL (report)'))},OR("
              f"{g('Report final')}=\"\",{tol(g('ECL final'), g('Report final'))})),"
              f"\"OK\",\"DIFFERS\")))")
    last = first + max(len(L), 1) - 1
    ws.cell(14, 1, "Totals").font = B.f_sub
    B.header(ws, 15, 1, ["Stage", "Positions", "On balance", "ECL (this workbook)",
                         "ECL (report)", "Difference", "ECL final", "Report final"])
    col = lambda n: _rng(first, U[n], last, U[n])
    for k, s in enumerate([1, 2, 3, "Total"]):
        rr = 16 + k
        B.put(ws, rr, 1, s, bold=s == "Total")
        if s == "Total":
            for c in range(2, 9):
                B.put(ws, rr, c, f"=SUM({_rng(16, c, 18, c)})", "int" if c == 2 else "amt",
                      bold=True)
            continue
        B.put(ws, rr, 2, f"=COUNTIF({col('Stage')},{s})", "int")
        for c, n in ((3, "On balance"), (4, "ECL (this workbook)"), (5, "ECL (report)"),
                     (7, "ECL final"), (8, "Report final")):
            B.put(ws, rr, c, f"=SUMIFS({col(n)},{col('Stage')},{s})", "amt")
        B.put(ws, rr, 6, f"={_a(rr, 4, fixed=False)}-{_a(rr, 5, fixed=False)}", "amt")
    from openpyxl.formatting.rule import CellIsRule
    rng = f"{_col_letter(U['Check'])}{first}:{_col_letter(U['Check'])}{last}"
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"OK"'],
                                                  fill=B.Fill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"DIFFERS"'],
                                                  fill=B.Fill("solid", fgColor="FFC7CE")))
    B.widths(ws, {_col_letter(U[n]): w for n, w in (
        ("Contract", 15), ("Contract (source id)", 20), ("Customer", 12),
        ("Customer name", 28), ("Product", 14), ("Portfolio", 15), ("Rating", 9),
        ("How the rating was set", 30), ("Stage", 7), ("Why", 30), ("On balance", 15),
        ("LGD", 9), ("12-month PD", 12), ("Lifetime PD", 12),
        ("ECL (this workbook)", 17), ("ECL (report)", 17), ("Difference", 11),
        ("Overlay", 11), ("ECL final", 15), ("Report final", 15), ("Check", 10))})
