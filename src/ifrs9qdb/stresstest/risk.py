"""Measuring a scenario, and the macro outlook as a whole.

    drivers        a scenario's change split by main variable (Shapley
                   values): each variable's own moves and what they move
                   through the linked effects, adding up to the total
    sensitivity    the change for a one-standard-deviation adverse move in
                   each variable, with its linked effects; a variable's
                   change over a range of sizes; a scenario at both ends of
                   the PD link's range
    ecl_at_risk    thousands of five-year paths from the macro system, each
                   priced through the three channels, with customer defaults
                   drawn in each: the distribution of the provision and of
                   the year's credit loss, its percentiles and expected
                   shortfall, where each named scenario sits, and who
                   carries the tail
    reverse        what it takes to raise the provision by a given amount:
                   how far each scenario must be scaled, how large a shock
                   each variable needs, and how often the macro system
                   produces it

Every number but the Monte Carlo (and the reverse search, which full
repricing then confirms) is a full repricing. Pricing thousands of paths in
full is too slow, so the book is decomposed once (``build_proxy``) and each
path read from it; ``proxy_check`` compares it with full repricing on the
library every time. The default draws come from one stream R draws
identically (``macro.prng``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..macro.prng import uniforms
from ..stress import pd_chain
from .library import library
from .scenarios import Scenario, shape_path
from .transmission import (RunContext, _k, _share, channel_paths, price_channels,
                           scenario_paths, transmit)

__all__ = ["drivers", "sensitivity", "sensitivity_curve", "pd_link_range", "Proxy",
           "build_proxy", "proxy_eval", "paths_matrix", "ecl_at_risk", "proxy_check",
           "reverse"]


def _total(rc: RunContext, scn: Scenario, slope_scale: float = 1.0) -> dict:
    """A scenario's change by channel, without the reporting."""
    sim = scenario_paths(scn, rc.system, rc.H)
    pc = price_channels(rc, sim["paths"], scn.restructured, slope_scale)
    ok = rc.base["ecl"].notna().to_numpy() & np.isfinite(pc["after"])
    b = float(np.sum(rc.base["ecl"].to_numpy(float)[ok]))
    a_pd = float(np.sum(pc["after_pd"][ok]))
    a_st = float(np.sum(pc["after_staging"][ok]))
    a = float(np.sum(pc["after"][ok]))
    return {"change": a - b, "pd": a_pd - b, "staging": a_st - a_pd, "collateral": a - a_st}


# ------------------------------------------------------------- drivers --
def _part(scn: Scenario, keep: str | None, H: int) -> Scenario:
    """The scenario with one main variable's moves kept and every other
    main variable held at baseline in the years it sets: the share of every
    path that comes from ``keep`` (the system is linear, so the parts' paths
    add up to the scenario's)."""
    main = {}
    for v, p in scn.given(H).items():
        main[v] = p if v == keep else [None if x is None else 0.0 for x in p]
    return Scenario(name=scn.name, main=main, linked=scn.linked,
                    restructured=scn.restructured if keep == "_deferral" else 0.0)


def drivers(rc: RunContext, scn, P: "Proxy | None" = None, max_full: int = 3,
            progress=None) -> dict:
    """The scenario's change split by its main variables (and the deferral
    programme), as Shapley values: each part's average contribution over
    every order the parts could be added in, so the parts add up to the
    total exactly. A part is the main variable's own moves and everything
    they move through the linked effects, with the other main variables held
    where the scenario sets them -- the parts' paths add up to the
    scenario's. Up to ``max_full`` parts every combination is priced in full;
    beyond, the combinations are read from the proxy and scaled to the full
    price of the whole."""
    import itertools
    import math
    if isinstance(scn, dict):
        scn = Scenario.from_dict(scn)
    given = scn.given(rc.H)
    parts = [(v, rc.system.label(v)) for v in given]
    if scn.restructured > 0:
        parts.append(("_deferral", "Deferral programme"))
    n = len(parts)
    if not n:
        return {"ok": False, "reason": "The scenario sets no variable."}
    paths = {}
    for v, _ in parts:
        if v != "_deferral":
            paths[v] = scenario_paths(_part(scn, v, rc.H), rc.system, rc.H)["paths"]
    zero = {u: np.zeros(rc.H) for u in rc.system.order}

    def combo(S):
        tot = {u: np.zeros(rc.H) for u in rc.system.order}
        for v in S:
            if v in paths:
                for u in tot:
                    tot[u] = tot[u] + np.asarray(paths[v].get(u, zero[u]), float)
        return tot, (scn.restructured if "_deferral" in S else 0.0)

    full = n <= max_full
    if not full and P is None:
        P = build_proxy(rc)
    names = [v for v, _ in parts]
    cache: dict = {}

    def value(S: frozenset) -> float:
        if S in cache:
            return cache[S]
        if not S:
            cache[S] = 0.0
            return 0.0
        pth, r = combo(S)
        if full:
            pc = price_channels(rc, pth, r)
            ok = rc.base["ecl"].notna().to_numpy() & np.isfinite(pc["after"])
            cache[S] = float(np.sum(pc["after"][ok]) - np.sum(rc.base["ecl"].to_numpy(float)[ok]))
        else:
            cache[S] = float(proxy_eval(P, paths_matrix(rc, P, pth), r)["total"][0])
        if progress:
            progress(len(cache), 2 ** n)
        return cache[S]

    phi = np.zeros(n)
    f = math.factorial
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            w = f(k) * f(n - k - 1) / f(n)
            for T in itertools.combinations(others, k):
                S = frozenset(names[j] for j in T)
                phi[i] += w * (value(S | {names[i]}) - value(S))
    total = _total(rc, scn)["change"]
    whole = value(frozenset(names))
    if not full and whole:
        phi = phi * total / whole
    alone = [value(frozenset({v})) * (total / whole if (not full and whole) else 1.0)
             for v in names]
    out = pd.DataFrame({"variable": names, "label": [lb for _, lb in parts],
                        "change": phi, "alone": alone})
    return {"ok": True, "parts": out.sort_values("change", key=abs, ascending=False)
            .reset_index(drop=True), "total": total, "exact": full,
            "pricings": len(cache) - 1}


# --------------------------------------------------------- sensitivity --
def _one(rc: RunContext, v: str, size_sd: float, shape: str, linked: bool,
         years: int = 2) -> Scenario:
    sd = rc.system.sd(v)
    size = rc.system.adverse(v) * float(size_sd) * (sd if np.isfinite(sd) else 1.0)
    return Scenario(name=rc.system.label(v), main={v: shape_path(shape, size, rc.H, years)},
                    linked=None if linked else [])


def sensitivity(rc: RunContext, size_sd: float = 1.0, shape: str = "one_year",
                linked: bool = True, progress=None) -> dict:
    """An adverse move of ``size_sd`` standard deviations in each variable,
    on its own (with its linked effects unless ``linked`` is false),
    ranked by the change in ECL."""
    rows = []
    for i, v in enumerate(rc.system.order):
        scn = _one(rc, v, size_sd, shape, linked)
        t = _total(rc, scn)
        move = next(x for x in scn.main[v] if x is not None)
        rows.append({"variable": v, "label": rc.system.label(v), "unit": rc.system.unit(v),
                     "sd": rc.system.sd(v), "move": move, "change": t["change"],
                     "change_pct": 100 * t["change"] / rc.base_total if rc.base_total else np.nan,
                     "pd": t["pd"], "staging": t["staging"], "collateral": t["collateral"]})
        if progress:
            progress(i + 1, len(rc.system.order))
    tab = pd.DataFrame(rows).sort_values("change", key=abs, ascending=False)
    return {"ok": True, "table": tab.reset_index(drop=True), "size_sd": size_sd,
            "shape": shape, "linked": linked}


def sensitivity_curve(rc: RunContext, v: str, sizes=(-3, -2, -1, 0, 1, 2, 3),
                      shape: str = "one_year", linked: bool = True) -> pd.DataFrame:
    """The change in ECL as one variable moves from a favourable to an
    adverse shock of each size (in standard deviations; positive is
    adverse)."""
    rows = []
    for a in sizes:
        if a == 0:
            rows.append({"size_sd": 0.0, "move": 0.0, "change": 0.0})
            continue
        scn = _one(rc, v, a, shape, linked)
        move = next(x for x in scn.main[v] if x is not None)
        rows.append({"size_sd": float(a), "move": move, "change": _total(rc, scn)["change"]})
    return pd.DataFrame(rows)


def pd_link_range(rc: RunContext, scn) -> dict:
    """A scenario priced with the PD model's slope on non-oil growth at each
    end of the range QDB's default history supports (stress.yml pd_link)."""
    if isinstance(scn, dict):
        scn = Scenario.from_dict(scn)
    pl = rc.cfg.get("pd_link") or {}
    b = float(pl.get("slope", -0.0414))
    lo, hi = (float(x) for x in pl.get("range", [-0.019, -0.048]))
    out = {"slope": b, "range": [lo, hi], "central": _total(rc, scn)["change"]}
    out["low"] = _total(rc, scn, lo / b)["change"]
    out["high"] = _total(rc, scn, hi / b)["change"]
    return out


# ------------------------------------------------------------------- proxy --
# The engine's ECL for a Stage 1 or 2 contract is
#
#     ECL = LGD x sum_t EAD(t) x [cumPD(t+1) - cumPD(t)] / (1 + EIR)^(t/12)
#
# and the StPD spreads each year's default probability m(y) evenly over its
# months, so the provision is linear in the yearly m's:
#
#     ECL = LGD x sum_y m(y) x W(y),   W(y) = sum over year y of EAD/disc/12
#
# with weights W that do not depend on the PD at all. A path changes the m's
# (the PD chain's drivers year by year), the chance of a Stage 1 contract
# moving to Stage 2 (through its 12-month PD), and LGD (collateral) -- so
# with W computed once for each contract and stage, and each curve on a grid
# of driver moves, a path is priced to the engine's own arithmetic without
# repricing. The grid holds each year's default rate among the survivors
# (m(y) / survival), read at that year's move: a year's defaults then depend
# on the path's earlier years through survival, as in the chain itself.

@dataclass
class Proxy:
    """The book, decomposed for pricing thousands of paths."""
    segments: list
    seg_of_row: np.ndarray
    ok: np.ndarray
    stage0: np.ndarray
    external: np.ndarray
    onb: np.ndarray
    cnet: np.ndarray
    lgd_base: float
    lgd_floor: float
    stage3_zero: bool
    cap: bool
    W1: np.ndarray                 # (n,)   Stage 1, year 1
    W2: np.ndarray                 # (n, H) Stage 2, years 1..H
    Q: np.ndarray                  # (n, G, H) each year's default rate among survivors, on the grid
    SG: np.ndarray                 # (n, G) survival to the end of year H on the grid
    R: np.ndarray                  # (n, G) Stage 2 years after H, weighted
    RM: np.ndarray                 # (n, G) the PD mass after H
    xs_int: np.ndarray
    xs_ext: np.ndarray
    pb: np.ndarray                 # base 12-month PD
    elig: np.ndarray               # Stage 1 rows that can move
    p12: float
    k: float
    coll: list                     # collateral variables, in path order after the two drivers
    coll_share: np.ndarray         # (C, n)
    customer: np.ndarray
    base_total: float = 0.0
    base_seg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    built_with: dict = field(default_factory=dict)


def _year_m(cum, Y: int) -> np.ndarray:
    c = np.asarray(cum, float)
    out = np.zeros(Y)
    for y in range(Y):
        lo, hi = 12 * y, 12 * (y + 1)
        if lo >= c.size - 1:
            break
        out[y] = c[min(hi, c.size - 1)] - c[lo]
    return out


def _pricer(rc: RunContext):
    """compute_ecl on the run's contracts with one curve for every row, LGD
    of one and no cap: the weights W."""
    from ..etl.report import compute_ecl
    frame = rc.ing.copy()
    frame["payment_frequency"] = frame["pay_freq"]
    have = frame["cid"].notna().to_numpy()
    ctx = rc.inputs.ctx.copy(cap=False)

    def run(stage, cum):
        f = frame.copy()
        f["stage"] = stage
        out = np.full(len(f), np.nan)
        r = compute_ecl(ctx, f[have], lgd=np.ones(int(have.sum())), cum=[cum] * int(have.sum()))
        out[have] = r["ecl"].to_numpy()
        return out
    return run


def build_proxy(rc: RunContext, progress=None) -> Proxy:
    """Decompose the book: the weights W for each stage, and the yearly
    default probabilities of every curve on a grid of moves in each PD
    model's driver (one PD chain rebuild per grid point)."""
    from ..etl.report import _KEYSEP
    cfg = rc.cfg.get("risk") or {}
    ing, H = rc.ing, rc.H
    n = len(ing)
    step = [0]

    def tick(msg):
        step[0] += 1
        if progress:
            progress(step[0], msg)

    ctx0 = rc.inputs.ctx if rc.base_curves is None else rc.inputs.ctx.copy(stpd=rc.base_curves)
    curve_len = max((len(v) for v in ctx0.stpd.values()), default=13)
    Y = max(H + 1, (curve_len - 1 + 11) // 12)
    run = _pricer(rc)
    W2 = np.zeros((n, Y))
    for y in range(Y):
        t = np.arange(curve_len)
        W2[:, y] = run(2, np.clip((t - 12 * y) / 12.0, 0.0, 1.0))
        tick(f"Weights, year {y + 1}")
    W1 = run(1, np.clip(np.arange(curve_len) / 12.0, 0.0, 1.0))
    tick("Weights, Stage 1")

    keys = np.empty(n, dtype=object)
    for i, (pf, r, rt) in enumerate(zip(ing["portfolio"], ing["rating"].astype(str),
                                        ing["rating_type"])):
        t = str(rt)
        t = "2" if (t.startswith("External") or t in ("2", "2.0")) else "1"
        b = ctx0.rat2bucket.get(f"{t}{_KEYSEP}{r}")
        keys[i] = None if b is None or (isinstance(b, float) and np.isnan(b)) else f"{pf}{_KEYSEP}{b}"

    xs = np.asarray(cfg.get("driver_grid", [-20, -15, -10, -6, -4, -2, -1, 0, 1, 2, 4]), float)
    pl = rc.cfg.get("pd_link") or {}
    drv = rc.drivers_internal.get(pl.get("internal", "NONOIL_GDP_QAT"))

    def grid(kind):
        if (kind == "int" and drv is None) or (kind == "ext" and not rc.drivers_external):
            return np.array([0.0, 1.0]), [ctx0.stpd, ctx0.stpd]
        sets = []
        for x in xs:
            if x == 0:
                sets.append(ctx0.stpd)
                continue
            if kind == "int":
                col, mult, _ = drv
                ch = pd_chain(rc.inputs, rc.report, rc.run_path, weight_mode="hold",
                              mev_deltas={col: [x / mult] * H})
            else:
                ch = pd_chain(rc.inputs, rc.report, rc.run_path, weight_mode="hold",
                              gcc_deltas={k: [x] * H for k in rc.drivers_external.values()})
            if not ch.get("ok"):
                raise RuntimeError(ch.get("reason", "The PD chain could not be rebuilt."))
            sets.append(ch["curves"])
            tick(f"PD chain, {'non-oil GDP' if kind == 'int' else 'GCC growth'} {x:+g}")
        return xs.copy(), sets

    xs_i, sets_i = grid("int")
    xs_e, sets_e = grid("ext")
    G = max(len(xs_i), len(xs_e))

    def pad(x_, s_):
        x_, s_ = list(x_), list(s_)
        while len(s_) < G:
            x_.append(x_[-1] + 1e-9 * len(s_))
            s_.append(s_[-1])
        return np.array(x_), s_
    xs_i, sets_i = pad(xs_i, sets_i)
    xs_e, sets_e = pad(xs_e, sets_e)

    memo: dict = {}
    T = np.full((n, G, H), np.nan)
    R = np.zeros((n, G))
    RM = np.zeros((n, G))
    ext = rc.external
    for j in range(G):
        M = np.full((n, Y), np.nan)
        for i in range(n):
            key = keys[i]
            if key is None:
                continue
            e = bool(ext[i])
            k2 = (j, e, key)
            if k2 not in memo:
                cv = (sets_e if e else sets_i)[j].get(key)
                memo[k2] = None if cv is None else _year_m(cv, Y)
            if memo[k2] is not None:
                M[i] = memo[k2]
        T[:, j, :] = M[:, :H]
        R[:, j] = np.nansum(M[:, H:] * W2[:, H:], axis=1)
        RM[:, j] = np.nansum(M[:, H:], axis=1)
    tick("Curves on the grid")
    surv = 1.0 - np.concatenate([np.zeros((n, G, 1)), np.cumsum(T, axis=2)[:, :, :-1]], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = np.where(surv > 1e-12, T / surv, 1.0)
    Q = np.clip(Q, 0.0, 1.0)
    SG = 1.0 - np.nansum(T, axis=2)

    ctx = rc.inputs.ctx
    segs = list(pd.unique(rc.segment))
    sidx = {s: i for i, s in enumerate(segs)}
    srow = np.array([sidx[s] for s in rc.segment])
    j0 = int(np.argmin(np.abs(xs_i)))
    pb = np.where(ext, T[:, int(np.argmin(np.abs(xs_e))), 0], T[:, j0, 0])
    s = rc.cfg.get("staging") or {}
    elig = (rc.stage0 == 1) & np.isfinite(pb) & (pb > 0) & (pb < 1)
    if s.get("internal_only", True):
        elig &= ~ext
    coll = list(rc.coll_share)
    base_ecl = rc.base["ecl"].to_numpy(float)
    P = Proxy(segments=segs, seg_of_row=srow,
              ok=np.isfinite(base_ecl) & (np.isfinite(T[:, 0, 0]) | (rc.stage0 == 3)),
              stage0=rc.stage0.copy(), external=ext.copy(),
              onb=ing["on_balance"].to_numpy(float),
              cnet=ing["collateral_net"].fillna(0.0).to_numpy(float),
              lgd_base=float(ctx.lgd_base), lgd_floor=float(ctx.lgd_floor),
              stage3_zero=str(ctx.stage3) == "zero", cap=bool(ctx.cap),
              W1=np.nan_to_num(W1), W2=np.nan_to_num(W2[:, :H]), Q=Q, SG=SG, R=R, RM=RM,
              xs_int=xs_i, xs_ext=xs_e, pb=pb, elig=elig, p12=float(s.get("p12", 0.063)),
              k=_k(rc.cfg), coll=coll,
              coll_share=np.vstack([_share(rc, c) for c in coll]) if coll else np.zeros((0, n)),
              customer=ing["customer"].astype(str).to_numpy())
    e0 = _price_paths(P, np.zeros((1, H, 2 + len(coll))), np.zeros(1))
    P.base_total = float(e0["ecl"][0].sum())
    P.base_seg = np.bincount(srow, weights=e0["ecl"][0], minlength=len(segs))
    P.built_with = {"pricings": step[0],
                    "base_gap": P.base_total - float(np.nansum(base_ecl[P.ok]))}
    return P


def _interp_index(x, xs):
    x = np.clip(np.asarray(x, float), xs[0], xs[-1])
    j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    w = (x - xs[j]) / np.maximum(xs[j + 1] - xs[j], 1e-12)
    return j, w


def _price_paths(P: Proxy, D: np.ndarray, restr, on=("staging", "collateral"),
                 chunk: int = 64) -> dict:
    """Every contract's ECL in each path (n_paths x n), and each row's
    stressed 12-month PD. D is (paths, H, 2 + collateral variables): the
    internal driver, GCC growth, then each collateral variable."""
    on = set(on)
    n = len(P.onb)
    H = P.Q.shape[2]
    G = P.Q.shape[1]
    restr = np.broadcast_to(np.asarray(restr, float), (D.shape[0],))
    ar = np.arange(n)
    ecl_all, pd12_all = [], []
    for c0 in range(0, D.shape[0], chunk):
        Dc = D[c0:c0 + chunk]
        rc_ = restr[c0:c0 + chunk]
        npth = Dc.shape[0]
        ji, wi = _interp_index(Dc[:, :, 0], P.xs_int)
        je, we = _interp_index(Dc[:, :, 1], P.xs_ext)
        jj = np.where(P.external[None, :, None], je[:, None, :], ji[:, None, :])
        ww = np.where(P.external[None, :, None], we[:, None, :], wi[:, None, :])
        hh = np.arange(H)[None, None, :]
        I = ar[None, :, None]
        q = (1 - ww) * P.Q[I, jj, hh] + ww * P.Q[I, np.minimum(jj + 1, G - 1), hh]
        q = np.clip(np.nan_to_num(q), 0.0, 1.0)
        surv = np.concatenate([np.ones((npth, n, 1)), np.cumprod(1.0 - q, axis=2)[:, :, :-1]],
                              axis=2)
        m = surv * q
        SH = surv[:, :, -1] * (1.0 - q[:, :, -1])           # survival to the end of year H
        jH, wH = jj[:, :, -1], ww[:, :, -1]
        jH1 = np.minimum(jH + 1, G - 1)
        lin = lambda A: (1 - wH) * A[ar[None, :], jH] + wH * A[ar[None, :], jH1]
        sg = lin(P.SG)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(sg > 1e-12, SH / sg, 0.0)
        rest = lin(P.R) * scale
        rmass = lin(P.RM) * scale
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(rmass > 0, np.clip(SH / rmass, 0.0, 1.0), 1.0)
        rest = rest * room
        s1 = m[:, :, 0] * P.W1[None, :]
        s2 = (m * P.W2[None, :, :]).sum(axis=2) + rest
        dp = np.zeros((npth, n))
        if "staging" in on:
            ps = m[:, :, 0]
            e = P.elig[None, :] & (ps > 0) & (ps < 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                dz = np.where(e, norm.ppf(np.clip(ps, 1e-12, 1 - 1e-12))
                              - norm.ppf(np.clip(P.pb, 1e-12, 1 - 1e-12))[None, :], 0.0)
            dp = np.where(e, np.clip(norm.cdf(norm.ppf(P.p12) + P.k * dz) - P.p12
                                     + rc_[:, None], 0.0, 1.0), 0.0)
        cn = np.tile(P.cnet, (npth, 1))
        if "collateral" in on and P.coll:
            mult = np.ones((npth, n))
            for ci in range(len(P.coll)):
                lvl = np.cumprod(1.0 + Dc[:, :, 2 + ci] / 100.0, axis=1) - 1.0
                tr = np.minimum(0.0, lvl.min(axis=1))
                mult += P.coll_share[ci][None, :] * tr[:, None]
            cn = np.maximum(0.0, cn * mult)
        onb = P.onb[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            lgd = np.where(onb > 0, P.lgd_base * np.maximum(P.lgd_floor, (onb - cn) / onb), 1.0)
        ecl = lgd * np.where(P.stage0[None, :] == 1, s1 + dp * (s2 - s1), s2)
        s3 = np.where(onb > 0, onb, 0.0) if not P.stage3_zero else np.zeros_like(cn)
        ecl = np.where(P.stage0[None, :] == 3, s3, ecl)
        if P.cap:
            ecl = np.where(onb > 0, np.minimum(ecl, onb), ecl)
        ecl = np.where(P.ok[None, :], ecl, 0.0)
        ecl_all.append(ecl)
        pd12_all.append(m[:, :, 0])
    return {"ecl": np.vstack(ecl_all), "pd12": np.vstack(pd12_all)}


def proxy_eval(P: Proxy, D: np.ndarray, restr=0.0, channels: bool = False,
               rows: np.ndarray | None = None) -> dict:
    """The change in ECL for a batch of paths: in total and by segment, and
    -- with ``channels`` -- each channel's part, added in the waterfall's
    order."""
    r = _price_paths(P, D, restr)
    S = len(P.segments)
    by = np.vstack([np.bincount(P.seg_of_row, weights=e, minlength=S) for e in r["ecl"]]) \
        - P.base_seg[None, :]
    out = {"total": by.sum(axis=1), "by_segment": by, "pd12": r["pd12"]}
    if channels:
        sel = np.arange(D.shape[0]) if rows is None else rows
        rs = np.broadcast_to(np.asarray(restr, float), (D.shape[0],))[sel]
        prev = np.zeros(len(sel))
        ch = {}
        for name, on in (("PD model", ()), ("Stage moves", ("staging",)),
                         ("Collateral", ("staging", "collateral"))):
            t = _price_paths(P, D[sel], rs, on)["ecl"].sum(axis=1) - P.base_total
            ch[name] = t - prev
            prev = t
        out["channels"] = ch
        out["channel_rows"] = sel
    return out


def paths_matrix(rc: RunContext, P: Proxy, paths: dict) -> np.ndarray:
    """One path set (variable -> deviations) as the proxy reads it."""
    ch = channel_paths(rc, paths)
    cols = [ch["internal"], ch["external"]] + [ch[c] for c in P.coll]
    return np.stack(cols, axis=1)[None, :rc.H, :]


def _draws(rc: RunContext, P: Proxy, n: int, seed: int) -> np.ndarray:
    A = rc.system.draw(n, rc.H, seed=seed)
    idx = {v: i for i, v in enumerate(rc.system.order)}
    pl = rc.cfg.get("pd_link") or {}
    out = np.zeros((n, rc.H, 2 + len(P.coll)))
    for j, v in enumerate([pl.get("internal", "NONOIL_GDP_QAT"), pl.get("external", "GDP_GCC")]
                          + P.coll):
        if v in idx:
            out[:, :, j] = A[:, idx[v], :]
    return out


def ecl_at_risk(rc: RunContext, P: Proxy | None = None, n: int | None = None,
                seed: int | None = None, threshold: float | None = None,
                scenarios=None, progress=None) -> dict:
    """The distribution of the provision (and of the year's credit loss)
    over ``n`` five-year paths drawn from the macro system."""
    cfg = rc.cfg.get("risk") or {}
    n = int(n or cfg.get("paths", 2000))
    seed = int(seed or cfg.get("seed", 20261027))
    P = P or build_proxy(rc, progress=progress)
    D = _draws(rc, P, n, seed)
    ev = proxy_eval(P, D)
    base = rc.base_total
    change = ev["total"]
    ecl = base + change
    pcts = [float(x) for x in cfg.get("percentiles", [95, 99])]

    def tail(x, q):
        v = float(np.percentile(x, q))
        es = float(x[x >= v].mean()) if np.any(x >= v) else v
        return v, es

    rows = []
    for q in pcts:
        v, es = tail(ecl, q)
        rows.append({"measure": f"ECL at {q:g}%", "value": v, "change": v - base,
                     "expected_shortfall": es, "es_change": es - base})
    q99 = max(pcts)
    v99 = float(np.percentile(ecl, q99))
    tailm = ecl >= v99
    seg = pd.DataFrame({"segment": P.segments, "base": P.base_seg,
                        "mean_change": ev["by_segment"].mean(axis=0),
                        "tail_change": ev["by_segment"][tailm].mean(axis=0)})
    tot_tail = float(seg["tail_change"].sum())
    seg["tail_share_pct"] = 100 * seg["tail_change"] / tot_tail if tot_tail else np.nan
    tail_idx = np.where(tailm)[0]
    samp = np.arange(0, n, max(1, n // 200))
    chs = proxy_eval(P, D, channels=True, rows=np.unique(np.concatenate([tail_idx, samp])))
    sel = chs["channel_rows"]
    in_tail, in_samp = np.isin(sel, tail_idx), np.isin(sel, samp)
    chan = pd.DataFrame({"channel": list(chs["channels"]),
                         "mean_change": [float(v[in_samp].mean()) for v in chs["channels"].values()],
                         "tail_change": [float(v[in_tail].mean()) if in_tail.any() else np.nan
                                         for v in chs["channels"].values()]})

    loss, loss_rows, cust_tail, el = None, [], pd.DataFrame(), None
    if cfg.get("default_draws", True):
        expo = P.onb.copy()
        perf = P.ok & (P.stage0 < 3) & (expo > 0)
        worst = pd.Series(P.stage0).groupby(P.customer).transform("max").to_numpy()
        perf &= worst < 3
        ids, inv = np.unique(P.customer[perf], return_inverse=True)
        C = len(ids)
        w = expo[perf]
        ead = np.bincount(inv, weights=w, minlength=C)
        lgd0 = rc.base["lgd"].to_numpy(float)[perf]
        lgd_c = np.bincount(inv, weights=w * np.nan_to_num(lgd0), minlength=C) / np.maximum(ead, 1e-12)
        pdm = np.vstack([np.bincount(inv, weights=w * row[perf], minlength=C)
                         for row in ev["pd12"]]) / np.maximum(ead, 1e-12)[None, :]
        pd0 = np.bincount(inv, weights=w * np.nan_to_num(rc.pd12[perf]), minlength=C) / \
            np.maximum(ead, 1e-12)
        u = uniforms(n * C, seed + 7).reshape(n, C)
        Lmat = (u < pdm) * (lgd_c * ead)[None, :]
        loss = Lmat.sum(axis=1)
        el = float(np.sum(pd0 * lgd_c * ead))
        for q in pcts:
            v, es = tail(loss, q)
            loss_rows.append({"measure": f"Credit loss at {q:g}%", "value": v,
                              "expected_shortfall": es})
        lv = float(np.percentile(loss, q99))
        tl = loss >= lv
        segc = np.zeros(C, dtype=int)
        segc[inv] = P.seg_of_row[perf]
        cust_tail = (pd.DataFrame({"customer": ids, "exposure": ead,
                                   "segment": [P.segments[s] for s in segc],
                                   "pd12": pd0, "lgd": lgd_c,
                                   "tail_loss": Lmat[tl].mean(axis=0) if tl.any() else 0.0})
                     .sort_values("tail_loss", ascending=False).head(25).reset_index(drop=True))
    named = []
    for s in (scenarios if scenarios is not None else library(rc.H)):
        d = float(proxy_eval(P, paths_matrix(rc, P, scenario_paths(s, rc.system, rc.H)["paths"]),
                             s.restructured)["total"][0])
        named.append({"id": s.id, "scenario": s.name, "category": s.category,
                      "proxy_change": d, "percentile": float(100 * np.mean(change <= d))})
    out = {"ok": True, "paths": n, "seed": seed, "base": base, "ecl": ecl, "change": change,
           "summary": pd.DataFrame(rows),
           "by_segment": seg.sort_values("tail_change", ascending=False).reset_index(drop=True),
           "by_channel": chan, "named": pd.DataFrame(named), "credit_loss": loss,
           "loss_summary": pd.DataFrame(loss_rows), "customers": cust_tail,
           "expected_loss": el}
    if threshold is not None:
        out["p_exceed"] = float(np.mean(ecl > float(threshold)))
    return out


def proxy_check(rc: RunContext, P: Proxy, scenarios=None) -> pd.DataFrame:
    """The proxy against full repricing on the named scenarios."""
    rows = []
    for s in (scenarios if scenarios is not None else library(rc.H)):
        sim = scenario_paths(s, rc.system, rc.H)
        prox = float(proxy_eval(P, paths_matrix(rc, P, sim["paths"]), s.restructured)["total"][0])
        full = _total(rc, s)["change"]
        rows.append({"id": s.id, "scenario": s.name, "full": full, "proxy": prox,
                     "gap": prox - full,
                     "gap_pct_of_full": 100 * (prox - full) / full if full else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- reverse --
def _solve(f, target: float, hi: float = 10.0, grid: int = 40) -> float | None:
    """The smallest scale in (0, hi] where f reaches target, or None."""
    xs = np.linspace(0, hi, grid + 1)[1:]
    vals = np.array([f(x) for x in xs])
    hit = np.where(vals >= target)[0]
    if not len(hit):
        return None
    j = int(hit[0])
    lo_, hi_ = (0.0 if j == 0 else xs[j - 1]), xs[j]
    for _ in range(40):
        mid = (lo_ + hi_) / 2
        if f(mid) >= target:
            hi_ = mid
        else:
            lo_ = mid
        if hi_ - lo_ < 1e-4:
            break
    return float(hi_)


def _scaled(s: Scenario, a: float, H: int) -> Scenario:
    return Scenario(name=s.name, main={v: [None if x is None else a * x for x in p]
                                       for v, p in s.given(H).items()},
                    linked=s.linked, restructured=min(1.0, a * s.restructured))


def reverse(rc: RunContext, P: Proxy, target: float, scenarios=None, n: int | None = None,
            seed: int | None = None, confirm: int = 3, max_scale: float = 10.0) -> dict:
    """What it takes to raise the provision by ``target``:

    * each scenario scaled -- the factor its main variables (and deferral
      share) must be multiplied by, smallest first, the closest ones
      confirmed by full repricing;
    * each variable alone -- the adverse shock in year 1, with its linked
      effects, in standard deviations and in its own unit;
    * how often the macro system's paths raise it that much."""
    H = rc.H
    scns = list(scenarios if scenarios is not None else library(H))
    srows = []
    for s in scns:
        sim = scenario_paths(s, rc.system, H)
        D1 = paths_matrix(rc, P, sim["paths"])

        def f(a, D1=D1, r=s.restructured):
            return float(proxy_eval(P, a * D1, min(1.0, a * r))["total"][0])
        a = _solve(f, target, max_scale)
        srows.append({"id": s.id, "scenario": s.name, "category": s.category,
                      "scale": a, "at_one": f(1.0), "full_change": None})
    tab = pd.DataFrame(srows)
    found = tab[tab["scale"].notna()].sort_values("scale")
    for i in found.index[:confirm]:
        s = next(x for x in scns if x.id == tab.at[i, "id"] and x.name == tab.at[i, "scenario"])
        a = float(tab.at[i, "scale"])
        pts = []
        for _ in range(4):
            fa = _total(rc, _scaled(s, a, H))["change"]
            pts.append((a, fa))
            if abs(fa - target) <= 0.01 * abs(target) or len(pts) > 3:
                break
            if len(pts) == 1:
                a2 = a * target / fa if fa > 0 else a * 1.2
            else:
                (a0, f0), (a1, f1) = pts[-2], pts[-1]
                a2 = a1 + (target - f1) * (a1 - a0) / (f1 - f0) if f1 != f0 else a1 * 1.1
            a = float(np.clip(a2, 0.5 * pts[-1][0], 2.0 * pts[-1][0]))
        tab.at[i, "scale"] = pts[-1][0]
        tab.at[i, "full_change"] = pts[-1][1]
    tab = tab.sort_values("scale", na_position="last").reset_index(drop=True)

    vrows = []
    for v in rc.system.order:
        sd = rc.system.sd(v)
        if not np.isfinite(sd) or sd <= 0:
            continue
        unit_scn = _one(rc, v, 1.0, "one_year", True)
        sim = scenario_paths(unit_scn, rc.system, H)
        D1 = paths_matrix(rc, P, sim["paths"])

        def g(a, D1=D1):
            return float(proxy_eval(P, a * D1)["total"][0])
        a = _solve(g, target, max_scale)
        vrows.append({"variable": v, "label": rc.system.label(v), "unit": rc.system.unit(v),
                      "size_sd": a, "move": None if a is None else
                      a * rc.system.adverse(v) * sd, "at_one_sd": g(1.0)})
    vt = pd.DataFrame(vrows).sort_values("size_sd", na_position="last").reset_index(drop=True)

    cfg = rc.cfg.get("risk") or {}
    n = int(n or cfg.get("paths", 2000))
    seed = int(seed or cfg.get("seed", 20261027))
    ch = proxy_eval(P, _draws(rc, P, n, seed))["total"]
    p = float(np.mean(ch >= target))
    return {"ok": True, "target": float(target), "scenarios": tab, "variables": vt,
            "probability": p, "one_in": (1.0 / p) if p > 0 else None, "paths": n}
