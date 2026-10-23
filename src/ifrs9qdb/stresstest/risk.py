"""Measuring the risk of a scenario, and of the macro outlook as a whole.

    shapley        a combined scenario's change split by shock: each shock's
                   average added ECL over every order the shocks could be
                   added in, so the parts add up to the total; the gap
                   between the total and the shocks priced alone is the
                   interaction
    sensitivities  the change for a one-standard-deviation adverse move in
                   each variable that reaches the book, and by segment
    ecl_at_risk    thousands of five-year paths from the macro model, each
                   priced through the transmission layers, with customer
                   defaults drawn in each: the distribution of the
                   provision and of the year's credit loss, its percentiles
                   and expected shortfall, where each named scenario sits,
                   and who carries the tail
    reverse        the most likely scenario (nearest the baseline in the
                   joint history) that produces a given loss

Every number but the Monte Carlo is a full repricing. Pricing thousands of
paths in full is too slow, so ``ecl_at_risk`` prices the book once on a grid
of moves per channel and segment (``build_proxy``), reads each path from the
grid, and checks the grid against full repricing on the named scenarios
(``proxy_check``). The paths and the default draws come from one stream R
draws identically (``macro.prng``).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..macro.catalogue import concepts as _concepts
from ..macro.prng import uniforms
from ..stress import pd_chain, reprice
from .config import segment_loadings
from .scenarios import Scenario, Shock, library, scenario_paths
from .transmission import (RunContext, _level_move, _sd, drawdown, external_notches,
                           factor_shift, implied_stage, notch_ratings, rate_notches,
                           stressed_pd12, transmit)

__all__ = ["shapley", "sensitivities", "linked_concepts", "Proxy", "build_proxy",
           "proxy_eval", "ecl_at_risk", "ALL_CHANNELS", "proxy_check", "reverse", "player_label"]


def _label(concept: str) -> str:
    c = _concepts()
    hit = c[c["concept"] == concept]
    return hit["label"].iloc[0] if len(hit) else concept


def player_label(s: Shock) -> str:
    unit = " sd" if s.unit == "sd" else ""
    if s.shape == "custom":
        moves = ", ".join(f"{v:+g}" for v in (s.path or []) if v)
        return f"{_label(s.concept)} ({moves}{unit})"
    words = {"one_year": "one year", "held": "held", "build": f"over {s.years} years",
             "recover": f"recovering over {s.years} years"}.get(s.shape, s.shape)
    return f"{_label(s.concept)} ({s.size:+g}{unit}, {words})"


# ------------------------------------------------------------------ Shapley --
def shapley(rc: RunContext, scn: Scenario, max_exact: int = 6, samples: int = 24,
            seed: int | None = None, progress=None) -> dict:
    """The scenario's change split by its parts: each shock, and the
    overrides, the parameter levers and the IFRS 9 weights when it has them.
    Exact over every subset up to ``max_exact`` parts (2^n repricings), by
    ``samples`` random orders beyond."""
    if isinstance(scn, dict):
        scn = Scenario.from_dict(scn)
    players = [("shock", i, player_label(s)) for i, s in enumerate(scn.shocks)]
    if scn.overrides:
        players.append(("overrides", None, "Paths set by hand"))
    if scn.levers:
        players.append(("levers", None, "Parameter levers"))
    if scn.scenario_weights:
        players.append(("weights", None, "IFRS 9 scenario weights"))
    n = len(players)
    if n == 0:
        return {"ok": False, "reason": "The scenario has no shocks."}
    cache: dict = {frozenset(): 0.0}

    def v(S: frozenset) -> float:
        if S in cache:
            return cache[S]
        sub = Scenario(
            name=scn.name, shocks=[scn.shocks[p[1]] for j, p in enumerate(players)
                                   if j in S and p[0] == "shock"],
            knock_on=scn.knock_on,
            overrides=scn.overrides if any(players[j][0] == "overrides" for j in S) else {},
            levers=scn.levers if any(players[j][0] == "levers" for j in S) else {},
            weights=scn.weights,
            scenario_weights=scn.scenario_weights
            if any(players[j][0] == "weights" for j in S) else None)
        r = transmit(rc, sub, waterfall=False)
        if not r.get("ok"):
            raise RuntimeError(r.get("reason"))
        cache[S] = float(r["delta"])
        if progress:
            progress(len(cache))
        return cache[S]

    phi = np.zeros(n)
    exact = n <= max_exact
    if exact:
        f = math.factorial
        for i in range(n):
            others = [j for j in range(n) if j != i]
            for r in range(len(others) + 1):
                w = f(r) * f(n - r - 1) / f(n)
                for S in itertools.combinations(others, r):
                    S = frozenset(S)
                    phi[i] += w * (v(S | {i}) - v(S))
    else:
        u = uniforms(samples * n, int(seed or rc.cfg.get("risk", {}).get("seed", 20261023)))
        for k in range(samples):
            order = list(np.argsort(u[k * n:(k + 1) * n], kind="stable"))
            S = frozenset()
            for i in order:
                phi[i] += (v(S | {i}) - v(S)) / samples
                S = S | {i}
    total = v(frozenset(range(n)))
    alone = np.array([v(frozenset({i})) for i in range(n)])
    out = pd.DataFrame({"part": [p[2] for p in players], "kind": [p[0] for p in players],
                        "alone": alone, "shapley": phi})
    out["share_pct"] = 100 * out["shapley"] / total if total else np.nan
    return {"ok": True, "parts": out.sort_values("shapley", key=abs, ascending=False)
            .reset_index(drop=True), "total": total, "sum_alone": float(alone.sum()),
            "interaction": float(total - alone.sum()), "exact": exact,
            "repricings": len(cache) - 1}


# ------------------------------------------------------------- sensitivity --
def linked_concepts(rc: RunContext) -> dict:
    """Every concept that reaches the book, with the direction of an
    adverse move (-1 a fall, +1 a rise) and the channel."""
    out = {}
    for c, (_, _, active) in rc.drivers_internal.items():
        if active:
            out[c] = (-1, "PD model driver")
    for c in rc.drivers_external:
        out.setdefault(c, (-1, "PD model driver (external)"))
    tot: dict = {}
    for seg in pd.unique(rc.segment):
        ext = seg == (rc.cfg.get("factor") or {}).get("external_segment", "External book")
        for c, b in segment_loadings(rc.cfg, seg, ext).items():
            tot[c] = tot.get(c, 0.0) + b
    for c, b in tot.items():
        if c not in out and b != 0 and c in rc.model.concepts:
            out[c] = (-1 if b > 0 else 1, "Credit factor")
    for c in rc.coll_share:
        out.setdefault(c, (-1, "Collateral"))
    rcfg = rc.cfg.get("rates") or {}
    for c in (rcfg.get("variable"), rcfg.get("follows")):
        if c:
            out.setdefault(c, (1, "Rates"))
    x = (rc.cfg.get("ead") or {}).get("variable")
    if x:
        out.setdefault(x, (-1, "Exposure"))
    for c in ((rc.cfg.get("external") or {}).get("notches_per_sd") or {}):
        out.setdefault(c, (-1, "External book"))
    return out


def sensitivities(rc: RunContext, size_sd: float = 1.0, shape: str = "held",
                  knock_on: bool = False, progress=None) -> dict:
    """The change for a ``size_sd`` adverse move in each linked variable on
    its own (held over the horizon, the rest at baseline unless
    ``knock_on``), ranked; and the same by segment."""
    rows, heat = [], []
    for k, (c, (sign, channel)) in enumerate(linked_concepts(rc).items()):
        scn = Scenario(name=c, shocks=[Shock(c, sign * size_sd, "sd", shape)],
                       knock_on=knock_on)
        r = transmit(rc, scn, waterfall=False)
        if progress:
            progress(k + 1)
        if not r.get("ok"):
            continue
        rows.append({"concept": c, "label": _label(c), "channel": channel,
                     "move": f"{'+' if sign > 0 else '-'}{size_sd:g} sd",
                     "move_native": sign * size_sd * _sd(rc, c),
                     "change": r["delta"], "change_pct": r["pct"]})
        for _, s in r["by_segment"].iterrows():
            heat.append({"concept": c, "label": _label(c), "segment": s["segment"],
                         "change": s["change"], "change_pct": s["change_pct"]})
    t = pd.DataFrame(rows)
    if len(t):
        t = t.sort_values("change", key=abs, ascending=False).reset_index(drop=True)
    return {"ok": True, "table": t, "heat": pd.DataFrame(heat)}


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
# (the PD chain's drivers year by year, the credit factor's probit shift,
# the downgrades), the stage (and so the horizon: Stage 1 has year 1 only),
# LGD (collateral) and EAD (drawdown) -- so with W computed once for each
# contract and stage, and each curve's yearly m's on a grid of driver moves,
# a path is priced without repricing, and to the engine's own arithmetic.
# What is left out: a curve capped at 1, and a discount rate that resets
# (rates: discount), which the full repricing covers.

@dataclass
class Proxy:
    """The book, decomposed for pricing thousands of paths."""
    segments: list
    seg_of_row: np.ndarray
    ok: np.ndarray                 # rows priced
    stage0: np.ndarray
    onb: np.ndarray
    cnet: np.ndarray
    lgd_base: float
    lgd_floor: float
    stage3_zero: bool
    cap: bool
    W1: np.ndarray                 # (n,)   Stage 1, year 1
    W2: np.ndarray                 # (n, H) Stage 2, years 1..H
    T: np.ndarray                  # (n, K, G, H) yearly m by notches and driver grid
    R: np.ndarray                  # (n, K, G) Stage 2 years after H, weighted
    RM: np.ndarray                 # (n, K, G) the PD mass after H, unweighted
    xs_int: np.ndarray
    xs_ext: np.ndarray
    k_rho: np.ndarray              # sqrt(rho / (1 - rho)) per row
    base_total: float
    base_seg: np.ndarray
    rate_rows: np.ndarray
    ead_rows: np.ndarray
    coll_share: dict
    loadings: np.ndarray           # (S, k) per sd
    concept_index: dict
    gcc_weights: dict
    driver: tuple | None           # (concept, multiplier)
    ladder_lp: dict                # rating type -> log midpoints
    rtype: np.ndarray
    grade0: np.ndarray
    customer: np.ndarray
    built_with: dict = field(default_factory=dict)


def _year_m(cum, Y: int) -> np.ndarray:
    """A zero-prepended monthly curve's default probability year by year."""
    c = np.asarray(cum, float)
    out = np.zeros(Y)
    for y in range(Y):
        lo, hi = 12 * y, 12 * (y + 1)
        if lo >= c.size - 1:
            break
        out[y] = c[min(hi, c.size - 1)] - c[lo]
    return out


def _pricer(rc: RunContext):
    """compute_ecl on the run's contracts with a curve per row, LGD of one
    and no cap: the weights W."""
    from ..etl.report import compute_ecl
    ing = rc.ing
    frame = ing.copy()
    frame["payment_frequency"] = frame["pay_freq"]
    have = frame["cid"].notna().to_numpy()
    ctx = rc.inputs.ctx.copy(cap=False)

    def run(stage, cum):
        f = frame.copy()
        f["stage"] = stage
        out = np.full(len(f), np.nan)
        r = compute_ecl(ctx, f[have], lgd=np.ones(int(have.sum())),
                        cum=[cum] * int(have.sum()))
        out[have] = r["ecl"].to_numpy()
        return out
    return run


def build_proxy(rc: RunContext, progress=None) -> Proxy:
    """Decompose the book: the weights W for each stage (one pricing per
    year of the longest contract), the yearly default probabilities of every
    curve on a grid of PD-model-driver moves (one PD chain rebuild per grid
    point), and every contract's curve after 0..K notches."""
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
        cum = np.clip((t - 12 * y) / 12.0, 0.0, 1.0)
        W2[:, y] = run(2, cum)
        tick(f"Weights, year {y + 1}")
    W1 = run(1, np.clip(np.arange(curve_len) / 12.0, 0.0, 1.0))
    tick("Weights, Stage 1")

    # every row's curve key after k notches (k = 0..K-1)
    mx_ext = int((rc.cfg.get("external") or {}).get("max_notches", 6))
    K = max(5, mx_ext + 1)
    rating0 = ing["rating"].astype(str).to_numpy().astype(object)
    keys = np.empty((n, K), dtype=object)
    for k in range(K):
        rat = notch_ratings(rc, rating0, np.ones(n, dtype=bool), k) if k else rating0
        for i, (pf, r, rt) in enumerate(zip(ing["portfolio"], rat, ing["rating_type"])):
            t = str(rt)
            t = "2" if (t.startswith("External") or t in ("2", "2.0")) else "1"
            b = ctx0.rat2bucket.get(f"{t}{_KEYSEP}{r}")
            keys[i, k] = None if b is None or (isinstance(b, float) and np.isnan(b)) \
                else f"{pf}{_KEYSEP}{b}"

    def grid_sets(kind):
        xs = np.asarray(cfg.get("driver_grid",
                                [-10, -8, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]), float)
        sets = []
        for x in xs:
            if x == 0:
                sets.append(ctx0.stpd)
                continue
            if kind == "int":
                c, (col, mult, _) = drv
                ch = pd_chain(rc.inputs, rc.report, rc.run_path, weight_mode="hold",
                              mev_deltas={col: [x / mult] * H})
            else:
                ch = pd_chain(rc.inputs, rc.report, rc.run_path, weight_mode="hold",
                              gcc_deltas={k: [x] * H for k in rc.drivers_external.values()})
            sets.append(ch["curves"])
            tick(f"PD chain, {'non-oil GDP' if kind == 'int' else 'GCC growth'} {x:+g}")
        return xs, sets

    drv_list = [(c, v) for c, v in rc.drivers_internal.items() if v[2]]
    drv = drv_list[0] if drv_list else None
    xs_i, sets_i = grid_sets("int") if drv else (np.array([0.0]), [ctx0.stpd])
    xs_e, sets_e = grid_sets("ext") if rc.drivers_external else (np.array([0.0]), [ctx0.stpd])
    G = max(len(xs_i), len(xs_e))

    def pad(xs, sets):
        # the shorter grid repeats its last point, so one tensor holds both
        while len(sets) < G:
            xs = np.append(xs, xs[-1] + 1e-9 * len(sets))
            sets.append(sets[-1])
        return xs, sets
    xs_i, sets_i = pad(xs_i, list(sets_i))
    xs_e, sets_e = pad(xs_e, list(sets_e))

    ym_cache: dict = {}

    def ym(j, ext, key):
        k2 = (j, ext, key)
        if k2 not in ym_cache:
            cv = (sets_e if ext else sets_i)[j].get(key)
            ym_cache[k2] = None if cv is None else _year_m(cv, Y)
        return ym_cache[k2]

    T = np.full((n, K, G, H), np.nan)
    R = np.zeros((n, K, G))
    RM = np.zeros((n, K, G))
    ext = rc.external
    for k in range(K):
        for j in range(G):
            M = np.full((n, Y), np.nan)
            for i in range(n):
                key = keys[i, k]
                if key is None:
                    continue
                m = ym(j, bool(ext[i]), key)
                if m is not None:
                    M[i] = m
            T[:, k, j, :] = M[:, :H]
            R[:, k, j] = np.nansum(M[:, H:] * W2[:, H:], axis=1)
            RM[:, k, j] = np.nansum(M[:, H:], axis=1)
    tick("Curves by notch and driver")

    # the base: the decomposition must reproduce the engine's own price
    stage0 = ing["stage"].fillna(2).astype(int).to_numpy()
    onb = ing["on_balance"].to_numpy(float)
    cn = ing["collateral_net"].fillna(0.0).to_numpy(float)
    ctx = rc.inputs.ctx
    segs = list(pd.unique(rc.segment))
    sidx = {s: i for i, s in enumerate(segs)}
    srow = np.array([sidx[s] for s in rc.segment])

    cat = list(rc.model.concepts)
    ci = {c: i for i, c in enumerate(cat)}
    L = np.zeros((len(segs), len(cat)))
    driver_c = {c for c, v in rc.drivers_internal.items() if v[2]}
    ext_name = (rc.cfg.get("factor") or {}).get("external_segment", "External book")
    for a_, s_ in enumerate(segs):
        e_ = s_ == ext_name
        skip = set(rc.drivers_external) if e_ else driver_c
        for c, b in segment_loadings(rc.cfg, s_, e_).items():
            if c in ci and c not in skip:
                L[a_, ci[c]] = b

    from ..stress import frozen_config
    fz = frozen_config(rc.run_path)
    prices = ((fz.get("raw") or {}).get("external_gcc_country_prices") or {}) if fz.get("ok") else {}
    tot = sum(float(np.atleast_1d(v)[0]) for v in prices.values()) or 1.0
    gw = {c: float(np.atleast_1d(prices.get(country, [0]))[0]) / tot
          for c, country in rc.drivers_external.items()}
    lad = {}
    for r_, v in rc.ladders.items():
        lp = np.log(v)
        lad[r_] = (lp[:-1] + lp[1:]) / 2.0
    rts = ing["rating_type"].map(lambda v: 2 if (str(v).startswith("External")
                                                  or str(v) in ("2", "2.0")) else 1).to_numpy()
    base_ecl = rc.base["ecl"].to_numpy(float)
    okrow = np.isfinite(base_ecl) & np.isfinite(T[:, 0, 0, 0])
    P = Proxy(segments=segs, seg_of_row=srow, ok=okrow, stage0=stage0, onb=onb, cnet=cn,
              lgd_base=float(ctx.lgd_base), lgd_floor=float(ctx.lgd_floor),
              stage3_zero=str(ctx.stage3) == "zero", cap=bool(ctx.cap),
              W1=np.nan_to_num(W1), W2=np.nan_to_num(W2[:, :H]), T=T, R=R, RM=RM,
              xs_int=xs_i, xs_ext=xs_e, k_rho=np.sqrt(rc.rho / (1 - rc.rho)),
              base_total=0.0, base_seg=np.zeros(len(segs)),
              rate_rows=rate_notches(rc, {})["rows"], ead_rows=drawdown(rc, {})[0],
              coll_share={c: _share(rc, c) for c in rc.coll_share},
              loadings=L, concept_index=ci, gcc_weights=gw,
              driver=(drv[0], drv[1][1]) if drv else None, ladder_lp=lad, rtype=rts,
              grade0=rc.grade0.copy(), customer=ing["customer"].astype(str).to_numpy())
    D0 = np.zeros((1, H, len(cat)))
    e0 = _price_paths(P, rc, D0)
    P.base_total = float(e0["ecl"][0].sum())
    P.base_seg = np.bincount(srow, weights=e0["ecl"][0], minlength=len(segs))
    P.built_with = {"pricings": step[0],
                    "base_gap": P.base_total - float(np.nansum(base_ecl[okrow]))}
    return P


def _share(rc, c):
    from .transmission import rc_share
    return rc_share(rc, c)


def _interp_index(x, xs):
    """Each value's grid cell and its weight towards the next point, clamped
    at the grid's ends."""
    x = np.clip(np.asarray(x, float), xs[0], xs[-1])
    j = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
    w = (x - xs[j]) / np.maximum(xs[j + 1] - xs[j], 1e-12)
    return j, w


def _path_inputs(P: Proxy, rc: RunContext, D: np.ndarray, on: set) -> dict:
    """What each path does to each channel."""
    n_p, H = D.shape[0], rc.H
    ci = P.concept_index
    sd = np.array([_sd(rc, c) for c in rc.model.concepts])
    rcfg = rc.cfg.get("rates") or {}
    D = D.copy()
    v, f = rcfg.get("variable"), rcfg.get("follows")
    rv = D[:, :, ci[v]].copy() if v in ci else np.zeros((n_p, H))
    if f in ci:
        none = ~np.any(rv != 0, axis=1)
        rv[none] = D[none][:, :, ci[f]]
        if v in ci:
            D[none, :, ci[v]] = rv[none]
    out = {}
    out["d_int"] = (D[:, :, ci[P.driver[0]]] / P.driver[1]) if (P.driver and P.driver[0] in ci
                                                                and "drivers" in on) \
        else np.zeros((n_p, H))
    g = np.zeros((n_p, H))
    if "drivers" in on:
        for c, w in P.gcc_weights.items():
            if c in ci:
                g += w * D[:, :, ci[c]]
    out["d_ext"] = g
    out["Z"] = np.einsum("sk,nhk->nsh", P.loadings / sd[None, :], D) if "factor" in on \
        else np.zeros((n_p, len(P.segments), H))
    out["k_rate"] = np.clip(np.round(float(rcfg.get("notches_per_point", 0)) *
                                     np.maximum(0, rv.max(axis=1))), 0, 4).astype(int) \
        if "rates" in on else np.zeros(n_p, dtype=int)
    ecfg = rc.cfg.get("external") or {}
    idx = np.zeros(n_p)
    if "external" in on:
        for c, w in (ecfg.get("notches_per_sd") or {}).items():
            if c in ci:
                idx += float(w) * np.minimum(0.0, (D[:, :, ci[c]] / _sd(rc, c)).min(axis=1))
    out["k_ext"] = np.clip(np.round(-idx), 0, int(ecfg.get("max_notches", 6))).astype(int)
    xcfg = rc.cfg.get("ead") or {}
    xv = xcfg.get("variable")
    if "exposure" in on and xv in ci:
        zc = (D[:, :, ci[xv]] / _sd(rc, xv)).min(axis=1)
        out["pct"] = np.where(zc < 0, -float(xcfg.get("drawdown_per_sd", 0)) * zc, 0.0)
    else:
        out["pct"] = np.zeros(n_p)
    lv = {}
    if "collateral" in on:
        for c in P.coll_share:
            if c in ci:
                lvl = np.cumprod(1.0 + D[:, :, ci[c]] / 100.0, axis=1) - 1.0
                j = np.abs(lvl).argmax(axis=1)
                lv[c] = 100.0 * lvl[np.arange(n_p), j]
    out["levels"] = lv
    return out


ALL_CHANNELS = ("drivers", "factor", "staging", "collateral", "rates", "exposure", "external")


def _price_paths(P: Proxy, rc: RunContext, D: np.ndarray, on=ALL_CHANNELS,
                 chunk: int = 64) -> dict:
    """Every contract's ECL in each path (n_paths x n), and each performing
    row's stressed 12-month PD (for the credit-loss draws)."""
    on = set(on)
    n = len(P.onb)
    H = rc.H
    K, G = P.T.shape[1], P.T.shape[2]
    sicr = int((rc.cfg.get("staging") or {}).get("sicr_notches", 3))
    ecl_all, pd12_all = [], []
    ar = np.arange(n)
    for c0 in range(0, D.shape[0], chunk):
        Dc = D[c0:c0 + chunk]
        x = _path_inputs(P, rc, Dc, on)
        npth = Dc.shape[0]
        ji, wi = _interp_index(x["d_int"], P.xs_int)          # (p, H)
        je, we = _interp_index(x["d_ext"], P.xs_ext)
        ext = rc.external
        jj = np.where(ext[None, :, None], je[:, None, :], ji[:, None, :])   # (p, n, H)
        ww = np.where(ext[None, :, None], we[:, None, :], wi[:, None, :])
        kk = np.zeros((npth, n), dtype=int)
        kk[:, P.rate_rows] = x["k_rate"][:, None]
        kk[:, ext] = np.maximum(kk[:, ext], x["k_ext"][:, None])
        kk = np.minimum(kk, K - 1)
        hh = np.arange(H)[None, None, :]
        I = ar[None, :, None]

        def gather(k):
            a = P.T[I, k[:, :, None], jj, hh]
            b = P.T[I, k[:, :, None], np.minimum(jj + 1, G - 1), hh]
            return (1 - ww) * a + ww * b
        m_k = gather(kk)
        m_0 = gather(np.zeros_like(kk))[:, :, 0] if "staging" in on else None
        # the credit factor's probit shift, year by year
        Zr = x["Z"][:, P.seg_of_row, :]                       # (p, n, H)
        sh = -P.k_rho[None, :, None] * Zr

        def shift(m, s):
            mm = np.clip(m, 1e-12, 1 - 1e-12)
            out = norm.cdf(norm.ppf(mm) + s)
            return np.where(np.isfinite(m) & (m > 0), out, np.nan_to_num(m))
        mk = shift(m_k, sh) if "factor" in on else m_k
        stage = np.tile(P.stage0, (npth, 1))
        if "staging" in on:
            p12 = shift(m_0, sh[:, :, 0]) if "factor" in on else m_0
            grade = P.grade0[None, :].repeat(npth, axis=0).copy()
            for r_, mids in P.ladder_lp.items():
                rows = P.rtype == r_
                if rows.any():
                    grade[:, rows] = np.searchsorted(mids, np.log(np.clip(
                        np.nan_to_num(p12[:, rows], nan=1e-4), 1e-9, 1.0)))
            implied = np.maximum(0, grade - P.grade0[None, :])
            to2 = (P.stage0[None, :] == 1) & (implied + kk >= sicr)
            stage = np.where(to2, 2, stage)
        onb = np.tile(P.onb, (npth, 1))
        if "exposure" in on:
            onb = onb * (1.0 + np.where(P.ead_rows[None, :], x["pct"][:, None] / 100.0, 0.0))
        cn = np.tile(P.cnet, (npth, 1))
        mult = np.ones((npth, n))
        for c, lvl in x["levels"].items():
            mult += P.coll_share[c][None, :] * lvl[:, None] / 100.0
        cn = np.maximum(0.0, cn * mult)
        with np.errstate(divide="ignore", invalid="ignore"):
            lgd = np.where(onb > 0, P.lgd_base * np.maximum(P.lgd_floor, (onb - cn) / onb), 1.0)
        e = onb / np.where(P.onb[None, :] != 0, P.onb[None, :], 1.0)
        # a cumulative PD cannot pass 1: once it reaches it, later years add
        # nothing (the StPD's cap, and shift_curve's)
        mk = np.nan_to_num(mk)
        S = np.cumsum(mk, axis=2)
        mk = np.clip(np.minimum(mk, 1.0 - (S - mk)), 0.0, None)
        SH = mk.sum(axis=2)
        jH, wH = jj[:, :, -1], ww[:, :, -1]
        jH1 = np.minimum(jH + 1, G - 1)
        rest = (1 - wH) * P.R[ar[None, :], kk, jH] + wH * P.R[ar[None, :], kk, jH1]
        rmass = (1 - wH) * P.RM[ar[None, :], kk, jH] + wH * P.RM[ar[None, :], kk, jH1]
        with np.errstate(divide="ignore", invalid="ignore"):
            room = np.where(rmass > 0, np.clip((1.0 - SH) / rmass, 0.0, 1.0), 1.0)
        rest = rest * room
        s1 = mk[:, :, 0] * P.W1[None, :]
        s2 = (mk * P.W2[None, :, :]).sum(axis=2) + rest
        ecl = lgd * e * np.where(stage == 1, s1, s2)
        s3 = np.where(onb > 0, onb, 0.0) if not P.stage3_zero else np.zeros_like(onb)
        ecl = np.where(stage == 3, s3, ecl)
        if P.cap:
            ecl = np.where(onb > 0, np.minimum(ecl, onb), ecl)
        ecl = np.where(P.ok[None, :], ecl, 0.0)
        ecl_all.append(ecl)
        pd12_all.append(mk[:, :, 0])
    return {"ecl": np.vstack(ecl_all), "pd12": np.vstack(pd12_all)}


def proxy_eval(P: Proxy, rc: RunContext, D: np.ndarray, channels: bool = False,
               rows: np.ndarray | None = None) -> dict:
    """The change in ECL for a batch of deviation paths D (n x H x k, native
    units, the model's concepts): in total and by segment, and -- with
    ``channels`` -- each channel's part, added in the waterfall's order."""
    r = _price_paths(P, rc, D)
    S = len(P.segments)
    by = np.vstack([np.bincount(P.seg_of_row, weights=e, minlength=S) for e in r["ecl"]]) \
        - P.base_seg[None, :]
    out = {"total": by.sum(axis=1), "by_segment": by, "pd12": r["pd12"]}
    if channels:
        sel = np.arange(D.shape[0]) if rows is None else rows
        order = [("PD model drivers", {"drivers"}),
                 ("Credit factor", {"drivers", "factor"}),
                 ("Ratings and stages", {"drivers", "factor", "staging"}),
                 ("Collateral", {"drivers", "factor", "staging", "collateral"}),
                 ("Rates", {"drivers", "factor", "staging", "collateral", "rates"}),
                 ("Exposure", {"drivers", "factor", "staging", "collateral", "rates",
                               "exposure"}),
                 ("External book", set(ALL_CHANNELS))]
        prev = np.zeros(len(sel))
        ch = {}
        for name, on in order:
            t = _price_paths(P, rc, D[sel], on)["ecl"].sum(axis=1) - P.base_total
            ch[name] = t - prev
            prev = t
        out["channels"] = ch
        out["channel_rows"] = sel
    return out


def ecl_at_risk(rc: RunContext, P: Proxy | None = None, n: int | None = None,
                seed: int | None = None, threshold: float | None = None,
                scenarios=None, progress=None) -> dict:
    """The distribution of the provision (and of the year's credit loss)
    over ``n`` five-year paths drawn from the macro model."""
    cfg = rc.cfg.get("risk") or {}
    n = int(n or cfg.get("paths", 2000))
    seed = int(seed or cfg.get("seed", 20261023))
    P = P or build_proxy(rc, progress=progress)
    D = rc.model.simulate(n, rc.H, seed=seed, bootstrap=bool(cfg.get("bootstrap", True)))
    ev = proxy_eval(P, rc, D)
    base = float(np.nansum(rc.base["ecl"]))
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

    # the channels, on the tail paths and on a sample for the mean
    tail_idx = np.where(tailm)[0]
    samp = np.arange(0, n, max(1, n // 200))
    chs = proxy_eval(P, rc, D, channels=True, rows=np.unique(np.concatenate([tail_idx, samp])))
    sel = chs["channel_rows"]
    in_tail = np.isin(sel, tail_idx)
    in_samp = np.isin(sel, samp)
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
    for s in (scenarios if scenarios is not None else library()):
        d = float(proxy_eval(P, rc, _paths_matrix(rc, s))["total"][0])
        named.append({"scenario": s.name, "proxy_change": d,
                      "percentile": float(100 * np.mean(change <= d))})
    out = {"ok": True, "paths": n, "seed": seed, "base": base, "ecl": ecl, "change": change,
           "summary": pd.DataFrame(rows),
           "by_segment": seg.sort_values("tail_change", ascending=False).reset_index(drop=True),
           "by_channel": chan, "named": pd.DataFrame(named), "credit_loss": loss,
           "loss_summary": pd.DataFrame(loss_rows), "customers": cust_tail,
           "expected_loss": el, "proxy": P}
    if threshold is not None:
        out["p_exceed"] = float(np.mean(ecl > float(threshold)))
    return out


def _paths_matrix(rc: RunContext, s) -> np.ndarray:
    sp = scenario_paths(s, rc.model, rc.H)["paths"]
    Ds = np.zeros((1, rc.H, len(rc.model.concepts)))
    ci = {c: i for i, c in enumerate(rc.model.concepts)}
    for c, v in sp.items():
        if c in ci:
            Ds[0, :, ci[c]] = np.asarray(v, float)[:rc.H]
    return Ds


def proxy_check(rc: RunContext, P: Proxy, scenarios=None) -> pd.DataFrame:
    """The proxy against full repricing on the named scenarios."""
    rows = []
    for s in (scenarios if scenarios is not None else library()):
        prox = float(proxy_eval(P, rc, _paths_matrix(rc, s))["total"][0])
        full = transmit(rc, s, waterfall=False)
        if not full.get("ok"):
            continue
        rows.append({"scenario": s.name, "full": full["delta"], "proxy": prox,
                     "gap": prox - full["delta"],
                     "gap_pct_of_full": 100 * (prox - full["delta"]) / full["delta"]
                     if full["delta"] else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- reverse --
def reverse(rc: RunContext, P: Proxy, target: float, max_iter: int = 40,
            confirm: bool = True) -> dict:
    """The most likely path set -- the one nearest the baseline in the joint
    history -- whose change in ECL is ``target``. Linearise the proxy, step
    along the covariance-weighted gradient (the most likely direction for a
    linear loss), scale to the target, then confirm with full repricing."""
    H, k = rc.H, len(rc.model.concepts)
    sd = np.array([rc.model.sd[c] for c in rc.model.concepts])
    _, C = rc.model.stacked(H)
    eps = 0.5
    m = H * k
    Dp = np.zeros((2 * m, H, k))
    for j in range(m):
        h, i = divmod(j, k)
        Dp[2 * j, h, i] = eps * sd[i]
        Dp[2 * j + 1, h, i] = -eps * sd[i]
    f = proxy_eval(P, rc, Dp)["total"]
    g = (f[0::2] - f[1::2]) / (2 * eps)                 # per sd, stacked h*k + i
    if not np.any(g != 0):
        return {"ok": False, "reason": "No macro variable moves this book."}
    direction = C @ g
    direction = direction / np.sqrt(g @ C @ g)          # unit Mahalanobis length

    def at(a):
        x = a * direction
        D = (x.reshape(H, k) * sd)[None, :, :]
        return float(proxy_eval(P, rc, D)["total"][0]), D[0]

    lo, hi = 0.0, 1.0
    while at(hi)[0] < target and hi < 64:
        hi *= 2
    if at(hi)[0] < target:
        return {"ok": True, "found": False, "reason": "No plausible macro path reaches "
                "that loss through the channels configured.", "max_change": at(hi)[0],
                "distance": hi}
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if at(mid)[0] >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    a = hi
    prox, D = at(a)

    def full_at(a_):
        _, D_ = at(a_)
        p_ = {c: D_[:, j] for j, c in enumerate(rc.model.concepts)}
        r_ = transmit(rc, Scenario(name="Reverse stress", knock_on=False),
                      waterfall=True, paths=p_)
        return (float(r_["delta"]) if r_.get("ok") else float("nan")), r_, D_

    full = None
    if confirm:
        # the proxy found the direction and a first scale; full repricing
        # settles the scale (secant steps on the distance along it)
        fa, full, D = full_at(a)
        pts = [(0.0, 0.0), (a, fa)]
        for _ in range(6):
            if not np.isfinite(fa) or abs(fa - target) <= 0.01 * abs(target):
                break
            (a0, f0), (a1, f1) = pts[-2], pts[-1]
            a_new = a1 + (target - f1) * (a1 - a0) / (f1 - f0) if f1 != f0 else a1 * 1.2
            a_new = float(np.clip(a_new, 0.25 * a1, 4.0 * a1))
            fa, full, D = full_at(a_new)
            pts.append((a_new, fa))
            a = a_new
        full = full if full and full.get("ok") else None
    years = list(range(rc.first_year, rc.first_year + H))
    tab = []
    for j, c in enumerate(rc.model.concepts):
        if np.max(np.abs(D[:, j])) < 1e-6 * max(1.0, sd[j]):
            continue
        tab.append({"concept": c, "label": _label(c),
                    "worst_sd": float(D[:, j][np.argmax(np.abs(D[:, j]))] / sd[j]),
                    **{str(y): float(D[h, j]) for h, y in enumerate(years)}})
    tab = pd.DataFrame(tab)
    if len(tab):
        tab = tab.reindex(tab["worst_sd"].abs().sort_values(ascending=False).index)
    return {"ok": True, "found": True, "target": target, "distance": a,
            "probability": float(norm.sf(a)), "one_in": float(1 / norm.sf(a)) if norm.sf(a) > 0
            else float("inf"), "proxy_change": prox,
            "full_change": full["delta"] if full else None,
            "proxy_at_first_scale": prox,
            "paths": tab.reset_index(drop=True), "result": full}
