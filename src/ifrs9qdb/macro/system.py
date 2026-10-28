"""Linked effects: how a scenario's main variables move the others.

A recursive system estimated on yearly history (``data/system.yml`` names
the variables, their series and each equation's candidate drivers):

    global      world GDP, the US policy rate (set by a scenario) and the oil
                price (moved by world GDP)
    Qatar       real government spending, non-oil GDP, real private credit,
                property prices
    GCC         real GDP growth of the six countries (the external book)

A variable can move the ones below it in the same year, and any variable
can move any other a year later -- the lag structure that lets a credit
shock reach non-oil GDP, then PD and stages (Miyajima, Khandelwal and Santos
for GCC banks; Adrian, Berrospide and Lafarguette; the ECB's macroprudential
stress test extension).

Each equation averages over every subset of its candidate drivers with BIC
weights (Bayesian model averaging), with country effects on the GCC panel.
A model where a driver takes the wrong sign is dropped, so each coefficient
is the evidence-weighted average of the sign-consistent models: a weak link
is shrunk rather than switched on or off.

A scenario works in deviations from the baseline, so intercepts and country
effects drop out:

    d_v(t) = sum over drivers  b x d_driver(t or t-1)  +  b_own x d_v(t-1)

Every number is a deterministic function of the store's vintage.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

__all__ = ["MacroSystem", "Equation", "fit_system", "system_spec", "load_panel",
           "fit_equation", "actual_col"]

_SPEC = Path(__file__).resolve().parent / "data" / "system.yml"


def system_spec(path=None) -> dict:
    return yaml.safe_load(Path(path or _SPEC).read_text(encoding="utf-8")) or {}


@dataclass
class Equation:
    var: str
    terms: list                       # [{"driver", "lag", "coef", "pip", "se"}]
    resid_sd: float
    n: int
    countries: int
    years: tuple
    models: int
    span: dict = field(default_factory=dict)   # term key -> largest departure seen

    def coef(self, driver: str, lag: int) -> float:
        for t in self.terms:
            if t["driver"] == driver and t["lag"] == lag:
                return float(t["coef"])
        return 0.0


def _term(driver: str, lag: int) -> str:
    return f"{driver}@{lag}"


def _annual(store, sid: str, vintage=None) -> pd.Series:
    try:
        a = store.annual(sid, vintage)
    except Exception:
        return pd.Series(dtype=float)
    a.index = a.index.astype(int)
    return a.sort_index().astype(float)


def _last_actual(store, sid: str, vintage, years) -> int:
    """The series' last actual year: the source's own marker where it gives
    one (the WEO does), else the year before the current one. Projections
    are kept for the baseline but never used to estimate."""
    from .sources import period_year
    from .store import last_actual_guess
    try:
        s = store.series(vintage)
        la = s.loc[s["series_id"] == sid, "last_actual"]
        la = period_year(la.iloc[0]) if len(la) and str(la.iloc[0]).strip() else None
    except Exception:
        la = None
    return int(la) if la is not None else last_actual_guess(list(years))


def actual_col(v: str) -> str:
    return f"_actual_{v}"


def _transform(a: pd.Series, how: str) -> pd.Series:
    from .store import to_concept_unit
    return to_concept_unit(a, how or "none")


def _static_series(name: str, project_root=None, transform: str = "none") -> pd.Series:
    """A package history file (year, value), for a variable the store does
    not have yet."""
    import io
    for d in ([Path(project_root) / "data-raw" / "static"] if project_root else []) + \
            [Path(__file__).resolve().parent.parent / "static"]:
        f = d / name
        if f.is_file():
            lines = [ln for ln in f.read_text(encoding="utf-8").splitlines()
                     if not ln.startswith("#")]
            t = pd.read_csv(io.StringIO("\n".join(lines)))
            a = pd.Series(t["value"].to_numpy(float), index=t["year"].astype(int))
            return _transform(a, transform)
    return pd.Series(dtype=float)


def load_panel(store, spec: dict | None = None, vintage=None, project_root=None) -> pd.DataFrame:
    """Every system variable as a long frame: one row per country and year
    (``c`` = the ISO code, ``G`` for the world), the variable's value in its
    system unit, and ``_actual_<var>`` -- whether that year is an outturn
    rather than a projection. Global variables repeat on every country's
    row."""
    spec = spec or system_spec()
    countries = list(spec.get("panel") or ["QAT"])
    years = range(1980, 2041)
    glob, pan, last = {}, {}, {}

    def one(sid, how, defl=None):
        raw = _annual(store, sid, vintage)
        s = _transform(raw, how)
        la = _last_actual(store, sid, vintage, raw.index)
        if defl:
            cpi = _annual(store, defl, vintage)
            s = (s - cpi.reindex(s.index)).dropna()
            la = min(la, _last_actual(store, defl, vintage, cpi.index))
        return s, la

    for v, m in (spec.get("variables") or {}).items():
        if m.get("series"):
            glob[v], last[(v, "G")] = one(m["series"], m.get("transform"))
        elif m.get("panel_series"):
            per = {}
            for c in (m.get("countries") or countries):
                per[c], last[(v, c)] = one(m["panel_series"].format(c=c), m.get("transform"),
                                           m["deflate"].format(c=c) if m.get("deflate") else None)
            pan[v] = per
        elif m.get("concept"):
            a = store.concept_annual(m["concept"], vintage, project_root)
            s = pd.Series(a["value"].to_numpy(float), index=a["year"].astype(int))
            la = int(a.loc[a["actual"], "year"].max()) if a["actual"].any() else None
            if s.dropna().empty and m.get("static"):
                s = _static_series(m["static"], project_root, m.get("transform", "pct_change"))
                la = int(s.index.max()) if len(s) else None
            pan[v] = {"QAT": s}
            last[(v, "QAT")] = la if la is not None else 0
    rows = []
    for c in ["G"] + countries:
        for y in years:
            r = {"c": c, "y": y}
            for v, s in glob.items():
                r[v] = s.get(y, np.nan)
                r[actual_col(v)] = y <= last[(v, "G")]
            for v, per in pan.items():
                r[v] = per.get(c, pd.Series(dtype=float)).get(y, np.nan) if c != "G" else np.nan
                r[actual_col(v)] = c != "G" and y <= last.get((v, c), 0)
            rows.append(r)
    d = pd.DataFrame(rows)
    vals = [v for v in list(glob) + list(pan)]
    keep = d[vals].notna().any(axis=1)
    return d[keep].reset_index(drop=True)


class _Fit:
    """Ordinary least squares, the few results model averaging needs."""

    def __init__(self, y: pd.Series, X: pd.DataFrame):
        Xv = X.to_numpy(float)
        yv = y.to_numpy(float)
        beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
        resid = yv - Xv @ beta
        n, k = Xv.shape
        self.ssr = float(resid @ resid)
        self.df_resid = max(n - k, 1)
        s2 = self.ssr / self.df_resid
        cov = s2 * np.linalg.pinv(Xv.T @ Xv)
        self.params = pd.Series(beta, index=X.columns)
        self.bse = pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=X.columns)
        llf = -n / 2.0 * (np.log(2 * np.pi) + np.log(max(self.ssr, 1e-300) / n) + 1.0)
        self.bic = float(-2.0 * llf + k * np.log(n))


def fit_equation(panel: pd.DataFrame, var: str, meta: dict, signs: dict,
                 start_year: int = 2001) -> Equation | None:
    """Model-average one equation over every subset of its candidates."""
    is_global = bool(meta.get("series"))
    d = panel[panel["c"] == "G"] if is_global else panel[panel["c"] != "G"]
    d = d.sort_values(["c", "y"]).copy()
    # outturns only: a projection is the forecaster's model, not evidence
    for col in [c for c in d.columns if not c.startswith("_actual_") and c not in ("c", "y")]:
        a = actual_col(col)
        if a in d.columns:
            d.loc[~d[a].astype(bool), col] = np.nan
    own = str(meta.get("own_lag", "always"))
    always, cands = [], []
    lagged = lambda col: d.groupby("c")[col].shift(1)
    d[_term(var, 1)] = lagged(var)
    if own == "always":
        always.append(_term(var, 1))
    elif own == "candidate":
        cands.append(_term(var, 1))
    for drv in meta.get("same") or []:
        d[_term(drv, 0)] = d[drv]
        cands.append(_term(drv, 0))
    for drv in meta.get("lag") or []:
        d[_term(drv, 1)] = lagged(drv)
        cands.append(_term(drv, 1))
    d = d[d["y"] >= int(meta.get("start_year", start_year))]
    cols = [var] + always + cands
    dd = d.dropna(subset=cols)
    if len(dd) < len(cols) + 5:
        return None
    multi = dd["c"].nunique() > 1
    fe = pd.get_dummies(dd["c"], drop_first=True, dtype=float) if multi \
        else pd.DataFrame(index=dd.index)
    res = []
    for k in range(len(cands) + 1):
        for S in itertools.combinations(cands, k):
            X = pd.concat([pd.Series(1.0, index=dd.index, name="const"), fe,
                           dd[always + list(S)]], axis=1)
            m = _Fit(dd[var], X)
            bad = False
            for t in S:
                drv = t.split("@")[0]
                sg = signs.get(drv)
                if drv != var and sg and np.sign(m.params[t]) != sg:
                    bad = True
                    break
            if bad:
                continue
            res.append((m.bic, S, m.params, m.bse, m.ssr / max(m.df_resid, 1)))
    if not res:
        return None
    b = np.array([r[0] for r in res])
    w = np.exp(-(b - b.min()) / 2)
    w /= w.sum()
    terms = []
    for t in always + cands:
        drv, lag = t.split("@")
        inc = [(wi, r) for wi, r in zip(w, res) if t in always or t in r[1]]
        pip = 1.0 if t in always else float(sum(wi for wi, _ in inc))
        coef = float(sum(wi * r[2].get(t, 0.0) for wi, r in zip(w, res)))
        se = float(sum(wi * r[3][t] for wi, r in inc) / sum(wi for wi, _ in inc)) if inc else float("nan")
        terms.append({"driver": drv, "lag": int(lag), "coef": coef, "pip": pip, "se": se})
    sig2 = float(sum(wi * r[4] for wi, r in zip(w, res)))
    span = {t: float((dd[t] - dd[t].mean()).abs().max()) for t in always + cands}
    return Equation(var=var, terms=terms, resid_sd=float(np.sqrt(sig2)), n=int(len(dd)),
                    countries=int(dd["c"].nunique()), years=(int(dd["y"].min()), int(dd["y"].max())),
                    models=len(res), span=span)


@dataclass
class MacroSystem:
    """The fitted system: its variables in recursive order and their
    equations."""
    order: list
    meta: dict
    equations: dict
    panel: pd.DataFrame
    vintage: str | None = None
    signs: dict = field(default_factory=dict)

    # ------------------------------------------------------------- basics --
    def label(self, v: str) -> str:
        return str((self.meta.get(v) or {}).get("label", v))

    def unit(self, v: str) -> str:
        return str((self.meta.get(v) or {}).get("unit", ""))

    def has_equation(self, v: str) -> bool:
        return v in self.equations

    def given_only(self) -> list:
        return [v for v in self.order if v not in self.equations]

    def sd(self, v: str) -> float:
        """The standard deviation of a variable's yearly history (Qatar's for
        a panel variable), the size of a one-sd move."""
        x = self.series(v, actual_only=True)
        return float(x.std(ddof=1)) if len(x) > 3 else float("nan")

    def series(self, v: str, actual_only: bool = False, country: str | None = None) -> pd.Series:
        """A variable's yearly values (Qatar's, or ``country``'s, for a panel
        variable), outturns only with ``actual_only``."""
        p = self.panel
        c = "G" if (self.meta.get(v) or {}).get("series") else (country or "QAT")
        rows = p[p["c"] == c]
        if v not in rows:
            return pd.Series(dtype=float)
        if actual_only and actual_col(v) in rows:
            rows = rows[rows[actual_col(v)].astype(bool)]
        return pd.Series(rows[v].to_numpy(float), index=rows["y"].astype(int)).dropna()

    def last_actual(self, v: str) -> int | None:
        x = self.series(v, actual_only=True)
        return int(x.index.max()) if len(x) else None

    def display(self, v: str) -> pd.DataFrame:
        """A variable as the screens chart it -- year, value, actual: the
        world's for a global variable, Qatar's for a Qatar one, the panel's
        average for a GCC one."""
        p = self.panel
        m = self.meta.get(v) or {}
        if m.get("series"):
            rows = p[p["c"] == "G"]
        elif str(m.get("level", "")).upper() == "GCC":
            rows = p[p["c"] != "G"]
        else:
            rows = p[p["c"] == "QAT"]
        if v not in rows:
            return pd.DataFrame({"year": [], "value": [], "actual": []})
        a = actual_col(v)
        d = pd.DataFrame({"year": rows["y"].astype(int), "value": rows[v].astype(float),
                          "actual": rows[a].astype(bool) if a in rows else True}).dropna()
        out = d.groupby("year").agg(value=("value", "mean"), actual=("actual", "all"))
        return out.reset_index()

    def baseline(self, v: str, years) -> list:
        """The store's values for ``years`` (projections ahead of the last
        actual year): the baseline a scenario's deviations are added to."""
        d = self.display(v).set_index("year")["value"]
        return [float(d[y]) if y in d.index else None for y in years]

    def adverse(self, v: str) -> int:
        """The direction of an adverse move: -1 a fall, +1 a rise."""
        return -int(self.signs.get(v, 1)) if self.signs else -1

    # ----------------------------------------------------------- simulate --
    def simulate(self, given: dict, linked=None, H: int = 5, shocks=None) -> dict:
        """Every variable's path under a scenario.

        ``given`` maps a main variable to its yearly deviations (a number per
        year; ``None`` leaves that year to its equation). ``linked`` is the
        set of other variables allowed to move (``None``: all of them); a
        variable outside it stays at baseline and passes nothing on.
        ``shocks`` adds equation residuals (Monte Carlo draws), an array
        (n_vars, H) in ``self.order``.

        Returns ``paths`` (variable -> array), ``role`` (main / linked /
        off), ``flags`` (variable -> years where a driver went beyond the
        largest move in the data the equation was fitted on) and
        ``recommended`` (the variables a main variable reaches)."""
        main = {v for v, p in (given or {}).items()
                if p is not None and any(x is not None for x in p)}
        allow = set(self.order) if linked is None else set(linked)
        P = {v: np.zeros(H) for v in self.order}
        set_ = {v: np.zeros(H, dtype=bool) for v in self.order}
        for v, path in (given or {}).items():
            if v not in P:
                continue
            for t, x in enumerate(list(path)[:H]):
                if x is not None and np.isfinite(float(x)):
                    P[v][t] = float(x)
                    set_[v][t] = True
        flags = {v: [] for v in self.order}
        idx = {v: i for i, v in enumerate(self.order)}
        for t in range(H):
            for v in self.order:
                if set_[v][t] or v not in self.equations:
                    continue
                if v not in main and v not in allow:
                    continue
                eq = self.equations[v]
                val = 0.0
                beyond = False
                for term in eq.terms:
                    lag = term["lag"]
                    if t - lag < 0:
                        continue
                    x = P[term["driver"]][t - lag]
                    val += term["coef"] * x
                    sp = eq.span.get(_term(term["driver"], lag))
                    if sp and abs(x) > sp and abs(term["coef"] * x) > 1e-9:
                        beyond = True
                if shocks is not None:
                    val += float(shocks[idx[v], t])
                P[v][t] = val
                if beyond and abs(val) > 1e-9:
                    flags[v].append(t)
        role = {}
        for v in self.order:
            role[v] = "main" if v in main else ("linked" if (v in allow and v in self.equations
                                                              and np.any(np.abs(P[v]) > 1e-9))
                                                else "off")
        return {"paths": P, "role": role, "flags": flags,
                "recommended": self.recommend(given, H)}

    def about(self, v: str) -> str:
        """What a variable is and what moves it, in words (system.yml)."""
        return " ".join(str((self.meta.get(v) or {}).get("about", "")).split())

    def explain(self, given: dict, linked=None, H: int = 5) -> pd.DataFrame:
        """Why each filled value is what it is: for every variable-year the
        equations filled, each driver's part (coefficient x the driver's
        value that year or the year before; the variable's own last year is
        its momentum). The parts add up to the filled value exactly."""
        sim = self.simulate(given, linked, H)
        P = sim["paths"]
        set_ = {v: [p is not None and j < len(p) and p[j] is not None
                    for j in range(H)] for v, p in (given or {}).items()
                for p in [list(p or [])]}
        rows = []
        for v in self.order:
            if v not in self.equations or sim["role"].get(v) == "off":
                continue
            for t in range(H):
                if set_.get(v, [False] * H)[t]:
                    continue
                for term in self.equations[v].terms:
                    lag = term["lag"]
                    if t - lag < 0:
                        continue
                    part = term["coef"] * float(P[term["driver"]][t - lag])
                    if abs(part) < 1e-9:
                        continue
                    rows.append({"variable": v, "year": t + 1, "driver": term["driver"],
                                 "lag": lag, "coef": term["coef"],
                                 "driver_value": float(P[term["driver"]][t - lag]),
                                 "part": part})
        return pd.DataFrame(rows, columns=["variable", "year", "driver", "lag", "coef",
                                           "driver_value", "part"])

    def recommend(self, given: dict, H: int = 5) -> list:
        """The variables the main variables reach through the equations."""
        main = {v for v, p in (given or {}).items()
                if p is not None and any(x is not None for x in p)}
        if not main:
            return []
        probe = {v: [1.0] * H for v in main}
        out = self.simulate_raw(probe, H)
        return [v for v in self.order if v not in main and v in self.equations
                and np.any(np.abs(out[v]) > 1e-9)]

    def simulate_raw(self, given: dict, H: int = 5) -> dict:
        P = {v: np.zeros(H) for v in self.order}
        for v, path in given.items():
            P[v] = np.asarray(path, float)[:H].copy()
        for t in range(H):
            for v in self.order:
                if v in given or v not in self.equations:
                    continue
                P[v][t] = sum(term["coef"] * P[term["driver"]][t - term["lag"]]
                              for term in self.equations[v].terms if t - term["lag"] >= 0)
        return P

    # ------------------------------------------------------------ draws ----
    def draw(self, n: int, H: int = 5, seed: int = 20261027) -> np.ndarray:
        """Monte Carlo paths (n, n_vars, H) in deviations: the global
        variables bootstrapped jointly from their own history (whole years,
        so world GDP, oil and the US rate keep their co-movement), every
        equation adding a normal residual at its estimated size."""
        rng = np.random.default_rng(seed)
        gv = [v for v in self.given_only()]
        hist = pd.concat({v: self.series(v, actual_only=True) for v in gv}, axis=1).dropna()
        hist = hist[hist.index >= 1991]
        dev = (hist - hist.mean()).to_numpy()
        out = np.zeros((n, len(self.order), H))
        idx = {v: i for i, v in enumerate(self.order)}
        for k in range(n):
            pick = rng.integers(0, len(dev), H)
            given = {v: list(dev[pick, j]) for j, v in enumerate(gv)}
            sh = np.zeros((len(self.order), H))
            for v, eq in self.equations.items():
                sh[idx[v]] = rng.normal(0.0, eq.resid_sd, H)
            P = self.simulate(given, None, H, shocks=sh)["paths"]
            for v in self.order:
                out[k, idx[v]] = P[v]
        return out

    # --------------------------------------------------------- describe ----
    def table(self) -> pd.DataFrame:
        """One row per equation term: what moves what, by how much, and the
        evidence for it."""
        rows = []
        for v in self.order:
            eq = self.equations.get(v)
            if eq is None:
                rows.append({"variable": self.label(v), "driver": "Set by the scenario",
                             "timing": "", "effect": np.nan, "inclusion": np.nan,
                             "per": "", "sample": ""})
                continue
            for t in eq.terms:
                rows.append({
                    "variable": self.label(v),
                    "driver": "Own last year" if t["driver"] == v else self.label(t["driver"]),
                    "timing": "same year" if t["lag"] == 0 else "a year earlier",
                    "effect": t["coef"], "inclusion": t["pip"],
                    "per": f"per {self.unit(t['driver'])}".replace("per pp", "per pp").strip(),
                    "sample": f"{eq.years[0]}–{eq.years[1]}, {eq.n} obs, "
                              f"{eq.countries} {'country' if eq.countries == 1 else 'countries'}"})
        return pd.DataFrame(rows)


def fit_system(store, vintage=None, project_root=None, spec: dict | None = None) -> MacroSystem:
    """Fit every equation of the system on the store's vintage."""
    spec = spec or system_spec()
    meta = spec.get("variables") or {}
    signs = {k: int(v) for k, v in (spec.get("signs") or {}).items()}
    panel = load_panel(store, spec, vintage, project_root)
    order = list(meta)
    pos = {v: i for i, v in enumerate(order)}
    for v, m in meta.items():
        for drv in m.get("same") or []:
            if pos.get(drv, 10 ** 6) >= pos[v]:
                raise ValueError(f"{drv} moves {v} in the same year but comes after it "
                                 "in system.yml; the order must be recursive.")
    eqs = {}
    for v, m in meta.items():
        if not (m.get("same") or m.get("lag")):
            continue
        if v not in panel.columns:
            continue
        eq = fit_equation(panel, v, m, signs, int(spec.get("start_year", 2001)))
        if eq is not None:
            eqs[v] = eq
    return MacroSystem(order=order, meta=meta, equations=eqs, panel=panel,
                       vintage=vintage or store.latest(), signs=signs)
