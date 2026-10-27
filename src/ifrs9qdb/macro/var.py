"""An earlier macro model, kept for comparison: scenarios now take their
linked effects from ``macro.system`` (a recursive system estimated equation
by equation on outturns).

A Bayesian VAR(1) on the core concepts' yearly values, each standardised by
its own history:

    x(t) = c + A x(t-1) + e(t),     e ~ N(0, Sigma)

A Minnesota prior pulls each equation towards its own lag (at the concept's
``persistence``) and the other variables' lags towards zero, which is what
makes a model of fifteen variables estimable on thirty years. A series that
starts late (non-oil GDP from 2015) is filled back by EM on the others before
the fit, and a concept with too little history to estimate is carried as its
own AR(1) at its catalogue persistence, related to nothing.

A scenario lives in DEVIATIONS from the baseline path, d(t) = A d(t-1) + e(t),
d(0) = 0 -- the intercept and the baseline drop out, so the model never needs
to agree with the IMF's forecast, only with history's co-movements. Over H
years the stacked deviations are D = M e with M's blocks A^(h-j), so

    Cov(D) = M (I_H (x) Sigma) M'

and the effects of a set of fixed paths d_S on the rest are the conditional mean

    E[D_u | D_S = d_S] = Cov_uS Cov_SS^-1 d_S

-- the shocked variables keep exactly the path given, everything else moves
as history says it moves with them. The same covariance gives the scenario's
severity: the distance m = sqrt(d_S' Cov_SS^-1 d_S) is how many standard
deviations the scenario sits from the baseline along its own direction, and
1 - Phi(m) the chance of a move at least that severe in that direction.

Every number here is a deterministic function of the store's vintage, so R
and Python agree to rounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .catalogue import concepts as _concepts
from .prng import normals, uniforms

__all__ = ["MacroModel", "fit_macro_model", "em_fill", "minnesota_var1"]


def em_fill(X: np.ndarray, iters: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Missing cells of a (T x k) matrix filled with their conditional mean
    under a multivariate normal fitted by EM. Columns are assumed
    standardised."""
    X = np.array(X, dtype=float)
    miss = ~np.isfinite(X)
    if not miss.any():
        return X
    Z = np.where(miss, 0.0, X)
    for _ in range(iters):
        mu = Z.mean(axis=0)
        C = np.cov(Z, rowvar=False, bias=True) + 1e-6 * np.eye(Z.shape[1])
        Znew = Z.copy()
        for t in np.where(miss.any(axis=1))[0]:
            m, o = miss[t], ~miss[t]
            if not o.any():
                Znew[t, m] = mu[m]
                continue
            Coo = C[np.ix_(o, o)]
            Cmo = C[np.ix_(m, o)]
            Znew[t, m] = mu[m] + Cmo @ np.linalg.solve(Coo, Z[t, o] - mu[o])
        step = np.max(np.abs(Znew - Z))
        Z = Znew
        if step < tol:
            break
    return Z


def minnesota_var1(X: np.ndarray, persistence: np.ndarray, tightness: float = 0.2,
                   cross: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior-mean VAR(1) coefficients under a Minnesota prior, equation
    by equation. Returns (A, c, residuals). X is (T x k), standardised."""
    T, k = X.shape
    Y, L = X[1:], X[:-1]
    R = np.column_stack([np.ones(T - 1), L])
    A = np.zeros((k, k))
    c = np.zeros(k)
    E = np.zeros_like(Y)
    for i in range(k):
        y = Y[:, i]
        # residual scale from the variable's own AR(1)
        Ri = np.column_stack([np.ones(T - 1), L[:, i]])
        b = np.linalg.lstsq(Ri, y, rcond=None)[0]
        s2 = max(float(np.mean((y - Ri @ b) ** 2)), 1e-6)
        prior_mean = np.zeros(k + 1)
        prior_mean[1 + i] = float(persistence[i])
        prior_var = np.full(k + 1, (tightness * cross) ** 2)
        prior_var[1 + i] = tightness ** 2
        prior_var[0] = 100.0
        P = np.diag(1.0 / prior_var)
        post = np.linalg.solve(R.T @ R / s2 + P, R.T @ y / s2 + P @ prior_mean)
        c[i] = post[0]
        A[i] = post[1:]
        E[:, i] = y - R @ post
    return A, c, E


@dataclass
class MacroModel:
    """The fitted model. ``concepts`` are every concept it knows, the VAR's
    first; ``sd`` and ``mean`` the native-unit scaling; ``A`` and ``Sigma``
    in standard deviations; ``resid`` the VAR's standardised residuals (one
    row a year) for a bootstrap."""
    concepts: list[str]
    var_concepts: list[str]
    mean: dict
    sd: dict
    A: np.ndarray
    Sigma: np.ndarray
    resid: np.ndarray
    years: tuple = (None, None)
    info: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.concepts)

    def index(self, c: str) -> int:
        return self.concepts.index(c)

    def stacked(self, H: int) -> tuple[np.ndarray, np.ndarray]:
        """(M, Cov) for H years; the deviation of concept i in year h is
        entry h*k + i."""
        k = self.k
        Apow = [np.eye(k)]
        for _ in range(H - 1):
            Apow.append(self.A @ Apow[-1])
        Mm = np.zeros((H * k, H * k))
        for h in range(H):
            for j in range(h + 1):
                Mm[h * k:(h + 1) * k, j * k:(j + 1) * k] = Apow[h - j]
        big = np.kron(np.eye(H), self.Sigma)
        return Mm, Mm @ big @ Mm.T

    def _stack(self, paths: dict, H: int) -> tuple[list[int], np.ndarray]:
        idx, vals = [], []
        for c, path in paths.items():
            if c not in self.concepts:
                continue
            i = self.index(c)
            p = np.asarray(path, dtype=float)[:H]
            for h in range(H):
                idx.append(h * self.k + i)
                vals.append((p[h] if h < p.size else 0.0) / self.sd[c])
        return idx, np.asarray(vals)

    def condition(self, paths: dict, H: int = 5) -> dict:
        """Every concept's deviation path given the fixed ones (native units).
        Concepts given keep exactly their path."""
        idx, x = self._stack(paths, H)
        out = {c: np.zeros(H) for c in self.concepts}
        if not idx:
            return out
        _, C = self.stacked(H)
        Css = C[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx))
        w = np.linalg.lstsq(Css, x, rcond=None)[0]
        full = C[:, idx] @ w
        for c in self.concepts:
            i = self.index(c)
            out[c] = np.array([full[h * self.k + i] for h in range(H)]) * self.sd[c]
        for c, p in paths.items():          # exactly as given
            if c in out:
                p = np.asarray(p, dtype=float)[:H]
                out[c] = np.concatenate([p, np.zeros(H - p.size)]) if p.size < H else p
        return out

    def severity(self, paths: dict, H: int = 5) -> dict:
        """How far the fixed paths sit from the baseline: the distance m in
        standard deviations, the one-sided probability 1 - Phi(m) of a move
        at least as severe in that direction, and its reciprocal."""
        idx, x = self._stack(paths, H)
        if not idx or not np.any(np.abs(x) > 0):
            return {"distance": 0.0, "probability": 1.0, "one_in": 1.0}
        _, C = self.stacked(H)
        Css = C[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx))
        m2 = float(x @ np.linalg.lstsq(Css, x, rcond=None)[0])
        m = float(np.sqrt(max(m2, 0.0)))
        p = float(norm.sf(m))
        return {"distance": m, "probability": p,
                "one_in": float(1.0 / p) if p > 0 else float("inf")}

    def simulate(self, n: int, H: int = 5, seed: int = 20261023,
                 bootstrap: bool = True) -> np.ndarray:
        """``n`` deviation paths, (n x H x k) in native units. With
        ``bootstrap`` each year's shock vector is a past year's residual
        vector drawn at random (keeping history's fat tails and its
        co-movement); concepts outside the VAR draw their own normal."""
        k, kv = self.k, len(self.var_concepts)
        L = np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(k))
        u = uniforms(n * H, seed)
        z = normals(n * H * k, seed + 1).reshape(n, H, k)
        D = np.zeros((n, H, k))
        T = self.resid.shape[0]
        use_boot = bootstrap and kv > 0 and T >= 5
        for h in range(H):
            if use_boot:
                pick = np.minimum((u[h::H][:n] * T).astype(int), T - 1)
                e = np.zeros((n, k))
                e[:, :kv] = self.resid[pick][:, :kv]
                e[:, kv:] = (z[:, h, :] @ L.T)[:, kv:]
            else:
                e = z[:, h, :] @ L.T
            prev = D[:, h - 1, :] if h else np.zeros((n, k))
            D[:, h, :] = prev @ self.A.T + e
        scale = np.array([self.sd[c] for c in self.concepts])
        return D * scale


def fit_macro_model(store, concepts=None, vintage=None, project_root=None,
                    min_years: int = 12, tightness: float = 0.2, cross: float = 0.5,
                    start_year: int = 1990, extra=None) -> MacroModel:
    """Fit the model on the store's actual yearly values of ``concepts``
    (the catalogue's core ones by default). ``extra`` are concepts carried
    as independent AR(1)s whatever their history -- every concept a
    scenario shocks must be in the model to have a severity."""
    cat = _concepts(project_root)
    core = list(concepts or cat.loc[cat["core"], "concept"])
    extra = [c for c in (extra or []) if c not in core]
    pers = dict(zip(cat["concept"], cat["persistence"]))
    mat = store.concept_matrix(core + extra, vintage, project_root)
    var_c, indep = [], []
    for c in core + extra:
        n = int(mat[c].notna().sum()) if c in mat.columns else 0
        (var_c if (c in core and n >= min_years) else indep).append(c)
    mean, sd = {}, {}
    for c in var_c + indep:
        s = mat[c].dropna() if c in mat.columns else pd.Series(dtype=float)
        mean[c] = float(s.mean()) if len(s) else 0.0
        sd[c] = float(store.concept_sd(c, vintage, project_root))
        if not np.isfinite(sd[c]) or sd[c] <= 0:
            sd[c] = 1.0

    k = len(var_c) + len(indep)
    A = np.zeros((k, k))
    S = np.zeros((k, k))
    resid = np.zeros((0, len(var_c)))
    y0 = y1 = None
    if var_c:
        m = mat[var_c]
        have = m.notna().sum(axis=1)
        years = m.index[(have >= max(1, len(var_c) // 2)) & (m.index >= start_year)]
        y0, y1 = int(years.min()), int(years.max())
        m = m.loc[y0:y1]
        X = np.column_stack([(m[c].to_numpy() - mean[c]) / sd[c] for c in var_c])
        Xf = em_fill(X)
        Av, _, E = minnesota_var1(Xf, np.array([pers.get(c, 0.0) for c in var_c]),
                                  tightness, cross)
        rad = max(abs(np.linalg.eigvals(Av)))
        if rad >= 0.98:                      # deviations must die out
            Av = Av * (0.95 / rad)
        Sv = E.T @ E / E.shape[0]
        Sv = 0.9 * Sv + 0.1 * np.diag(np.diag(Sv))
        kv = len(var_c)
        A[:kv, :kv] = Av
        S[:kv, :kv] = Sv
        resid = E
    for j, c in enumerate(indep):
        i = len(var_c) + j
        rho = float(np.clip(pers.get(c, 0.0), 0.0, 0.95))
        A[i, i] = rho
        S[i, i] = 1.0 - rho ** 2
    return MacroModel(concepts=var_c + indep, var_concepts=var_c, mean=mean, sd=sd,
                      A=A, Sigma=S, resid=resid, years=(y0, y1),
                      info={"independent": indep, "min_years": min_years,
                            "tightness": tightness, "cross": cross,
                            "vintage": vintage or store.latest()})
