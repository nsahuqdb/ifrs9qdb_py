"""A random stream R and Python draw identically.

The Park-Miller "minimal standard" generator, x(n+1) = 48271 x(n) mod
(2^31 - 1), computed with a multiplication split into 16-bit halves so every
intermediate is an exact integer below 2^53 -- exact in a double, so R's
numeric vectors give the same stream as numpy's. A Monte Carlo run is then
the same path for path in both engines and a parity test can compare them.

Blocks are generated vectorised: the first block sequentially, every later
block as the first block times a^(k B) mod m.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

__all__ = ["M", "A", "mulmod", "uniforms", "normals"]

M = 2147483647.0
A = 48271.0
_B = 4096


def mulmod(a, x):
    """(a * x) mod M for integer-valued doubles below M, exactly."""
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    xh = np.floor(x / 65536.0)
    xl = x - xh * 65536.0
    t = np.mod(a * xh, M)
    return np.mod(t * 65536.0 + a * xl, M)


def _powmod(a: float, n: int) -> float:
    r, b = 1.0, float(a)
    while n > 0:
        if n & 1:
            r = float(mulmod(r, b))
        b = float(mulmod(b, b))
        n >>= 1
    return r


def uniforms(n: int, seed: int) -> np.ndarray:
    """``n`` uniforms in (0, 1) from ``seed`` (1 <= seed < 2^31 - 1)."""
    s = float(int(seed) % int(M)) or 1.0
    n = int(n)
    first = np.empty(min(n, _B))
    x = s
    for i in range(first.size):
        x = float(mulmod(A, x))
        first[i] = x
    out = [first]
    if n > _B:
        step = _powmod(A, _B)
        mult = step
        k = 1
        while k * _B < n:
            out.append(mulmod(mult, first))
            mult = float(mulmod(mult, step))
            k += 1
    return np.concatenate(out)[:n] / M


def normals(n: int, seed: int) -> np.ndarray:
    return ndtri(uniforms(n, seed))
