"""Questions answered through the surrogate, at a cost no simulator campaign could afford.

    from surrogate.analysis import sobol_indices, soil_sampler, monte_carlo
    S = sobol_indices(f, lo, hi, n=2048, log_cols=[7])      # which inputs drive the output?
    mc = monte_carlo(sur, base_case, n=10_000)              # spread of K(f) from soil uncertainty

Sobol indices (variance-based sensitivity). Vary the inputs at random over their ranges and
ask how much of the output's variance each input explains:
  S1_i  first-order index: the share explained by input i alone;
  ST_i  total index: the share in which input i takes part at all, alone or with others.
ST_i - S1_i measures interactions; an input with ST_i ~ 0 can be fixed without loss.
Estimated with the Saltelli scheme: two quasi-random sample matrices A and B, and for each
input a matrix AB_i equal to A with column i taken from B: n (d + 2) model evaluations
(S1: Saltelli et al. 2010; ST: Jansen 1999). With d = 8 soil inputs and n = 2,048 that is
20,480 predictions: about 16 days of single-core MISS3D runs for a 45 x 15 m raft (66 s each),
seconds for the surrogate.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import qmc

from .portable import INPUTS, NOMINAL


def _to_box(U, lo, hi, log_cols=()):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    X = lo + U * (hi - lo)
    for j in log_cols:                       # log-uniform, as g_factor was sampled in dataset v1
        X[:, j] = np.exp(np.log(lo[j]) + U[:, j] * (np.log(hi[j]) - np.log(lo[j])))
    return X


def sobol_indices(f, lo, hi, n=2048, log_cols=(), seed=0, n_boot=200):
    """S1 and ST of every output of f over the box [lo, hi] (inputs independent, uniform;
    log-uniform for log_cols).

    f maps inputs (m, d) to outputs (m, k) (or (m,)). Returns a dict with S1, ST (d, k) and
    their bootstrap standard errors S1_se, ST_se, plus the number of evaluations."""
    d = len(lo)
    U = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
    A, B = _to_box(U[:, :d], lo, hi, log_cols), _to_box(U[:, d:], lo, hi, log_cols)
    AB = np.repeat(A[None], d, axis=0)
    for i in range(d):
        AB[i, :, i] = B[:, i]
    out = f(np.vstack([A, B, AB.reshape(-1, d)]))
    out = out.reshape(len(out), -1)
    out = out - out[:2 * n].mean(axis=0)     # centred: same expectations, far less estimator noise
    fA, fB, fAB = out[:n], out[n:2 * n], out[2 * n:].reshape(d, n, -1)

    def estimate(idx):
        a, b, ab = fA[idx], fB[idx], fAB[:, idx]
        V = np.var(np.vstack([a, b]), axis=0)
        V = np.where(V > 0, V, np.nan)
        s1 = np.mean(b[None] * (ab - a[None]), axis=1) / V
        st = 0.5 * np.mean((a[None] - ab) ** 2, axis=1) / V
        return s1, st

    S1, ST = estimate(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = [estimate(rng.integers(0, n, n)) for _ in range(n_boot)]
    return {"S1": S1, "ST": ST, "S1_se": np.std([b[0] for b in boots], axis=0),
            "ST_se": np.std([b[1] for b in boots], axis=0), "n_eval": len(out)}


def soil_box(sur):
    """The ranges of the 8 soil inputs in the training data (thicknesses, g_factor)."""
    return sur.x_min[:8].copy(), sur.x_max[:8].copy()


def with_footprint(soil, L_x, L_y):
    """Soil inputs (m, 8) + a fixed raft -> full inputs (m, 10)."""
    soil = np.atleast_2d(soil)
    return np.column_stack([soil, np.full(len(soil), float(L_x)), np.full(len(soil), float(L_y))])


def soil_sampler(base=None, cov_thickness=0.10, g_factor_spread=1.25, seed=0):
    """Random soils around a base case: each thickness normal with the given coefficient of
    variation (truncated at 10 % of the base value), g log-normal with 2 standard deviations equal to a
    factor g_factor_spread. Returns draw(n) -> inputs (n, 10). The raft stays fixed."""
    base = dict(NOMINAL if base is None else base)
    x0 = np.array([base[k] for k in INPUTS], float)
    rng = np.random.default_rng(seed)
    s_ln = np.log(g_factor_spread) / 2

    def draw(n):
        X = np.repeat(x0[None], n, axis=0)
        X[:, :7] = np.maximum(x0[:7] * (1 + cov_thickness * rng.standard_normal((n, 7))), 0.1 * x0[:7])
        X[:, 7] = x0[7] * np.exp(s_ln * rng.standard_normal(n))
        return X
    return draw


def monte_carlo(sur, base=None, n=10_000, q=(2.5, 16, 50, 84, 97.5), chunk=2000, **sampler_kw):
    """K(f) for n random soils around `base`: percentiles q of Re K and Im K (and of the static
    values), and the time taken. Returns a dict."""
    draw = soil_sampler(base, **sampler_kw)
    X = draw(n)
    t0 = time.perf_counter()
    K = np.concatenate([sur.predict(X[i:i + chunk]).K for i in range(0, n, chunk)])
    seconds = time.perf_counter() - t0
    return {"X": X, "q": q, "re": np.percentile(K.real, q, axis=0), "im": np.percentile(K.imag, q, axis=0),
            "static": np.percentile(K[:, 0, :].real, q, axis=0), "seconds": seconds, "n": n}
