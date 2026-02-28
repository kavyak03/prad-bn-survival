"""DoWhy-style ATE estimation (lightweight, sklearn-based).

We treat 'treatment' as setting a module (or gene) to high vs low and estimate:
  ATE = E[Y(1) - Y(0)]
using observational adjustment on available confounders (clinical covariates).

This is not a full doWhy dependency; it's a doWhy-*style* pipeline:
- Define treatment T, outcome Y, confounders Z
- Identify backdoor adjustment (assumed: adjust for provided Z)
- Estimate ATE via AIPW (doubly robust)
- Refute via placebo (shuffle T)

Outputs are intended for *credible interview demos*, not clinical claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class AteResult:
    ate: float
    se: float
    n: int
    method: str
    placebo_ate: float


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def estimate_ate_aipw(
    T: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    clip: float = 0.01,
    seed: int = 7,
) -> AteResult:
    """Estimate ATE with Augmented IPW (binary treatment, binary outcome)."""
    rng = np.random.default_rng(seed)
    T = T.astype(int).ravel()
    Y = Y.astype(float).ravel()
    n = len(Y)
    if Z is None:
        Z = np.zeros((n, 1), dtype=float)

    # Propensity model e(Z) = P(T=1|Z)
    prop = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    prop.fit(Z, T)
    e = prop.predict_proba(Z)[:, 1]
    e = np.clip(e, clip, 1 - clip)

    # Outcome models m1(Z)=E[Y|T=1,Z], m0(Z)=E[Y|T=0,Z]
    out1 = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    out0 = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("lr", LogisticRegression(max_iter=2000, solver="lbfgs"))])

    out1.fit(Z[T == 1], Y[T == 1])
    out0.fit(Z[T == 0], Y[T == 0])

    m1 = out1.predict_proba(Z)[:, 1]
    m0 = out0.predict_proba(Z)[:, 1]

    # AIPW influence function
    psi = (m1 - m0) + (T * (Y - m1) / e) - ((1 - T) * (Y - m0) / (1 - e))
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))

    # placebo refuter: shuffle T
    T_shuf = rng.permutation(T)
    prop.fit(Z, T_shuf)
    e2 = np.clip(prop.predict_proba(Z)[:, 1], clip, 1 - clip)
    out1.fit(Z[T_shuf == 1], Y[T_shuf == 1])
    out0.fit(Z[T_shuf == 0], Y[T_shuf == 0])
    m1b = out1.predict_proba(Z)[:, 1]
    m0b = out0.predict_proba(Z)[:, 1]
    psi_b = (m1b - m0b) + (T_shuf * (Y - m1b) / e2) - ((1 - T_shuf) * (Y - m0b) / (1 - e2))
    placebo = float(np.mean(psi_b))

    return AteResult(ate=ate, se=se, n=n, method="AIPW(logistic)", placebo_ate=placebo)


def treatment_from_bins(x_bins: np.ndarray, low: int = 0, high: int = 2) -> np.ndarray:
    """Define binary treatment by comparing extreme bins and dropping middle bins.

    Returns array with same length; middle bins marked as -1.
    """
    x = x_bins.astype(int).ravel()
    T = np.full_like(x, -1)
    T[x == low] = 0
    T[x == high] = 1
    return T
