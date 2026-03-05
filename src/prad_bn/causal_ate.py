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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd

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

    T = np.asarray(T).astype(int).ravel()
    Y = np.asarray(Y).astype(float).ravel()
    n = len(Y)

    if Z is None:
        # No confounders: use an empty design matrix (n x 0)
        Z = np.zeros((n, 0), dtype=float)

    # --- Guardrails ---
    if len(np.unique(T)) < 2:
        raise ValueError("Treatment T has <2 classes (all 0s or all 1s). Cannot estimate ATE.")
    if np.sum(T == 1) < 2 or np.sum(T == 0) < 2:
        raise ValueError("Not enough samples in one treatment arm after filtering.")

    # Build models consistently for both propensity and outcome models
    if isinstance(Z, pd.DataFrame):
        # Identify numeric vs categorical columns
        num_cols = [c for c in Z.columns if pd.api.types.is_numeric_dtype(Z[c])]
        cat_cols = [c for c in Z.columns if c not in num_cols]

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        pre = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, num_cols),
                ("cat", categorical_pipe, cat_cols),
            ],
            remainder="drop",
        )

        prop = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        out1 = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        out0 = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        Z_fit = Z  # keep as DF for ColumnTransformer

    else:
        # Numpy / array-like: median-impute then scale for each model
        Z_arr = np.asarray(Z, dtype=float)

        # If Z came in as (n,) make it (n,1)
        if Z_arr.ndim == 1:
            Z_arr = Z_arr.reshape(-1, 1)

        # Impute once, then models can scale
        imp = SimpleImputer(strategy="median")
        Z_imp = imp.fit_transform(Z_arr)

        prop = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        out1 = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        out0 = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ])

        Z_fit = Z_imp  # numeric matrix, no NaNs now

    # --- Propensity model e(Z)=P(T=1|Z) ---
    prop.fit(Z_fit, T)
    e = prop.predict_proba(Z_fit)[:, 1]
    e = np.clip(e, clip, 1 - clip)

    # --- Outcome models m1(Z)=E[Y|T=1,Z], m0(Z)=E[Y|T=0,Z] ---
    # Fit on each arm
    out1.fit(Z_fit[T == 1], Y[T == 1])
    out0.fit(Z_fit[T == 0], Y[T == 0])

    m1 = out1.predict_proba(Z_fit)[:, 1]
    m0 = out0.predict_proba(Z_fit)[:, 1]

    # --- AIPW influence function ---
    psi = (m1 - m0) + (T * (Y - m1) / e) - ((1 - T) * (Y - m0) / (1 - e))
    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))

    # --- Placebo refuter: shuffle T ---
    T_shuf = rng.permutation(T)

    if len(np.unique(T_shuf)) < 2 or np.sum(T_shuf == 1) < 2 or np.sum(T_shuf == 0) < 2:
        placebo = float("nan")
    else:
        prop.fit(Z_fit, T_shuf)
        e2 = np.clip(prop.predict_proba(Z_fit)[:, 1], clip, 1 - clip)

        out1.fit(Z_fit[T_shuf == 1], Y[T_shuf == 1])
        out0.fit(Z_fit[T_shuf == 0], Y[T_shuf == 0])

        m1b = out1.predict_proba(Z_fit)[:, 1]
        m0b = out0.predict_proba(Z_fit)[:, 1]

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
