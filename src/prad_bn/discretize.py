"""
Discretization utilities.

We discretize:
- survival time -> 5 Kaplan–Meier-style risk strata via quantile bins
- gene expression -> binned integers for discrete Bayesian Networks

Note: bn-cuts is mentioned in the project narrative as a supervised BN-oriented discretizer.
In this toy repo we provide:
1) A simple quantile discretizer for genes (robust, easy)
2) A hook where you can plug in bn-cuts if you want
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def survival_to_km_groups(time_days: np.ndarray, q: int = 5) -> np.ndarray:
    """
    Convert survival time into q risk strata using quantile bins.
    Lower time -> higher risk group (by default we define group 4 as worst survival).
    """
    s = pd.Series(time_days)
    # qcut assigns 0..q-1 from low to high; we invert so that "high risk" means lower survival
    groups = pd.qcut(s, q=q, labels=False, duplicates="drop").to_numpy()
    # invert so: short survival -> larger label
    groups = (groups.max() - groups).astype(int)
    return groups


def discretize_expression_quantile(expr: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Discretize continuous expression into n_bins using quantiles.
    Returns integer matrix of shape (n_samples, n_genes) with values 0..n_bins-1.
    """
    # KBinsDiscretizer with strategy='quantile' works feature-wise
    enc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    disc = enc.fit_transform(expr).astype(int)
    return disc


def maybe_supervised_bncuts(expr: np.ndarray, y: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Optional supervised discretization placeholder.
    If you install and prefer bn-cuts (or MDLPC-like) discretization,
    replace this implementation with that library call.

    For now, we fall back to quantile discretization.
    """
    # TODO: plug bn-cuts here if desired
    return discretize_expression_quantile(expr, n_bins=n_bins)
