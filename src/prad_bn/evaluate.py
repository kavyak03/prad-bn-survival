"""
Evaluation helpers:
- Binary AUC (group vs rest) for stability
- Optional multiclass AUC (one-vs-rest)
- Kaplan–Meier curves + CoxPH (sanity checks) on simulated data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter


def auc_group_vs_rest(y_true: np.ndarray, posteriors: np.ndarray, group: int = 0) -> float:
    """
    Stable binary AUC: group vs rest using posterior prob of `group`.
    """
    y_bin = (y_true == group).astype(int)
    # posterior column for that group (assumes labels 0..K-1)
    p = posteriors[:, group]
    return float(roc_auc_score(y_bin, p))


def auc_multiclass_ovr(y_true: np.ndarray, posteriors: np.ndarray) -> float:
    """
    One-vs-rest multiclass AUC (macro).
    """
    return float(roc_auc_score(y_true, posteriors, multi_class="ovr", average="macro"))


def plot_km_by_group(time_days: np.ndarray, event: np.ndarray, groups: np.ndarray, outpath: str) -> None:
    """
    Kaplan–Meier survival curves by risk group label.
    """
    kmf = KaplanMeierFitter()
    plt.figure()
    for g in sorted(np.unique(groups)):
        mask = groups == g
        kmf.fit(time_days[mask], event_observed=event[mask], label=f"group {g}")
        kmf.plot_survival_function()

    plt.xlabel("Days")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier curves by discretized risk group (toy)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def fit_coxph(time_days: np.ndarray, event: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """
    Cox proportional hazards on a single covariate (risk group).
    Returns model summary dataframe.
    """
    df = pd.DataFrame({"time": time_days, "event": event, "risk_group": groups})
    cph = CoxPHFitter()
    cph.fit(df, duration_col="time", event_col="event")
    return cph.summary
