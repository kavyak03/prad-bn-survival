
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def auc_group4_vs_rest(y_true: np.ndarray, prob_group4: np.ndarray) -> float:
    """Binary AUC: worst-risk group (max label, typically 4) vs rest."""
    worst = int(np.max(y_true))
    y_bin = (y_true == worst).astype(int)
    return float(roc_auc_score(y_bin, prob_group4))


def km_step_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple Kaplan–Meier step function estimator.
    Returns (t, S(t)) arrays suitable for plt.step(where="post").
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # sort by time
    order = np.argsort(times)
    t = times[order]
    e = events[order]

    uniq_times = np.unique(t)
    n_at_risk = len(t)
    surv = 1.0
    T = [0.0]
    S = [1.0]

    for ut in uniq_times:
        # deaths at time ut
        d = int(np.sum((t == ut) & (e == 1)))
        # censored at time ut (doesn't directly change survival, but reduces risk set after ut)
        c = int(np.sum((t == ut) & (e == 0)))

        if n_at_risk > 0 and d > 0:
            surv *= (1.0 - d / n_at_risk)
            T.append(float(ut))
            S.append(float(surv))

        # update risk set after this time
        n_at_risk -= (d + c)

    return np.array(T), np.array(S)


def plot_km_by_group(time_days: np.ndarray, event: np.ndarray, km_group: np.ndarray, outpath: str) -> None:
    """
    Plot KM curves by discretized KM group (toy sanity check visualization).
    """
    plt.figure()
    groups = np.unique(km_group)
    for g in groups:
        mask = km_group == g
        T, S = km_step_curve(time_days[mask], event[mask])
        plt.step(T, S, where="post", label=f"Group {int(g)} (n={mask.sum()})")
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.title("Toy Kaplan–Meier curves by discretized risk group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_auc_over_iters(aucs: List[float], outpath: str) -> None:
    plt.figure()
    x = np.arange(1, len(aucs) + 1)
    plt.plot(x, aucs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("AUC (worst group vs rest)")
    plt.title("AUC over iterations (worst group vs rest)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_sampling_probs(prob: np.ndarray, top_k: int, outpath: str) -> None:
    idx = np.argsort(prob)[::-1][:top_k]
    plt.figure(figsize=(10, 4))
    plt.bar([f"G{i}" for i in idx], prob[idx])
    plt.xlabel("Gene")
    plt.ylabel("Sampling probability")
    plt.title(f"Top-{top_k} genes by sampling probability")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
