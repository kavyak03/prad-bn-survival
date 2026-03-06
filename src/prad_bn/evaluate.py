from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def auc_group4_vs_rest(y_true_group: np.ndarray, y_score: np.ndarray, group: int | None = None) -> float:
    """
    Compute ROC AUC for worst survival group vs rest.

    Parameters
    ----------
    y_true_group : np.ndarray
        Integer group labels (typically 0..4).
    y_score : np.ndarray
        Predicted risk/probability for the worst group.
    group : int | None
        Positive class group. If None, uses max(y_true_group).

    Returns
    -------
    float
        ROC AUC, or 0.5 if undefined.
    """
    y_true_group = np.asarray(y_true_group).astype(int).ravel()
    y_score = np.asarray(y_score).astype(float).ravel()

    if y_true_group.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true_group={y_true_group.shape[0]} vs y_score={y_score.shape[0]}"
        )

    if group is None:
        group = int(np.max(y_true_group))

    y_bin = (y_true_group == group).astype(int)

    if len(np.unique(y_bin)) < 2:
        return 0.5

    try:
        return float(roc_auc_score(y_bin, y_score))
    except Exception:
        return 0.5


def plot_auc_over_iters(aucs: Sequence[float], outpath: str) -> None:
    aucs = np.asarray(list(aucs), dtype=float)

    plt.figure(figsize=(7.0, 4.2))
    plt.plot(np.arange(1, len(aucs) + 1), aucs, linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("AUC (worst-group vs rest)")
    plt.title("AUC over iterative BN sampling")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_sampling_probs(probs: np.ndarray, top_k: int, outpath: str) -> None:
    probs = np.asarray(probs, dtype=float).ravel()
    if probs.size == 0:
        return

    k = min(int(top_k), probs.size)
    idx = np.argsort(probs)[::-1][:k]
    vals = probs[idx]

    plt.figure(figsize=(8.0, 4.5))
    plt.bar(np.arange(k), vals)
    plt.xticks(np.arange(k), [f"G{i}" for i in idx], rotation=60, ha="right")
    plt.ylabel("Sampling probability")
    plt.title(f"Top {k} feature sampling probabilities")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_km_by_group(time_days: np.ndarray, event: np.ndarray, km_group: np.ndarray, outpath: str) -> None:
    """
    Plot Kaplan-Meier curves by discrete group.
    Falls back to a placeholder image if lifelines is unavailable.
    """
    t = np.asarray(time_days, dtype=float).ravel()
    e = np.asarray(event, dtype=int).ravel()
    g = np.asarray(km_group, dtype=int).ravel()

    try:
        from lifelines import KaplanMeierFitter
    except Exception as exc:
        plt.figure(figsize=(7.0, 4.0))
        plt.text(
            0.5,
            0.5,
            f"lifelines not available:\n{exc}",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        return

    plt.figure(figsize=(7.2, 4.8))
    kmf = KaplanMeierFitter()

    for grp in np.unique(g):
        mask = g == grp
        if mask.sum() < 5:
            continue
        kmf.fit(t[mask], event_observed=e[mask], label=f"Group {int(grp)} (n={int(mask.sum())})")
        kmf.plot(ci_show=False)

    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.title("Kaplan-Meier curves by risk group")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_right_skewed_expression(
    expr: np.ndarray,
    outpath: str,
    max_points: int = 250_000,
    seed: int = 7,
) -> None:
    """
    Histogram showing right-skewed RNA-seq expression distribution.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (samples x genes).
    outpath : str
        Output image path.
    max_points : int
        Max number of expression values to sample for plotting speed.
    seed : int
        RNG seed for reproducible subsampling.
    """
    rng = np.random.default_rng(seed)

    X = np.asarray(expr, dtype=float)
    flat = X.reshape(-1)
    flat = flat[np.isfinite(flat)]

    if flat.size == 0:
        return

    if flat.size > max_points:
        flat = rng.choice(flat, size=max_points, replace=False)

    vals = np.log1p(np.maximum(flat, 0.0))

    plt.figure(figsize=(7.0, 4.5))
    plt.hist(vals, bins=80)
    plt.xlabel("log1p(TPM)")
    plt.ylabel("Count")
    plt.title("Right-skewed RNA-seq expression distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_block_correlated_modules(
    expr: np.ndarray,
    outpath: str,
    n_genes: int = 350,
    seed: int = 7,
) -> None:
    """
    Correlation heatmap for top-variable genes to visualize block/module structure.

    Parameters
    ----------
    expr : np.ndarray
        Expression matrix (samples x genes).
    outpath : str
        Output image path.
    n_genes : int
        Number of top-variable genes to include in the heatmap.
    seed : int
        Included for API consistency / future reproducibility extensions.
    """
    _ = seed  # currently unused

    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(expr, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 3:
        return

    # 1) Compute per-gene variance safely
    var = np.nanvar(X, axis=0)
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) Pick top variable genes
    k = min(int(n_genes), X.shape[1])
    idx = np.argsort(var)[::-1][:k]
    Xg = X[:, idx]

    # 3) Drop any genes that are entirely non-finite
    keep_cols = np.isfinite(Xg).any(axis=0)
    Xg = Xg[:, keep_cols]
    if Xg.shape[1] < 2:
        return

    # 4) Impute remaining NaNs using column medians
    #    If a median itself is NaN, replace it with 0
    with np.errstate(all="ignore"):
        col_med = np.nanmedian(Xg, axis=0)
    col_med = np.nan_to_num(col_med, nan=0.0, posinf=0.0, neginf=0.0)

    bad = ~np.isfinite(Xg)
    if np.any(bad):
        Xg = Xg.copy()
        Xg[bad] = np.take(col_med, np.where(bad)[1])

    # 5) Remove zero-variance columns after imputation
    sd = np.std(Xg, axis=0)
    keep_cols = sd > 0
    Xg = Xg[:, keep_cols]
    if Xg.shape[1] < 2:
        return

    # 6) Z-score for stable correlation computation
    mu = np.mean(Xg, axis=0)
    sd = np.std(Xg, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xz = (Xg - mu) / sd

    # 7) Gene-gene correlation matrix
    C = np.corrcoef(Xz, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

    # 8) Try hierarchical clustering for nicer block ordering
    order = np.arange(C.shape[0])
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        D = 1.0 - np.clip(C, -1.0, 1.0)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        Z = linkage(squareform(D, checks=False), method="average")
        order = leaves_list(Z)
    except Exception:
        # Fallback: leave unclustered if scipy unavailable
        pass

    C_ord = C[np.ix_(order, order)]

    # 9) Plot heatmap
    plt.figure(figsize=(7.0, 6.0))
    im = plt.imshow(
        C_ord,
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.title(f"Gene correlation heatmap (top-variable genes, n={C_ord.shape[0]})")
    plt.xlabel("Genes (clustered)")
    plt.ylabel("Genes (clustered)")
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()