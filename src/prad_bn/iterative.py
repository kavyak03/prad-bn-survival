"""
Iterative Bayesian Network learning with AUC-driven feature sampling.

Core loop:
1) sample 15-20 genes using adaptive probabilities
2) fit BN (genes + outcome) with constraints (max_parents)
3) prune to nodes connected to outcome
4) refit BN on pruned set
5) infer posteriors and compute AUC
6) update feature sampling probabilities based on performance feedback

This is a toy. The goal is to demonstrate the concept clearly.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import networkx as nx

from .bn_model import fit_bn_discrete, prune_to_outcome_connected, predict_outcome_posteriors
from .evaluate import auc_group_vs_rest


@dataclass
class IterConfig:
    n_iters: int = 25
    subset_size: int = 20
    max_parents: int = 3
    outcome_name: str = "Outcome"
    target_group_for_auc: int = 0
    lr: float = 0.25  # how aggressively to update sampling probs


@dataclass
class IterResult:
    best_auc: float
    best_features: list[int]
    best_names: list[str]
    history: list[dict]


def run_iterative_bn(
    X_disc: np.ndarray,
    y: np.ndarray,
    seed: int = 7,
    cfg: IterConfig = IterConfig(),
) -> IterResult:
    """
    Args:
      X_disc: (n_samples, n_genes) int
      y: (n_samples,) int (0..K-1)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_genes = X_disc.shape

    # adaptive sampling probabilities over genes
    probs = np.ones(n_genes, dtype=float) / n_genes

    best_auc = -1.0
    best_features = []
    best_names = []
    history: list[dict] = []

    # create combined data function
    def make_data(feat_idx: np.ndarray) -> tuple[np.ndarray, list[str], int]:
        names = [f"G{j}" for j in feat_idx] + [cfg.outcome_name]
        outcome_idx = len(names) - 1
        data = np.column_stack([X_disc[:, feat_idx], y])
        return data, names, outcome_idx

    for t in range(cfg.n_iters):
        feat_idx = rng.choice(n_genes, size=cfg.subset_size, replace=False, p=probs)
        data, names, outcome_idx = make_data(feat_idx)

        # fit BN on subset
        fit = fit_bn_discrete(data, state_names=names, max_parents=cfg.max_parents, seed=int(seed + t))

        # prune to outcome-connected nodes (including outcome)
        pruned_data, pruned_names = prune_to_outcome_connected(data, names, fit.graph, cfg.outcome_name)

        # re-fit BN on pruned set (often smaller and more interpretable)
        pruned_fit = fit_bn_discrete(
            pruned_data,
            state_names=pruned_names,
            max_parents=cfg.max_parents,
            seed=int(seed + 1000 + t),
        )

        # infer posteriors
        out_idx2 = pruned_names.index(cfg.outcome_name)
        posts = predict_outcome_posteriors(pruned_fit.model, pruned_data, outcome_idx=out_idx2)

        # evaluate AUC (group vs rest)
        auc = auc_group_vs_rest(y, posts, group=cfg.target_group_for_auc)

        # track best
        improved = auc > best_auc
        if improved:
            best_auc = auc
            # map pruned gene names back to original indices
            used_gene_names = [n for n in pruned_names if n != cfg.outcome_name]
            used_gene_idx = [int(n[1:]) for n in used_gene_names]  # "G123" -> 123
            best_features = used_gene_idx
            best_names = pruned_names

        # update sampling probabilities (simple bandit-like update)
        # genes present in this iteration get a small reward if improved, else a small penalty
        delta = cfg.lr * (1.0 if improved else -0.5)
        probs[feat_idx] = np.clip(probs[feat_idx] * np.exp(delta), 1e-8, 1.0)
        probs = probs / probs.sum()

        history.append({
            "iter": t,
            "auc": float(auc),
            "improved": bool(improved),
            "subset_size": int(cfg.subset_size),
            "pruned_vars": int(len(pruned_names)),
            "kept_genes": int(len(pruned_names) - 1),
        })

    return IterResult(best_auc=best_auc, best_features=best_features, best_names=best_names, history=history)
