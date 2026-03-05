
"""
Iterative Bayesian Network learning with AUC-driven feature sampling.

Demo BN backend: Naive Bayes BN (Outcome -> genes) using pgmpy DiscreteBayesianNetwork.
Core idea: feedback-driven adaptive feature sampling loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from prad_bn.evaluate import auc_group4_vs_rest


@dataclass
class IterConfig:
    n_iters: int = 25
    subset_size: int = 20
    outcome_name: str = "Outcome"

    # Stabilized adaptive sampler knobs (prevent probability collapse)
    temperature: float = 1.0     # higher = more exploration
    epsilon: float = 0.05        # epsilon-greedy mixing with uniform
    p_min: float = 1e-4          # probability floor
    ema_alpha: float = 0.15      # smoothing for feature scores
    reward_scale: float = 6.0    # how strongly AUC reward updates scores

    # BN parameter smoothing
    prior_type: str = "BDeu"
    equivalent_sample_size: int = 5


@dataclass
class IterResult:
    aucs: List[float]
    best_auc: float
    best_features: List[int]
    best_model_edges: List[Tuple[str, str]]
    final_sampling_probs: np.ndarray


def _naive_bayes_structure(
    columns: List[str],
    outcome: str,
    covariate_cols: Optional[List[str]] = None,
) -> DiscreteBayesianNetwork:
    """Create a simple BN structure.

    Default demo structure is Naive Bayes: Outcome -> genes.

    If covariates are provided, treat them as *confounders* for outcome by adding:
      covariate -> Outcome
    while keeping:
      Outcome -> genes

    This makes the BN prediction for Outcome condition on both genes and covariates.
    """
    covariate_cols = covariate_cols or []
    cov_set = set(covariate_cols)
    gene_cols = [c for c in columns if c != outcome and c not in cov_set]
    edges = [(outcome, g) for g in gene_cols] + [(z, outcome) for z in covariate_cols]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(columns)
    return model


def _prune_to_outcome_connected(model: DiscreteBayesianNetwork, outcome: str) -> DiscreteBayesianNetwork:
    G = nx.Graph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(model.edges())
    if outcome not in G:
        return model
    comp = nx.node_connected_component(G, outcome)
    pruned_edges = [(u, v) for (u, v) in model.edges() if u in comp and v in comp]
    pruned = DiscreteBayesianNetwork(pruned_edges)
    pruned.add_nodes_from(list(comp))
    return pruned


def _infer_prob_worst(model: DiscreteBayesianNetwork, df: pd.DataFrame, outcome: str) -> np.ndarray:
    infer = VariableElimination(model)
    probs = np.zeros(df.shape[0], dtype=float)
    worst = int(np.max(df[outcome].astype(int)))

    for i, row in df.iterrows():
        evidence = {col: int(row[col]) for col in df.columns if col != outcome}
        try:
            q = infer.query(variables=[outcome], evidence=evidence, show_progress=False)
            probs[i] = float(q.values[worst]) if worst < len(q.values) else float(q.values[-1])
        except Exception:
            q = infer.query(variables=[outcome], show_progress=False)
            probs[i] = float(q.values[-1])
    return probs


def run_iterative_bn(
    X_disc: np.ndarray,
    y: np.ndarray,
    cfg: IterConfig,
    seed: int = 7,
    Z_disc: Optional[np.ndarray] = None,
    z_names: Optional[List[str]] = None,
) -> IterResult:
    rng = np.random.default_rng(seed)
    n_samples, n_genes = X_disc.shape

    # Feature scores updated via EMA; probabilities derived from softmax(scores/temperature)
    scores = np.zeros(n_genes, dtype=float)
    probs = np.ones(n_genes, dtype=float) / n_genes

    best_auc = 0.0
    best_features: List[int] = []
    best_edges: List[Tuple[str, str]] = []
    aucs: List[float] = []
    failures = 0

    if Z_disc is not None and Z_disc.shape[0] != n_samples:
        raise ValueError(f"Z_disc must have same number of rows as X_disc (got {Z_disc.shape[0]} vs {n_samples})")
    z_names = z_names or ([f"Z{i}" for i in range(Z_disc.shape[1])] if Z_disc is not None else [])

    for t in range(cfg.n_iters):
        feat_idx = rng.choice(n_genes, size=min(cfg.subset_size, n_genes), replace=False, p=probs)
        gene_cols = [f"G{j}" for j in feat_idx]
        cov_cols = list(z_names) if Z_disc is not None else []
        cols = gene_cols + cov_cols + [cfg.outcome_name]

        blocks = [X_disc[:, feat_idx]]
        if Z_disc is not None:
            blocks.append(Z_disc)
        blocks.append(y.reshape(-1, 1))

        data = np.column_stack(blocks).astype(int)
        df = pd.DataFrame(data, columns=cols)

        try:
            model = _naive_bayes_structure(cols, cfg.outcome_name, covariate_cols=cov_cols)
            model.fit(
                df,
                estimator=BayesianEstimator,
                prior_type=str(cfg.prior_type),
                equivalent_sample_size=int(cfg.equivalent_sample_size),
            )

            model = _prune_to_outcome_connected(model, cfg.outcome_name)
            df2 = df[list(model.nodes())]
            model.fit(
                df2,
                estimator=BayesianEstimator,
                prior_type=str(cfg.prior_type),
                equivalent_sample_size=int(cfg.equivalent_sample_size),
            )

            p_worst = _infer_prob_worst(model, df2, cfg.outcome_name)
            auc = auc_group4_vs_rest(y, p_worst)
        except Exception:
            failures += 1
            auc = 0.5
            model = None

        aucs.append(float(auc))
        # Feedback update (stabilized): update scores, then softmax + epsilon-greedy + floor
        reward = float(auc - 0.5)
        # small negative rewards shouldn't collapse; clip
        reward = max(-0.05, min(0.30, reward))
        # delta for sampled features
        delta = np.zeros(n_genes, dtype=float)
        delta[feat_idx] = reward
        scores = (1.0 - cfg.ema_alpha) * scores + cfg.ema_alpha * (cfg.reward_scale * delta)

        # Softmax with temperature
        T = max(1e-6, float(cfg.temperature))
        s = scores / T
        s = s - np.max(s)
        p = np.exp(s)
        p = p / np.sum(p)

        # Mix with uniform for exploration
        eps = float(cfg.epsilon)
        p = (1.0 - eps) * p + eps * (np.ones_like(p) / n_genes)

        # Probability floor
        p = np.maximum(p, float(cfg.p_min))
        probs = p / np.sum(p)

        if auc > best_auc:
            best_auc = auc
            best_features = feat_idx.tolist()
            best_edges = list(model.edges()) if model is not None else []

    if failures:
        print(f"[iterative] Note: {failures}/{cfg.n_iters} iterations fell back to AUC=0.5 due to fit/inference issues")

    return IterResult(
        aucs=aucs,
        best_auc=float(best_auc),
        best_features=best_features,
        best_model_edges=best_edges,
        final_sampling_probs=probs,
    )
