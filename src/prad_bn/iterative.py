
"""
Iterative Bayesian Network learning with AUC-driven feature sampling.

Demo BN backend: Naive Bayes BN (Outcome -> genes) using pgmpy DiscreteBayesianNetwork.
Core idea: feedback-driven adaptive feature sampling loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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


@dataclass
class IterResult:
    aucs: List[float]
    best_auc: float
    best_features: List[int]
    best_model_edges: List[Tuple[str, str]]
    final_sampling_probs: np.ndarray


def _naive_bayes_structure(columns: List[str], outcome: str) -> DiscreteBayesianNetwork:
    gene_cols = [c for c in columns if c != outcome]
    edges = [(outcome, g) for g in gene_cols]
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


def run_iterative_bn(X_disc: np.ndarray, y: np.ndarray, cfg: IterConfig, seed: int = 7) -> IterResult:
    rng = np.random.default_rng(seed)
    n_samples, n_genes = X_disc.shape

    probs = np.ones(n_genes, dtype=float) / n_genes

    best_auc = 0.0
    best_features: List[int] = []
    best_edges: List[Tuple[str, str]] = []
    aucs: List[float] = []
    failures = 0

    for t in range(cfg.n_iters):
        feat_idx = rng.choice(n_genes, size=min(cfg.subset_size, n_genes), replace=False, p=probs)
        cols = [f"G{j}" for j in feat_idx] + [cfg.outcome_name]
        data = np.column_stack([X_disc[:, feat_idx], y]).astype(int)
        df = pd.DataFrame(data, columns=cols)

        try:
            model = _naive_bayes_structure(cols, cfg.outcome_name)
            model.fit(df, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)

            model = _prune_to_outcome_connected(model, cfg.outcome_name)
            df2 = df[list(model.nodes())]
            model.fit(df2, estimator=BayesianEstimator, prior_type='BDeu', equivalent_sample_size=5)

            p_worst = _infer_prob_worst(model, df2, cfg.outcome_name)
            auc = auc_group4_vs_rest(y, p_worst)
        except Exception:
            failures += 1
            auc = 0.5
            model = None

        aucs.append(float(auc))

        # Feedback update: reward above chance
        reward = max(0.0, auc - 0.5)
        if reward > 0:
            probs[feat_idx] *= (1.0 + 15.0 * reward)
            probs /= probs.sum()
        else:
            probs = 0.99 * probs + 0.01 * (np.ones_like(probs) / n_genes)
            probs /= probs.sum()

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
