"""
Bayesian Network wrapper using pomegranate.

We:
- learn a BN from discrete data (genes + outcome)
- expose inference to get posterior probabilities of outcome
- expose an adjacency graph for pruning to outcome-connected nodes
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import networkx as nx

try:
    from pomegranate import BayesianNetwork
except Exception as e:  # pragma: no cover
    BayesianNetwork = None  # type: ignore


@dataclass
class BNFitResult:
    model: object
    state_names: list[str]
    graph: nx.DiGraph


def _require_pomegranate():
    if BayesianNetwork is None:
        raise ImportError(
            "pomegranate is required for this repo. Install with: pip install -r requirements.txt"
        )


def fit_bn_discrete(data: np.ndarray, state_names: list[str], max_parents: int = 3, seed: int = 7) -> BNFitResult:
    """
    Fit a Bayesian Network to discrete data.

    Args:
      data: (n_samples, n_vars) int matrix, last column can be outcome
      state_names: list of variable names (len == n_vars)
      max_parents: constraint used by pomegranate structure learning
    """
    _require_pomegranate()
    # pomegranate's structure learning API can vary across versions.
    # `from_samples` is the most common.
    model = BayesianNetwork.from_samples(
        data,
        algorithm="exact",
        state_names=state_names,
        max_parents=max_parents,
        random_state=seed,
    )

    # Build a directed graph of the learned structure
    g = nx.DiGraph()
    g.add_nodes_from(state_names)

    # edges are between State objects; map to names
    try:
        edges = model.structure
        # model.structure is list of parent indices; convert to edges parent->child
        for child_idx, parents in enumerate(edges):
            for p in parents:
                g.add_edge(state_names[p], state_names[child_idx])
    except Exception:
        # fallback: try model.edges if present
        try:
            for p, c in model.edges:
                g.add_edge(p.name, c.name)
        except Exception:
            pass

    return BNFitResult(model=model, state_names=state_names, graph=g)


def prune_to_outcome_connected(
    data: np.ndarray,
    names: list[str],
    graph: nx.DiGraph,
    outcome_name: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Keep only nodes that have a directed path to the outcome (including outcome).

    This is the "clinician-friendly" pruning step:
    remove features not connected to the outcome in the learned structure.
    """
    if outcome_name not in names:
        return data, names

    keep = set([outcome_name])
    # any node that can reach outcome
    for n in names:
        if n == outcome_name:
            continue
        try:
            if nx.has_path(graph, n, outcome_name):
                keep.add(n)
        except nx.NetworkXError:
            continue

    keep_list = [n for n in names if n in keep]
    idx = [names.index(n) for n in keep_list]
    return data[:, idx], keep_list


def predict_outcome_posteriors(model, X: np.ndarray, outcome_idx: int) -> np.ndarray:
    """
    For each sample, return posterior distribution over the outcome node.

    Returns:
      post: (n_samples, n_outcome_states) float
    """
    _require_pomegranate()
    posts = []
    for i in range(X.shape[0]):
        sample = X[i]
        # predict_proba accepts full samples; we can provide evidence for all nodes except outcome
        evidence = sample.copy()
        # use None for unknown outcome
        evidence[outcome_idx] = None  # type: ignore
        dists = model.predict_proba(evidence)

        out_dist = dists[outcome_idx]
        # pomegranate distributions expose .parameters[0] as dict of state->prob
        params = out_dist.parameters[0]
        # states might be ints; sort by key
        keys = sorted(params.keys())
        posts.append([params[k] for k in keys])
    return np.asarray(posts)
