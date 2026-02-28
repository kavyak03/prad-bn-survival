"""BN what-if analysis via do-interventions (graph mutilation).

This is a *model-based* do-operator:
- remove incoming edges into intervened node(s)
- set their CPDs deterministically
- infer outcome distribution

On synthetic data, this aligns more closely with the DGP.
On TCGA, treat as sensitivity analysis unless you model confounders and identification explicitly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


@dataclass(frozen=True)
class DoEffect:
    variable: str
    low: int
    high: int
    p_worst_low: float
    p_worst_high: float
    delta: float


def _card(model: DiscreteBayesianNetwork, var: str) -> int:
    cpd = model.get_cpds(var)
    if cpd is None:
        raise ValueError(f"No CPD for {var}")
    return int(cpd.variable_card)


def do_intervene(model: DiscreteBayesianNetwork, do: Dict[str, int]) -> DiscreteBayesianNetwork:
    m = model.copy()
    for var, val in do.items():
        parents = list(m.get_parents(var))
        if parents:
            m.remove_edges_from([(p, var) for p in parents])
        card = _card(m, var)
        vals = np.zeros((card, 1))
        vals[int(val), 0] = 1.0
        new = TabularCPD(variable=var, variable_card=card, values=vals)
        old = m.get_cpds(var)
        if old is not None:
            m.remove_cpds(old)
        m.add_cpds(new)
    return m


def predict(model: DiscreteBayesianNetwork, outcome: str, evidence: Optional[Dict[str,int]] = None) -> np.ndarray:
    inf = VariableElimination(model)
    q = inf.query([outcome], evidence=evidence or {}, show_progress=False)
    return np.asarray(q.values, dtype=float)


def do_effect_on_worst(model: DiscreteBayesianNetwork, outcome: str, var: str, low: int = 0, high: Optional[int] = None) -> DoEffect:
    if high is None:
        high = _card(model, var) - 1
    worst = _card(model, outcome) - 1
    p_low = predict(do_intervene(model, {var: low}), outcome)
    p_high = predict(do_intervene(model, {var: high}), outcome)
    return DoEffect(variable=var, low=low, high=high, p_worst_low=float(p_low[worst]), p_worst_high=float(p_high[worst]), delta=float(p_high[worst]-p_low[worst]))


def rank_effects(model: DiscreteBayesianNetwork, outcome: str, variables: List[str], top_k: int = 10) -> List[DoEffect]:
    eff=[do_effect_on_worst(model, outcome, v) for v in variables]
    eff.sort(key=lambda e: abs(e.delta), reverse=True)
    return eff[:top_k]
