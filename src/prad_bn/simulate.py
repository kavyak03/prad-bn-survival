"""
Simulation utilities for PRAD-like toy data.

Key ideas:
- Right-skewed expression (log-normal / truncated)
- Block-wise correlation to mimic pathways/modules
- Weak survival-associated signal injected into a small gene set
- Long-tailed survival times + optional censoring
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SimConfig:
    n_samples: int = 120
    n_genes: int = 2000          # keep toy-friendly; conceptually maps to 20k
    n_blocks: int = 20
    block_corr: float = 0.6
    expr_scale: float = 10.0     # roughly 0-10 for "highly expressed" toy range
    signal_genes: int = 8
    signal_strength: float = 0.8
    censoring_rate: float = 0.35
    survival_scale_days: float = 1000.0


def _make_block_cov(n_genes: int, n_blocks: int, block_corr: float) -> np.ndarray:
    """
    Create a block-diagonal covariance-like matrix (correlation structure).
    """
    # Partition genes into blocks as evenly as possible
    sizes = [n_genes // n_blocks] * n_blocks
    for i in range(n_genes % n_blocks):
        sizes[i] += 1

    cov = np.eye(n_genes)
    start = 0
    for bsz in sizes:
        end = start + bsz
        # within-block correlation
        cov[start:end, start:end] = block_corr
        np.fill_diagonal(cov[start:end, start:end], 1.0)
        start = end
    return cov


def simulate_expression(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate right-skewed expression with block-wise correlation.
    Approach:
    1) sample multivariate normal with block correlation
    2) exponentiate -> log-normal-like right skew
    3) rescale into a convenient numeric range
    """
    cov = _make_block_cov(cfg.n_genes, cfg.n_blocks, cfg.block_corr)

    # sample correlated latent variables
    latent = rng.multivariate_normal(
        mean=np.zeros(cfg.n_genes),
        cov=cov,
        size=cfg.n_samples,
        check_valid="ignore",
    )

    # induce right skew (log-normal-ish)
    expr = np.exp(latent / 2.0)

    # rescale to a toy "TPM-ish" range
    expr = expr / np.percentile(expr, 95) * cfg.expr_scale
    expr = np.clip(expr, 0.0, cfg.expr_scale)
    return expr


def simulate_survival(cfg: SimConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate long-tailed survival times with optional censoring.

    Returns:
      time_days: (n,) float
      event_observed: (n,) int (1=event/death, 0=censored)
    """
    # exponential => long-tailed time-to-event
    time_days = rng.exponential(scale=cfg.survival_scale_days, size=cfg.n_samples)

    # censoring indicator
    event_observed = (rng.random(cfg.n_samples) > cfg.censoring_rate).astype(int)

    # if censored, reduce observed time a bit (simple toy mechanism)
    censor_factor = rng.uniform(0.4, 1.0, size=cfg.n_samples)
    time_days = np.where(event_observed == 0, time_days * censor_factor, time_days)
    return time_days, event_observed


def inject_survival_signal(expr: np.ndarray, km_group: np.ndarray, cfg: SimConfig, rng: np.random.Generator):
    """
    Inject survival-associated signal into a subset of genes.

    We shift expression *monotonically* with risk group so discretization preserves the signal.
    """
    expr = expr.copy()
    n_samples, n_genes = expr.shape
    n_signal = int(getattr(cfg, 'signal_genes', 6))
    strength = float(getattr(cfg, 'signal_strength', 2.0))
    signal_genes = rng.choice(n_genes, size=min(n_signal, n_genes), replace=False)
    # Normalize group to [0,1] so the shift is smooth
    g = km_group.astype(float)
    if g.max() > 0:
        g = g / g.max()
    # Apply shift: higher risk -> higher expression in signal genes
    shift = strength * (g - g.mean())
    for sg in signal_genes:
        expr[:, sg] = expr[:, sg] + shift + rng.normal(0, 0.05, size=n_samples)
    return expr, signal_genes

def simulate_prad_like_dataset(cfg: SimConfig, seed: int = 7) -> dict:
    """
    End-to-end simulation:
      expression -> survival time/censoring -> risk groups -> inject weak signal
    """
    rng = np.random.default_rng(seed)

    expr = simulate_expression(cfg, rng)
    time_days, event = simulate_survival(cfg, rng)

    # risk groups will be computed downstream using quantile binning
    # (we return time_days for that)
    return {
        "expr": expr,
        "time_days": time_days,
        "event": event,
    }
