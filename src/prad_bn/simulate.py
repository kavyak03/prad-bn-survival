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
    use_expr_survival: bool = True
    n_signal_blocks: int = 3          # number of correlated blocks that affect survival
    block_signal_genes: int = 6       # genes per signal block
    survival_beta: float = 0.6        # effect size on log-hazard


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

def simulate_survival_from_expression(expr: np.ndarray, cfg: SimConfig, rng: np.random.Generator):
    """Simulate survival where risk is driven by *distributed, block-level* gene programs.

    This avoids a 'single magic gene' and better matches real tumor biology where pathways matter.
    Mechanism:
    - Pick several blocks
    - Within each, pick a few genes
    - Compute a latent risk score as the mean expression of those genes (standardized)
    - Generate time-to-event with hazard scaled by exp(beta * risk)

    Returns:
      time_days, event_observed, signal_gene_indices
    """
    n_samples, n_genes = expr.shape
    n_blocks = int(max(1, cfg.n_blocks))
    # Build block index ranges (same partitioning as _make_block_cov)
    sizes = [n_genes // n_blocks] * n_blocks
    for i in range(n_genes % n_blocks):
        sizes[i] += 1
    ranges=[]
    s=0
    for bsz in sizes:
        ranges.append((s, s+bsz))
        s += bsz

    n_sig_blocks = int(min(cfg.n_signal_blocks, n_blocks))
    chosen_blocks = rng.choice(n_blocks, size=n_sig_blocks, replace=False)

    sig_genes=[]
    for b in chosen_blocks:
        start,end = ranges[b]
        k = int(min(cfg.block_signal_genes, end-start))
        if k <= 0:
            continue
        sig = rng.choice(np.arange(start,end), size=k, replace=False)
        sig_genes.extend(list(sig))
    sig_genes = np.array(sorted(set(sig_genes)), dtype=int)

    # latent risk: mean of signal genes, standardized
    risk = expr[:, sig_genes].mean(axis=1) if len(sig_genes) else expr.mean(axis=1)
    risk = (risk - risk.mean()) / (risk.std() + 1e-8)

    # time-to-event: exponential with hazard scaled by exp(beta*risk)
    base_scale = float(cfg.survival_scale_days)
    beta = float(cfg.survival_beta)
    # shorter times for higher risk
    scale = base_scale / np.exp(beta * risk)
    time_days = rng.exponential(scale=scale)

    event_observed = (rng.random(n_samples) > cfg.censoring_rate).astype(int)
    censor_factor = rng.uniform(0.4, 1.0, size=n_samples)
    time_days = np.where(event_observed == 0, time_days * censor_factor, time_days)
    return time_days, event_observed, sig_genes


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

    if getattr(cfg, 'use_expr_survival', False):
        time_days, event, signal_genes_survival = simulate_survival_from_expression(expr, cfg, rng)
    else:
        time_days, event = simulate_survival(cfg, rng)
        signal_genes_survival = np.array([], dtype=int)

    # risk groups will be computed downstream using quantile binning
    # (we return time_days for that)
    return {
        "expr": expr,
        "time_days": time_days,
        "event": event,
        "signal_genes_survival": signal_genes_survival,
    }


def simulate_clinical_covariates(
    n_samples: int,
    rng: np.random.Generator,
    *,
    age_mean: float = 65.0,
    age_sd: float = 8.0,
) -> dict:
    """Simulate simple PRAD-like clinical covariates.

    Returns a dict with keys: age, stage, gleason.

    Notes:
    - This is intentionally simple and interview-demo friendly.
    - Stage is categorical {I, II, III, IV}.
    - Gleason is correlated with stage.
    """
    age = rng.normal(loc=age_mean, scale=age_sd, size=n_samples)
    age = np.clip(age, 40.0, 90.0)

    stage_levels = np.array(["I", "II", "III", "IV"], dtype=object)
    stage = rng.choice(stage_levels, size=n_samples, p=[0.25, 0.45, 0.22, 0.08])

    # Gleason loosely correlated with stage
    stage_to_base = {"I": 6.0, "II": 7.0, "III": 8.0, "IV": 9.0}
    base = np.array([stage_to_base[s] for s in stage], dtype=float)
    gleason = base + rng.normal(0.0, 0.6, size=n_samples)
    gleason = np.clip(np.round(gleason), 6.0, 10.0)

    return {"age": age, "stage": stage, "gleason": gleason}


def apply_covariate_confounding_to_survival(
    time_days: np.ndarray,
    covariates: dict,
    rng: np.random.Generator,
    *,
    gamma_age: float = 0.25,
    gamma_stage: float = 0.35,
    gamma_gleason: float = 0.25,
) -> np.ndarray:
    """Optionally make survival depend on clinical covariates.

    This simulates confounding: clinical covariates affect outcome, and may also be
    correlated with expression in real data.

    Mechanism:
      time' = time / exp(gamma * risk(covariates))
    So higher age/stage/gleason => shorter survival.
    """
    t = np.asarray(time_days, dtype=float).copy()

    age = np.asarray(covariates.get("age"), dtype=float)
    gleason = np.asarray(covariates.get("gleason"), dtype=float)
    stage = np.asarray(covariates.get("stage"), dtype=object)

    age_z = (age - np.nanmean(age)) / (np.nanstd(age) + 1e-8)
    gl_z = (gleason - np.nanmean(gleason)) / (np.nanstd(gleason) + 1e-8)

    stage_map = {"I": 0.0, "II": 1.0, "III": 2.0, "IV": 3.0}
    st = np.array([stage_map.get(str(s), 0.0) for s in stage], dtype=float)
    st_z = (st - np.mean(st)) / (np.std(st) + 1e-8)

    risk = gamma_age * age_z + gamma_stage * st_z + gamma_gleason * gl_z
    # mild randomness so effect isn't perfectly deterministic
    risk = risk + rng.normal(0.0, 0.05, size=risk.shape[0])

    t = t / np.exp(risk)
    t = np.clip(t, 1.0, None)
    return t

