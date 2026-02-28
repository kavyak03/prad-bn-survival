#!/usr/bin/env python3
"""End-to-end runner for PRAD BN Survival demo.

Supports:
- Synthetic PRAD-like simulation (default): right-skewed expression, block-correlated modules, censored survival
- TCGA-PRAD mode (via UCSC Xena): end-to-end run on real PRAD expression + phenotype
- Optional: model-based do-interventions on fitted BN (gene-level)
- Optional: module-level causal/ATE analysis with clinical adjustment (doWhy-style)

Run:
  python -m scripts.run_pipeline --outdir runs/demo --seed 7
  python -m scripts.run_pipeline --data tcga --endpoint PFI --outdir runs/tcga_pfi --seed 7

Note: Requires Python >= 3.10 (pgmpy typing).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import json

import numpy as np

# Make src/ importable when running without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prad_bn.simulate import SimConfig, simulate_prad_like_dataset, inject_survival_signal
from prad_bn.discretize import survival_to_km_groups, maybe_supervised_bncuts
from prad_bn.iterative import IterConfig, run_iterative_bn
from prad_bn.evaluate import plot_km_by_group, plot_auc_over_iters, plot_sampling_probs

from prad_bn.moduleize import modules_from_blocks, module_eigengenes, discretize_modules
from prad_bn.causal_bn import rank_effects as rank_bn_effects
from prad_bn.causal_ate import treatment_from_bins, estimate_ate_aipw
from prad_bn.tcga import TcgaConfig, load_tcga_prad

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/demo", help="Output directory for artifacts")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--data", type=str, default="sim", choices=["sim", "tcga"], help="Data source")
    ap.add_argument("--endpoint", type=str, default="PFI", choices=["PFI", "OS"], help="TCGA endpoint (tcga mode)")

    ap.add_argument("--n_genes", type=int, default=2000, help="Gene count (sim: toy; tcga: top-variable genes)")
    ap.add_argument("--subset_size", type=int, default=20)
    ap.add_argument("--iters", type=int, default=25)

    # Stabilized sampler knobs (leave defaults unless you want more/less exploration)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--p_min", type=float, default=1e-4)
    ap.add_argument("--ema_alpha", type=float, default=0.15)

    # Synthetic realism knobs
    ap.add_argument("--inject_signal", action="store_true", help="Also inject monotonic signal after KM binning (legacy demo hook)")

    # Causal / explainability
    ap.add_argument("--causal_bn", action="store_true", help="Model-based do-interventions on BN (gene-level)")
    ap.add_argument("--module_causal", action="store_true", help="Module-level analysis (modules->outcome) + ATE-style estimates")
    ap.add_argument("--n_blocks", type=int, default=8, help="Number of modules (blocks) for module analysis")
    ap.add_argument("--module_bins", type=int, default=3, help="Discretization bins for modules")
    ap.add_argument("--tcga_cache", type=str, default="data/tcga", help="Cache dir for TCGA downloads (tcga mode)")
    return ap.parse_args()


def _fit_naive_bayes_bn(X_disc: np.ndarray, y: np.ndarray, feat_idx: np.ndarray, outcome: str = "Outcome") -> DiscreteBayesianNetwork:
    cols = [f"G{j}" for j in feat_idx] + [outcome]
    data = np.column_stack([X_disc[:, feat_idx], y]).astype(int)
    import pandas as pd
    df = pd.DataFrame(data, columns=cols)
    edges = [(outcome, f"G{j}") for j in feat_idx]
    model = DiscreteBayesianNetwork(edges)
    model.add_nodes_from(cols)
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=5)
    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load data
    covariates = None
    covariate_names = []
    if args.data == "sim":
        sim_cfg = SimConfig(
            n_samples=120,
            n_genes=args.n_genes,
            n_blocks=10,
            block_corr=0.65,
            expr_scale=10.0,
            signal_genes=10,
            signal_strength=1.2,
            censoring_rate=0.35,
            use_expr_survival=True,
            n_signal_blocks=3,
            block_signal_genes=6,
            survival_beta=0.7,
        )
        sim = simulate_prad_like_dataset(sim_cfg, seed=args.seed)
        expr, time_days, event = sim["expr"], sim["time_days"], sim["event"]
    else:
        tcfg = TcgaConfig(cache_dir=args.tcga_cache, n_genes=args.n_genes, endpoint=args.endpoint, random_seed=args.seed)
        tcga = load_tcga_prad(tcfg)
        expr, time_days, event = tcga["expr"], tcga["time_days"], tcga["event"]
        covariates = tcga.get("covariates", None)
        covariate_names = tcga.get("covariate_names", [])
        # TCGA expression can be large; clip extreme tails for discretization stability
        expr = np.clip(expr, np.nanpercentile(expr, 0.5), np.nanpercentile(expr, 99.5))

    # 2) Discretize survival into 5 KM-like groups (quantile bins)
    km_group = survival_to_km_groups(time_days, q=5)

    # Optional legacy hook: inject monotonic signal AFTER KM binning (kept for continuity)
    if args.data == "sim" and args.inject_signal:
        rng = np.random.default_rng(args.seed)
        sim_cfg = SimConfig(n_genes=expr.shape[1])  # minimal
        expr, _ = inject_survival_signal(expr, km_group, sim_cfg, rng)

    # 3) Discretize gene expression into bins suitable for BN
    X_disc = maybe_supervised_bncuts(expr, km_group, n_bins=3)

    # 4) Iterative BN learning with stabilized adaptive sampling
    iter_cfg = IterConfig(
        n_iters=args.iters,
        subset_size=args.subset_size,
        outcome_name="Outcome",
        temperature=args.temperature,
        epsilon=args.epsilon,
        p_min=args.p_min,
        ema_alpha=args.ema_alpha,
    )
    result = run_iterative_bn(X_disc, km_group, iter_cfg, seed=args.seed)

    # 5) Write artifacts
    np.savetxt(f"{args.outdir}/km_group.csv", km_group, delimiter=",", fmt="%d")
    np.savetxt(f"{args.outdir}/time_days.csv", time_days, delimiter=",")
    np.savetxt(f"{args.outdir}/event.csv", event, delimiter=",", fmt="%d")

    with open(f"{args.outdir}/best_edges.txt", "w") as f:
        for u, v in result.best_model_edges:
            f.write(f"{u} -> {v}\n")

    meta = {
        "data": args.data,
        "endpoint": args.endpoint if args.data == "tcga" else None,
        "n_samples": int(expr.shape[0]),
        "n_genes": int(expr.shape[1]),
        "best_auc": float(result.best_auc),
        "best_features": list(map(int, result.best_features)),
        "sampler": {"temperature": args.temperature, "epsilon": args.epsilon, "p_min": args.p_min, "ema_alpha": args.ema_alpha},
        "covariates": covariate_names,
    }
    with open(f"{args.outdir}/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # 6) Plots
    plot_auc_over_iters(result.aucs, f"{args.outdir}/auc_over_iters.png")
    plot_km_by_group(time_days, event, km_group, f"{args.outdir}/km_curves_by_group.png")
    plot_sampling_probs(result.final_sampling_probs, top_k=20, outpath=f"{args.outdir}/top_sampling_probs.png")

    # 7) Optional: BN do-intervention what-if (model-based)
    if args.causal_bn:
        try:
            bn = _fit_naive_bayes_bn(X_disc, km_group, np.array(result.best_features, dtype=int), outcome="Outcome")
            vars_ = [f"G{j}" for j in result.best_features]
            eff = rank_bn_effects(bn, outcome="Outcome", variables=vars_, top_k=min(10, len(vars_)))
            out = {
                "note": "Model-based do-interventions inside the learned BN (sensitivity). Treat as causal only if assumptions hold.",
                "effects": [e.__dict__ for e in eff],
            }
            with open(f"{args.outdir}/causal_bn_effects.json", "w") as f:
                json.dump(out, f, indent=2)
        except Exception as e:
            with open(f"{args.outdir}/causal_bn_effects_error.txt", "w") as f:
                f.write(str(e))

    # 8) Optional: Module-level doWhy-style ATE with clinical adjustment
    if args.module_causal:
        specs = modules_from_blocks(n_genes=expr.shape[1], n_blocks=args.n_blocks)
        M = module_eigengenes(expr, specs)
        M_disc = discretize_modules(M, n_bins=args.module_bins)
        module_names = [s.name for s in specs]

        worst = int(np.max(km_group))
        Y = (km_group == worst).astype(int)

        results = []
        for j, name in enumerate(module_names):
            T = treatment_from_bins(M_disc[:, j], low=0, high=args.module_bins-1)
            keep = T != -1
            if keep.sum() < 40:
                continue
            Z = covariates[keep] if covariates is not None else None
            res = estimate_ate_aipw(T[keep], Y[keep], Z=Z, seed=args.seed)
            results.append({"module": name, "ate": res.ate, "se": res.se, "n": res.n, "method": res.method, "placebo_ate": res.placebo_ate})

        results.sort(key=lambda r: abs(r["ate"]), reverse=True)
        out = {
            "note": "DoWhy-style ATE: treatment=module-high vs module-low; outcome=worst KM risk indicator; adjusted by available covariates (if any).",
            "endpoint": args.endpoint if args.data == "tcga" else "simulated",
            "covariates": covariate_names,
            "results_top": results[:10],
            "n_modules_tested": len(results),
        }
        with open(f"{args.outdir}/module_ate_results.json", "w") as f:
            json.dump(out, f, indent=2)

    print("\n=== PRAD BN Survival Pipeline ===")
    print(f"Data: {args.data} | Samples: {expr.shape[0]} | Genes: {expr.shape[1]}")
    if args.data == "tcga":
        print(f"Endpoint: {args.endpoint} | Covariates used for ATE: {', '.join(covariate_names) if covariate_names else '(none found)'}")
    print(f"Best AUC (worst-group vs rest): {result.best_auc:.3f}")
    print(f"Artifacts written to: {args.outdir}\n")


if __name__ == "__main__":
    main()
