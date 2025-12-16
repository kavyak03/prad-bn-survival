
"""
End-to-end pipeline runner
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from prad_bn.simulate import SimConfig, simulate_prad_like_dataset, inject_survival_signal
from prad_bn.discretize import survival_to_km_groups, maybe_supervised_bncuts
from prad_bn.iterative import IterConfig, run_iterative_bn
from prad_bn.evaluate import plot_km_by_group, plot_auc_over_iters, plot_sampling_probs


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/demo", help="Output directory for artifacts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_genes", type=int, default=2000, help="Toy gene count (conceptually stands in for 20k)")
    ap.add_argument("--subset_size", type=int, default=20)
    ap.add_argument("--iters", type=int, default=25)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # 1) simulate PRAD-like dataset
    sim_cfg = SimConfig(
        n_samples=120,
        n_genes=args.n_genes,
        n_blocks=10,
        block_corr=0.6,
        expr_scale=10.0,
        signal_genes=10,
        signal_strength=3.0,
        censoring_rate=0.35,
    )
    sim = simulate_prad_like_dataset(sim_cfg, seed=args.seed)
    expr, time_days, event = sim['expr'], sim['time_days'], sim['event']

    # 2) discretize survival into 5 KM-like groups
    km_group = survival_to_km_groups(time_days, q=5)

    # Inject weak survival-associated signal into a few hidden genes (ground truth)
    rng = np.random.default_rng(args.seed)
    expr, signal_genes = inject_survival_signal(expr, km_group, sim_cfg, rng)


    # 3) discretize gene expression into 5 bins
    # (bn-cuts hook exists, but we keep it simple and deterministic here)
    X_disc = maybe_supervised_bncuts(expr, km_group, n_bins=3)

    # 4) iterative BN learning with AUC feedback
    iter_cfg = IterConfig(n_iters=args.iters, subset_size=args.subset_size, outcome_name="Outcome")
    result = run_iterative_bn(X_disc, km_group, iter_cfg, seed=args.seed)

    # 5) write artifacts
    np.savetxt(f"{args.outdir}/km_group.csv", km_group, delimiter=",", fmt="%d")
    np.savetxt(f"{args.outdir}/time_days.csv", time_days, delimiter=",")
    np.savetxt(f"{args.outdir}/event.csv", event, delimiter=",", fmt="%d")

    with open(f"{args.outdir}/best_edges.txt", "w") as f:
        for u, v in result.best_model_edges:
            f.write(f"{u} -> {v}\n")

    # 6) plots
    plot_auc_over_iters(result.aucs, f"{args.outdir}/auc_over_iters.png")
    plot_km_by_group(time_days, event, km_group, f"{args.outdir}/km_curves_by_group.png")
    plot_sampling_probs(result.final_sampling_probs, top_k=20, outpath=f"{args.outdir}/top_sampling_probs.png")

    # Print outputs
    print("\n=== PRAD BN Survival Pipeline (pgmpy backend) ===")
    print(f"Samples: {expr.shape[0]} | Genes (toy): {expr.shape[1]}")
    print(f"Signal genes injected (hidden ground truth): {sorted([int(g) for g in signal_genes])}")
    print(f"Best AUC (group 0 vs rest): {result.best_auc:.3f}")
    print(f"Best kept genes (subset) size={len(result.best_features)} | example genes: {sorted(result.best_features)[:15]}")
    print(f"Artifacts written to: {args.outdir}\n")


if __name__ == "__main__":
    main()
