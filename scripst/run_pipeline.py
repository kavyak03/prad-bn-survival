"""
End-to-end pipeline runner.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from prad_bn.simulate import SimConfig, simulate_prad_like_dataset, inject_survival_signal
from prad_bn.discretize import survival_to_km_groups, maybe_supervised_bncuts
from prad_bn.iterative import IterConfig, run_iterative_bn
from prad_bn.evaluate import auc_group_vs_rest, auc_multiclass_ovr, plot_km_by_group, fit_coxph
from prad_bn.utils import ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/demo", help="Output directory for artifacts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_genes", type=int, default=2000, help="Toy gene count (conceptually stands in for 20k)")
    ap.add_argument("--subset_size", type=int, default=20)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) simulate
    sim_cfg = SimConfig(n_genes=args.n_genes)
    sim = simulate_prad_like_dataset(sim_cfg, seed=args.seed)
    expr = sim["expr"]
    time_days = sim["time_days"]
    event = sim["event"]

    # 2) outcome engineering: 5 KM-style groups
    groups = survival_to_km_groups(time_days, q=5)

    # 3) inject weak survival-associated signal (optional but demonstrates learnability)
    expr2, signal_genes = inject_survival_signal(expr, groups, sim_cfg, np.random.default_rng(args.seed))

    # 4) discretize expression for BN
    X_disc = maybe_supervised_bncuts(expr2, groups, n_bins=5)

    # 5) iterative BN learning
    iter_cfg = IterConfig(n_iters=args.iters, subset_size=args.subset_size)
    result = run_iterative_bn(X_disc, groups, seed=args.seed, cfg=iter_cfg)

    # 6) save summary
    save_json({
        "best_auc_group0_vs_rest": result.best_auc,
        "best_feature_indices": result.best_features,
        "signal_genes": [int(x) for x in signal_genes],
        "iter_history": result.history,
    }, f"{args.outdir}/summary.json")

    # 7) quick extra eval artifacts
    plot_km_by_group(time_days, event, groups, outpath=f"{args.outdir}/km_by_group.png")
    cox_summary = fit_coxph(time_days, event, groups)
    cox_summary.to_csv(f"{args.outdir}/coxph_summary.csv", index=True)

    # print interview-friendly outputs
    print("\n=== Toy PRAD BN Survival Pipeline ===")
    print(f"Samples: {expr.shape[0]} | Genes (toy): {expr.shape[1]}")
    print(f"Signal genes injected (hidden ground truth): {sorted([int(g) for g in signal_genes])}")
    print(f"Best AUC (group 0 vs rest): {result.best_auc:.3f}")
    print(f"Best kept genes after pruning (count={len(result.best_features)}): {sorted(result.best_features)[:20]}")
    print(f"Artifacts written to: {args.outdir}\n")


if __name__ == "__main__":
    main()
