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

from prad_bn.simulate import (
    SimConfig,
    simulate_prad_like_dataset,
    inject_survival_signal,
    simulate_clinical_covariates,
    apply_covariate_confounding_to_survival,
)
from prad_bn.discretize import survival_to_km_groups, maybe_supervised_bncuts
from prad_bn.iterative import IterConfig, run_iterative_bn
from prad_bn.evaluate import plot_km_by_group, plot_auc_over_iters, plot_sampling_probs

from prad_bn.moduleize import modules_from_blocks, module_eigengenes, discretize_modules
from prad_bn.causal_bn import rank_effects as rank_bn_effects
from prad_bn.causal_ate import treatment_from_bins, estimate_ate_aipw
from prad_bn.tcga import TcgaConfig, load_tcga_prad
from prad_bn.clinical_gdc import ClinicalMap, load_gdc_covariates_aligned

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator


def discretize_clinical_covariates(cov_df, names):
    """Discretize age, stage, and gleason into small integer bins for a discrete BN.

    Missing/unknown values are mapped to 0.
    Returns (Z_disc, z_names_used).
    """
    import numpy as np
    import pandas as pd

    if cov_df is None or not names:
        return None, []

    df = cov_df if isinstance(cov_df, pd.DataFrame) else pd.DataFrame(cov_df, columns=names)

    used = []
    cols = []

    # age: tertiles -> {1,2,3} (missing->0)
    if "age" in df.columns:
        s = pd.to_numeric(df["age"], errors="coerce")
        b = pd.qcut(s, q=3, duplicates="drop").cat.codes  # -1 indicates NaN
        b = b.astype(int)
        miss = b == -1
        b = b + 1
        b[miss] = 0
        cols.append(b.to_numpy())
        used.append("age")

    # stage: map I/II/III/IV -> {1,2,3,4} (unknown->0)
    if "stage" in df.columns:
        s = df["stage"].astype(str).str.upper()

        def _stage_to_int(x: str) -> int:
            if x.startswith("I") and not x.startswith("II") and not x.startswith("III") and not x.startswith("IV"):
                return 1
            if x.startswith("II") and not x.startswith("III") and not x.startswith("IV"):
                return 2
            if x.startswith("III") and not x.startswith("IV"):
                return 3
            if x.startswith("IV"):
                return 4
            return 0

        b = s.map(_stage_to_int).fillna(0).astype(int)
        cols.append(b.to_numpy())
        used.append("stage")

    # gleason: buckets (<=6, 7, >=8) -> {1,2,3} (missing->0)
    if "gleason" in df.columns:
        g = pd.to_numeric(df["gleason"], errors="coerce")
        b = pd.cut(g, bins=[0, 6, 7, 10], labels=[1, 2, 3], include_lowest=True)
        b = pd.Series(b).astype("float").fillna(0).astype(int)
        cols.append(b.to_numpy())
        used.append("gleason")

    if not cols:
        return None, []

    return np.column_stack(cols).astype(int), used


def covariates_to_numeric_for_ate(cov_df, names):
    """Convert covariates into a numeric design matrix for ATE estimation."""
    import pandas as pd
    import numpy as np

    if cov_df is None or not names:
        return None, []

    df = cov_df if isinstance(cov_df, pd.DataFrame) else pd.DataFrame(cov_df, columns=names)

    mats = []
    out_names = []

    if "age" in df.columns:
        s = pd.to_numeric(df["age"], errors="coerce")
        mats.append(s.fillna(s.median()).to_numpy().reshape(-1, 1))
        out_names.append("age")

    if "gleason" in df.columns:
        s = pd.to_numeric(df["gleason"], errors="coerce")
        mats.append(s.fillna(s.median()).to_numpy().reshape(-1, 1))
        out_names.append("gleason")

    if "stage" in df.columns:
        st = df["stage"].astype(str).fillna("NA")
        d = pd.get_dummies(st, prefix="stage", dummy_na=False)
        mats.append(d.to_numpy().astype(float))
        out_names.extend(list(d.columns))

    if not mats:
        return None, []

    Z = np.concatenate(mats, axis=1).astype(float)
    return Z, out_names


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="runs/demo", help="Output directory for artifacts")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--data", type=str, default="sim", choices=["sim", "tcga"], help="Data source")
    ap.add_argument("--endpoint", type=str, default="PFI", choices=["PFI", "OS"], help="TCGA endpoint (tcga mode)")

    ap.add_argument("--n_genes", type=int, default=2000, help="Gene count (sim: toy; tcga: top-variable genes)")
    ap.add_argument("--subset_size", type=int, default=20)
    ap.add_argument("--iters", type=int, default=25)

    # Optional: user-provided GDC clinical file (TCGA mode) for age/stage/gleason covariates
    ap.add_argument(
        "--clinical_file",
        type=str,
        default=None,
        help="(tcga mode) Optional GDC clinical TSV/CSV/ZIP used to extract age/stage/gleason and align to TCGA samples",
    )
    ap.add_argument("--clinical_id_col", type=str, default=None, help="Override ID column name in clinical file")
    ap.add_argument("--clinical_age_col", type=str, default=None, help="Override age column name in clinical file")
    ap.add_argument("--clinical_stage_col", type=str, default=None, help="Override stage column name in clinical file")
    ap.add_argument("--clinical_gleason_col", type=str, default=None, help="Override gleason column name in clinical file")

    # Stabilized sampler knobs (leave defaults unless you want more/less exploration)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--p_min", type=float, default=1e-4)
    ap.add_argument("--ema_alpha", type=float, default=0.15)

    # Optional: include available covariates as confounders in BN inference
    ap.add_argument("--bn_covariates", action="store_true", help="Include covariates as parents of Outcome in the BN")
    ap.add_argument("--bn_ess", type=int, default=5, help="BN equivalent sample size (BDeu smoothing)")

    # Optional: simulate clinical covariates for synthetic runs
    ap.add_argument("--sim_covariates", action="store_true", help="(sim mode) simulate age/stage/gleason")
    ap.add_argument("--sim_confound_survival", action="store_true", help="(sim mode) if sim_covariates, make survival depend on covariates")

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
    sample_ids = None

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

        if args.sim_covariates:
            import pandas as pd
            rng = np.random.default_rng(args.seed)
            cov = simulate_clinical_covariates(expr.shape[0], rng)
            covariates = pd.DataFrame(cov)
            covariate_names = list(covariates.columns)
            if args.sim_confound_survival:
                time_days = apply_covariate_confounding_to_survival(time_days, cov, rng)

    else:
        tcfg = TcgaConfig(cache_dir=args.tcga_cache, n_genes=args.n_genes, endpoint=args.endpoint, random_seed=args.seed)
        tcga = load_tcga_prad(tcfg)
        expr, time_days, event = tcga["expr"], tcga["time_days"], tcga["event"]
        covariates = tcga.get("covariates", None)
        covariate_names = tcga.get("covariate_names", [])
        sample_ids = tcga.get("sample_ids", None)

        # ---- FIX #1: Ensure expr is samples x genes (not genes x samples) ----
        # We expect len(time_days) == n_samples
        n_surv = len(time_days)
        if expr.shape[0] != n_surv and expr.shape[1] == n_surv:
            # Looks like genes x samples -> transpose
            expr = expr.T

        # Prefer covariates from user-provided GDC clinical file (Xena phenotype often lacks stage/gleason)
        if args.clinical_file:
            if sample_ids is None:
                raise RuntimeError("TCGA loader did not return sample_ids; cannot align clinical covariates")

            cov_np, cov_names, msg = load_gdc_covariates_aligned(
                clinical_file=args.clinical_file,
                sample_ids=np.asarray(sample_ids),
                mapping=ClinicalMap(
                    id_col=args.clinical_id_col,
                    age_col=args.clinical_age_col,
                    stage_col=args.clinical_stage_col,
                    gleason_col=args.clinical_gleason_col,
                ),
            )
            print("[clinical]", msg)
            if cov_np is not None and cov_names:
                covariates = cov_np
                covariate_names = cov_names

        # TCGA expression can be large; clip extreme tails for discretization stability
        expr = np.clip(expr, np.nanpercentile(expr, 0.5), np.nanpercentile(expr, 99.5))

        # Final sanity: expr rows must match survival vectors
        if expr.shape[0] != len(time_days) or expr.shape[0] != len(event):
            raise RuntimeError(
                f"Alignment error: expr has {expr.shape[0]} rows but time_days has {len(time_days)} and event has {len(event)}. "
                "Expression must be samples x genes."
            )

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
        equivalent_sample_size=args.bn_ess,
    )

    Z_disc, z_used = (None, [])
    if args.bn_covariates and covariates is not None and covariate_names:
        Z_disc, z_used = discretize_clinical_covariates(covariates, covariate_names)

    result = run_iterative_bn(X_disc, km_group, iter_cfg, seed=args.seed, Z_disc=Z_disc, z_names=z_used)

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
        "covariates_available": covariate_names,
        "bn_used_covariates": bool(args.bn_covariates and covariates is not None and covariate_names),
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
        Y_all = (km_group == worst).astype(int)

        Z_all, Z_names = covariates_to_numeric_for_ate(covariates, covariate_names)

        results = []
        for j, name in enumerate(module_names):
            T_all = treatment_from_bins(M_disc[:, j], low=0, high=args.module_bins - 1)

            keep = (T_all != -1)
            if keep.sum() < 40:
                continue

            # ---- FIX #2: mask ONCE, consistently ----
            T = T_all[keep]
            Y = Y_all[keep]
            Z_keep = (Z_all[keep] if Z_all is not None else None)

            res = estimate_ate_aipw(T, Y, Z=Z_keep, seed=args.seed)
            results.append(
                {"module": name, "ate": res.ate, "se": res.se, "n": res.n, "method": res.method, "placebo_ate": res.placebo_ate}
            )

        results.sort(key=lambda r: abs(r["ate"]), reverse=True)
        out = {
            "note": "DoWhy-style ATE: treatment=module-high vs module-low; outcome=worst KM risk indicator; adjusted by available covariates (if any).",
            "endpoint": args.endpoint if args.data == "tcga" else "simulated",
            "covariates": Z_names,
            "results_top": results[:10],
            "n_modules_tested": len(results),
        }
        with open(f"{args.outdir}/module_ate_results.json", "w") as f:
            json.dump(out, f, indent=2)

    print("\n=== PRAD BN Survival Pipeline ===")
    print(f"Data: {args.data} | Samples: {expr.shape[0]} | Genes: {expr.shape[1]}")
    if covariate_names:
        print(f"Covariates available: {', '.join(covariate_names)}")
    else:
        print("Covariates available: (none)")
    if args.data == "tcga":
        print(f"Endpoint: {args.endpoint}")
    if args.bn_covariates:
        print(f"BN covariates enabled: {'yes' if (covariates is not None and covariate_names) else 'requested but none available'}")
    print(f"Best AUC (worst-group vs rest): {result.best_auc:.3f}")
    print(f"Artifacts written to: {args.outdir}\n")


if __name__ == "__main__":
    main()