# PRAD Adaptive Bayesian Network Survival + Causal Extensions

This repo is an implementation of a high‑p / low‑n survival modeling pipeline inspired by TCGA‑PRAD:

- **Synthetic PRAD-like simulator (default)**: right‑skewed RNA‑seq‑like expression, **block‑correlated modules**, long‑tailed **censored** survival
- **Survival outcome engineering**: discretize survival time into **5 Kaplan–Meier-style risk strata** (quantile bins)
- **Bayesian Network modeling**: discretize expression into bins and fit compact BN models
- **Adaptive feature sampling loop**: iteratively sample small gene subsets and update sampling probabilities using AUC feedback
- **Causal / explainability add‑ons**:
  - **Model-based do-interventions** inside the learned BN (gene-level sensitivity)
  - **Module-level doWhy-style ATE** with (optional) clinical adjustment (TCGA mode)

> **Why the synthetic-first design?**  
> It keeps the repo runnable anywhere and lets you explain the method clearly.  
> Real TCGA runs are supported as an optional mode and are expected to have **modest** results (which is realistic and credible).

---


## Architecture at a glance (3 layers)

> **Layer 1 (Core):** discretized survival → iterative BN learning → adaptive feature sampling  
> **Layer 2 (Model-based “what-if”):** BN do-interventions (graph mutilation + deterministic CPDs)  
> **Layer 3 (doWhy-style causal validation):** module-level ATE with clinical adjustment + refutation

```text
                 ┌───────────────────────────────────────────────────────┐
                 │                    Data Sources                        │
                 │  (A) Synthetic PRAD-like   (B) TCGA PRAD via Xena       │
                 └───────────────┬───────────────────────────┬───────────┘
                                 │                           │
                                 v                           v
                    ┌──────────────────────┐     ┌──────────────────────┐
                    │  Preprocess / Align  │     │  Preprocess / Align  │
                    │  - expr matrix X     │     │  - expr + survival     │
                    │  - time/event        │     │  - clinical covariates │
                    └───────────┬──────────┘     └───────────┬──────────┘
                                \____________________________/
                                              │
                                              v
                         ┌────────────────────────────────────────┐
                         │      Outcome Engineering (Layer 1)      │
                         │  survival → Kaplan–Meier risk groups     │
                         └──────────────────────┬─────────────────┘
                                                │
                                                v
                         ┌────────────────────────────────────────┐
                         │      Discretize Expression (Layer 1)    │
                         │   genes/modules → bins (small-n ready)  │
                         └──────────────────────┬─────────────────┘
                                                │
                                                v
                  ┌────────────────────────────────────────────────────────┐
                  │     Iterative BN + Adaptive Feature Sampling (Layer 1) │
                  │  sample subset → fit constrained BN → AUC → update p   │
                  └──────────────────────┬─────────────────────────────────┘
                                         │
                ┌────────────────────────┼────────────────────────┐
                │                        │                        │
                v                        v                        v
┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│ Outputs (Layer 1)         │  │ BN do-interventions        │  │ Module ATE (doWhy-style)   │
│ - KM curves               │  │ (Layer 2, optional)         │  │ (Layer 3, optional)        │
│ - AUC over iterations     │  │ - P(Y|do(gene/bin))         │  │ - Treatment: module high/low│
│ - sampling probabilities  │  │ - Δ P(worst-risk) rankings  │  │ - Adjust for clinical Z     │
└───────────────────────────┘  └───────────────────────────┘  │ - AIPW ATE + placebo test   │
                                                               └───────────────────────────┘
```

## Requirements

- **Python ≥ 3.10** (pgmpy uses modern typing)
- Recommended: create an isolated virtual environment.

Install:

```bash
py -3.11 -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Quickstart (Synthetic Demo)

Run the default synthetic demo:

```bash
python -m scripts.run_pipeline --outdir runs/demo --seed 7
```

This writes:
- `auc_over_iters.png` — AUC improving over iterations
- `top_sampling_probs.png` — stabilized top‑20 sampling probabilities (no “one gene = 1.0” collapse)
- `km_curves_by_group.png` — KM curves by risk stratum
- `best_edges.txt`, `run_metadata.json`

Run pipeline fill usage:
run_pipeline.py [-h] [--outdir OUTDIR] [--seed SEED] [--data {sim,tcga}] [--endpoint {PFI,OS}] [--n_genes N_GENES]
                       [--subset_size SUBSET_SIZE] [--iters ITERS] [--temperature TEMPERATURE] [--epsilon EPSILON]
                       [--p_min P_MIN] [--ema_alpha EMA_ALPHA] [--inject_signal] [--causal_bn] [--module_causal]
                       [--n_blocks N_BLOCKS] [--module_bins MODULE_BINS] [--tcga_cache TCGA_CACHE]

### How does the sampling probability loop run

The adaptive sampler is **stabilized** to prevent premature collapse:
- score updates in logit/score space + **softmax temperature**
- **EMA smoothing**
- **epsilon-greedy** exploration
- probability floor `p_min`

You can tune these knobs:

```bash
python -m scripts.run_pipeline --temperature 1.2 --epsilon 0.08 --ema_alpha 0.12 --p_min 1e-4
```

---

## TCGA Mode (End-to-End on Real PRAD)

This mode downloads public data from **UCSC Xena** and runs the same pipeline.

```bash
python -m scripts.run_pipeline \
  --data tcga \
  --endpoint PFI \
  --n_genes 2000 \
  --tcga_cache data/tcga \
  --outdir runs/tcga_pfi \
  --seed 7
```c

Notes:
- TCGA runs are slower.
- PRAD survival endpoints can be noisy; **modest AUC is normal**.
- We subset to the **top-variable genes** to keep it runnable.

---

## Causal / Explainability Extensions

### 1) Gene-level do-intervention (model-based BN sensitivity)

Runs **do-operator interventions inside the learned BN**:

```bash
python -m scripts.run_pipeline --outdir runs/demo_causal --seed 7 --causal_bn
```

Output:
- `causal_bn_effects.json` (top genes ranked by Δ P(worst risk) under do-intervention)

**Important caveat:**  
On TCGA this should be described as *model-based sensitivity / what-if analysis* unless you add a full causal identification strategy.

---

### 2) Module-level doWhy-style ATE (recommended “causal story” for TCGA)

This is the most relevant “SCM-ish” extension for your use case:
- build **module eigengenes** from correlated blocks
- define treatment as **module-high vs module-low** (drop mid bin)
- define outcome as **worst KM risk vs rest**
- estimate **ATE** using **AIPW (doubly robust)** with clinical adjustment if covariates are available
- run a simple **placebo refuter** (shuffle treatment)

Run on TCGA:

```bash
python -m scripts.run_pipeline \
  --data tcga \
  --endpoint PFI \
  --n_genes 2000 \
  --module_causal \
  --n_blocks 8 \
  --module_bins 3 \
  --outdir runs/tcga_pfi_ate \
  --seed 7
```

Output:
- `module_ate_results.json` containing:
  - covariates actually found/used (age/gleason/stage if present)
  - top modules by |ATE|
  - placebo ATE (should be near 0 if the estimate is not spurious)

**How to describe this in interviews (accurate + strong):**
- “I implemented a doWhy-style ATE pipeline at the module level with backdoor adjustment on available clinical covariates (when present).”
- “This gives an interpretable ‘pathway shift’ effect on risk, and it’s more stable than gene-level effects.”

---

## What changed vs the original version

1) **More realistic synthetic survival**: survival can be driven by *distributed pathway-like signal* (modules), avoiding “one magic gene”
2) **Stabilized adaptive sampler**: prevents probability collapse; improves run-to-run behavior
3) **TCGA mode via Xena**: optional end-to-end real PRAD run
4) **Causal extensions**:
   - BN do-intervention sensitivity
   - module-level doWhy-style ATE (AIPW + placebo refuter)

---

## Repo layout

- `src/prad_bn/simulate.py` — realistic PRAD-like simulator
- `src/prad_bn/iterative.py` — adaptive BN loop (stabilized sampler)
- `src/prad_bn/tcga.py` — TCGA PRAD loader (UCSC Xena)
- `src/prad_bn/causal_bn.py` — do-intervention inside BN (sensitivity)
- `src/prad_bn/causal_ate.py` — doWhy-style ATE (AIPW + placebo)
- `src/prad_bn/moduleize.py` — modules/eigengenes for module-level causal modeling
- `scripts/run_pipeline.py` — CLI entry point (sim + tcga + causal)
