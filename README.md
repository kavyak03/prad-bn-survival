# PRAD-BN-Survival

**Bayesian Network--Based Causal Survival Modeling in Prostate Cancer**

A reproducible pipeline for discovering gene modules associated with
prostate cancer survival using **Bayesian networks, causal inference,
and TCGA PRAD transcriptomics**.

The repository demonstrates how **structure learning + causal
inference** can identify survival-associated gene modules while
adjusting for **clinical confounders (age, tumor stage, Gleason
score)**.

The project supports both:

-   **Synthetic simulations** for validating causal inference behavior
-   **Real TCGA PRAD analysis** using transcriptomics and GDC clinical
    data

------------------------------------------------------------------------

# Project Motivation

Gene expression signatures are often correlated with survival but may be
**confounded by clinical variables** such as tumor stage or patient age.

This repository explores two complementary causal inference strategies:

1.  **Doubly Robust ATE estimation**
2.  **Bayesian Network do-interventions**

to identify gene modules that may **causally influence patient
survival**.

------------------------------------------------------------------------

# High Level Pipeline

Gene Expression Data\
↓\
Gene Subset Sampling\
↓\
Discretization\
↓\
Bayesian Network Structure Learning\
↓\
Outcome Prediction / Survival Grouping

Branches: - Module Causal Effect (ATE) - BN Intervention Analysis

------------------------------------------------------------------------

# Bayesian Network Intuition

Typical causal structure:

Age ─┐\
Stage ─┼──► Survival\
Gleason ┘

Gene_1 ──►\
Gene_2 ──► Survival\
Gene_3 ──►

Causal inference estimates:

P(Survival \| do(Gene_i = high))

instead of the observational:

P(Survival \| Gene_i = high)

------------------------------------------------------------------------

# Causal Inference Modes

  -----------------------------------------------------------------------
  Mode                    Flag                    Description
  ----------------------- ----------------------- -----------------------
  Baseline survival       *(default)*             BN structure search +
  grouping                                        Kaplan--Meier grouping

  Module causal inference `--module_causal`       Doubly robust ATE
                                                  estimation for gene
                                                  modules

  Bayesian network        `--causal_bn`           Estimate causal effects
  interventions                                   using BN
                                                  do-interventions
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Optional Covariates

Covariates can be incorporated into the pipeline.

### Available Covariates

From GDC clinical data:

-   Age at diagnosis\
-   Tumor stage\
-   Gleason score

Enable covariate adjustment in the BN with:

--bn_covariates

------------------------------------------------------------------------

# Synthetic Data Experiments

The simulation framework allows controlled causal experiments.

### Simulated Covariates

--sim_covariates

Adds simulated:

-   Age\
-   Stage\
-   Gleason

### Simulated Confounding

--sim_confound_survival

Creates ground truth:

Covariates → Survival\
Gene Modules → Survival

This allows testing whether causal inference methods recover unbiased
gene effects.

------------------------------------------------------------------------

# Installation

Clone the repository.

git clone https://github.com/YOUR_USERNAME/prad-bn-survival.git\
cd prad-bn-survival

Create a virtual environment.

Windows:

py -m venv .venv\
.venv`\Scripts`{=tex}`\activate`{=tex}

Linux / Mac:

python3 -m venv .venv\
source .venv/bin/activate

Install dependencies.

pip install -r requirements.txt

------------------------------------------------------------------------

# Example Runs

## 1. Synthetic Demo

python -m scripts.run_pipeline --outdir runs/demo --seed 7

------------------------------------------------------------------------

## 2. Synthetic + Covariates

python -m scripts.run_pipeline\
--sim_covariates\
--sim_confound_survival\
--outdir runs/demo_cov

------------------------------------------------------------------------

## 3. TCGA PRAD Survival Analysis

python -m scripts.run_pipeline\
--data tcga\
--endpoint PFI\
--n_genes 8000\
--tcga_cache data/tcga\
--outdir runs/tcga
--seed 7\
--iters 60\
--subset_size 30\

------------------------------------------------------------------------

## 4. TCGA + Clinical Covariates + causal inference

Download the TCGA PRAD clinical file from GDC.

python -m scripts.run_pipeline\
--data tcga\
--endpoint PFI\
--clinical_file data/tcga/clinical.zip\
--bn_covariates\
--module_causal\
--n_genes 8000\
--tcga_cache data/tcga\
--seed 7\
--iters 60\
--subset_size 30\
--outdir runs/tcga_clinical

------------------------------------------------------------------------

## 5. Full Causal Analysis with bayesian network interventions

python -m scripts.run_pipeline\
--data tcga\
--endpoint PFI\
--clinical_file data/tcga/clinical.zip\
--bn_covariates\
--module_causal\
--causal_bn\
--n_genes 8000\
--tcga_cache data/tcga\
--seed 7\
--iters 60\
--subset_size 30\
--outdir runs/tcga_causal

This performs:

1.  BN structure learning\
2.  Module-level causal ATE estimation\
3.  Bayesian network intervention analysis

------------------------------------------------------------------------

# Output Files

Results are written to the runs/ directory.

Typical outputs:

runs/\
├── best_gene_sets.csv\
├── survival_groups.csv\
├── causal_module_effects.csv\
├── bn_structure.json\
└── plots/

------------------------------------------------------------------------

# Repository Structure

scripts/\
run_pipeline.py

src/prad_bn/\
tcga.py\
clinical_gdc.py\
iterative.py\
causal_effects.py

docs/\
bn_structure.png

------------------------------------------------------------------------

# Scientific Contributions

• Bayesian network structure learning on transcriptomics\
• Doubly robust causal effect estimation\
• Integration of clinical confounders from GDC\
• Simulation framework for causal validation

------------------------------------------------------------------------

# Future Improvements

-   Graph neural networks for pathway discovery\
-   Time-to-event causal modeling\
-   Multi-omics integration\
-   Automatic DAG visualization
