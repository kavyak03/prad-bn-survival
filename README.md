# BioGraphX

BioGraphX is a modular probabilitistic discovery framework for discovering biologically meaningful gene signals from high-dimensional omics data using iterative Bayesian network modeling and causal inference. It supports both simulated and real-world transcriptomic workflows, identifies feature subsets associated with outcome risk, and extends those discoveries into module-level causal effect estimation with clinical covariate adjustment. The framework is designed to be reusable across disease areas and phenotypes, not just a single cancer use case. It combines interpretable probabilistic modeling, robust data engineering, and biologically grounded feature discovery in one end-to-end pipeline.

**Bayesian Network--Based Causal Survival Modeling in Prostate Cancer**

This repo showcases a reproducible pipeline for discovering gene modules associated with
prostate cancer survival using **Bayesian networks, causal inference,
and TCGA PRAD transcriptomics** through BioGraphX. It demonstrates how **structure learning + causal
inference** can identify survival-associated gene modules while
adjusting for **clinical confounders (age, tumor stage, Gleason
score)**.

The project supports:

• Synthetic simulations for validating causal inference behavior\
• Real TCGA PRAD analysis using transcriptomics and GDC clinical data

------------------------------------------------------------------------

# Project Motivation

Gene expression signatures are often correlated with survival but may be
**confounded by clinical variables** such as tumor stage or patient age.

This repository explores two complementary causal inference strategies:

1.  Doubly Robust ATE estimation\
2.  Bayesian Network do-interventions

to identify gene modules that may **causally influence patient
survival**.

------------------------------------------------------------------------

# Pipeline Overview

Below is the conceptual workflow implemented in this repository.

    TCGA / Simulated Gene Expression
                    │
                    ▼
            Expression QC
       (distribution + correlation)
                    │
                    ▼
            Gene Subset Sampling
                    │
                    ▼
            Expression Discretization
                    │
                    ▼
       Iterative Bayesian Network Learning
                    │
                    ▼
         Survival Risk Group Discovery
                    │
            ┌───────┴────────┐
            ▼                ▼
     Module-level        BN do-intervention
     causal analysis     sensitivity analysis
     (ATE estimation)    P(Y | do(Gene))

This design allows both **predictive modeling and causal
interpretation**.

------------------------------------------------------------------------

# Bayesian Network Intuition

Typical causal structure:

Age ─┐\
Stage ─┼──► Survival\
Gleason ┘

Gene_1 ──►\
Gene_2 ──► Survival\
Gene_3 ──►

Instead of estimating:

P(Survival \| Gene)

the model attempts to approximate:

P(Survival \| do(Gene))

which reflects **intervention-style causal effects**.

------------------------------------------------------------------------

# Expression QC and Biological Structure

When running with **TCGA data**, the pipeline automatically generates QC
plots in the run directory.

### Right-Skewed Expression Distribution

`tcga_expr_right_skew.png`

RNA-seq expression values typically follow a **right-skewed
distribution**, where most genes have low expression and a few are
highly expressed.

This confirms the data behaves as expected for transcriptomic
measurements.

------------------------------------------------------------------------

### Block-Correlated Gene Modules

`tcga_expr_block_corr.png`

A clustered gene--gene correlation heatmap reveals **co-regulated
transcriptional modules**.

These correlated gene blocks often correspond to:

• shared regulatory programs\
• pathway activation\
• tumor subtype biology

This structure motivates the **module-level causal analysis** performed
later.

------------------------------------------------------------------------

### Heatmap Gene Subset

To keep visualization interpretable and computationally efficient, the
heatmap uses **top variable genes only**.

Default:

--qc_corr_genes 350

Example override:

python -m scripts.run_pipeline\
--data tcga\
--endpoint PFI\
--n_genes 8000\
--qc_corr_genes 500

Typical settings:

  Genes           Effect
  --------------- -------------------------
  200--300        faster visualization
  350 (default)   balanced clarity
  500--600        richer module structure
  \>1000          not recommended

------------------------------------------------------------------------

# Optional Covariates

Clinical covariates can be incorporated into the analysis.

Available from GDC clinical data:

• Age at diagnosis\
• Tumor stage\
• Gleason score

Enable adjustment in the BN using:

--bn_covariates

This allows the network to model relationships such as:

Age → Survival\
Stage → Survival\
Gleason → Survival

------------------------------------------------------------------------

# Synthetic Data Experiments

Simulation enables controlled causal validation.

### Simulated Covariates

--sim_covariates

Adds simulated:

• Age\
• Stage\
• Gleason

### Simulated Confounding

--sim_confound_survival

Creates ground-truth relationships:

Covariates → Survival\
Gene modules → Survival

This allows testing whether causal methods recover unbiased effects.

------------------------------------------------------------------------

# Installation

Clone the repository.

git clone https://github.com/YOUR_USERNAME/prad-bn-survival.git\
cd prad-bn-survival

Create virtual environment.

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

### Synthetic Demo

python -m scripts.run_pipeline --outdir runs/demo --seed 7

------------------------------------------------------------------------

### Synthetic + Covariates

python -m scripts.run_pipeline\
--sim_covariates\
--sim_confound_survival\
--outdir runs/demo_cov

------------------------------------------------------------------------

### TCGA PRAD Analysis

python -m scripts.run_pipeline\
--data tcga\
--endpoint PFI\
--n_genes 8000\
--tcga_cache data/tcga\
--seed 7\
--iters 60\
--subset_size 30\
--outdir runs/tcga

------------------------------------------------------------------------

### TCGA + Clinical Covariates

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

### Full Causal Analysis

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

------------------------------------------------------------------------

# Output Files

Results are written to the runs/ directory.

Typical outputs:

runs/ ├── best_gene_sets.csv\
├── survival_groups.csv\
├── causal_module_effects.csv\
├── bn_structure.json\
├── tcga_expr_right_skew.png\
├── tcga_expr_block_corr.png\
└── plots/

------------------------------------------------------------------------

# Scientific Contributions

• Bayesian network structure learning on transcriptomics\
• Doubly robust causal effect estimation\
• Integration of clinical confounders from GDC\
• Simulation framework for causal validation

------------------------------------------------------------------------

# Future Improvements

• Graph neural networks for pathway discovery\
• Time-to-event causal modeling\
• Multi-omics integration\
• Automatic DAG visualization
