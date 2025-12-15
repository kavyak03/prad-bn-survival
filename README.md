# Bayesian netowrk model for predicting survival in prostate cancer patients

Implementation of an interpretable, probabilistic pipeline inspired by TCGA-PRAD:
- **Simulate** PRAD-like gene expression (right-skewed, block-correlated modules)
- **Simulate** long-tailed survival times + optional censoring
- **Discretize** survival into **5 Kaplan–Meier-style risk strata** (quantile bins)
- **Discretize** gene expression into bins suitable for Bayesian Networks
- **Iteratively learn** compact Bayesian Networks on small feature subsets (15–20 genes),
  with constraints and a **feedback-driven adaptive feature sampling loop**
- **Infer** per-patient posterior risk probabilities and evaluate with **AUC** (binary group-vs-rest and optional multiclass)
- Optional: **Kaplan–Meier curves** and **CoxPH** sanity checks on the simulated labels


## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate  # windows
pip install -r requirements.txt
```

### 2) Run the full pipeline
```bash
python scripts/run_pipeline.py --outdir runs/demo --seed 7
```

This will:
- generate data  (simulated data can be substituted with data of your choice in the same format)
- run the iterative BN loop
- print AUC
- save plots + artifacts to `runs/demo`

## Repo structure
- `src/prad_bn/` core library
- `scripts/run_pipeline.py` end-to-end runnable entrypoint
- `runs/` outputs (created at runtime)

## Notes on dependencies
This repo uses **pomegranate** for Bayesian Network structure learning/inference.
If you hit installation issues, try:
- `pip install "pomegranate>=1.0.0"` (or a compatible version for your platform)
- or use conda/mamba.

## License
MIT

