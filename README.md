# Bayesian netowrk model for predicting survival in prostate cancer patients

Implementation of an interpretable, probabilistic pipeline inspired by TCGA-PRAD data:
- **Simulate** PRAD-like gene expression (right-skewed, block-correlated modules)
- **Simulate** long-tailed survival times + optional censoring
- **Discretize** survival into **5 Kaplan–Meier-style risk strata** (quantile bins)
- **Discretize** gene expression into bins suitable for Bayesian Networks
- **Iteratively learn** compact Bayesian Networks on small feature subsets (15–20 genes),
  with constraints and a **feedback-driven adaptive feature sampling loop**
- **Infer** per-patient posterior risk probabilities and evaluate with **AUC** (binary group-vs-rest and optional multiclass)
- Optional: **Kaplan–Meier curves** and **CoxPH** sanity checks on the simulated labels

> ⚠️ This is a pedagogical repo. It uses synthetic data by default so you can run it anywhere.

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
- generate data
- run the iterative BN loop
- print AUC
- save plots + artifacts to `runs/demo`

## Repo structure
- `src/prad_bn/` core library
- `scripts/run_pipeline.py` end-to-end runnable entrypoint
- `runs/` outputs (created at runtime)

## Main concepts
1. **Outcome engineering**: convert survival to 5 KM-like risk strata (`discretize.py`)
2. **Simulation**: block-correlated genes + weak outcome signal (`simulate.py`)
3. **Discretization**: bins for BN CPDs (`discretize.py`)
4. **Iterative BN**: sample 15–20 genes, constrain graph, prune to outcome-connected subgraph, evaluate, update sampling (`iterative.py`)
5. **Inference**: posterior `P(risk_group | gene_bins)` (`bn_model.py`)
6. **Evaluation**: AUC + KM/Cox sanity checks (`evaluate.py`)

## License
MIT

