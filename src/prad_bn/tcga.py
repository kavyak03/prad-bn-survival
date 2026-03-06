"""TCGA-PRAD data loader via UCSC Xena public hubs.

This keeps the repo lightweight: it downloads gene expression + phenotype at runtime and caches locally.

Notes:
- This is an *end-to-end demo* for interviews; results can be modest and that is expected.
- We use Xena Toil RSEM (or TCGA gene expression matrix) depending on availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Iterable

import io
import pandas as pd
import numpy as np
import requests
import gzip


XENA_TOIL_RSEM_URL = "https://toil.xenahubs.net/download/tcga_RSEM_gene_tpm.gz"

XENA_PHENO_URLS = [
    "https://pancanatlas.xenahubs.net/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
]

XENA_SURV_URLS = [
    "https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp.gz",
    "https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download/Survival_SupplementalTable_S1_20171025_xena_sp",
]

@dataclass(frozen=True)
class TcgaConfig:
    cache_dir: str = "data/tcga"
    tumor: str = "TCGA-PRAD"
    n_genes: int = 2000  # top variable genes within PRAD
    endpoint: str = "PFI"  # OS or PFI (if available)
    random_seed: int = 7

def _detect_tcga_id_col(df: pd.DataFrame) -> str:
    """
    Detect the column containing TCGA sample IDs (e.g., 'TCGA-XX-YYYY...') by content.
    Falls back to common names if needed.
    """
    def looks_like_tcga_id(s: str) -> bool:
        return isinstance(s, str) and s.startswith("TCGA-") and len(s) >= 12

    # content-based detection
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(80).tolist()
        if not vals:
            continue
        hits = sum(looks_like_tcga_id(v) for v in vals)
        if hits >= 5:
            return c

    # common-name fallback
    for c in ["sample", "sampleID", "Sample", "id", "ID"]:
        if c in df.columns:
            return c

    raise RuntimeError(f"Could not detect TCGA sample id column. Columns: {list(df.columns)[:40]}")

def _read_tsv_auto(path: Path) -> pd.DataFrame:
    """
    Read TSV that may or may not be gzipped, regardless of filename extension.
    Detect gzip by magic bytes 1f 8b.
    """
    with open(path, "rb") as f:
        magic = f.read(2)

    is_gz = (magic == b"\x1f\x8b")
    if is_gz:
        return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    return pd.read_csv(path, sep="\t", low_memory=False)

def _download_first(urls: List[str], dest: Path) -> Path:
    last_err = None
    for url in urls:
        try:
            return _download(url, dest)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download from all mirrors. Last error: {last_err}")

def _read_expr_subset_gz_stream(expr_path: Path, prad_samples: List[str]) -> pd.DataFrame:
    """
    Stream-read huge gzipped Xena expression TSV and keep only PRAD sample columns.
    Memory-stable version: preallocates a float32 matrix and fills row-by-row
    (avoids Python lists + final vstack allocation).
    Returns: DataFrame indexed by gene, columns = matched samples, dtype=float32
    """
    # --- header: determine columns ---
    with gzip.open(expr_path, "rt") as f:
        header = f.readline().rstrip("\n")
    cols = header.split("\t")
    gene_col = cols[0]
    expr_cols = cols[1:]

    prad_set = set(map(str, prad_samples))
    prad_12 = {str(s)[:12] for s in prad_samples}

    matched = [c for c in expr_cols if c in prad_set]
    if len(matched) < 50:
        matched = [c for c in expr_cols if c[:12] in prad_12]

    if len(matched) < 50:
        raise RuntimeError(f"Too few PRAD samples matched in expression header (matched={len(matched)}).")

    pos = {c: i for i, c in enumerate(expr_cols)}
    keep_pos = [pos[c] for c in matched]  # positions within row_vals = parts[1:]

    # --- count genes (rows) without loading whole file ---
    n_genes = 0
    with gzip.open(expr_path, "rt") as f:
        _ = f.readline()
        for _ in f:
            n_genes += 1

    # --- preallocate matrix ---
    n_samples = len(matched)
    mat = np.empty((n_genes, n_samples), dtype=np.float32)
    genes: List[str] = [None] * n_genes  # type: ignore

    # --- fill ---
    i = 0
    with gzip.open(expr_path, "rt") as f:
        _ = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            genes[i] = parts[0]
            row_vals = parts[1:]

            # fill selected samples
            for j, p in enumerate(keep_pos):
                v = row_vals[p]
                if v == "" or v.lower() == "nan":
                    mat[i, j] = np.nan
                else:
                    # float32 conversion
                    mat[i, j] = np.float32(v)
            i += 1

    expr = pd.DataFrame(mat, index=pd.Index(genes, name="gene"), columns=matched)
    return expr

def _download(url: str, dest: Path) -> Path:
    """
    Download URL to dest if not already present.
    Adds sanity check to avoid caching HTML/XML error pages (common with S3 403).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    import requests

    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # sanity check: detect HTML/XML error pages cached as "data"
    with open(dest, "rb") as f:
        head = f.read(400).lower()

    if (b"<html" in head) or (b"accessdenied" in head) or (b"<?xml" in head):
        # delete bad cache so next run can retry
        try:
            dest.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Downloaded error page instead of data from {url}")

    return dest

def _read_tsv_gz(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)

def _prad_samples_from_pheno(pheno: pd.DataFrame, tumor: str) -> pd.DataFrame:
    """
    Robustly subset Xena phenotype table to PRAD samples.

    Handles multiple phenotype formats:
    - classic pancan phenotype tables
    - minimal phenotype tables containing `_primary_disease`
    """

    pheno = pheno.copy()

    # Ensure sample column exists
    if "sample" not in pheno.columns:
        pheno = pheno.rename(columns={pheno.columns[0]: "sample"})

    pheno["sample"] = pheno["sample"].astype(str)

    # --- 1. Preferred: use _primary_disease column ---
    if "_primary_disease" in pheno.columns:
        disease = pheno["_primary_disease"].astype(str).str.upper()
        mask = disease.str.contains("PROSTATE", na=False) | disease.str.contains("PRAD", na=False)

        prad = pheno.loc[mask].copy()

        if len(prad) < 100:
            raise RuntimeError(
                "PRAD filtering using `_primary_disease` returned too few samples."
            )

        return prad

    # --- 2. Try generic cohort/cancer columns ---
    cand_cols = []
    for c in pheno.columns:
        cl = c.lower()
        if any(k in cl for k in ["cancer", "cohort", "study", "project", "disease", "tumor"]):
            cand_cols.append(c)

    for c in cand_cols:
        s = pheno[c].astype(str).str.upper()
        mask = s.str.contains("PRAD", na=False) | s.str.contains("PROSTATE", na=False)

        if mask.sum() > 100:
            return pheno.loc[mask].copy()

    raise RuntimeError(
        "Could not identify PRAD samples in phenotype table. "
        f"Available columns: {list(pheno.columns)}"
    )

def _pick_endpoint(pheno: pd.DataFrame, endpoint: str) -> Tuple[np.ndarray, np.ndarray]:
    endpoint = endpoint.upper().strip()
    # Common Xena dense phenotype columns: OS.time, OS, PFI.time, PFI, DSS, DFI
    # We'll try a few variants
    candidates = []
    if endpoint == "OS":
        candidates = [("OS.time", "OS"), ("OS.time", "OS.event"), ("OS_time", "OS"), ("overall_survival_time", "overall_survival")]
    elif endpoint == "PFI":
        candidates = [("PFI.time", "PFI"), ("PFI.time", "PFI.event"), ("PFI_time", "PFI")]
    else:
        raise ValueError("endpoint must be OS or PFI")

    for tcol, ecol in candidates:
        if tcol in pheno.columns and ecol in pheno.columns:
            t = pd.to_numeric(pheno[tcol], errors="coerce").to_numpy()
            e = pd.to_numeric(pheno[ecol], errors="coerce").to_numpy()
            e = (e > 0).astype(int)
            return t, e

    # fallback: try any columns containing endpoint substrings
    time_cols = [c for c in pheno.columns if endpoint.lower() in c.lower() and "time" in c.lower()]
    event_cols = [c for c in pheno.columns if endpoint.lower() in c.lower() and ("event" in c.lower() or c.lower().endswith(endpoint.lower()))]
    if time_cols and event_cols:
        t = pd.to_numeric(pheno[time_cols[0]], errors="coerce").to_numpy()
        e = pd.to_numeric(pheno[event_cols[0]], errors="coerce").to_numpy()
        e = (e > 0).astype(int)
        return t, e

    raise ValueError(f"Could not find {endpoint} time/event columns in phenotype table.")

def _extract_covariates(pheno: pd.DataFrame) -> pd.DataFrame:
    # keep a small, defensible set if present. Missing columns are ignored.
    cand = [
        "age_at_initial_pathologic_diagnosis",
        "age_at_diagnosis",
        "gleason_score",
        "ajcc_pathologic_tumor_stage",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "tumor_stage",
        "race",
    ]
    keep = [c for c in cand if c in pheno.columns]
    Z = pheno[keep].copy() if keep else pd.DataFrame(index=pheno.index)
    # encode categoricals
    for c in Z.columns:
        if Z[c].dtype == object:
            Z[c] = Z[c].astype(str).fillna("NA")
    # One-hot encode categoricals, keep numeric as is
    if not Z.empty:
        Z = pd.get_dummies(Z, drop_first=True)
        for c in Z.columns:
            Z[c] = pd.to_numeric(Z[c], errors="coerce")
        Z = Z.fillna(Z.median(numeric_only=True))
    return Z

def load_tcga_prad(cfg: TcgaConfig) -> Dict[str, object]:
    cache = Path(cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    # --- Download expression (cached) ---
    expr_path = _download(XENA_TOIL_RSEM_URL, cache / "tcga_RSEM_gene_tpm.gz")

    # --- Download phenotype with URL fallbacks (cached) ---
    pheno_path = _download_first(XENA_PHENO_URLS, cache / "TCGA_phenotype_dense.tsv.gz")
    pheno = _read_tsv_auto(pheno_path)

    # --- Select PRAD samples from phenotype ---
    prad = _prad_samples_from_pheno(pheno, cfg.tumor)
    if "sample" not in prad.columns:
        raise RuntimeError(f"_prad_samples_from_pheno must return a DataFrame with a 'sample' column. Got columns: {list(prad.columns)[:40]}")

    samples = prad["sample"].astype(str).tolist()

    # --- OOM-safe: stream-read expression subset (genes x samples), then transpose to (samples x genes) ---
    expr = _read_expr_subset_gz_stream(expr_path, samples)  # genes x matched_samples
    expr_prad = expr.T  # matched_samples x genes

    # --- Align phenotype rows to expression sample ids ---
    prad_idx = prad.copy()
    prad_idx["sample"] = prad_idx["sample"].astype(str)
    prad_idx = prad_idx.set_index("sample")

    pheno_keys = []
    for sid in expr_prad.index.astype(str):
        if sid in prad_idx.index:
            pheno_keys.append(sid)
        else:
            # fallback by 12-char TCGA case id
            pheno_keys.append(sid[:12])

    prad_aligned = prad_idx.loc[pheno_keys].copy()
    prad_aligned.index = expr_prad.index  # keep expression sample IDs
    prad_aligned.index.name = "sample"    # CRITICAL: ensures reset_index() produces 'sample'

    # --- Build phenotype frame used for endpoint selection ---
    ph_for_endpoint = prad_aligned.reset_index()

    # Defensive rename if pandas still uses an unexpected name
    if "sample" not in ph_for_endpoint.columns:
        for cand in ["index", "level_0", "Sample", "sampleID", "sample_id"]:
            if cand in ph_for_endpoint.columns:
                ph_for_endpoint = ph_for_endpoint.rename(columns={cand: "sample"})
                break
    if "sample" not in ph_for_endpoint.columns:
        raise RuntimeError(
            f"Could not construct 'sample' column after phenotype alignment. Columns: {list(ph_for_endpoint.columns)[:40]}"
        )

    # --- Endpoint selection: try phenotype first; if missing (PFI etc.), merge TCGA-CDR survival table ---
    try:
        time_days, event = _pick_endpoint(ph_for_endpoint, cfg.endpoint)
    except ValueError:
        # download survival table (contains PFI/OS/DFI/DSS endpoints)
        surv_path = _download_first(
            XENA_SURV_URLS,
            cache / "Survival_SupplementalTable_S1_20171025_xena_sp"
        )
        surv = _read_tsv_auto(surv_path)

        # detect sample id column in survival table, rename to "sample"
        id_col = _detect_tcga_id_col(surv)
        surv = surv.rename(columns={id_col: "sample"}).copy()
        surv["sample"] = surv["sample"].astype(str)

        # normalize join keys (12-char TCGA case id)
        ph_for_endpoint["sample_12"] = ph_for_endpoint["sample"].astype(str).str[:12]
        surv["sample_12"] = surv["sample"].astype(str).str[:12]

        merged = ph_for_endpoint.merge(
            surv.drop_duplicates("sample_12"),
            on="sample_12",
            how="left",
            suffixes=("", "_surv"),
        )

        # retry endpoint selection with merged survival columns
        time_days, event = _pick_endpoint(merged, cfg.endpoint)

    # --- Filter missing survival values ---
    mask = np.isfinite(time_days)
    expr_prad = expr_prad.loc[mask].copy()
    prad_aligned = prad_aligned.loc[mask].copy()
    time_days = time_days[mask]
    event = event[mask]

    # --- Convert to matrix ---
    X = expr_prad.to_numpy(dtype=np.float32)
    gene_names_all = expr_prad.columns.to_numpy()

    # --- Filter unusable genes ---
    # keep genes that have at least one finite value
    has_signal = np.isfinite(X).any(axis=0)

    # remove genes with zero or undefined variance
    vars_all = np.nanvar(X, axis=0)
    has_variance = np.isfinite(vars_all) & (vars_all > 0)

    valid = has_signal & has_variance

    X = X[:, valid]
    gene_names_all = gene_names_all[valid]
    vars_all = vars_all[valid]

    # --- Choose top variable genes within PRAD ---
    top = np.argsort(vars_all)[::-1][: cfg.n_genes]
    X = X[:, top]
    gene_names = gene_names_all[top]

    # --- Covariates (from phenotype-aligned table) ---
    Z = _extract_covariates(prad_aligned.reset_index())

    return {
        "expr": X,
        "gene_names": gene_names,
        "time_days": time_days.astype(float),
        "event": event.astype(int),
        "covariates": Z.to_numpy(dtype=float) if not Z.empty else None,
        "covariate_names": Z.columns.to_list() if not Z.empty else [],
        "sample_ids": expr_prad.index.to_numpy(),
    }
