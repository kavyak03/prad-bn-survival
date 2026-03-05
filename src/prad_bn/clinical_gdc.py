"""GDC clinical covariate loader (TCGA PRAD) with robust sample alignment.

This module is intentionally lightweight and defensive:
  - Accepts TSV/CSV files OR a ZIP that contains a TSV/CSV.
  - Detects TCGA sample/case identifier columns by content ("TCGA-").
  - Extracts a small set of covariates (age, stage, gleason) if present.
  - Aligns covariates to an array of TCGA expression sample IDs using 12-char case IDs.

Output is a numeric numpy matrix Z (n_samples x k) and covariate names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import zipfile

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClinicalMap:
    id_col: Optional[str] = None
    age_col: Optional[str] = None
    stage_col: Optional[str] = None
    gleason_col: Optional[str] = None


def _read_table(path: Path) -> pd.DataFrame:
    """Read TSV/CSV with minimal guessing."""
    # try TSV then CSV
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _load_gdc_table(clinical_file: str) -> pd.DataFrame:
    p = Path(clinical_file)
    if not p.exists():
        raise FileNotFoundError(f"clinical_file not found: {p}")

    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            # pick the first TSV/CSV-like file
            members = [m for m in z.namelist() if m.lower().endswith((".tsv", ".txt", ".csv"))]
            if not members:
                # some GDC zips contain files without extensions; fall back to first file
                members = z.namelist()
            if not members:
                raise RuntimeError("clinical ZIP is empty")
            name = members[0]
            with z.open(name) as f:
                # try TSV then CSV
                try:
                    return pd.read_csv(f, sep="\t", low_memory=False)
                except Exception:
                    f.seek(0)
                    return pd.read_csv(f, low_memory=False)

    return _read_table(p)


def _looks_like_tcga_id(x: str) -> bool:
    return isinstance(x, str) and x.startswith("TCGA-") and len(x) >= 12


def _detect_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(200).tolist()
        if not vals:
            continue
        hits = sum(_looks_like_tcga_id(v) for v in vals)
        if hits >= 5:
            return c

    # common GDC names
    for c in ["submitter_id", "case_submitter_id", "bcr_patient_barcode", "sample", "case_id"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"Could not detect TCGA id column in clinical file. Columns: {list(df.columns)[:60]}")


def _find_col(df: pd.DataFrame, override: Optional[str], candidates: List[str], substr: List[str]) -> Optional[str]:
    if override and override in df.columns:
        return override
    lower = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k.lower() in lower:
            return lower[k.lower()]
    for c in df.columns:
        cl = c.lower()
        if any(s in cl for s in substr):
            return c
    return None


def _stage_to_ordinal(x: str) -> int:
    if not isinstance(x, str):
        return 0
    s = x.upper().strip()
    # normalize common prefixes
    s = s.replace("STAGE", "").replace(" ", "")
    # keep leading roman numeral group
    if s.startswith("IV"):
        return 4
    if s.startswith("III"):
        return 3
    if s.startswith("II"):
        return 2
    if s.startswith("I"):
        return 1
    return 0


def load_gdc_covariates_aligned(
    clinical_file: str,
    sample_ids: np.ndarray,
    mapping: ClinicalMap = ClinicalMap(),
) -> Tuple[Optional[np.ndarray], List[str], str]:
    """Load GDC clinical covariates and align to expression sample_ids.

    Returns:
      (Z, names, msg)
      - Z: float matrix (n_samples x k) aligned to sample_ids, or None if nothing extracted
      - names: covariate names (length k)
      - msg: human-readable status
    """
    raw = _load_gdc_table(clinical_file)

    id_col = mapping.id_col or _detect_id_col(raw)
    df = raw.copy()
    df[id_col] = df[id_col].astype(str)
    df["case_12"] = df[id_col].str[:12]

    age_col = _find_col(
        df,
        mapping.age_col,
        candidates=["age_at_diagnosis", "age_at_index", "age"],
        substr=["age_at", "age"],
    )
    stage_col = _find_col(
        df,
        mapping.stage_col,
        candidates=["ajcc_pathologic_stage", "tumor_stage", "pathologic_stage", "clinical_stage"],
        substr=["stage"],
    )
    gleason_col = _find_col(
        df,
        mapping.gleason_col,
        candidates=["gleason_score", "gleason", "gleason_sum"],
        substr=["gleason"],
    )

    cols = []
    names: List[str] = []

    if age_col is not None:
        cols.append(pd.to_numeric(df[age_col], errors="coerce"))
        names.append("age")
    if stage_col is not None:
        cols.append(df[stage_col].astype(str).map(_stage_to_ordinal))
        names.append("stage")
    if gleason_col is not None:
        cols.append(pd.to_numeric(df[gleason_col], errors="coerce"))
        names.append("gleason")

    if not cols:
        return None, [], f"No covariate columns found in clinical file (checked age/stage/gleason). id_col={id_col!r}"

    cov = pd.concat(cols, axis=1)
    cov.columns = names

    # Build lookup by 12-char case id
    cov["case_12"] = df["case_12"].values
    cov = cov.drop_duplicates("case_12").set_index("case_12")

    samp = pd.Series(sample_ids.astype(str))
    samp_12 = samp.str[:12]

    aligned = cov.reindex(samp_12.values)
    matched = int(aligned.notna().any(axis=1).sum())

    # Impute missing with median (per column) and ensure float
    for c in names:
        med = pd.to_numeric(aligned[c], errors="coerce").median()
        aligned[c] = pd.to_numeric(aligned[c], errors="coerce").fillna(med)

    Z = aligned[names].to_numpy(dtype=float)
    return Z, names, f"Loaded covariates {names}; matched {matched}/{len(sample_ids)} samples (by 12-char case id)."
