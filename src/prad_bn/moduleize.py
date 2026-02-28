"""Module-level features for more stable, biologically meaningful interventions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass(frozen=True)
class ModuleSpec:
    name: str
    gene_indices: List[int]

def modules_from_blocks(n_genes: int, n_blocks: int = 8) -> List[ModuleSpec]:
    block_size = int(np.ceil(n_genes / n_blocks))
    specs=[]
    for b in range(n_blocks):
        s=b*block_size
        e=min(n_genes,(b+1)*block_size)
        if s>=e: break
        specs.append(ModuleSpec(name=f"M{b}", gene_indices=list(range(s,e))))
    return specs

def module_eigengenes(expr: np.ndarray, specs: List[ModuleSpec]) -> np.ndarray:
    n=expr.shape[0]
    M=np.zeros((n,len(specs)),dtype=float)
    for i,spec in enumerate(specs):
        X=expr[:, spec.gene_indices]
        X=X - X.mean(axis=0, keepdims=True)
        # SVD-based PC1 projection
        U,S,Vt=np.linalg.svd(X, full_matrices=False)
        M[:,i]=U[:,0]*S[0]
    return M

def quantile_bin(x: np.ndarray, n_bins:int=3) -> np.ndarray:
    qs=np.quantile(x, np.linspace(0,1,n_bins+1))
    for k in range(1,len(qs)):
        if qs[k] <= qs[k-1]:
            qs[k]=qs[k-1]+1e-9
    return np.clip(np.digitize(x, qs[1:-1], right=False), 0, n_bins-1).astype(int)

def discretize_modules(M: np.ndarray, n_bins:int=3) -> np.ndarray:
    out=np.zeros_like(M,dtype=int)
    for j in range(M.shape[1]):
        out[:,j]=quantile_bin(M[:,j], n_bins=n_bins)
    return out
