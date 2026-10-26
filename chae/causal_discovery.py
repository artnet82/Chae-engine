"""Open-source causal discovery: PC + Granger only."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- #
# PC algorithm
# --------------------------------------------------------------------------- #
def discover_pc(data, alpha: float = 0.05, labels: Optional[Sequence[str]] = None,
                strength_scale: float = 1.0) -> nx.DiGraph:
    """Run the PC algorithm on a numeric matrix and return a DiGraph."""
    try:
        from pcalg_py.pc import pc
        from pcalg_py.indep_test import gaussCItest, SuffStat
    except ImportError as exc:
        raise ImportError("pip install -e '.[discovery]'") from exc

    arr = np.asarray(data, dtype=float)
    n_samples, n_features = arr.shape
    labels = list(labels) if labels else [f"V{i}" for i in range(n_features)]
    
    # Считаем корреляционную матрицу и добавляем микро-шум для стабильности
    corr = np.corrcoef(arr.T) + np.eye(n_features) * 1e-8
    suff = SuffStat(C=corr, n=n_samples)
    
    # Запускаем PC-алгоритм
    cpdag = pc(suff, gaussCItest, alpha=alpha, labels=labels)["graph"]

    g = nx.DiGraph()
    g.add_nodes_from(labels)
    for u, v in cpdag.edges():
        try:
            pc_val = float(abs(corr[labels.index(u), labels.index(v)]))
        except Exception:
            pc_val = 0.5
        g.add_edge(u, v, strength=round(min(1.0, pc_val) * strength_scale, 4), evidence="PC-algorithm")
    return g


# --------------------------------------------------------------------------- #
# Granger causality
# --------------------------------------------------------------------------- #
def discover_granger(df, maxlag: int = 3, alpha: float = 0.05,
                     strength_scale: float = 1.0) -> nx.DiGraph:
    """Pairwise Granger causality test → DiGraph."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as exc:
        raise ImportError("pip install -e '.[timeseries]'") from exc

    cols = list(df.columns)
    g = nx.DiGraph()
    g.add_nodes_from(cols)
    
    for target in cols:
        for source in cols:
            if source == target:
                continue
                
            # ВАЖНО: конвертируем в numpy массив, чтобы statsmodels не сбоил в Linux/CI
            pair = df[[target, source]].dropna().to_numpy()
            if len(pair) <= maxlag + 2:
                continue
                
            try:
                # Подавляем предупреждения statsmodels, которые ломают CI
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
            except Exception:
                continue
                
            # Собираем все p-value из результатов
            pvals = []
            for lag in res:
                for test_name, stats in res[lag][0].items():
                    p = stats[1]
                    if isinstance(p, (int, float)) and not np.isnan(p):
                        pvals.append(float(p))
                        
            # Если хотя бы один лаг показал значимость — добавляем ребро
            if pvals and min(pvals) < alpha:
                p_min = min(pvals)
                g.add_edge(source, target,
                           strength=round(min(1.0, 1.0 - p_min) * strength_scale, 4),
                           evidence=f"Granger (min p={p_min:.3g})")
    return g


# --------------------------------------------------------------------------- #
# Merge into CausalGraph
# --------------------------------------------------------------------------- #
def merge_into_causal(causal_core, discovered: nx.DiGraph, evidence_prefix: str = "discovered") -> int:
    """Copy discovered edges into a CausalCore / CausalGraph."""
    cg = causal_core.cg if hasattr(causal_core, "cg") else causal_core
    n = 0
    for u, v, d in discovered.edges(data=True):
        cg.add_cause(u, v, float(d.get("strength", 0.5)),
                     f"{evidence_prefix}:{d.get('evidence', 'discovered')}")
        n += 1
    return n