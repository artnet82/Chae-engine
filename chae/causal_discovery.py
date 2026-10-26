"""Open-source causal discovery: PC + Granger only."""
from __future__ import annotations

from typing import Optional, Sequence

import networkx as nx
import numpy as np


def discover_pc(data, alpha: float = 0.05, labels: Optional[Sequence[str]] = None,
                strength_scale: float = 1.0) -> nx.DiGraph:
    try:
        from pcalg_py.pc import pc
        from pcalg_py.indep_test import gaussCItest, SuffStat
    except ImportError as exc:
        raise ImportError("pip install -e '.[discovery]'") from exc

    arr = np.asarray(data, dtype=float)
    n_samples, n_features = arr.shape
    labels = list(labels) if labels else [f"V{i}" for i in range(n_features)]
    corr = np.corrcoef(arr.T) + np.eye(n_features) * 1e-8
    suff = SuffStat(C=corr, n=n_samples)
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


def discover_granger(df, maxlag: int = 3, alpha: float = 0.05,
                     strength_scale: float = 1.0) -> nx.DiGraph:
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
            pair = df[[target, source]].dropna()
            if len(pair) <= maxlag + 2:
                continue
            try:
                res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
            except Exception:
                continue
            pvals = [
                float(stats[1])
                for lag in res
                for stats in res[lag][0].values()
                if isinstance(stats[1], (int, float)) and not np.isnan(stats[1])
            ]
            if pvals and min(pvals) < alpha:
                p_min = min(pvals)
                g.add_edge(source, target,
                           strength=round(min(1.0, 1.0 - p_min) * strength_scale, 4),
                           evidence=f"Granger (min p={p_min:.3g})")
    return g


def merge_into_causal(causal_core, discovered: nx.DiGraph, evidence_prefix: str = "discovered") -> int:
    cg = causal_core.cg if hasattr(causal_core, "cg") else causal_core
    n = 0
    for u, v, d in discovered.edges(data=True):
        cg.add_cause(u, v, float(d.get("strength", 0.5)),
                     f"{evidence_prefix}:{d.get('evidence', 'discovered')}")
        n += 1
    return n