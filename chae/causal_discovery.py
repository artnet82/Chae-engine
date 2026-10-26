"""Open-source causal discovery: PC + Granger only."""
from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# Granger causality (custom OLS implementation)
# --------------------------------------------------------------------------- #
def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
    """Return Residual Sum of Squares for OLS regression y ~ X."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coef
    return float(np.sum((y - y_pred) ** 2))


def discover_granger(df, maxlag: int = 3, alpha: float = 0.05,
                     strength_scale: float = 1.0) -> nx.DiGraph:
    """Pairwise Granger causality test via F-test → DiGraph."""
    try:
        from scipy.stats import f as f_dist
    except ImportError as exc:
        raise ImportError("pip install scipy") from exc

    cols = list(df.columns)
    g = nx.DiGraph()
    g.add_nodes_from(cols)

    for target in cols:
        for source in cols:
            if source == target:
                continue
            
            # Принудительно приводим к float и numpy
            pair = df[[target, source]].dropna().to_numpy(dtype=float)
            n = len(pair)
            if n <= maxlag + 2:
                continue

            y = pair[:, 0]
            x = pair[:, 1]

            # Сдвигаем временные ряды (лаги)
            Y = y[maxlag:]
            # Ограниченная модель: только прошлое Y
            X_r = np.column_stack(
                [np.ones(n - maxlag)] + [y[maxlag - k - 1 : n - k - 1] for k in range(maxlag)]
            )
            # Неограниченная модель: прошлое Y + прошлое X
            X_u = np.column_stack(
                [X_r] + [x[maxlag - k - 1 : n - k - 1] for k in range(maxlag)]
            )

            rss_r = _ols_rss(X_r, Y)
            rss_u = _ols_rss(X_u, Y)

            df1 = maxlag
            df2 = n - maxlag - X_u.shape[1]
            
            if df2 <= 0 or rss_u <= 1e-12:
                continue

            # F-статистика
            f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
            # P-value через распределение Фишера
            p_val = float(f_dist.sf(f_stat, df1, df2))

            if p_val < alpha:
                strength = float(min(1.0, 1.0 - p_val) * strength_scale)
                g.add_edge(source, target, strength=round(strength, 4),
                           evidence=f"Granger (p={p_val:.3g})")
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