# tests/test_discovery.py
import numpy as np
import pandas as pd
import networkx as nx
from chae.causal import CausalCore
from chae.causal_discovery import discover_granger, discover_pc, merge_into_causal


def test_granger_recovers_chain():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(0, 1, n)
    b = 0.8 * a + rng.normal(0, 0.2, n)
    c = 0.7 * b + rng.normal(0, 0.2, n)
    g = discover_granger(pd.DataFrame({"a": a, "b": b, "c": c}), maxlag=3, alpha=0.01)
    edges = set(g.edges())
    assert ("a", "b") in edges or ("b", "c") in edges


def test_pc_runs_on_synthetic():
    rng = np.random.default_rng(42)
    n = 300
    x = rng.normal(0, 1, n)
    y = 0.9 * x + rng.normal(0, 0.3, n)
    z = 0.8 * y + rng.normal(0, 0.3, n)
    g = discover_pc(np.column_stack([x, y, z]), alpha=0.05, labels=["x", "y", "z"])
    assert set(g.nodes()) == {"x", "y", "z"}


def test_merge_into_causal():
    core = CausalCore(seed_priors=False)
    g = nx.DiGraph()
    g.add_edge("a", "b", strength=0.7, evidence="test")
    assert merge_into_causal(core, g) == 1
    assert core.cg.graph["a"]["b"]["strength"] == 0.7