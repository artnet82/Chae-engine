import numpy as np
import pandas as pd
import pytest
from chae import CHAE


def test_think_returns_plan():
    res = CHAE().think("Why does rain make the ground wet?")
    assert res["query"].startswith("Why") and len(res["plan"]["steps"]) >= 2


def test_tool_registration():
    e = CHAE()
    e.tools.register("echo", lambda query="": f"echo:{query}")
    assert e.tools.has("echo") and e.tools.call("echo", {"query": "hi"}) == "echo:hi"


def test_memory_grows():
    e = CHAE()
    before = e.memory.stats()
    for q in ["rain causes wet ground", "fire causes smoke", "lack of sleep causes fatigue"]:
        e.think(q)
    assert e.memory.stats()["episodic"] > before["episodic"]


def test_discover_from_data():
    e = CHAE()
    rng = np.random.default_rng(0)
    n = 500  # Увеличили выборку
    a = rng.normal(0, 1, n)
    # Уменьшили шум
    df = pd.DataFrame({"a": a, "b": 0.8 * a + rng.normal(0, 0.05, n)})
    out = e.discover_from_data(df, method="granger", maxlag=2)
    assert out["edges_discovered"] >= 1


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        CHAE().discover_from_data(None, method="neural_granger")