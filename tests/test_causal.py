# tests/test_causal.py
from chae.causal import CausalCore, CausalGraph


def test_seeded_priors():
    assert CausalCore().reason("why does rain happen?")["confidence"] > 0.5


def test_intervene_rain_false():
    assert CausalCore().intervene("rain", False)["direct_effects"]["wet_ground"] == 0.0


def test_counterfactual_fire():
    assert CausalCore().counterfactual("fire", "no_fire")["comparison"]["smoke"]["difference"] > 0


def test_path_strength_chain():
    cg = CausalGraph()
    cg.add_cause("a", "b", 0.5)
    cg.add_cause("b", "c", 0.8)
    assert abs(cg.path_strength("a", "c") - 0.4) < 1e-9