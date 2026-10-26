"""Causal chain + discovery demo."""
import numpy as np
import pandas as pd

from chae import CHAE, CHAEConfig


def main() -> None:
    engine = CHAE(CHAEConfig(seed_priors=True))
    rng = np.random.default_rng(0)
    n = 200
    fitness = rng.normal(0, 1, n)
    disease_risk = 0.7 * fitness + rng.normal(0, 0.3, n)
    lifespan = 0.8 * disease_risk + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"fitness": fitness, "disease_risk": disease_risk, "lifespan": lifespan})

    out = engine.discover_from_data(df, method="granger", maxlag=2)
    print(f"[discovery] {out['edges_discovered']} edges found")

    for q in ["exercise causes fitness", "what causes fatigue", "lack of sleep causes fatigue"]:
        r = engine.think(q)
        print(f"\nQ: {q}\n  conf={r['causal_confidence']:.2f}\n  {r['explanation']}")

    print("\nFinal graph edges:")
    for e in engine.causal.cg.summary()["edges"]:
        print(f"  {e['source']} -> {e['target']} ({e['strength']:.2f}, {e['evidence']})")


if __name__ == "__main__":
    main()