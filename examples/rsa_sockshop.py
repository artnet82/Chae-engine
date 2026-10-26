"""End-to-end RCA on Sock Shop (RCAEval RE1).

Download RE1 from https://zenodo.org/records/14504481, then:
    python examples/rca_sockshop.py --data-dir data/RE1 --serve
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chae import CHAE, CHAEConfig
from chae.causal_discovery import discover_granger, discover_pc, merge_into_causal
from chae.visualization import serve_in_background


def pick_case(root: str) -> str:
    cands = [p for p in glob.glob(os.path.join(root, "**"), recursive=True)
             if os.path.isdir(p) and "sock" in p.lower()]
    leaves = [p for p in cands if glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)]
    if leaves:
        return leaves[0]
    for p in glob.glob(os.path.join(root, "**"), recursive=True):
        if os.path.isdir(p) and glob.glob(os.path.join(p, "**", "*.csv"), recursive=True):
            return p
    raise FileNotFoundError(f"no CSV case under {root}")


def load_metrics(case: str) -> pd.DataFrame:
    for f in sorted(glob.glob(os.path.join(case, "**", "*.csv"), recursive=True)):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        num = df.select_dtypes(include="number")
        num = num.loc[:, num.nunique() > 1].dropna(axis=1, how="all")
        if num.shape[1] >= 3 and num.shape[0] >= 20:
            num = num.rename(columns={c: f"m{i}" for i, c in enumerate(num.columns)})
            print(f"[data] {f}  shape={num.shape}")
            return num
    raise ValueError("no suitable numeric CSV")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--maxlag", type=int, default=3)
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args()

    case = pick_case(args.data_dir)
    print(f"[case] {case}")
    df = load_metrics(case)

    engine = CHAE(CHAEConfig(perception_backend="sentence-transformers", seed_priors=False))

    print("[discovery] Granger ...")
    g = discover_granger(df, maxlag=args.maxlag)
    print(f"  → {len(g.edges)} edges")
    merge_into_causal(engine.causal, g, evidence_prefix="granger")

    try:
        print("[discovery] PC ...")
        sample = df.sample(n=min(300, len(df)), random_state=0)
        g = discover_pc(sample, alpha=0.05, labels=list(df.columns))
        print(f"  → {len(g.edges)} edges")
        merge_into_causal(engine.causal, g, evidence_prefix="pc")
    except Exception as exc:
        print(f"  [skip] PC: {exc}")

    sink = list(df.columns)[-1]
    res = engine.think(f"why did {sink} spike?")
    print("\n=== RCA ===")
    print("confidence :", f"{res['causal_confidence']:.3f}")
    print("explanation:", res["explanation"])

    cg = engine.causal.cg.graph
    if sink in cg:
        preds = sorted([(p, float(cg[p][sink]["strength"])) for p in cg.predecessors(sink)],
                       key=lambda x: x[1], reverse=True)
        print(f"\nTop upstream causes of {sink}:")
        for n, s in preds[:10]:
            print(f"  {n:>8s}  strength={s:.3f}")

    if not args.no_viz:
        print(f"\n[viz] http://127.0.0.1:{args.port}")
        serve_in_background(engine.causal, port=args.port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()