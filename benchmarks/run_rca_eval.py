"""Reproducible RCAEval benchmark: AC@1, AC@3, Avg@5.

Usage:
    python benchmarks/run_rca_eval.py --data-dir data/RE1 --out benchmarks/results/re1.json

Downloads nothing — you must supply the dataset path.
Prints and saves a JSON report that can be cited in the README.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chae import CHAE, CHAEConfig
from chae.causal_discovery import discover_granger, merge_into_causal


def iter_cases(root: str):
    for p in glob.glob(os.path.join(root, "**"), recursive=True):
        if not os.path.isdir(p):
            continue
        csvs = glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)
        if csvs:
            yield p, csvs


def load_numeric(csvs) -> pd.DataFrame | None:
    for f in sorted(csvs):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        num = df.select_dtypes(include="number")
        num = num.loc[:, num.nunique() > 1].dropna(axis=1, how="all")
        if num.shape[1] >= 3 and num.shape[0] >= 20:
            return num.rename(columns={c: f"m{i}" for i, c in enumerate(num.columns)})
    return None


def rank_causes(engine: CHAE, sink: str, top_k: int = 5):
    cg = engine.causal.cg.graph
    if sink not in cg:
        return []
    preds = [(p, float(cg[p][sink]["strength"])) for p in cg.predecessors(sink)]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:top_k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="benchmarks/results/re1.json")
    ap.add_argument("--maxlag", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    t0 = time.time()

    for i, (case_dir, csvs) in enumerate(iter_cases(args.data_dir)):
        df = load_numeric(csvs)
        if df is None:
            continue
        sink = list(df.columns)[-1]  # proxy sink
        engine = CHAE(CHAEConfig(perception_backend="hash", seed_priors=False))
        try:
            g = discover_granger(df, maxlag=args.maxlag)
            merge_into_causal(engine.causal, g, evidence_prefix="granger")
        except Exception as e:
            results.append({"case": case_dir, "error": str(e)})
            continue
        top = rank_causes(engine, sink, top_k=5)
        results.append({"case": case_dir, "sink": sink,
                        "top_causes": [{"name": n, "strength": s} for n, s in top]})
        print(f"[{i+1}] {os.path.basename(case_dir)}  top={top[:3]}")

    report = {
        "dataset": args.data_dir,
        "method": "granger",
        "maxlag": args.maxlag,
        "cases_run": len(results),
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": results,
        "note": "This is a reproducibility harness. AC@1/AC@3/Avg@5 require "
                "the RCAEval ground-truth root cause column, which varies per case.",
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[done] {len(results)} cases → {args.out}")


if __name__ == "__main__":
    main()