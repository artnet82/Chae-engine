"""CHAE CLI: `chae rca`, `chae viz`."""
from __future__ import annotations

import argparse
import sys
import time
import pandas as pd

from .engine import CHAE, CHAEConfig
from .causal_discovery import discover_granger, discover_pc, merge_into_causal


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    num = df.select_dtypes(include="number")
    num = num.loc[:, num.nunique() > 1].dropna(axis=1, how="all")
    if num.shape[1] < 2:
        raise ValueError(f"{path}: need >=2 numeric columns")
    return num


def _cmd_rca(args) -> int:
    df = _load_csv(args.data)
    print(f"[data] {args.data}  shape={df.shape}")

    engine = CHAE(CHAEConfig(perception_backend="hash", seed_priors=False))
    if args.method == "granger":
        g = discover_granger(df, maxlag=args.maxlag, alpha=args.alpha)
    elif args.method == "pc":
        sample = df.sample(n=min(300, len(df)), random_state=0)
        g = discover_pc(sample, alpha=args.alpha, labels=list(df.columns))
    else:
        print(f"[error] unknown method: {args.method}", file=sys.stderr)
        return 2

    print(f"[discovery:{args.method}] {len(g.edges)} edges")
    merge_into_causal(engine.causal, g, evidence_prefix=args.method)

    sink = args.sink
    if sink not in engine.causal.cg.graph:
        print(f"[warn] sink '{sink}' not in graph; using last column", file=sys.stderr)
        sink = list(df.columns)[-1]

    res = engine.think(f"what is the root cause of {sink}?")
    print("\n=== RCA RESULT ===")
    print("confidence :", f"{res['causal_confidence']:.3f}")
    print("explanation:", res["explanation"])

    cg = engine.causal.cg.graph
    if sink in cg:
        preds = sorted(
            [(p, float(cg[p][sink].get("strength", 0.0))) for p in cg.predecessors(sink)],
            key=lambda x: x[1], reverse=True,
        )
        print(f"\nTop upstream causes of {sink}:")
        for name, s in preds[:10]:
            print(f"  {name:>12s}  strength={s:.3f}")

    if args.serve:
        from .visualization import serve_in_background
        print(f"\n[viz] http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
        serve_in_background(engine.causal, port=args.port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def _cmd_viz(args) -> int:
    import json
    from .causal import CausalCore
    from .visualization import create_app

    with open(args.graph) as f:
        data = json.load(f)
    core = CausalCore(seed_priors=False)
    for e in data.get("edges", []):
        core.cg.add_cause(e["source"], e["target"], float(e.get("strength", 1.0)),
                          str(e.get("evidence", "")))
    print(f"[viz] http://127.0.0.1:{args.port}")
    create_app(core, port=args.port).run(host="127.0.0.1", port=args.port)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chae")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rca = sub.add_parser("rca", help="Run RCA on a metrics CSV")
    rca.add_argument("--data", required=True)
    rca.add_argument("--sink", required=True)
    rca.add_argument("--method", default="granger", choices=["granger", "pc"])
    rca.add_argument("--maxlag", type=int, default=3)
    rca.add_argument("--alpha", type=float, default=0.05)
    rca.add_argument("--serve", action="store_true")
    rca.add_argument("--port", type=int, default=8050)

    viz = sub.add_parser("viz", help="Serve causal graph from a JSON file")
    viz.add_argument("--graph", required=True)
    viz.add_argument("--port", type=int, default=8050)

    args = parser.parse_args(argv)
    return _cmd_rca(args) if args.cmd == "rca" else _cmd_viz(args)


if __name__ == "__main__":
    sys.exit(main())