"""Interactive causal-graph visualization (Flask + vis.js).

Start a local server:

    python -m chae.visualization

or embed in your own app:

    from chae.visualization import create_app
    app = create_app(causal_core=causal, port=8050)
    app.run(debug=True)

Endpoints:
    GET /            — HTML page with vis.js network
    GET /api/graph   — JSON {nodes:[...], edges:[...]}
    GET /api/summary — text summary of the graph
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

# vis.js via CDN keeps zero Python-side JS dependencies.
_VIS_JS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>CHAE — Causal Graph</title>
  <script src="__VIS_JS_CDN__"></script>
  <style>
    html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; }
    #header { padding: 10px 16px; background: #0f172a; color: #f8fafc; }
    #header h1 { margin: 0; font-size: 18px; }
    #header small { color: #94a3b8; }
    #network { width: 100%; height: calc(100vh - 56px); }
  </style>
</head>
<body>
  <div id="header">
    <h1>CHAE — Causal Graph</h1>
    <small>__SUMMARY__</small>
  </div>
  <div id="network"></div>
  <script>
    const graphData = __GRAPH_JSON__;
    const container = document.getElementById('network');

    const nodes = new vis.DataSet(graphData.nodes.map(n => ({
      id: n.id,
      label: n.label,
      title: n.title || n.label,
      shape: n.shape || 'dot',
      size: n.size || 18,
      color: n.color || '#38bdf8',
      font: { color: '#0f172a', size: 14 }
    })));

    const edges = new vis.DataSet(graphData.edges.map(e => ({
      from: e.source,
      to: e.target,
      label: e.label || '',
      arrows: 'to',
      title: e.title || '',
      width: e.width || 1,
      color: { color: e.color || '#64748b', highlight: '#f97316' },
      smooth: { type: 'dynamic' }
    })));

    const data = { nodes, edges };
    const options = {
      physics: { stabilization: true, barnesHut: { gravitationalConstant: -8000 } },
      interaction: { hover: true, tooltipDelay: 100 },
      layout: { improvedLayout: true }
    };
    new vis.Network(container, data, options);
  </script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
def graph_to_vis_json(causal_core, max_strength: float = 1.0) -> Dict[str, Any]:
    """Convert a ``CausalCore`` / ``CausalGraph`` into vis.js-friendly JSON."""
    cg = causal_core.cg if hasattr(causal_core, "cg") else causal_core
    g = cg.graph

    nodes: List[Dict[str, Any]] = []
    for n in g.nodes:
        in_deg = g.in_degree(n)
        out_deg = g.out_degree(n)
        size = 16 + min(18, (in_deg + out_deg) * 2)
        # root causes are orange, intermediate nodes blue, sinks green
        if in_deg == 0 and out_deg > 0:
            color = "#f97316"
        elif out_deg == 0 and in_deg > 0:
            color = "#22c55e"
        else:
            color = "#38bdf8"
        nodes.append(
            {
                "id": n,
                "label": str(n),
                "title": f"{n}\nin-degree={in_deg}, out-degree={out_deg}",
                "size": size,
                "color": color,
            }
        )

    edges: List[Dict[str, Any]] = []
    for u, v, d in g.edges(data=True):
        s = float(d.get("strength", 1.0))
        ev = str(d.get("evidence", ""))
        edges.append(
            {
                "source": u,
                "target": v,
                "label": f"{s:.2f}",
                "title": f"strength={s:.2f}\nevidence={ev or 'n/a'}",
                "width": max(1.0, s * 6.0 * max_strength if max_strength else s * 6.0),
                "color": "#94a3b8",
            }
        )
    return {"nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------- #
def create_app(causal_core, host: str = "127.0.0.1", port: int = 8050):
    """Build a Flask app serving the causal graph."""
    try:
        from flask import Flask, jsonify, request  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "flask is required for visualization. "
            "Install with: pip install -e '.[viz]'"
        ) from exc

    app = Flask(__name__)

    @app.get("/")
    def index():
        graph_json = json.dumps(graph_to_vis_json(causal_core))
        try:
            summary = causal_core.cg.summary()
            summary_txt = f"nodes={summary['nodes']}, edges={len(summary['edges'])}"
        except Exception:
            summary_txt = ""
        html = (
            _HTML_TEMPLATE.replace("__VIS_JS_CDN__", _VIS_JS_CDN)
            .replace("__GRAPH_JSON__", graph_json)
            .replace("__SUMMARY__", summary_txt)
        )
        return html

    @app.get("/api/graph")
    def api_graph():
        return jsonify(graph_to_vis_json(causal_core))

    @app.get("/api/summary")
    def api_summary():
        try:
            return jsonify(causal_core.cg.summary())
        except Exception:
            return jsonify({"error": "causal_core has no .cg.summary()"}), 400

    return app


# --------------------------------------------------------------------------- #
def serve_in_background(causal_core, host: str = "127.0.0.1", port: int = 8050) -> threading.Thread:
    """Start the viz server in a daemon thread and return it."""
    app = create_app(causal_core, host, port)
    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()
    return t


if __name__ == "__main__":  # pragma: no cover
    from .causal import CausalCore

    core = CausalCore()
    core.cg.add_cause("fitness", "disease_risk", 0.7, "demo")
    core.cg.add_cause("disease_risk", "lifespan", 0.8, "demo")
    app = create_app(core)
    print("CHAE viz → http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=8050, debug=False)