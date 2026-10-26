"""Interactive causal-graph visualization (Flask + vis.js)."""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

_VIS_JS_CDN = "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" /><title>CHAE — Causal Graph</title>
<script src="__VIS_JS_CDN__"></script>
<style>html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
#header{padding:10px 16px;background:#0f172a;color:#f8fafc}
#header h1{margin:0;font-size:18px}#header small{color:#94a3b8}
#network{width:100%;height:calc(100vh - 56px)}</style></head>
<body><div id="header"><h1>CHAE — Causal Graph</h1><small>__SUMMARY__</small></div>
<div id="network"></div>
<script>
const graphData = __GRAPH_JSON__;
const container = document.getElementById('network');
const nodes = new vis.DataSet(graphData.nodes.map(n => ({
  id:n.id,label:n.label,title:n.title||n.label,shape:n.shape||'dot',
  size:n.size||18,color:n.color||'#38bdf8',font:{color:'#0f172a',size:14}})));
const edges = new vis.DataSet(graphData.edges.map(e => ({
  from:e.source,to:e.target,label:e.label||'',arrows:'to',title:e.title||'',
  width:e.width||1,color:{color:e.color||'#64748b',highlight:'#f97316'},
  smooth:{type:'dynamic'}})));
new vis.Network(container,{nodes,edges},{
  physics:{stabilization:true,barnesHut:{gravitationalConstant:-8000}},
  interaction:{hover:true,tooltipDelay:100},layout:{improvedLayout:true}});
</script></body></html>
"""


def graph_to_vis_json(causal_core) -> Dict[str, Any]:
    cg = causal_core.cg if hasattr(causal_core, "cg") else causal_core
    g = cg.graph
    nodes: List[Dict[str, Any]] = []
    for n in g.nodes:
        ind, outd = g.in_degree(n), g.out_degree(n)
        color = "#f97316" if ind == 0 and outd > 0 else "#22c55e" if outd == 0 and ind > 0 else "#38bdf8"
        nodes.append({"id": n, "label": str(n),
                      "title": f"{n}\nin={ind}, out={outd}",
                      "size": 16 + min(18, (ind + outd) * 2), "color": color})
    edges = [
        {"source": u, "target": v, "label": f"{float(d.get('strength',1.0)):.2f}",
         "title": f"strength={float(d.get('strength',1.0)):.2f}\nevidence={d.get('evidence','')}",
         "width": max(1.0, float(d.get("strength", 1.0)) * 6), "color": "#94a3b8"}
        for u, v, d in g.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def create_app(causal_core, host: str = "127.0.0.1", port: int = 8050):
    from flask import Flask, jsonify
    app = Flask(__name__)

    @app.get("/")
    def index():
        try:
            summary = causal_core.cg.summary()
            stxt = f"nodes={summary['nodes']}, edges={len(summary['edges'])}"
        except Exception:
            stxt = ""
        return (_HTML_TEMPLATE
                .replace("__VIS_JS_CDN__", _VIS_JS_CDN)
                .replace("__GRAPH_JSON__", json.dumps(graph_to_vis_json(causal_core)))
                .replace("__SUMMARY__", stxt))

    @app.get("/api/graph")
    def api_graph():
        return jsonify(graph_to_vis_json(causal_core))

    @app.get("/api/summary")
    def api_summary():
        return jsonify(causal_core.cg.summary())

    return app


def serve_in_background(causal_core, host: str = "127.0.0.1", port: int = 8050) -> threading.Thread:
    app = create_app(causal_core, host, port)
    t = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True)
    t.start()
    return t