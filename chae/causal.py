"""Explicit causal world-model: graph, reasoning, intervention, counterfactual."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class CausalGraph:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_cause(self, cause: str, effect: str, strength: float = 1.0, evidence: str = "") -> None:
        if self.graph.has_edge(cause, effect):
            d = self.graph[cause][effect]
            d["strength"] = max(float(d.get("strength", 0.0)), float(strength))
            ev = str(d.get("evidence", ""))
            if evidence and evidence not in ev:
                d["evidence"] = (ev + "; " + evidence).strip("; ")
        else:
            self.graph.add_edge(cause, effect, strength=float(strength), evidence=evidence)

    def causes_of(self, effect: str) -> List[Tuple[str, float]]:
        if not self.graph.has_node(effect):
            return []
        return [(u, float(self.graph[u][effect]["strength"])) for u in self.graph.predecessors(effect)]

    def effects_of(self, cause: str) -> List[Tuple[str, float]]:
        if not self.graph.has_node(cause):
            return []
        return [(v, float(self.graph[cause][v]["strength"])) for v in self.graph.successors(cause)]

    def path_strength(self, source: str, target: str) -> float:
        if source not in self.graph or target not in self.graph:
            return 0.0
        best = 0.0
        try:
            for path in nx.all_simple_paths(self.graph, source, target):
                prod = 1.0
                for u, v in zip(path, path[1:]):
                    prod *= float(self.graph[u][v]["strength"])
                best = max(best, prod)
        except nx.NetworkXNoPath:
            return 0.0
        return best

    def summary(self) -> Dict[str, Any]:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": [
                {"source": u, "target": v,
                 "strength": float(d.get("strength", 1.0)),
                 "evidence": str(d.get("evidence", ""))}
                for u, v, d in self.graph.edges(data=True)
            ],
        }


class CausalCore:
    def __init__(self, seed_priors: bool = True) -> None:
        self.cg = CausalGraph()
        if seed_priors:
            self._seed_priors()

    def _seed_priors(self) -> None:
        self.cg.add_cause("rain", "wet_ground", 0.9, "prior")
        self.cg.add_cause("fire", "smoke", 0.95, "prior")
        self.cg.add_cause("lack_of_sleep", "fatigue", 0.85, "prior")
        self.cg.add_cause("exercise", "fitness", 0.8, "prior")
        self.cg.add_cause("study", "knowledge", 0.7, "prior")

    def _match_nodes(self, text: str) -> List[str]:
        text_l, toks = text.lower(), set(_TOKEN_RE.findall(text.lower()))
        return [
            n for n in self.cg.graph.nodes
            if str(n).lower() in text_l
            or ("_" in str(n).lower() and all(p in toks for p in str(n).lower().split("_")))
            or str(n).lower() in toks
        ]

    def reason(self, query: str, episodic_hits=None, semantic_hits=None) -> Dict[str, Any]:
        matched = self._match_nodes(query)
        rels, expl, confs = [], [], []
        for node in matched:
            for u, v, d in self.cg.graph.out_edges(node, data=True):
                s, ev = float(d.get("strength", 1.0)), str(d.get("evidence", "")) or "prior"
                rels.append({"source": u, "target": v, "strength": s, "evidence": ev})
                expl.append(f"{u} -> {v} (strength={s:.2f}, evidence={ev})")
                confs.append(s)
            for u, v, d in self.cg.graph.in_edges(node, data=True):
                s, ev = float(d.get("strength", 1.0)), str(d.get("evidence", "")) or "prior"
                rels.append({"source": u, "target": v, "strength": s, "evidence": ev})
                expl.append(f"{u} -> {v} (strength={s:.2f}, evidence={ev})")
                confs.append(s)

        for hit in episodic_hits or []:
            c = hit.get("content")
            if not isinstance(c, dict):
                continue
            for e in c.get("entities_mentioned", []) or []:
                for node in matched:
                    if e == node or self.cg.graph.has_edge(e, node) or self.cg.graph.has_edge(node, e):
                        continue
                    self.cg.add_cause(e, node, 0.5, "episodic co-occurrence")
                    rels.append({"source": e, "target": node, "strength": 0.5, "evidence": "episodic co-occurrence"})
                    expl.append(f"{e} -> {node} (strength=0.50, evidence=episodic co-occurrence)")
                    confs.append(0.5)

        if semantic_hits:
            for r in semantic_hits.get("relations", []) or []:
                src, dst = r.get("source"), r.get("target")
                if src is None or dst is None or self.cg.graph.has_edge(src, dst):
                    continue
                self.cg.add_cause(src, dst, 0.5, "semantic relation")
                rels.append({"source": src, "target": dst, "strength": 0.5, "evidence": "semantic relation"})
                expl.append(f"{src} -> {dst} (strength=0.50, evidence=semantic relation)")
                confs.append(0.5)

        seen, uniq = set(), []
        for r in rels:
            k = (r["source"], r["target"])
            if k not in seen:
                seen.add(k)
                uniq.append(r)

        return {
            "confidence": float(max(confs)) if confs else 0.0,
            "explanation": "; ".join(expl),
            "relations": uniq,
            "graph": self.cg.summary(),
        }

    def intervene(self, variable: str, value: bool) -> Dict[str, Any]:
        direct, propagated = {}, {}
        if variable in self.cg.graph:
            factor = 1.0 if value else 0.0
            for v in self.cg.graph.successors(variable):
                direct[v] = float(self.cg.graph[variable][v]["strength"]) * factor
            for effect, base in direct.items():
                visited, stack = {variable, effect}, [effect]
                while stack:
                    cur = stack.pop()
                    for nxt in self.cg.graph.successors(cur):
                        if nxt in visited:
                            continue
                        visited.add(nxt)
                        s = float(self.cg.graph[cur][nxt]["strength"])
                        propagated[nxt] = propagated.get(nxt, 0.0) + base * s
                        stack.append(nxt)
        return {"intervention": {"variable": variable, "value": value},
                "direct_effects": direct, "propagated_effects": propagated}

    def counterfactual(self, event: str, alternative: str) -> Dict[str, Any]:
        def eff(x: str) -> Dict[str, float]:
            return {} if x not in self.cg.graph else {
                v: float(self.cg.graph[x][v]["strength"]) for v in self.cg.graph.successors(x)
            }
        e_eff, a_eff = eff(event), eff(alternative)
        comparison = {
            e: {"if_event": float(e_eff.get(e, 0.0)),
                "if_alternative": float(a_eff.get(e, 0.0)),
                "difference": float(e_eff.get(e, 0.0)) - float(a_eff.get(e, 0.0))}
            for e in set(e_eff) | set(a_eff)
        }
        return {"event": event, "alternative": alternative, "comparison": comparison}