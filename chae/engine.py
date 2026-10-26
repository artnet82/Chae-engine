"""CHAE engine (open core, no Pro plugins)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .causal import CausalCore
from .causal_discovery import discover_granger, discover_pc, merge_into_causal
from .memory import HierarchicalMemory
from .perception import make_perception
from .planner import SymbolicPlanner
from .tools import ToolRegistry


@dataclass
class CHAEConfig:
    max_working_memory: int = 16
    causal_confidence_threshold: float = 0.6
    enable_continual_learning: bool = True
    perception_backend: str = "hash"
    seed_priors: bool = True


class CHAE:
    def __init__(self, config: Optional[CHAEConfig] = None) -> None:
        self.config = config or CHAEConfig()
        self.perception = make_perception(self.config.perception_backend)
        self.tools = ToolRegistry()
        self.memory = HierarchicalMemory(self.perception, max_working=self.config.max_working_memory)
        self.causal = CausalCore(seed_priors=self.config.seed_priors)
        self.planner = SymbolicPlanner(self.tools)
        self._plugins: List[Any] = []  # Pro plugins register here

    def register_plugin(self, plugin: Any) -> None:
        """Pro plugins call this to hook into discovery/export/etc."""
        self._plugins.append(plugin)
        if hasattr(plugin, "on_register"):
            plugin.on_register(self)

    def _plugin_discover(self, data, method: str, **kw):
        for p in self._plugins:
            if hasattr(p, "discover") and p.supports(method):
                return p.discover(data, method, **kw)
        return None

    def think(self, query: str, context: Optional[Any] = None) -> Dict[str, Any]:
        emb = self.perception.embed(query)
        ep = self.memory.episodic.recall(emb, top_k=5)
        sem = self.memory.semantic.query(query)
        self.memory.working.push({"query": query, "context": context})

        cr = self.causal.reason(query, episodic_hits=ep, semantic_hits=sem)
        plan = self.planner.plan(query, cr, available_tools=self.tools.names())

        actions: List[Dict[str, Any]] = []
        for step in plan["steps"]:
            a = str(step.get("action", ""))
            if a.startswith("call_"):
                tname = a[len("call_"):]
                if self.tools.has(tname):
                    try:
                        out = self.tools.call(tname, {"query": query})
                    except Exception as exc:
                        out = f"tool error: {exc}"
                    actions.append({"tool": tname, "result": out})

        ents = {r["source"] for r in cr["relations"]} | {r["target"] for r in cr["relations"]}
        self.memory.episodic.store({"query": query, "entities_mentioned": sorted(ents)}, embedding=emb)
        if ents:
            self.memory.semantic.add_entities(
                sorted(ents), relations=[(r["source"], r["target"]) for r in cr["relations"]]
            )
        if self.config.enable_continual_learning:
            self.memory.procedural.record_skill(query, plan)

        return {"query": query, "causal_confidence": cr["confidence"],
                "explanation": cr["explanation"], "plan": plan,
                "actions": actions, "memory_stats": self.memory.stats()}

    def discover_from_data(self, data, method: str = "pc", **kw) -> Dict[str, Any]:
        plugin_result = self._plugin_discover(data, method, **kw)
        if plugin_result is not None:
            g = plugin_result
        elif method == "pc":
            g = discover_pc(data, **kw)
        elif method == "granger":
            g = discover_granger(data, **kw)
        else:
            raise ValueError(f"unknown discovery method: {method!r} (Pro method? install chae-pro)")
        n = merge_into_causal(self.causal, g, evidence_prefix=method)
        return {"method": method, "edges_discovered": len(g.edges),
                "edges_merged": n, "graph": g}

    def visualize(self, host: str = "127.0.0.1", port: int = 8050):
        from .visualization import create_app
        return create_app(self.causal, host=host, port=port)

    def intervene(self, variable: str, value: bool) -> Dict[str, Any]:
        return self.causal.intervene(variable, value)

    def counterfactual(self, event: str, alternative: str) -> Dict[str, Any]:
        return self.causal.counterfactual(event, alternative)