"""Four-layer hierarchical memory."""
from __future__ import annotations

import re
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from .perception import PerceptionLayer

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class WorkingMemory:
    def __init__(self, max_size: int = 16) -> None:
        self.max_size = max_size
        self._buf: deque = deque(maxlen=max_size)

    def push(self, item: Any) -> None:
        self._buf.append(item)

    def recent(self, n: int = 5) -> List[Any]:
        items = list(self._buf)
        return items[-n:] if n < len(items) else items

    def __len__(self) -> int:
        return len(self._buf)


class EpisodicMemory:
    def __init__(self, perception) -> None:
        self.perception = perception
        self._items: List[Dict[str, Any]] = []

    def store(self, content: Any, embedding: Optional[np.ndarray] = None) -> None:
        if embedding is None:
            embedding = self.perception.embed(content if isinstance(content, str) else str(content))
        self._items.append({"embedding": embedding, "content": content})

    def recall(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        scored = [
            {"content": it["content"], "embedding": it["embedding"],
             "score": self.perception.similarity(embedding, it["embedding"])}
            for it in self._items
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._items)


class SemanticMemory:
    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_entities(self, entities: List[str], relations: Optional[List[Tuple[str, str]]] = None) -> None:
        for e in entities:
            if not self.graph.has_node(e):
                self.graph.add_node(e)
        for src, dst in relations or []:
            w = self.graph.get_edge_data(src, dst, {}).get("weight", 0.0) + 1.0
            self.graph.add_edge(src, dst, weight=w)

    def query(self, text: str) -> Dict[str, Any]:
        text_l, toks = text.lower(), set(_TOKEN_RE.findall(text.lower()))
        matched = [
            n for n in self.graph.nodes
            if str(n).lower() in text_l
            or ("_" in str(n).lower() and all(p in toks for p in str(n).lower().split("_")))
            or str(n).lower() in toks
        ]
        mset = set(matched)
        rels = [{"source": u, "target": v} for u, v in self.graph.edges if u in mset or v in mset]
        return {"entities": matched, "relations": rels}


class ProceduralMemory:
    def __init__(self) -> None:
        self.skills: Counter = Counter()

    def record_skill(self, query: str, plan: Dict[str, Any]) -> None:
        key = " -> ".join(str(s.get("action", "")) for s in plan.get("steps", []) if s.get("action"))
        if key:
            self.skills[key] += 1

    def best_skills(self, top_k: int = 5) -> List[Tuple[str, int]]:
        return self.skills.most_common(top_k)


class HierarchicalMemory:
    def __init__(self, perception, max_working: int = 16) -> None:
        self.perception = perception
        self.working = WorkingMemory(max_working)
        self.episodic = EpisodicMemory(perception)
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()

    def stats(self) -> Dict[str, int]:
        return {
            "working": len(self.working),
            "episodic": len(self.episodic),
            "semantic_entities": self.semantic.graph.number_of_nodes(),
            "procedural_skills": len(self.procedural.skills),
        }