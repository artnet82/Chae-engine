"""Symbolic planner."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class SymbolicPlanner:
    def __init__(self, tools: Any = None) -> None:
        self.tools = tools

    def _available(self, override: Optional[List[str]]) -> List[str]:
        if override is not None:
            return list(override)
        if self.tools is not None:
            try:
                return list(self.tools.names())
            except Exception:
                return []
        return []

    def plan(self, query: str, causal_result: Dict[str, Any],
             available_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        tools = self._available(available_tools)
        tokens = set(_TOKEN_RE.findall(query.lower()))
        conf = float(causal_result.get("confidence", 0.0))
        steps: List[Dict[str, Any]] = [
            {"action": "explain", "detail": causal_result.get("explanation", "")}
        ]
        if conf < 0.5:
            steps.append({"action": "request_evidence", "detail": f"causal confidence {conf:.2f} below 0.50"})
        for t in tools:
            tl = t.lower()
            if tl in tokens or any(tok in tl for tok in tokens) or tl in query.lower():
                steps.append({"action": f"call_{t}", "detail": f"invoke tool '{t}'"})
        steps.append({"action": "conclude", "detail": "aggregate causal explanation"})
        return {"goal": query, "causal_confidence": conf, "steps": steps,
                "estimated_complexity": len(steps)}