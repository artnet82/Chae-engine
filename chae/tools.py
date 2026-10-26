"""Minimal tool registry."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, func: Callable[..., Any]) -> None:
        if not callable(func):
            raise TypeError(f"tool '{name}' must be callable")
        self._tools[name] = func

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def call(self, name: str, args: Optional[Dict[str, Any]] = None) -> Any:
        if name not in self._tools:
            raise KeyError(f"tool '{name}' is not registered")
        return self._tools[name](**(args or {}))