"""CHAE — Causal Hierarchical Agent Engine (open core)."""
from .engine import CHAE, CHAEConfig
from .perception import PerceptionLayer, SentenceTransformerPerception

__version__ = "0.2.1"

__all__ = ["CHAE", "CHAEConfig", "PerceptionLayer", "SentenceTransformerPerception", "__version__"]