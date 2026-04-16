"""
core/middleware/__init__.py
============================
Exports domain-agnostic middleware classes from vs-agent-core.
Domain agents import these and add their own domain-specific middleware.
"""

from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware
from core.middleware.semantic_cache import SemanticCacheMiddleware
from core.middleware.episodic_memory import EpisodicMemoryMiddleware

__all__ = [
    "BaseAgentMiddleware",
    "TracerMiddleware",
    "SemanticCacheMiddleware",
    "EpisodicMemoryMiddleware",
]
