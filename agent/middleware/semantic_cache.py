"""
semantic_cache.py — SemanticCacheMiddleware
=============================================
Caches agent responses in Pinecone by semantic similarity.

HOW IT WORKS:
  1. before_agent: embed the incoming query → search the cache__ namespace
     If cosine similarity > threshold → return cached answer immediately
     No LLM call, no tool calls, sub-100ms response

  2. after_agent: store the new question+answer in the cache__ namespace
     Next time a semantically similar question arrives, it hits the cache

THRESHOLDS:
  pharma domain:  0.97 (very strict — clinical accuracy matters)
  general domain: 0.88 (more lenient)

WHY PINECONE (not Redis)?
  Redis is faster but ephemeral — entries vanish when the container dies.
  Pinecone persists across AgentCore microVM cold starts and scales to
  millions of cached entries. The latency difference (5ms vs 50ms) is
  negligible compared to the LLM call we're avoiding (2-8s).

CACHE NAMESPACE:
  Pinecone index: clinical-agent (same index as episodic memory)
  Namespace: cache__   (double underscore to avoid collision with data namespaces)
"""

import logging
from langchain_core.messages import HumanMessage

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

# Similarity threshold by domain
_THRESHOLDS = {
    "pharma":  0.97,
    "general": 0.88,
}
_DEFAULT_THRESHOLD = 0.92


class SemanticCacheMiddleware(BaseAgentMiddleware):
    """
    Pinecone-backed semantic cache for agent responses.

    cache: SemanticCache instance (wraps Pinecone index)
    """

    def __init__(self, cache):
        self._cache = cache

    async def before_agent(self, state: dict) -> None:
        # Don't cache resume requests — they're continuations, not new queries
        if state.get("config", {}).get("configurable", {}).get("resume"):
            return

        messages = state.get("messages", [])
        query    = _last_human_message(messages)
        if not query:
            return

        domain    = state["config"].get("configurable", {}).get("domain", "pharma")
        threshold = _THRESHOLDS.get(domain, _DEFAULT_THRESHOLD)

        try:
            cached = await self._cache.lookup(query, threshold=threshold)
            if cached:
                log.info(f"[CACHE] HIT  threshold={threshold}  query='{query[:60]}'")
                state["cached"] = cached
            else:
                log.info(f"[CACHE] MISS  threshold={threshold}  query='{query[:60]}'")
        except Exception as exc:
            # Cache failure must never block the request
            log.warning(f"[CACHE] Lookup error (bypassing): {exc}")

    async def after_agent(self, state: dict) -> None:
        # Don't store cache entries for blocked, cached, or resume responses
        if state.get("blocked") or state.get("cached") or \
           state.get("config", {}).get("configurable", {}).get("resume"):
            return

        response = state.get("response", "")
        messages = state.get("messages", [])
        query    = _last_human_message(messages)

        if not query or not response:
            return

        try:
            await self._cache.store(query=query, answer=response)
            log.info(f"[CACHE] Stored  query='{query[:60]}'")
        except Exception as exc:
            log.warning(f"[CACHE] Store error: {exc}")


def _last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""
