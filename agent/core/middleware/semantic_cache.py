"""
semantic_cache.py — SemanticCacheMiddleware
=============================================
Semantic cache check (before) and store (after).
Cache REPLACES work — HIT means NO episodic search, NO tools, NO LLM.

Cache is layer 4 — after PII scrubbing and content filtering,
before episodic memory, tools, and the LLM.

SEMANTIC CACHING:
  Embeds query as vector, finds cached vectors above cosine similarity
  threshold (0.97 pharma, 0.88 general). Same question rephrased still hits.

BRIDGE DESIGN — scalar not dict:
  AgentCore: one request per container → no concurrency → scalar is safe.
  FastAPI/Gunicorn: concurrent requests → use dict keyed by run_id instead.

FIRE-AND-FORGET WRITE:
  Pinecone write runs in background daemon thread after response is sent.
  daemon=True prevents blocking container shutdown.

TTL=3600s (1 hour):
  Balances cache effectiveness vs clinical data freshness risk.

NEVER CACHE GUARDRAIL FALLBACKS:
  Caching "did not meet safety standards" would permanently block valid queries.

TRACER INTEGRATION:
  On cache HIT, annotates trace with:
    TracerMiddleware.update_trace(run_id, {
        "cache_hit":       True,
        "cache_namespace": "cache_pharma",
    })
  On cache MISS, cache_hit=False is the default in tracer — no annotation needed.
"""

import logging
import threading
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.cache import SemanticCache

log = logging.getLogger(__name__)

_GUARDRAIL_FALLBACK_MARKER = "did not meet safety and accuracy standards"


class SemanticCacheMiddleware(BaseAgentMiddleware):
    """
    Two-hook middleware: cache lookup (before_agent) and cache write (after_agent).

    Instance state (scalars — safe on AgentCore, see module docstring):
      _cache          — SemanticCache (Pinecone-backed)
      _human_message  — bridge: before → after
      _user_id        — bridge: before → after
    """

    def __init__(self, cache: SemanticCache):
        super().__init__()
        self._cache:         SemanticCache  = cache
        self._human_message: Optional[str] = None
        self._user_id:       str           = "anonymous"

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Cache lookup — short-circuit on HIT, annotate trace.

        SKIP CONDITIONS (return None, proceed):
          1. No human message
          2. Multi-turn conversation (answer is context-dependent)
          3. Empty question
          4. Lookup error (treat as MISS)

        HIT → return cached AIMessage + jump_to="end"
              Annotates trace: cache_hit=True, cache_namespace
        MISS → return None, stash question for after_agent
        """
        run_id         = self._get_run_id(runtime)
        messages       = state.get("messages", [])
        human_messages = [m for m in messages if getattr(m, "type", None) == "human"]

        if not human_messages:
            self._human_message = None
            return None

        if len(human_messages) > 1:
            # Multi-turn: cache lookup without full context would return wrong answer
            log.debug(f"[CACHE_MW] skip — multi-turn ({len(human_messages)} human messages)")
            self._human_message = None
            return None

        question = str(human_messages[0].content).strip()
        if not question:
            self._human_message = None
            return None

        user_id = (getattr(runtime, "context", None) or {}).get("user_id", "anonymous")
        self._human_message = question
        self._user_id       = user_id
        log.debug(f"[CACHE_MW] lookup  user={user_id}  question='{question[:80]}'")

        try:
            cached = self._cache.lookup(question, user_id=user_id)

            if cached:
                log.info(f"[CACHE_MW] HIT  user={user_id}  answer_len={len(cached)}")
                self._human_message = None   # clear bridge — no write needed on HIT

                # Annotate trace with cache hit details
                # cache_namespace tells us which Pinecone namespace served the hit
                # Useful for debugging: "cache_pharma" vs "cache_general"
                TracerMiddleware.update_trace(run_id, {
                    "cache_hit":       True,
                    "cache_namespace": getattr(self._cache, "namespace", "unknown"),
                })

                return {"messages": [AIMessage(content=cached)], "jump_to": "end"}

            log.debug(f"[CACHE_MW] MISS  user={user_id}")

        except Exception as exc:
            log.warning(f"[CACHE_MW] lookup error  user={user_id}  error={exc} — treating as MISS")

        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Cache write — store LLM answer for future hits.

        Clears bridge values immediately (before early returns) to prevent
        stale values leaking between requests.

        SKIP CONDITIONS (no write):
          1. No question stored (multi-turn, empty, or cache HIT)
          2. No AI answer in state
          3. Answer is a guardrail fallback (never cache error responses)
        """
        question = self._human_message
        user_id  = self._user_id

        # Clear bridge before any early return — prevents stale state
        self._human_message = None
        self._user_id       = "anonymous"

        if not question:
            log.debug("[CACHE_MW] after_agent skip — no question (multi-turn or HIT)")
            return None

        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            log.debug(f"[CACHE_MW] after_agent skip — no AI answer  user={user_id}")
            return None

        # Never cache guardrail fallbacks — would permanently block similar queries
        if _GUARDRAIL_FALLBACK_MARKER in answer:
            log.info(f"[CACHE_MW] skip — guardrail fallback, not caching  user={user_id}")
            return None

        log.debug(f"[CACHE_MW] storing  user={user_id}  question='{question[:80]}'")

        # daemon=True: does not block container shutdown
        threading.Thread(
            target = self._store_sync,
            args   = (question, answer, user_id),
            daemon = True,
        ).start()
        return None

    def _store_sync(self, question: str, answer: str, user_id: str, ttl: int = 3_600) -> None:
        """
        Write question-answer pair to Pinecone cache. Background daemon thread.

        ttl=3600 (1 hour): clinical data changes — new results, FDA decisions.
        1 hour balances freshness vs cache effectiveness for clinical domain.
        """
        try:
            self._cache.store(question, answer, user_id=user_id, ttl=ttl)
            log.info(f"[CACHE_MW] store complete  user={user_id}  answer_len={len(answer)}")
        except Exception as exc:
            # Non-fatal: failed write → next identical query goes to LLM
            log.warning(f"[CACHE_MW] store failed  user={user_id}  error={exc}")