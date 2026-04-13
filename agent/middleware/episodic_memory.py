"""
episodic_memory.py — EpisodicMemoryMiddleware
===============================================
Injects relevant past conversation context into the current query.

WHAT IS EPISODIC MEMORY?
  Humans remember specific past experiences. This middleware gives the agent
  the same ability. If a user previously asked about a specific patient
  scenario and the agent gave a specific answer, that context is available
  for future conversations even after the microVM restarts.

HOW IT WORKS:
  1. before_agent: search the episodic__ namespace in Pinecone for
     past answers semantically similar to the current query.
     If found, inject them as a SystemMessage above the user's message.

  2. after_agent: check if the response ends with "EPISODIC: YES"
     (the system prompt instructs the LLM to tag responses this way).
     If YES → store the Q+A pair in the episodic__ namespace.
     If NO  → don't store (generic knowledge doesn't need episodic memory).
     Strip the tag before returning — the user should never see it.

EPISODIC vs SEMANTIC CACHE:
  Semantic cache:   exact-ish match (0.97 similarity) → returns full cached answer
  Episodic memory:  looser match → injects context, LLM still generates fresh answer

  Cache is for performance. Episodic is for personalisation.

PINECONE NAMESPACE:
  Index:     clinical-agent (same as cache)
  Namespace: episodic__
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

_EPISODIC_TAG_PATTERN = re.compile(r'\n?EPISODIC:\s*(YES|NO)\s*$', re.IGNORECASE)
_TOP_K_EPISODIC       = 3
_SIMILARITY_THRESHOLD = 0.75


class EpisodicMemoryMiddleware(BaseAgentMiddleware):
    """
    Pinecone-backed episodic memory for personalised context injection.

    store: PineconeStore instance (wraps Pinecone episodic__ namespace)
    """

    def __init__(self, store):
        self._store = store

    async def before_agent(self, state: dict) -> None:
        # Don't inject episodic context for blocked or cached requests
        if state.get("blocked") or state.get("cached"):
            return

        messages = state.get("messages", [])
        query    = _last_human_message(messages)
        if not query:
            return

        try:
            memories = await self._store.search(
                query=query,
                top_k=_TOP_K_EPISODIC,
                threshold=_SIMILARITY_THRESHOLD,
            )
            if memories:
                ctx = "\n\n".join(memories)
                context_msg = SystemMessage(
                    content=(
                        f"RELEVANT PAST CONTEXT (from previous conversations):\n\n"
                        f"{ctx}\n\n"
                        f"Use this context if relevant to the current question."
                    )
                )
                # Insert context SystemMessage before the last HumanMessage
                for i in range(len(messages) - 1, -1, -1):
                    if isinstance(messages[i], HumanMessage):
                        messages.insert(i, context_msg)
                        break
                state["messages"]      = messages
                state["episodic_ctx"]  = ctx
                log.info(f"[EPISODIC] Injected {len(memories)} memories  query='{query[:60]}'")
        except Exception as exc:
            log.warning(f"[EPISODIC] Lookup error (bypassing): {exc}")

    async def after_agent(self, state: dict) -> None:
        response = state.get("response", "")
        if not response:
            return

        # Extract the EPISODIC: YES/NO tag the LLM appended
        match = _EPISODIC_TAG_PATTERN.search(response)

        if match:
            tag      = match.group(1).upper()
            clean    = _EPISODIC_TAG_PATTERN.sub("", response).strip()
            state["response"] = clean   # strip tag from user-visible response

            if tag == "YES":
                messages = state.get("messages", [])
                query    = _last_human_message(messages)
                if query:
                    try:
                        await self._store.store(question=query, answer=clean)
                        log.info(f"[EPISODIC] Stored memory  query='{query[:60]}'")
                    except Exception as exc:
                        log.warning(f"[EPISODIC] Store error: {exc}")


def _last_human_message(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content).strip()
    return ""
