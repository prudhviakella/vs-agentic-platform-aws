"""
episodic_memory.py — EpisodicMemoryMiddleware
===============================================
Gives the agent memory across sessions — it remembers what a specific user
asked in the past and uses that to enrich the current response.

WHY DO WE NEED THIS?
  Without episodic memory every conversation starts blank.

  Session 1:  User asks "What is the metformin dose for my patient with eGFR 25?"
              Agent gives a detailed answer.

  Session 2:  User asks "Is that dose still safe?"
              Agent: "Which dose? Which patient?" -- it has forgotten everything.

  With episodic memory:
  Session 2:  Agent injects prior context into the prompt automatically:
              "Based on our previous discussion about eGFR 25 dosing..."

HOW IS THIS DIFFERENT FROM SEMANTICCACHE?
  SemanticCache  → shared across ALL users, skips the agent entirely on HIT.
                   Stores generic answers. Purpose: saves cost.
  EpisodicMemory → PRIVATE per user, agent still runs with richer context.
                   Stores user-specific interactions. Purpose: personalisation.

THE CORE IDEA — RELEVANT RETRIEVAL, NOT FULL STATE PASSING:
  Stores past Q&A pairs in Pinecone and retrieves the TOP 3 MOST RELEVANT
  ones for the current question. Token cost is always bounded (~300 tokens)
  regardless of session count. Relevance beats recency.

HOW IT WORKS WITH SUMMARIZATIONMIDDLEWARE:
  State (LangGraph/Postgres) → working memory — compressed, bounded.
  Pinecone (episodic namespace) → long-term memory — permanent, searchable.
  SummarizationMiddleware compresses state but never touches Pinecone.
  Fresh relevant memories are injected on every new turn.

WHY THE LLM DECIDES WHAT TO STORE:
  "What is the normal eGFR range?" → generic → NO
  "What dose for my patient with eGFR 25?" → user-specific → YES
  Only the LLM can tell them apart. EPISODIC: YES/NO tag at zero extra cost.

TRACER INTEGRATION:
  Annotates the trace with:
    TracerMiddleware.update_trace(run_id, {
        "episodic_hits":   int,   — memories retrieved and injected
        "episodic_stored": bool,  — whether this Q&A was stored
    })
"""

import hashlib
import logging
import re
import threading
import time
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore

log = logging.getLogger(__name__)


class EpisodicMemoryMiddleware(BaseAgentMiddleware):
    """
    Retrieves relevant past Q&A pairs before the LLM runs (before_agent)
    and stores new ones after the LLM responds (after_agent).
    """

    _EPISODIC_TAG = re.compile(r"EPISODIC:\s*(YES|NO)", re.IGNORECASE)

    def __init__(self, store: BaseStore):
        """
        Args:
            store: PineconeStore shared with create_agent(store=...).
                   Same instance avoids a second Pinecone connection at cold start.
        """
        super().__init__()
        self._store = store

    def _get_user_id(self, runtime: Runtime) -> str:
        ctx = getattr(runtime, "context", {}) or {}
        return ctx.get("user_id", "anonymous")

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Retrieve relevant past memories and inject as a SystemMessage.

        WHY SystemMessage: visible to the LLM directly, no @dynamic_prompt needed.
        WHY query=question: semantic search for relevance, not recency.
        WHY limit=3: ~300 token overhead, always bounded.

        Annotates trace with episodic_hits count regardless of whether
        memories were found — 0 hits is useful data too.
        """
        run_id  = self._get_run_id(runtime)
        user_id = self._get_user_id(runtime)

        messages = state.get("messages", [])
        question = ""
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "human":
                question = str(msg.content).strip()
                break

        if not question:
            log.info(f"[EPISODIC] No question found — skipping search  user={user_id}")
            TracerMiddleware.update_trace(run_id, {"episodic_hits": 0})
            return None

        try:
            items = self._store.search(("episodic", user_id), query=question, limit=3)
            log.info(f"[EPISODIC] Searched  user={user_id}  hits={len(items)}")

            # Always annotate hit count — 0 is valid data
            TracerMiddleware.update_trace(run_id, {"episodic_hits": len(items)})

        except Exception:
            log.info("[EPISODIC] Search unavailable — proceeding without episodic context")
            TracerMiddleware.update_trace(run_id, {"episodic_hits": 0})
            return None

        if not items:
            log.info(f"[EPISODIC] No relevant memories found  user={user_id}")
            return None

        context_lines = []
        for i, item in enumerate(items, 1):
            text = item.value.get("text", "").strip()
            if text:
                context_lines.append(f"[Memory {i}]\n{text}")

        if not context_lines:
            return None

        context = "\n\n".join(context_lines)
        log.info(f"[EPISODIC] Injecting {len(context_lines)} memories  user={user_id}")

        return {
            "messages": [
                SystemMessage(content=f"Relevant past interactions with this user:\n\n{context}")
            ]
        }

    @staticmethod
    def _parse_storage_decision(answer: str) -> tuple[bool, str]:
        """
        Parse EPISODIC: YES/NO tag and strip it from the answer.

        WHY tag instead of separate classifier: zero extra cost — LLM is
        already running. The tag approach piggybacks on the existing call.

        WHY strip regardless of YES/NO: tag is always internal metadata,
        never shown to users.

        Returns (should_store, clean_answer).
        """
        match = EpisodicMemoryMiddleware._EPISODIC_TAG.search(answer)
        if not match:
            return False, answer

        should_store = match.group(1).upper() == "YES"
        clean = re.sub(
            r"\nEPISODIC:.*$", "",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        ).strip()
        return should_store, clean

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Strip EPISODIC tag from answer and optionally store Q&A in Pinecone.

        Annotates trace with episodic_stored=True/False so DynamoDB shows
        which sessions generated memorable content.

        WHY fire-and-forget: response is already streaming to user by the
        time after_agent runs — don't add Pinecone write latency.
        """
        run_id   = self._get_run_id(runtime)
        user_id  = self._get_user_id(runtime)
        messages = state.get("messages", [])

        question, answer = "", ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not answer:
                answer = str(msg.content)
            elif hasattr(msg, "type") and msg.type == "human" and not question:
                question = str(msg.content)
            if question and answer:
                break

        if not (question and answer):
            TracerMiddleware.update_trace(run_id, {"episodic_stored": False})
            return None

        should_store, clean_answer = self._parse_storage_decision(answer)

        # Always strip EPISODIC tag from the message the user sees
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                msg.content = clean_answer
                break

        # Annotate trace with storage decision
        TracerMiddleware.update_trace(run_id, {"episodic_stored": should_store})

        if not should_store:
            log.info(f"[EPISODIC] LLM tagged NO — skipping store  user={user_id}")
            return None

        log.info(f"[EPISODIC] LLM tagged YES — storing  user={user_id}")

        # daemon=True: killed on container shutdown, no zombie threads
        threading.Thread(
            target  = self._store_sync,
            args    = (user_id, question, clean_answer),
            daemon  = True,
        ).start()
        return None

    def _store_sync(self, user_id: str, question: str, answer: str) -> None:
        """
        Write Q&A pair to Pinecone. Background daemon thread.

        WHY MD5 entry_id: deterministic deduplication — same question
        overwrites instead of duplicating. MD5[:12] = 48 bits, sufficient
        collision resistance within one user's memory namespace.

        WHY truncate to 300 chars: storage efficiency. 300 chars gives
        enough gist for future context enrichment without ballooning costs.
        """
        try:
            entry_id = hashlib.md5(f"{user_id}{question}".encode()).hexdigest()[:12]
            self._store.put(
                ("episodic", user_id),
                entry_id,
                {
                    "text": f"Q: {question}\nA: {answer[:300]}",
                    "ts":   time.time(),
                },
            )
            log.info(f"[EPISODIC] Stored  user={user_id}  id={entry_id}")
        except Exception as exc:
            log.warning(f"[EPISODIC] Store failed: {exc}")