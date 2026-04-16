"""
semantic_cache_with_rules.py — SemanticCacheMiddlewareWithRules
================================================================
Extended version of SemanticCacheMiddleware that adds an intelligent
write policy — only high-quality, reusable answers are stored in cache.

The base SemanticCacheMiddleware (semantic_cache.py) is unchanged.
Swap it in by replacing SemanticCacheMiddleware with
SemanticCacheMiddlewareWithRules in agent/middleware/__init__.py.

Cache eligibility rules (all must pass to store):
  1. tool_count > 0         — LLM retrieved evidence (not memory-only)
  2. faithfulness >= 0.85   — answer is grounded (OutputGuardrail score)
  3. is_fallback == False   — answer was not blocked by OutputGuardrail
  4. len(answer) >= 100     — not a clarification / error message
  5. no patient signals     — question is generic, not patient-specific

Dynamic TTL based on question domain:
  regulatory / guidelines  → 7 days
  clinical trial results   → 1 day
  safety / market          → 1 hour
  default                  → 3 hours

State contract:
  OutputGuardrailMiddleware writes to state:
    _cache_faithfulness  (float)  — faithfulness score from LLM judge
    _cache_is_fallback   (bool)   — True if hard-block fallback was returned

  ActionGuardrailMiddleware writes to state:
    _cache_tool_count    (int)    — total tool calls made this request

  SemanticCacheMiddlewareWithRules reads all three in after_agent.

Why not use an LLM to decide?
  The signals above are deterministic and computed for free by existing
  middleware. An LLM caching-decision call would cost ~$0.001 and add
  ~500ms after every response — more than the cache saves on a MISS.
  Rule-based signals cover ~95% of cases correctly. LLM judgment belongs
  in an offline nightly batch job that evaluates stored entries and evicts
  low-quality ones, not on the hot path.
"""

import logging
import threading
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.cache import SemanticCache
from core.middleware.semantic_cache import SemanticCacheMiddleware

log = logging.getLogger(__name__)

# ── Patient-specific signals — questions mentioning individual patients
# should never be cached and reused across users (accuracy + HIPAA).
_PATIENT_SIGNALS = [
    "my patient", "the patient", "patient is", "patient has",
    "years old", "year-old", "year old",
    "her ", "his ", "they have", "she has", "he has",
    "my ", "i have", "i am",
]

# ── Domain TTL mapping (seconds)
_TTL_REGULATORY  = 7 * 24 * 3_600   # 7 days  — FDA/EMA guidelines
_TTL_TRIAL       = 1 * 24 * 3_600   # 1 day   — RCT results, endpoints
_TTL_SAFETY      = 1 * 3_600         # 1 hour  — recalls, warnings
_TTL_DEFAULT     = 3 * 3_600         # 3 hours — everything else

# ── Minimum answer length to be considered a real clinical answer
_MIN_ANSWER_LEN = 100


class SemanticCacheMiddlewareWithRules(SemanticCacheMiddleware):
    """
    Drop-in replacement for SemanticCacheMiddleware with an intelligent
    write policy. The lookup path (before_agent) is identical — only the
    store decision in after_agent is extended with eligibility rules.
    """

    def __init__(self, cache: SemanticCache):
        super().__init__(cache)

    # ── before_agent is inherited unchanged from SemanticCacheMiddleware ──

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        INTELLIGENT CACHE WRITE — apply eligibility rules before storing.

        Reads signals written by OutputGuardrailMiddleware and
        ActionGuardrailMiddleware from state, then decides whether this
        answer is worth caching.
        """
        question = self._human_message
        user_id  = self._user_id

        # Always clear — prevents leaking into subsequent warm invocations
        self._human_message = None
        self._user_id       = "anonymous"

        if not question:
            log.debug("[CACHE_RULES] skip — no question (multi-turn or cache HIT)")
            return None

        # ── Extract answer ─────────────────────────────────────────────────
        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            log.debug(f"[CACHE_RULES] skip — no AI answer in state  user={user_id}")
            return None

        # ── Apply eligibility rules ────────────────────────────────────────
        eligible, reason = self._is_eligible(question, answer, state)
        if not eligible:
            log.debug(f"[CACHE_RULES] skip — {reason}  user={user_id}")
            return None

        # ── Compute domain-aware TTL ───────────────────────────────────────
        ttl = self._compute_ttl(question)
        log.debug(
            f"[CACHE_RULES] eligible  user={user_id}"
            f"  question='{question[:80]}'  answer_len={len(answer)}  ttl={ttl}s"
        )

        threading.Thread(
            target=self._store_sync,
            args=(question, answer, user_id, ttl),
            daemon=True,
        ).start()
        return None

    # ── Eligibility rules ──────────────────────────────────────────────────

    def _is_eligible(
        self,
        question: str,
        answer:   str,
        state:    AgentState,
    ) -> tuple[bool, str]:
        """
        Return (eligible, reason_string).

        All rules must pass. Returns on the first failure so the log
        message identifies exactly which rule blocked the write.

        Rules and their rationale:

        Rule 1 — tool_count > 0
          The agent must have called at least one tool (search, graph,
          summariser). If tool_count is 0 the LLM answered from parametric
          memory, which violates the CORE BEHAVIOUR rule. Caching a
          memory-only answer serves ungrounded information at full speed
          to every future user — dangerous in a clinical domain.

        Rule 2 — faithfulness >= 0.85
          OutputGuardrailMiddleware computes a faithfulness score via an
          LLM-as-judge call. Answers below threshold are partially
          hallucinated. Caching a hallucination means it gets served
          instantly (bypassing all guardrails) to every future user.

        Rule 3 — not a fallback
          OutputGuardrail hard-blocks non-faithful answers and replaces
          them with a safe fallback message. Caching the fallback means
          the agent never retries — the user gets "I was unable to verify"
          served from cache forever.

        Rule 4 — answer length >= 100 chars
          Short answers are typically clarification questions generated by
          ask_user_input ("Could you specify which drug?") or error
          messages. These must not be cached — they are not clinical answers.

        Rule 5 — no patient-specific signals
          Questions referencing individual patients ("my patient has eGFR 35",
          "she is 67 years old") produce answers that are specific to one
          person. Serving that answer to another user asking a similar question
          is both clinically wrong and a HIPAA concern.
        """
        # Rule 1 — retrieval must have happened
        tool_count = state.get("_cache_tool_count", 0)
        if tool_count == 0:
            return False, f"tool_count=0 — LLM answered from memory, not grounded"

        # Rule 2 — answer must be grounded
        faithfulness = state.get("_cache_faithfulness", 1.0)
        if faithfulness < 0.85:
            return False, f"faithfulness={faithfulness:.2f} below 0.85 — hallucination risk"

        # Rule 3 — must not be a guardrail fallback
        is_fallback = state.get("_cache_is_fallback", False)
        if is_fallback:
            return False, "is_fallback=True — guardrail blocked the answer"

        # Rule 4 — must be a substantive answer
        if len(answer) < _MIN_ANSWER_LEN:
            return False, f"answer_len={len(answer)} < {_MIN_ANSWER_LEN} — likely a clarification"

        # Rule 5 — must not be patient-specific
        q_lower = question.lower()
        for signal in _PATIENT_SIGNALS:
            if signal in q_lower:
                return False, f"patient-specific signal '{signal.strip()}' detected"

        return True, "all rules passed"

    # ── Dynamic TTL ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ttl(question: str) -> int:
        """
        Return cache TTL in seconds based on how quickly the answer domain
        changes in clinical practice.

        Regulatory / guideline information changes slowly (FDA approvals,
        ADA guidelines are updated annually). Trial result data is stable
        once published but new publications can supersede it. Safety and
        market information (recalls, availability) can change overnight.
        """
        q = question.lower()

        # Regulatory / guideline — slowest to change
        if any(w in q for w in [
            "fda", "ema", "ich", "approval", "approved",
            "guideline", "guidelines", "recommendation", "label",
        ]):
            log.debug(f"[CACHE_RULES] TTL=7d (regulatory)")
            return _TTL_REGULATORY

        # Clinical trial results — stable once published
        if any(w in q for w in [
            "phase 1", "phase 2", "phase 3", "phase i", "phase ii", "phase iii",
            "rct", "randomized", "randomised", "trial", "endpoint",
            "efficacy", "hazard ratio", "p-value", "confidence interval",
        ]):
            log.debug(f"[CACHE_RULES] TTL=1d (trial results)")
            return _TTL_TRIAL

        # Safety / market — fastest to change
        if any(w in q for w in [
            "recall", "warning", "black box", "shortage", "availability",
            "current", "latest", "recent", "price", "cost",
        ]):
            log.debug(f"[CACHE_RULES] TTL=1h (safety/market)")
            return _TTL_SAFETY

        log.debug(f"[CACHE_RULES] TTL=3h (default)")
        return _TTL_DEFAULT
