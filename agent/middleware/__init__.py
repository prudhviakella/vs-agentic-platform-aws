"""
agent/middleware/__init__.py — Clinical Trial Agent Middleware Stack
=====================================================================

This file is the COMPOSITION ROOT for the middleware stack. Its only job is
to assemble the ordered list of middleware instances that gets passed to
create_agent(middleware=...) in agent.py.

WHAT IS MIDDLEWARE IN LANGCHAIN 1.0?
──────────────────────────────────────
Middleware wraps the agent node in a LangGraph graph. Think of it like
express.js middleware or Django middleware — a chain of processing layers
where each layer can inspect, modify, or short-circuit the request/response.

Each middleware class implements two hooks:
  before_agent(state, runtime) → called before the LLM runs (ingress)
  after_agent(state, runtime)  → called after the LLM produces output (egress)

The middleware list is applied like concentric rings around the agent:

  Request  ───►  Layer 1  ───►  Layer 2  ───► ... ───►  LLM
  Response ◄───  Layer 1  ◄───  Layer 2  ◄─── ... ◄───  LLM

  before_agent:  Layer 1 runs first  (outermost ring, ingress)
  after_agent:   Layer 1 runs last   (outermost ring, egress)

This means:
  - TracerMiddleware (layer 1) sees the RAW input and the FINAL output
  - OutputGuardrailMiddleware (layer 9) sees the LLM output FIRST on egress
    and can block it before any other layer touches it

WHY THIS ORDERING MATTERS:
───────────────────────────
The order is not arbitrary — each layer depends on the previous layers
having already run. Here's why each layer is in its position:

  LAYER 1 — TracerMiddleware (FIRST)
    Starts the trace timer and assigns a run_id BEFORE anything else.
    If placed after the cache, cache hits wouldn't be traced. Placing it
    first ensures 100% observability — every request gets a trace record.

  LAYER 2 — DomainPIIMiddleware (EARLY, before LLM or cache)
    Scrubs PII from the user's message before it touches anything:
    - Before the semantic cache (PII-containing queries must not be cached)
    - Before the LLM (PII must never appear in OpenAI's servers)
    - Before episodic memory (PII must not be stored in Pinecone)
    HIPAA compliance requires PII to be removed at the earliest possible point.

  LAYER 3 — ContentFilterMiddleware (EARLY, after PII)
    Blocks off-topic or toxic queries after PII is scrubbed.
    Runs before the cache (no point caching a blocked query) and before
    the LLM (blocked queries should never reach OpenAI).

  LAYER 4 — SemanticCacheMiddleware (EARLY, SHORT-CIRCUIT)
    If a semantically similar query was answered recently, return the
    cached answer immediately and SKIP ALL LAYERS BELOW.
    Placed here (not at layer 1) so PII and toxic content are checked first —
    we don't want to cache a PII-containing query or serve a cached answer
    to a blocked user.
    When cache HITS: layers 5-9 never run. This is the "HIT skips everything
    below" note in the stack diagram.

  LAYER 5 — EpisodicMemoryMiddleware (before LLM)
    Injects relevant past conversation context into the prompt before the
    LLM runs. Runs after the cache (cache hits skip memory retrieval — no
    point retrieving context for a cached answer) and after PII scrubbing
    (memory should not contain PII from the current query).

  LAYER 6 — SummarizationMiddleware (before_model, fires at 8_000 tokens)
    Compresses long conversation history when token count approaches limits.

    HOW IT WORKS (verified from source):
      Uses before_model() — runs before every LLM call, not just the first.
      On every LLM call it checks: total_tokens >= 8_000?
        NO  → returns None, history unchanged, LLM runs normally
        YES → calls GPT-4o-mini to summarise the oldest messages,
              then returns RemoveMessage(id=REMOVE_ALL_MESSAGES) +
              HumanMessage("Here is a summary...") + last 10 messages
      The RemoveMessage clears the full history and replaces it with
      one HumanMessage summary + the preserved recent messages.
      This is a REPLACEMENT, not an append — no spurious tokens in response.

    WHY 8_000 TOKENS:
      Production traces show typical sessions are 1–2 LLM turns (~2,300 tokens).
      The 8,000 token threshold means summarization fires at ~8 turns.
      Normal users never hit it. Heavy research sessions benefit from it.
      DynamoDB trace data confirms: turns=1 and turns=2 are the dominant pattern.

    WHY GPT-4o-mini FOR SUMMARIZATION:
      Summarization is a simple extraction task — "what happened in these messages?"
      GPT-4o-mini is 5x cheaper and fast enough for this. We save GPT-4o budget
      for the actual clinical reasoning.

  LAYER 7 — SingleClarificationHITLMiddleware (before LLM, one per conversation)
    Intercepts broad queries and forces the LLM to call ask_user_input for
    clarification. Runs near the bottom of before_agent so it sees the
    fully enriched state (after episodic memory injection) when deciding
    whether to inject the MANDATORY OVERRIDE SystemMessage.
    interrupt_on fires LangGraph's checkpoint pause mechanism before the
    tool actually executes — state.tasks[0].interrupts holds the question.

  LAYER 8 — ActionGuardrailMiddleware (COMMENTED OUT — see note below)
    Would enforce tool call limits (max calls per request, blocked tool patterns).
    Currently disabled — the prompt's TOOL USAGE section and max_tool_calls
    variable provide soft limits. Re-enable when the agent starts exceeding
    tool call budgets in production.

  LAYER 9 — OutputGuardrailMiddleware (LAST in before, FIRST in after)
    The output guard runs LAST in after_agent — closest to the user.
    Three-layer check on the LLM's answer:
      Code check:   check_medical_action_output() — regex, < 1ms
      Layer 2:      LLM faithfulness judge (GPT-4o-mini) — does the answer
                    contradict the retrieved evidence? Score 0.0–1.0.
      Layer 3:      LLM consistency check (GPT-4o-mini) — does the answer
                    contradict prior answers in this thread?
    Placed last in the chain so it sees the final answer exactly as it
    will be delivered to the user — not an intermediate version.

THRESHOLDS EXPLANATION:
────────────────────────
  faithfulness_threshold=0.00
    GPT-4o-mini scores faithfulness as 0.0–1.0 where 1.0 = perfectly faithful.
    Production data shows scores clustering around 0.80–1.00 for well-grounded
    answers. We set 0.00 (disabled) during initial infra_deployment to avoid blocking
    valid responses while we calibrate the expected score distribution.
    TODO: raise to 0.70 after 500+ production traces confirm the baseline.

  confidence_threshold=0.00
    Same reasoning — disabled during calibration phase.

TWO PACKAGES — WHY SPLIT BETWEEN core/ AND agent/?
─────────────────────────────────────────────────────
  core/middleware/   — domain-AGNOSTIC layers.
    These work for any agent in any domain: tracer, cache, episodic memory,
    summarization, HITL. A finance agent or a legal research agent could use
    these unchanged.

  agent/middleware/  — pharma-DOMAIN layers.
    PII patterns, toxic content rules, medical action output patterns, and
    faithfulness judging are all specific to clinical trial research. They
    would need to be rewritten for a different domain.

  This split follows the package architecture: vs-agent-core contains
  reusable infrastructure; the agent/ package contains domain logic.
  When building a new domain agent (e.g. legal), you import from core/
  and write new domain/ middleware files — you don't modify core/.
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware

# ── Domain-agnostic middleware from vs-agent-core ─────────────────────────
# These live in core/middleware/ — they have no pharma-specific logic
# and can be reused by any domain agent without modification.
from core.aws import get_trace_table_name
from core.middleware.tracer import TracerMiddleware
from core.middleware.semantic_cache import SemanticCacheMiddleware
from core.middleware.episodic_memory import EpisodicMemoryMiddleware
from core.cache import SemanticCache

# ── Pharma-domain middleware — lives in agent/middleware/ ─────────────────
# These contain clinical trial / pharma-specific rules.
# PII patterns: patient IDs, clinical trial numbers, medical record numbers
# Toxic check: clinical harm patterns (not generic toxicity)
# Output guardrail: medical action directives, faithfulness to trial evidence
from agent.middleware.pii import DomainPIIMiddleware
from agent.middleware.content_filter import ContentFilterMiddleware
from agent.middleware.action_guardrail import ActionGuardrailMiddleware
from agent.middleware.output_guardrail import OutputGuardrailMiddleware
from agent.middleware.hitl import SingleClarificationHITLMiddleware


def build_stack(domain: str, store, safety_llm, cache: SemanticCache) -> list:
    """
    Assemble and return the ordered 9-layer middleware stack.

    WHY A FUNCTION INSTEAD OF A MODULE-LEVEL LIST?
      Middleware instances are created with dependency-injected arguments
      (store, safety_llm, cache) that aren't available at import time.
      A function lets us defer instantiation until build_agent() has
      built all the dependencies.

    Args:
        domain:     "pharma" | "general" — controls cache namespace and
                    domain_frame in the prompt.

        store:      PineconeStore instance shared with create_agent(store=...).
                    EpisodicMemoryMiddleware uses this to upsert and query
                    the episodic memory namespace in Pinecone.

        safety_llm: ChatOpenAI(model="gpt-4o-mini") — used exclusively by
                    OutputGuardrailMiddleware for faithfulness and consistency
                    judging. Separate from the main gpt-4o reasoning LLM.

        cache:      SemanticCache (Pinecone-backed) injected from build_agent().
                    Pre-configured with the domain's similarity threshold and
                    namespace. SemanticCacheMiddleware uses this to check for
                    cache hits and write new entries.

    Returns:
        Ordered list of middleware instances. Passed directly to
        create_agent(middleware=...) in agent.py.
    """
    return [
        # ── Layer 1: Tracer ───────────────────────────────────────────────
        # FIRST — must see every request including cache hits and blocked queries.
        # Writes a trace record to DynamoDB with: run_id, user_id, timestamp,
        # input text, tools called, latency, and output summary.
        TracerMiddleware(dynamodb_table_name=get_trace_table_name()),

        # ── Layer 2: PII Scrubbing ────────────────────────────────────────
        # EARLY — PII must be removed before touching cache, LLM, or memory.
        # Pharma-domain patterns: patient IDs, clinical trial IDs, provider names.
        # Replaces matched patterns with [REDACTED_TYPE] placeholders.
        DomainPIIMiddleware(),

        # ── Layer 3: Content Filter ───────────────────────────────────────
        # EARLY — blocks off-topic or toxic queries before reaching the LLM.
        # Returns a blocked response immediately — no LLM call, no cache write.
        ContentFilterMiddleware(),

        # ── Layer 4: Semantic Cache ───────────────────────────────────────
        # SHORT-CIRCUIT — if a semantically similar query was answered recently,
        # return the cached answer and SKIP layers 5–9 entirely.
        # Threshold: 0.97 cosine similarity (pharma) — strict for clinical safety.
        SemanticCacheMiddleware(cache=cache),

        # ── Layer 5: Episodic Memory ──────────────────────────────────────
        # ENRICHMENT — retrieves relevant past conversations for this user
        # from Pinecone and injects them as a SystemMessage before the LLM runs.
        # After the LLM responds, decides whether to store the current exchange
        # (LLM-judged "is this worth keeping?").
        EpisodicMemoryMiddleware(store=store),

        # ── Layer 6: Summarization ────────────────────────────────────────
        # COMPRESSION — compresses long conversation history when token count
        # reaches 8,000 tokens (~8 LLM turns).
        #
        # WHY 8_000:
        #   Production traces show typical sessions are 1–2 LLM turns (~2,300 tokens).
        #   8,000 tokens = ~8 turns. Normal users never trigger this.
        #   Heavy research sessions (10+ turns) benefit from compression.
        #
        # HOW IT WORKS:
        #   Runs before_model() on every LLM call (not just the first).
        #   When triggered: calls GPT-4o-mini to summarise old messages,
        #   then REPLACES the full history with:
        #     RemoveMessage(REMOVE_ALL) + HumanMessage(summary) + last 10 messages
        #   This is a replacement, not an append — no spurious tokens in response.
        #   Verified from source: _build_new_messages() returns a HumanMessage,
        #   and before_model() returns RemoveMessage(id=REMOVE_ALL_MESSAGES) first.
        SummarizationMiddleware(
            model   = "openai:gpt-4o-mini",   # cheap model — summarization is simple extraction
            trigger = ("tokens", 8_000),       # fires at ~8 turns, never on typical 1–2 turn sessions
            keep    = ("messages", 10),        # preserve last 10 messages after summarization
        ),

        # ── Layer 7: HITL — Single Clarification Gate ─────────────────────
        # GATE — intercepts broad queries and forces clarification via the
        # ask_user_input tool before the LLM answers.
        #
        # Code-enforcement: _is_broad_query() detects broad queries and injects
        # SystemMessage("MANDATORY OVERRIDE: call ask_user_input").
        # More reliable than prompt-only instructions — GPT-4o's base training
        # to be helpful overrides vague prompt rules.
        #
        # interrupt_on fires LangGraph's checkpoint pause before the tool runs.
        # app.py reads the question/options via Path C (agent.aget_state()).
        # SingleClarificationHITLMiddleware(
        #     interrupt_on={"tool-hitl___ask_user_input": True},
        # ),
        HumanInTheLoopMiddleware(
            interrupt_on={"clarify___ask_user_input": True},
        ),

        # ── Layer 8: Action Guardrail (DISABLED) ──────────────────────────
        # Would enforce per-request tool call limits.
        # Disabled — fires too aggressively on legitimate multi-tool workflows.
        # Re-enable after tuning limits from production DynamoDB trace data.
        #
        # ActionGuardrailMiddleware(),

        # ── Layer 9: Output Guardrail ─────────────────────────────────────
        # LAST — closest to the user on egress. Three-layer output check:
        #   Code check:  check_medical_action_output() regex — < 1ms
        #   Layer 2:     LLM faithfulness judge (GPT-4o-mini) — ~500ms
        #   Layer 3:     LLM consistency check  (GPT-4o-mini) — ~500ms
        #
        # Thresholds set to 0.00 during calibration phase.
        # TODO: raise both to 0.70 after 500+ production traces.
        OutputGuardrailMiddleware(
            llm                    = safety_llm,
            faithfulness_threshold = 0.00,   # TODO: raise to 0.70 after calibration
            confidence_threshold   = 0.00,   # TODO: raise to 0.70 after calibration
        ),
    ]


# ── Public API ────────────────────────────────────────────────────────────────
# __all__ controls what "from agent.middleware import *" exports.
__all__ = [
    "TracerMiddleware",
    "DomainPIIMiddleware",
    "ContentFilterMiddleware",
    "SemanticCacheMiddleware",
    "EpisodicMemoryMiddleware",
    "ActionGuardrailMiddleware",
    "OutputGuardrailMiddleware",
    "SingleClarificationHITLMiddleware",
    "build_stack",
]