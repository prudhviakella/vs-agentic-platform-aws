"""
content_filter.py — ContentFilterMiddleware
=============================================

Domain-specific input guardrail — lives in AGENT middleware (agent/middleware/),
not in gateway middleware (platform/gateway/).

WHY HERE AND NOT AT THE GATEWAY:
  The platform gateway is domain-agnostic. It serves any agent — a pharma agent
  today, a finance or legal agent tomorrow. The gateway has no knowledge of what
  constitutes "harmful content" in any specific domain.

  What counts as toxic is a DOMAIN DECISION:
    Pharma agent:     "how to synthesise this compound" is dangerous → block
    Finance agent:    "how to front-run this trade" is dangerous → block
    Marketing agent:  "how to DDOS a competitor" is dangerous → block
    General assistant: a much narrower set of harmful patterns

  Putting domain toxic patterns in the gateway would mean:
    1. The gateway needs to import pharma-specific regex → tight coupling
    2. Every new domain agent requires a gateway code change → not scalable
    3. The gateway can't be shared across domains without custom rules per domain

  Solution: gateway handles domain-AGNOSTIC security (prompt injection),
  agent middleware handles domain-SPECIFIC content rules (toxic patterns).

  See guardrails.py for the pattern split:
    check_prompt_injection()  → called by GATEWAY (generic, domain-agnostic)
    check_toxic()             → called HERE (pharma-domain toxic patterns)

WHAT THIS HANDLES:
  - Pharma-domain toxic content (check_toxic from guardrails.py):
    - Violence against patients / persons
    - Dangerous synthesis instructions (weapons, poisons)

WHAT THIS DOES NOT HANDLE:
  - Prompt injection → already caught by gateway before the agent receives the request
  - PII → handled by DomainPIIMiddleware (layer 2, runs before this)
  - Off-topic queries → handled by the system prompt's domain framing
  - Output safety → handled by OutputGuardrailMiddleware (layer 9)

LAYER POSITION (layer 3) AND ITS IMPLICATIONS:
  ContentFilterMiddleware runs AFTER DomainPIIMiddleware (layer 2).
  This ordering matters: PII is scrubbed before toxic check.

  WHY: a query like "Dr Smith at MGH wants to know how to harm a patient"
  should have "Dr Smith at MGH" redacted first (PII), then the remaining
  "how to harm a patient" text is checked for toxic content.
  If we checked toxic before PII, we'd be logging the real name in the
  warning message ("Toxic blocked ... input='Dr Smith at MGH...'").

  ContentFilterMiddleware runs BEFORE SemanticCacheMiddleware (layer 4).
  WHY: no point doing a Pinecone lookup for a query we're going to block anyway.
  Blocked queries must never be cached — if they were, a future non-blocked
  query that's semantically similar might get a "blocked" response from cache.
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from agent.guardrails import check_toxic

log = logging.getLogger(__name__)


class ContentFilterMiddleware(BaseAgentMiddleware):
    """
    Checks the current user input against pharma-domain toxic content patterns.
    Blocks immediately on match — no LLM call, no tool call, no cache write.

    Only implements before_agent (no after_agent needed) because content
    filtering is purely an input concern. Output safety is handled separately
    by OutputGuardrailMiddleware at layer 9.
    """

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Check the current user input for toxic content. Block on match.

        WHY check messages[-1] (last message) instead of first HumanMessage:
          LangChain appends the new user message to state["messages"] BEFORE
          before_agent fires. So the current user input is always the last
          element in the list at this point in execution.

          Checking the first HumanMessage would only work on turn 1.
          On turn 2+, the first HumanMessage is the original query from
          turn 1 — checking it again is both wrong (already checked) and
          wasteful. The current question is always messages[-1].

        WHY skip if last message is not a HumanMessage:
          In a multi-turn conversation with tool calls, the last message
          might be a ToolMessage (tool result) or an AIMessage (LLM turn).
          Those are not user inputs — no toxic check needed.
          We only check content that came FROM the user.

        Returns:
          None                          — input is clean, proceed to next middleware
          {"messages": [...], "jump_to": "end"}  — toxic content, block immediately

        can_jump_to=["end"]: required to allow returning jump_to="end".
          Without this hook_config declaration, LangChain raises a config error
          when this middleware tries to short-circuit execution.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # Check the LAST message — always the current user input.
        # See docstring above for why last and not first-human.
        last_msg = messages[-1]
        if not (hasattr(last_msg, "type") and last_msg.type == "human"):
            # Last message is a ToolMessage or AIMessage — not user input, skip.
            return None

        user_content = str(last_msg.content)
        if not user_content.strip():
            # Empty message — nothing to check.
            return None

        # ── Pharma-domain toxic content check ────────────────────────────
        # check_toxic() runs _TOXIC_PATTERNS regex from guardrails.py.
        # Returns (True, "") for clean input, (False, reason) for toxic.
        # Runtime: < 1ms — pure regex, no LLM, no network.
        ok, reason = check_toxic(user_content)

        if not ok:
            # ── BLOCKED ───────────────────────────────────────────────────
            # Return a user-facing error message and jump to "end".
            # jump_to="end" skips all remaining middleware (layers 4–9)
            # and the agent — no LLM call, no tool calls, no cache write.
            #
            # WHY include the reason in the response:
            #   Opaque "blocked" messages frustrate legitimate users who
            #   accidentally triggered a pattern. Showing the reason helps
            #   them rephrase a valid clinical question if they were blocked
            #   by a false positive.
            #
            # WHY log as WARNING not ERROR:
            #   A blocked input is expected and handled — not an application
            #   error. WARNING means "unusual event worth noting" which is
            #   correct. ERROR should be reserved for unexpected failures.
            log.warning(
                f"[CONTENT_FILTER] Toxic blocked"
                f"  reason='{reason}'"
                f"  input='{user_content[:60]}'"
            )
            return {
                "messages": [AIMessage(content=(
                    "Your request could not be processed — it matches a prohibited "
                    f"content pattern for this domain. Reason: {reason}."
                ))],
                "jump_to": "end",
            }

        # ── PASSED ────────────────────────────────────────────────────────
        # Input is clean — return None (no state modification) so execution
        # continues to layer 4 (SemanticCacheMiddleware).
        log.info(f"[CONTENT_FILTER] Passed  input='{user_content[:60]}'")
        return None