"""
pii.py — DomainPIIMiddleware
==============================
One middleware instance covering ALL domain PII rules — input AND output.

WHY one class not three PIIMiddleware instances:
  LangChain 1.0 asserts all middleware names must be unique:
    assert len({m.name for m in middleware}) == len(middleware)
  Three PIIMiddleware() calls all share the same .name → AssertionError.
  Solution: one custom class, one name, handles all PII rules internally.

WHY here (domain middleware, not gateway):
  PII definition is domain-specific:
    Pharma:    patient email, credit card numbers
    Finance:   account numbers, SSN, routing numbers
    Marketing: phone numbers, postal codes
  Gateway is generic — it cannot know what counts as PII per domain.

Rules applied (pharma domain):
  INPUT  — email       → [REDACTED_EMAIL]
  INPUT  — credit card → ****-****-****-1234
  OUTPUT — email       → [REDACTED_EMAIL]
"""

import logging
import re
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

log = logging.getLogger(__name__)


class DomainPIIMiddleware(BaseAgentMiddleware):

    _EMAIL_PAT = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
    _CC_PAT    = re.compile(r"\b(?:\d[ \-]?){15,16}\b")

    @staticmethod
    def _redact_email(text: str) -> str:
        return DomainPIIMiddleware._EMAIL_PAT.sub("[REDACTED_EMAIL]", text)

    @staticmethod
    def _mask_cc(text: str) -> str:
        def _mask(m):
            digits = re.sub(r"\D", "", m.group())
            return "****-****-****-" + digits[-4:]
        return DomainPIIMiddleware._CC_PAT.sub(_mask, text)

    @staticmethod
    def _clean_input(text: str) -> str:
        text = DomainPIIMiddleware._redact_email(text)
        text = DomainPIIMiddleware._mask_cc(text)
        return text

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Redact PII from the latest user message before LLM sees it."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not (hasattr(last_msg, "type") and last_msg.type == "human"):
            return None

        original = str(last_msg.content)
        cleaned = self._clean_input(original)

        if cleaned == original:
            return None  # nothing changed — no state update needed

        log.info(f"[DOMAIN_PII] Input redacted  '{original[:40]}' → '{cleaned[:40]}'")
        # Create a new message object — don't mutate in place
        #redacted_msg = last_msg.copy(update={"content": cleaned})  # Pydantic v1
        redacted_msg = last_msg.model_copy(update={"content": cleaned})  # Pydantic v2

        # Return updated state — replace last message, keep the rest intact
        return {"messages": messages[:-1] + [redacted_msg]}

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Redact email from the final AI response before it reaches the user."""
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            return None

        original = str(last_msg.content)
        cleaned = self._redact_email(original)

        if cleaned == original:
            return None  # nothing changed — no state update needed

        log.info(f"[DOMAIN_PII] Output redacted  '{original[:40]}' → '{cleaned[:40]}'")

        # New message object — don't mutate in place (Pydantic v2)
        redacted_msg = last_msg.model_copy(update={"content": cleaned})

        # Return updated state
        return {"messages": messages[:-1] + [redacted_msg]}