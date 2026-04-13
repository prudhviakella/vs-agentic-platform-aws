"""
content_filter.py — ContentFilterMiddleware
=============================================
Blocks queries that are clearly outside the clinical research domain
BEFORE they reach the LLM. This saves tokens and prevents misuse.

Strategy: keyword blocklist for obviously off-topic content.
Anything ambiguous is allowed through — the LLM's system prompt
handles domain scoping for grey areas.

If a query is blocked, state["blocked"] = True and
state["block_reason"] = "..." is set. MiddlewareAgent checks this
and returns the block message without calling the LLM.
"""

import logging
import re

from langchain_core.messages import HumanMessage

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

# Patterns that are clearly off-topic for a clinical research assistant
_BLOCKED_PATTERNS = [
    # Code generation
    (r'\b(write|generate|create|code|program|script)\b.{0,30}\b(python|javascript|java|sql|html|css)\b', "general coding"),
    # Finance / crypto
    (r'\b(bitcoin|ethereum|crypto|stock\s+price|forex|trading)\b', "financial content"),
    # Entertainment
    (r'\b(movie|film|song|lyrics|recipe|cook|restaurant)\b', "off-topic entertainment"),
    # Politics
    (r'\b(election|vote|democrat|republican|political\s+party)\b', "political content"),
    # Explicit personal advice
    (r'\b(my\s+doctor|my\s+patient|prescribe\s+me|dose\s+for\s+me)\b', "personal medical advice"),
]

_BLOCK_MESSAGE = (
    "I'm a clinical research assistant focused on clinical trial data, "
    "drug efficacy, safety profiles, and biomedical knowledge graphs. "
    "I'm not able to help with {reason}. "
    "Please ask me about clinical trials, trial results, drugs, diseases, "
    "or related biomedical research topics."
)


class ContentFilterMiddleware(BaseAgentMiddleware):
    """
    Blocks clearly off-topic queries before they hit the LLM.
    Ambiguous queries pass through — the system prompt handles scoping.
    """

    async def before_agent(self, state: dict) -> None:
        messages = state.get("messages", [])
        if not messages:
            return

        # Get text of the last human message
        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = str(msg.content).lower()
                break

        if not query:
            return

        for pattern, reason in _BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                state["blocked"]      = True
                state["block_reason"] = reason
                state["cached"]       = _BLOCK_MESSAGE.format(reason=reason)
                log.info(f"[CONTENT_FILTER] Blocked: {reason}  query='{query[:60]}'")
                return

        log.debug(f"[CONTENT_FILTER] Allowed: '{query[:60]}'")
