"""
pii.py — DomainPIIMiddleware
=============================
Scrubs Personally Identifiable Information from user queries BEFORE
the LLM sees them. This ensures no real patient data is sent to OpenAI.

Patterns scrubbed:
  - Patient names (heuristic: Dr. / Mr. / Mrs. + capitalised words)
  - NHS / medical record numbers
  - Dates of birth
  - Email addresses
  - Phone numbers
  - UK / US postcodes

After scrubbing, state["pii_scrubbed"] = True if anything was replaced.
The original query is NOT stored — scrubbing is in-place and irreversible.
"""

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

# Patterns to scrub — ordered from most specific to least specific
_PATTERNS = [
    # Email
    (r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b', "[EMAIL]"),
    # Phone (UK/US)
    (r'\b(\+44|0044|0)[\s\-]?(\d[\s\-]?){9,10}\b', "[PHONE]"),
    (r'\b(\+1[\s\-]?)?\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}\b', "[PHONE]"),
    # Date of birth
    (r'\b(0?[1-9]|[12]\d|3[01])[/\-](0?[1-9]|1[0-2])[/\-](\d{2}|\d{4})\b', "[DOB]"),
    # NHS number (10 digits with optional spaces)
    (r'\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b', "[NHS_NUMBER]"),
    # Medical record numbers (MRN: 6-10 digits)
    (r'\bMRN[\s:#]?\d{6,10}\b', "[MRN]"),
    # UK postcode
    (r'\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b', "[POSTCODE]"),
    # US zipcode
    (r'\b\d{5}(-\d{4})?\b', "[ZIPCODE]"),
    # Patient names with title prefix
    (r'\b(Dr|Mr|Mrs|Ms|Miss|Prof)\.?\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?\b', "[PATIENT_NAME]"),
]


def _scrub(text: str) -> tuple[str, bool]:
    """Apply all PII patterns and return (scrubbed_text, was_modified)."""
    original = text
    for pattern, replacement in _PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text, text != original


class DomainPIIMiddleware(BaseAgentMiddleware):
    """
    Scrubs PII from the last HumanMessage in the conversation.
    Only the most recent user message is scrubbed — older messages
    have already been through the pipeline.
    """

    async def before_agent(self, state: dict) -> None:
        messages = state.get("messages", [])
        if not messages:
            return

        # Find the last HumanMessage and scrub it
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, HumanMessage):
                scrubbed, modified = _scrub(str(msg.content))
                if modified:
                    messages[i] = HumanMessage(content=scrubbed)
                    state["pii_scrubbed"] = True
                    log.info("[PII] Scrubbed PII from user message")
                break

        state["messages"] = messages
