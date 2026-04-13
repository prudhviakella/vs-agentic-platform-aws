"""
handler.py — hitl_lambda
=========================
MCP Lambda tool: Human-in-the-Loop ask_user_input.

Registered in MCP Gateway as tool name: ask_user_input

HOW HITL WORKS ON AGENTCORE:
  Initial call (LLM → tool):
    event = {"question": "Which trial?", "options": [...], "user_answer": ""}
    LangGraph middleware intercepts → graph PAUSES → interrupt event streamed to UI
    This Lambda is NOT actually invoked on initial call — interrupt happens in agent

  Resume call (user answered → POST /resume → agent):
    event = {"user_answer": "Pfizer BNT162b2", "question": null, "options": null}
    Returns user_answer as string → becomes ToolMessage → LLM continues

No external dependencies needed — pure Python.
"""

import json
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def handler(event: dict, context) -> str:
    """
    HITL tool handler.
    On resume: returns user_answer so LangGraph adds it as ToolMessage.
    """
    user_answer = event.get("user_answer", "")
    question    = event.get("question", "")

    log.info(
        f"[HITL] question='{str(question)[:60]}'"
        f"  user_answer='{str(user_answer)[:60]}'"
    )

    # Return the human answer — LangGraph adds this as ToolMessage
    # The LLM sees it and uses it to call search_tool with the specific answer
    return user_answer
