"""
handler.py — hitl_lambda
=========================
HumanInTheLoopMiddleware intercepts BEFORE this tool runs on first call.
This handler only runs on RESUME when user_answer is injected via Command.

Returns user_answer as string → becomes ToolMessage → LLM continues.
"""
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def handler(event: dict, context) -> str:
    user_answer = event.get("user_answer", "")
    question    = event.get("question", "")
    log.info(f"[HITL] question='{str(question)[:60]}'  user_answer='{str(user_answer)[:60]}'")
    return user_answer