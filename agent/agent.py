"""
agent.py — Clinical Trial Agent on AgentCore Runtime
======================================================
Deploy: agentcore configure --entrypoint agent.py && agentcore launch

STREAMING FLOW:
  User → POST /chat → Platform FastAPI
    → boto3 invoke_agent_runtime (AgentCore)
      → @app.entrypoint async generator
        → yields {"type": "token", "content": "..."} etc.
      → AgentCore auto-converts each yield to SSE: data: {...}\n\n
    → Platform forwards SSE stream to Chainlit
  Chainlit streams tokens in real time

HITL FLOW:
  Agent calls ask_user_input → HumanInTheLoopMiddleware intercepts
  → graph PAUSES → yields {"type": "interrupt", "question": ..., "options": [...]}
  → Platform forwards interrupt SSE to Chainlit
  → Chainlit shows option buttons
  → User clicks → POST /resume → Platform → AgentCore with resume payload
  → Agent continues with user_answer as ToolMessage
"""

import logging
import os
import time

from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext
from langgraph.types import Command

from agent.graph import build_agent
from agent.tools.mcp_client import get_mcp_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# ── Warm container singletons ─────────────────────────────────────────────
_agent = None
_tools = None


async def _ensure_agent():
    """Build agent once per microVM — reused across sessions in warm container."""
    global _agent, _tools
    if _agent is None:
        log.info("[AGENT] Cold start — building agent")
        _tools = await get_mcp_tools()
        _agent = await build_agent(tools=_tools)
        log.info(f"[AGENT] Ready  tools={[t.name for t in _tools]}")
    return _agent


# ── AgentCore entrypoint ──────────────────────────────────────────────────

@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main handler — async generator.
    AgentCore converts each yielded dict → SSE: data: {...}\n\n

    Event types yielded:
      {"type": "token",      "content": "The NCI-MATCH..."}
      {"type": "tool_start", "name": "search_tool"}
      {"type": "tool_end",   "name": "search_tool"}
      {"type": "interrupt",  "question": "...", "options": [...], "allow_freetext": true}
      {"type": "done",       "latency_ms": 12345}
      {"type": "error",      "message": "..."}
    """
    t0          = time.perf_counter()
    thread_id   = payload.get("thread_id", "default")
    domain      = payload.get("domain", "pharma")
    is_resume   = payload.get("resume", False)
    user_answer = payload.get("user_answer", "")
    message     = payload.get("message", "")
    session_id  = getattr(context, "session_id", None) or thread_id

    log.info(
        f"[HANDLER] {'resume' if is_resume else 'chat'}"
        f"  thread={thread_id}  session={session_id}"
    )

    try:
        agent = await _ensure_agent()

        config = {
            "configurable": {
                "thread_id": thread_id,
                "session_id": session_id,
                "domain": domain,
            }
        }

        if is_resume:
            input_data = Command(
                resume={
                    "decisions": [{
                        "type": "edit",
                        "edited_action": {
                            "name": "ask_user_input",
                            "args": {"user_answer": user_answer},
                        },
                    }]
                }
            )
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        async for event in _stream_events(agent, input_data, config):
            yield event

    except Exception as exc:
        log.exception(f"[HANDLER] Error: {exc}")
        yield {"type": "error", "message": str(exc)}
    finally:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[HANDLER] Complete  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


async def _stream_events(agent, input_data, config):
    """Translate LangGraph stream_events() into typed SSE dicts."""
    current_tool = None

    try:
        async for event in agent.astream_events(
            input_data, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── LLM token ─────────────────────────────────────────────────
            if kind == "on_chat_model_stream":
                content = getattr(data.get("chunk", {}), "content", "") or ""
                if content:
                    yield {"type": "token", "content": content}

            # ── Tool started ──────────────────────────────────────────────
            elif kind == "on_tool_start":
                current_tool = name
                tool_input   = data.get("input", {})

                if name == "ask_user_input":
                    # HITL — stream interrupt and stop
                    log.info(f"[STREAM] HITL interrupt  question='{str(tool_input.get('question',''))[:60]}'")
                    yield {
                        "type":          "interrupt",
                        "question":      tool_input.get("question", "Please clarify:"),
                        "options":       tool_input.get("options", []),
                        "allow_freetext": tool_input.get("allow_freetext", True),
                    }
                    return  # stop streaming — Platform holds for /resume
                else:
                    log.info(f"[STREAM] tool_start  name={name}")
                    yield {"type": "tool_start", "name": name}

            # ── Tool finished ─────────────────────────────────────────────
            elif kind == "on_tool_end":
                if current_tool and current_tool != "ask_user_input":
                    yield {"type": "tool_end", "name": current_tool}
                current_tool = None

            # ── LangGraph interrupt (backup detection) ────────────────────
            elif kind == "on_chain_end":
                output = data.get("output", {})
                if isinstance(output, dict) and output.get("__interrupt__"):
                    interrupts = output["__interrupt__"]
                    if interrupts:
                        v = interrupts[0].value
                        yield {
                            "type":          "interrupt",
                            "question":      v.get("question", "Please clarify:"),
                            "options":       v.get("options", []),
                            "allow_freetext": v.get("allow_freetext", True),
                        }
                        return

    except Exception as exc:
        log.exception(f"[STREAM] Error: {exc}")
        yield {"type": "error", "message": str(exc)}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
