"""
agent.py — Clinical Trial Agent on AgentCore Runtime
======================================================
Deploy:
    agentcore configure --entrypoint agent.py --region us-east-1
    agentcore launch

WHAT IS @app.entrypoint?
─────────────────────────
BedrockAgentCoreApp wraps this function and turns it into an HTTP server
that AgentCore Runtime knows how to call. When you run agentcore launch,
AgentCore builds a container from this file and deploys it to a serverless
microVM environment.

The function MUST be an async generator (uses yield, not return) so that
AgentCore can stream responses back to the caller in real time.

Each value you yield becomes one SSE line:
    yield {"type": "token", "content": "The NCI-MATCH..."}
    → data: {"type": "token", "content": "The NCI-MATCH..."}\n\n


WHAT IS A MICROVM?
───────────────────
AgentCore runs each USER SESSION in its own isolated micro virtual machine.
Think of it like a fresh container that spins up when a user starts talking,
stays alive for the conversation, and shuts down after 15 minutes of silence.

Consequences for our code:
  - Cold start: first request per microVM runs _ensure_agent() which builds
    the LangGraph graph, connects to MCP gateway, loads the system prompt.
    This takes 3-5 seconds.
  - Warm: subsequent requests in the SAME session reuse _agent (already built).
    This is why _agent and _tools are module-level singletons — they survive
    across multiple turns within ONE session.
  - Isolation: _agent from user A's microVM cannot leak into user B's microVM.


FULL SSE STREAMING FLOW:
─────────────────────────
  User types a question in Chainlit
        │
        ▼
  Chainlit  ──POST /chat──►  Platform FastAPI
        │                           │
        │                    boto3 invoke_agent_runtime()
        │                           │
        │                    AgentCore Runtime (this file)
        │                           │
        │                    LangGraph astream_events()
        │                           │
        │         ←─ SSE token ─────┘  (yield {"type":"token", ...})
        │         ←─ SSE token ─────    (yield {"type":"token", ...})
        │         ←─ SSE done  ─────    (yield {"type":"done",  ...})
        │
  Chainlit calls msg.stream_token() on each token
  → user sees text appearing in real time


HITL FLOW:
───────────
  User asks: "Tell me about the COVID vaccine trial"
        │
        ▼
  Agent calls search_tool → finds Pfizer BNT162b2 and Moderna mRNA-1273
        │
        ▼
  Agent calls ask_user_input(question="Which trial?", options=["Pfizer","Moderna"])
        │
        ▼
  HumanInTheLoopMiddleware intercepts → PAUSES the LangGraph graph
  Full graph state saved to Postgres checkpointer
        │
        ▼
  yield {"type": "interrupt", "question": ..., "options": [...]}
  return  ← stops streaming
        │
        ▼
  User clicks "Pfizer BNT162b2" in Chainlit
        │
        ▼
  POST /resume with user_answer="Pfizer BNT162b2"
        │
        ▼
  Command(resume=...) fed into LangGraph → graph resumes from saved state
  Agent calls search_tool("Pfizer BNT162b2") → answers specifically
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
# Built ONCE per microVM on cold start, reused for every subsequent request
# in the same session. Building the agent takes ~3-5s (connects to MCP
# gateway, fetches system prompt from Bedrock, sets up Postgres checkpointer).
# Caching avoids paying that cost on every message.
_agent = None
_tools = None


async def _ensure_agent():
    """
    Build the LangGraph agent on first call, return cached instance on all
    subsequent calls within the same microVM.

    Not thread-safe, but AgentCore processes one request at a time per
    microVM so concurrent cold starts cannot happen.
    """
    global _agent, _tools
    if _agent is None:
        log.info("[AGENT] Cold start — building agent")
        _tools = await get_mcp_tools()              # discover tools from MCP gateway
        _agent = await build_agent(tools=_tools)    # build LangGraph graph
        log.info(f"[AGENT] Ready  tools={[t.name for t in _tools]}")
    return _agent


# ── AgentCore entrypoint ──────────────────────────────────────────────────

@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main entry point for every request.

    This is an async generator — @app.entrypoint converts each yielded dict
    into one SSE line sent back to the Platform:
        data: {"type": "...", ...}\n\n

    payload fields (sent by Platform FastAPI):
        message     — user question (empty string on resume)
        thread_id   — LangGraph thread ID for Postgres state persistence
        domain      — "pharma" or "general" (controls system prompt framing)
        resume      — True when resuming after HITL, False for new messages
        user_answer — human answer to the HITL question (on resume only)

    context:
        AgentCore passes a RequestContext object here. We use getattr() with
        a fallback because the exact context type varies across SDK versions.
        thread_id from the payload is the reliable conversation identifier.
    """
    t0          = time.perf_counter()
    thread_id   = payload.get("thread_id", "default")
    domain      = payload.get("domain", "pharma")
    is_resume   = payload.get("resume", False)
    user_answer = payload.get("user_answer", "")
    message     = payload.get("message", "")

    # AgentCore microVM session ID — identifies this container instance.
    # Different from thread_id: session_id tracks the microVM lifetime,
    # thread_id tracks the LangGraph conversation state in Postgres.
    session_id = getattr(context, "session_id", None) or thread_id

    log.info(
        f"[HANDLER] {'resume' if is_resume else 'chat'}"
        f"  thread={thread_id}  session={session_id}"
    )

    try:
        agent = await _ensure_agent()

        # LangGraph uses thread_id to load/save conversation state from Postgres.
        # Same thread_id across /chat and /resume calls means LangGraph can
        # find the paused graph state and continue from where it stopped.
        config = {
            "configurable": {
                "thread_id": thread_id,
                "session_id": session_id,
                "domain":     domain,
            }
        }

        if is_resume:
            # RESUME AFTER HITL
            # ─────────────────
            # Standard LangGraph docs show: Command(resume="answer")
            # We use a richer format because HumanInTheLoopMiddleware expects
            # this specific structure to inject the user_answer as a ToolMessage
            # for the paused ask_user_input tool call.
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
            # NEW MESSAGE
            # Standard LangGraph message format. The agent receives this as
            # a HumanMessage at the start of the conversation turn.
            input_data = {"messages": [{"role": "user", "content": message}]}

        async for event in _stream_events(agent, input_data, config):
            yield event

    except Exception as exc:
        log.exception(f"[HANDLER] Error: {exc}")
        yield {"type": "error", "message": str(exc)}

    finally:
        # "done" is ALWAYS the last event — even after an error or HITL interrupt.
        # Platform and Chainlit use it as the signal that the SSE stream has ended.
        # Chainlit ignores "done" if it already handled an error or interrupt.
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[HANDLER] Complete  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


async def _stream_events(agent, input_data, config):
    """
    Translates raw LangGraph astream_events() output into typed SSE dicts.

    WHY astream_events() AND NOT astream()?
    ─────────────────────────────────────────
    astream() yields the full LangGraph state only after each NODE completes.
    The user sees nothing until the entire answer is generated — no streaming.

    astream_events(version="v2") fires a separate event for every internal
    action: each LLM token, each tool call start and end. This is what enables
    tokens to appear in the UI one by one as GPT-4o generates them.

    EVENT TYPES WE HANDLE:
    ───────────────────────
    on_chat_model_stream  → one token from GPT-4o
    on_tool_start         → agent decided to call a tool
    on_tool_end           → tool returned a result to the agent
    """
    current_tool = None

    try:
        async for event in agent.astream_events(
            input_data, config=config, version="v2"
        ):
            kind = event.get("event", "")
            name = event.get("name", "")
            data = event.get("data", {})

            # ── LLM generated one token ───────────────────────────────────
            # content is a string fragment (e.g. "The " "NCI-" "MATCH" "...")
            # We forward each immediately so the UI renders as it arrives.
            if kind == "on_chat_model_stream":
                content = getattr(data.get("chunk", {}), "content", "") or ""
                if content:
                    yield {"type": "token", "content": content}

            # ── Agent is calling a tool ───────────────────────────────────
            elif kind == "on_tool_start":
                current_tool = name
                tool_input   = data.get("input", {})

                if name == "ask_user_input":
                    # HITL interrupt — pause the stream here.
                    # LangGraph graph is paused inside HumanInTheLoopMiddleware.
                    # Graph state is saved to Postgres by the checkpointer.
                    # "return" exits this generator cleanly — handler's
                    # finally block then sends the "done" event.
                    log.info(
                        f"[STREAM] HITL interrupt"
                        f"  question='{str(tool_input.get('question',''))[:60]}'"
                    )
                    yield {
                        "type":           "interrupt",
                        "question":       tool_input.get("question", "Please clarify:"),
                        "options":        tool_input.get("options", []),
                        "allow_freetext": tool_input.get("allow_freetext", True),
                    }
                    return

                else:
                    log.info(f"[STREAM] tool_start  name={name}")
                    yield {"type": "tool_start", "name": name}

            # ── Tool returned a result ────────────────────────────────────
            # We do NOT yield the raw tool result — the LLM synthesises it
            # into the final answer. We just update the UI loading indicator.
            elif kind == "on_tool_end":
                if current_tool and current_tool != "ask_user_input":
                    yield {"type": "tool_end", "name": current_tool}
                current_tool = None

    except Exception as exc:
        log.exception(f"[STREAM] Error during streaming: {exc}")
        yield {"type": "error", "message": str(exc)}


if __name__ == "__main__":
    # Local testing only — agentcore launch handles startup in production
    app.run(host="0.0.0.0", port=8080)
