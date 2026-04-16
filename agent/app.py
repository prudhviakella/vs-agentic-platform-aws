"""
app.py — AgentCore Runtime Entrypoint
=======================================

This file is the ENTRY POINT for the AgentCore serverless container.
When AWS invokes your agent, it calls handler() via the @app.entrypoint decorator.

OVERALL FLOW:
  1. First invocation on a new thread_id → cold start (_ensure_agent builds everything)
  2. handler() decides: is this a new chat or a HITL resume?
  3. _stream_events() runs the LangGraph agent and yields typed event dicts
  4. Each yielded dict is serialised as an SSE event by AgentCore and sent to the caller

WHY AGENTCORE INSTEAD OF PLAIN FASTAPI?
  AgentCore gives us:
    - Serverless arm64 container with auto-scaling
    - One container per runtimeSessionId (clean state per user session)
    - Built-in SigV4 auth — callers must sign requests with IAM credentials
    - Native streaming via the yield pattern (no manual SSE boilerplate)
    - CloudWatch logging integration via the otel collector sidecar

HITL DESIGN — why it's more complex than you'd expect:
  Normally you'd think: call the tool → get the question/options back → yield them.
  But LangGraph's interrupt_on mechanism intercepts the tool call BEFORE on_tool_start
  fires in astream_events. So by the time we see any event, the tool has already been
  intercepted and the graph is paused — hitl_input is always empty from the event stream.

  The solution: after astream_events ends (the stream closes with no tokens),
  we call agent.aget_state() and read the interrupt value directly from the
  LangGraph checkpoint in Postgres. This is Path C and it is the primary path.

  We still handle three other paths as fallbacks:
    Path A — if interrupt_on is disabled and the HITL tool actually runs to completion
    Path B — if LangGraph emits __interrupt__ as an on_chain_end event (some versions)
    Path D — if LangGraph raises GraphInterrupt as an exception (older versions)

RESUME FLOW:
  After HITL, the UI POSTs to /resume with user_answer.
  The platform sends payload={resume: True, user_answer: "..."}
  handler() repairs any dangling tool calls in the Postgres checkpoint,
  then injects a "[HITL Answer]: X. Now search and answer." user message.
  HumanInTheLoopMiddleware.before_agent() detects the prior HITL ToolMessage
  in history and injects a SystemMessage preventing a second HITL call.

EPISODIC TAG STRIPPING:
  EpisodicMemoryMiddleware instructs the LLM to append "EPISODIC: YES/NO" to every
  response. EpisodicMemoryMiddleware.after_agent() strips it from state["messages"]
  before storing to Pinecone, but by then the tokens have already streamed out.
  Fix: _stream_events() buffers the last _TAIL_SIZE chars of accumulated tokens.
  Safe prefix (everything before the tail) is yielded immediately — streaming is
  preserved for 99% of the response. When the stream ends, the tail is flushed
  after stripping any EPISODIC tag. The tail size (60 chars) catches the worst case:
  "\nEPISODIC: NO1.01.0" ≈ 20 chars plus leading whitespace.
"""

import logging
import os
import re
import time

import boto3

from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agent.agent import build_agent
from agent.tools.mcp_client import get_mcp_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── AgentCore app instance ─────────────────────────────────────────────────
app = BedrockAgentCoreApp()

# ── Global agent cache ─────────────────────────────────────────────────────
_agent = None

# ── EPISODIC tag pattern ───────────────────────────────────────────────────
# Matches "\nEPISODIC: YES" or "\nEPISODIC: NO" optionally followed by
# numbers (faithfulness/consistency scores the prompt incorrectly appends).
# re.DOTALL not needed — $ anchors to end of string with re.MULTILINE off.
# Examples matched:
#   "\nEPISODIC: NO"            — clean case
#   "\nEPISODIC: NO1.01.0"      — scores appended (prompt bug)
#   "\n\nEPISODIC: YES 0.9 1.0" — spaces and floats
#   " \nEPISODIC: NO"           — leading space before newline
_EPISODIC_PATTERN = re.compile(
    r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$',
    re.IGNORECASE,
)

# Number of chars to hold back as a tail buffer for EPISODIC tag stripping.
# 60 chars safely covers: "\nEPISODIC: NO1.01.0" (≈20 chars) + whitespace + margin.
# The safe prefix (everything before the tail) is streamed immediately.
_TAIL_SIZE = 60


def _extract_hitl_input(raw_input: dict) -> dict:
    """
    Extract HITL tool args from the on_tool_start event data["input"].

    WHY this function exists:
      LangChain tools can be called in two ways:
        1. Native tool (in-process): data["input"] = {"question": "...", "options": [...]}
        2. MCP tool (via gateway):   data["input"] = {"arguments": {"question": "...", "options": [...]}}

      MCP wraps all tool arguments under an "arguments" key because MCP's
      JSON-RPC protocol uses a standard envelope format. We need to unwrap it.
    """
    if not raw_input:
        return {}
    if "arguments" in raw_input:
        return raw_input["arguments"]
    if "question" in raw_input or "options" in raw_input:
        return raw_input
    return raw_input


def _extract_interrupt_args(iv: dict) -> dict:
    """
    Extract question/options/allow_freetext from a LangGraph interrupt value.

    LangChain HumanInTheLoopMiddleware stores the interrupt value as:
      {"action_requests": [{"name": "...", "args": {"question": "...", "options": [...]}}]}
    We navigate to action_requests[0]["args"].
    """
    if not iv or not isinstance(iv, dict):
        return {}
    action_requests = iv.get("action_requests", [])
    if action_requests:
        return action_requests[0].get("args", {})
    return iv


async def _ensure_agent():
    """
    Build the agent on first call (cold start), return cached instance on warm calls.

    COLD START SEQUENCE:
      1. Load OpenAI API key from Secrets Manager
      2. Attach CloudWatch log handler
      3. Connect to MCP Gateway and discover tool definitions
      4. Build the LangGraph agent with middleware stack
    """
    global _agent
    if _agent is None:
        log.info("[APP] Cold start — building agent with MCP tools")

        if not os.environ.get("OPENAI_API_KEY"):
            import json
            ssm_prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(
                sm.get_secret_value(SecretId=f"{ssm_prefix}/openai")["SecretString"]
            )
            os.environ["OPENAI_API_KEY"] = secret.get("api_key") or secret.get("OPENAI_API_KEY", "")
            log.info("[APP] OpenAI API key loaded from Secrets Manager")

        try:
            import watchtower
            _rt_id   = os.environ.get("AGENT_RUNTIME_ID", "vs_agentcore_clinical_trial-gAwtrFAHxd")
            _lg_name = f"/aws/bedrock-agentcore/runtimes/{_rt_id}-DEFAULT"
            _cw      = watchtower.CloudWatchLogHandler(
                log_group_name    = _lg_name,
                log_stream_name   = "runtime-logs",
                boto3_client      = boto3.client("logs", region_name="us-east-1"),
                create_log_group  = False,
                create_log_stream = True,
            )
            _cw.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
            root = logging.getLogger()
            if not any(isinstance(h, watchtower.CloudWatchLogHandler) for h in root.handlers):
                root.addHandler(_cw)
                log.info(f"[APP] CloudWatch handler attached → {_lg_name}")
        except Exception as _cw_err:
            log.warning(f"[APP] CloudWatch handler skipped: {_cw_err}")

        mcp_tools = await get_mcp_tools()
        _agent    = await build_agent(
            domain       = "pharma",
            use_postgres = True,
            tools        = mcp_tools,
        )
        log.info(f"[APP] Agent ready  tools={[t.name for t in mcp_tools]}")
    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main handler called by AgentCore for every invocation.

    PAYLOAD STRUCTURE:
      New chat:    {"message": "...", "thread_id": "...", "domain": "pharma", "resume": False}
      HITL resume: {"user_answer": "...", "thread_id": "...", "domain": "pharma", "resume": True}

    EVENT TYPES YIELDED:
      {"type": "token",     "content": "The "}
      {"type": "tool_start","name": "tool-search___..."}
      {"type": "tool_end",  "name": "tool-search___..."}
      {"type": "interrupt", "question": "...", "options": [...]}
      {"type": "done",      "latency_ms": 12345}
      {"type": "error",     "message": "..."}
    """
    t0          = time.perf_counter()
    thread_id   = payload.get("thread_id", "default")
    domain      = payload.get("domain",    "pharma")
    is_resume   = payload.get("resume",    False)
    user_answer = payload.get("user_answer", "")
    message     = payload.get("message",   "")
    session_id  = getattr(context, "session_id", None) or thread_id

    log.info(f"[APP] {'resume' if is_resume else 'chat'}  thread={thread_id}")

    try:
        agent = await _ensure_agent()

        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": session_id,
                "domain":     domain,
            },
            "recursion_limit": 50,
        }

        agent_context = {
            "user_id":    thread_id,
            "session_id": thread_id,
            "domain":     domain,
        }

        if is_resume:
            # ── HITL Resume: repair dangling tool calls ────────────────────
            # When interrupt_on fires, the AIMessage with the tool_call is in
            # Postgres but there's no ToolMessage result for it. OpenAI requires
            # every tool_call to have a matching ToolMessage before the next
            # human turn — inject fake ToolMessages to satisfy this constraint.
            try:
                from langchain_core.messages import ToolMessage
                state    = await agent.aget_state(config)
                messages = state.values.get("messages", [])

                result_ids = {
                    getattr(m, "tool_call_id", None)
                    for m in messages
                    if hasattr(m, "tool_call_id")
                }
                dangling = [
                    tc
                    for msg in messages
                    for tc in getattr(msg, "tool_calls", [])
                    if tc.get("id") not in result_ids
                ]

                if dangling:
                    log.info(f"[APP] Repairing {len(dangling)} dangling tool call(s)")
                    await agent.aupdate_state(config, {
                        "messages": [
                            ToolMessage(
                                content=f"[Interrupted — user answered: {user_answer}]",
                                tool_call_id=tc["id"],
                                name=tc.get("name", "unknown"),
                            )
                            for tc in dangling
                        ]
                    })
            except Exception as e:
                log.warning(f"[APP] State repair failed (continuing): {e}")

            log.info(f"[APP] Resume  answer='{user_answer[:60]}'")
            input_data = {
                "messages": [{
                    "role":    "user",
                    "content": f"[HITL Answer]: {user_answer}. Now search and answer.",
                }]
            }
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        async for event in _stream_events(agent, input_data, config, agent_context):
            yield event

    except Exception as exc:
        log.exception(f"[APP] Handler error: {exc}")
        yield {"type": "error", "message": str(exc)}
    finally:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[APP] done  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


async def _stream_events(agent, input_data, config, agent_context: dict):
    """
    Run the LangGraph agent and yield typed event dicts for each event.

    EPISODIC TAG STRIPPING (tail buffer):
      EpisodicMemoryMiddleware instructs the LLM to append "EPISODIC: YES/NO"
      to every response so after_agent can decide whether to store the Q&A.
      The problem: tokens stream out one at a time. By the time after_agent
      runs to strip the tag from state, the tag has already been sent to the user:
        data: {"type":"token","content":"EP"}
        data: {"type":"token","content":"IS"}
        data: {"type":"token","content":"OD"}
        ...

      FIX: tail buffer.
        - Accumulate tokens into _tail_buffer
        - Whenever the buffer exceeds _TAIL_SIZE (60 chars), flush the safe
          prefix immediately (everything before the last 60 chars)
        - After the stream ends, strip the EPISODIC tag from the tail
          using _EPISODIC_PATTERN, then flush the cleaned tail
        - The user never sees the tag. Streaming is preserved for the entire
          response except the last 60 chars which are flushed at end.

      WHY 60 chars:
        Worst case tag: "\n\nEPISODIC: NO1.01.0" ≈ 22 chars.
        With extra whitespace and margin: 60 chars is safe.
        A larger buffer delays more of the response end — 60 is the right balance.

    FOUR HITL PATHS:
      Path A — on_tool_end (interrupt_on NOT active): tool ran to completion
      Path B — on_chain_end with __interrupt__: some LangGraph versions
      Path C — agent.aget_state() after stream (PRIMARY in production)
      Path D — GraphInterrupt exception (oldest LangGraph versions)
    """
    hitl_input      = {}
    interrupt_fired = False

    # ── Tail buffer for EPISODIC tag stripping ─────────────────────────────
    # Accumulates token content. Safe prefix is flushed immediately.
    # Last _TAIL_SIZE chars are held back until stream ends.
    _tail_buffer = ""

    def _flush_safe_prefix():
        """Yield everything except the last _TAIL_SIZE chars immediately."""
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]
            return safe
        return ""

    try:
        async for event in agent.astream_events(
            input_data,
            config=config,
            version="v2",
            context=agent_context,
        ):
            kind = event.get("event", "")
            name = event.get("name",  "")
            data = event.get("data",  {})

            # ── Token streaming ───────────────────────────────────────────
            # Accumulate into tail buffer, flush safe prefix immediately.
            # This preserves real-time streaming for the bulk of the response
            # while holding back the last 60 chars for EPISODIC tag stripping.
            if kind == "on_chat_model_stream":
                chunk   = data.get("chunk", {})
                content = getattr(chunk, "content", "")

                # Normalise both content formats (string and list[ContentBlock])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, str) and block:
                            _tail_buffer += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            if text:
                                _tail_buffer += text
                elif isinstance(content, str) and content:
                    _tail_buffer += content

                # Flush safe prefix — stream the bulk of the response in real time
                safe = _flush_safe_prefix()
                if safe:
                    yield {"type": "token", "content": safe}

            elif kind == "on_tool_start":
                is_hitl = "ask_user_input" in name
                if is_hitl:
                    raw        = data.get("input", {})
                    hitl_input = _extract_hitl_input(raw)
                    log.info(f"[APP] HITL tool starting  input={hitl_input}")
                else:
                    yield {"type": "tool_start", "name": name}

            elif kind == "on_tool_end":
                end_name = event.get("name", "")
                is_hitl  = "ask_user_input" in end_name
                if is_hitl:
                    # Path A — tool ran to completion (interrupt_on not active)
                    log.info(f"[APP] Path A — HITL tool ended  input={hitl_input}")
                    interrupt_fired = True

                    # Flush tail before yielding interrupt (no EPISODIC expected here
                    # since answer hasn't been generated yet, but flush for safety)
                    if _tail_buffer.strip():
                        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                        if clean:
                            yield {"type": "token", "content": clean}
                        _tail_buffer = ""

                    question = hitl_input.get("question", "")
                    options  = hitl_input.get("options", [])
                    if not question:
                        out = data.get("output", {})
                        if isinstance(out, dict):
                            question = out.get("question", "Please clarify:")
                            options  = out.get("options", [])
                    yield {
                        "type":           "interrupt",
                        "question":       question or "Please clarify:",
                        "options":        options,
                        "allow_freetext": hitl_input.get("allow_freetext", True),
                    }
                    break
                else:
                    yield {"type": "tool_end", "name": end_name}

            elif kind == "on_chain_end":
                # Path B — __interrupt__ surfaced as on_chain_end event
                output = data.get("output", {})
                if isinstance(output, dict) and "__interrupt__" in output:
                    interrupts = output["__interrupt__"]
                    if interrupts:
                        iv    = interrupts[0]
                        val   = iv.value if hasattr(iv, "value") else iv
                        args  = _extract_interrupt_args(val)
                        log.info(f"[APP] Path B — __interrupt__ in chain  args={args}")
                        interrupt_fired = True

                        # Flush tail before yielding interrupt
                        if _tail_buffer.strip():
                            clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                            if clean:
                                yield {"type": "token", "content": clean}
                            _tail_buffer = ""

                        yield {
                            "type":           "interrupt",
                            "question":       args.get("question", "Please clarify:"),
                            "options":        args.get("options", []),
                            "allow_freetext": args.get("allow_freetext", True),
                        }
                        return

    except Exception as exc:
        exc_type = type(exc).__name__
        if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
            # Path D — GraphInterrupt raised as exception
            log.info("[APP] Path D — GraphInterrupt exception")
            interrupt_fired = True

            # Flush tail before yielding interrupt
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""

            yield {
                "type":           "interrupt",
                "question":       hitl_input.get("question", "Please clarify:"),
                "options":        hitl_input.get("options", []),
                "allow_freetext": hitl_input.get("allow_freetext", True),
            }
        else:
            log.exception(f"[APP] Stream error: {exc}")

            # Flush any remaining tail on error
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""

            yield {"type": "error", "message": str(exc)}
        return

    # ── Flush tail buffer with EPISODIC tag stripped ───────────────────────
    # This runs after the async for loop ends (stream complete, no interrupt).
    # Strip "\nEPISODIC: YES/NO[scores]" from the end of the accumulated tail.
    # The EPISODIC tag is always at the very end of the LLM's response.
    #
    # Example tail buffer at this point:
    #   "...always consult healthcare professionals.\n\nEPISODIC: NO1.01.0"
    # After strip:
    #   "...always consult healthcare professionals."
    if _tail_buffer:
        clean_tail = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean_tail:
            yield {"type": "token", "content": clean_tail}
        elif _tail_buffer.strip() and not clean_tail:
            # The entire tail was the EPISODIC tag — nothing to yield.
            # This is the normal case when the response ends exactly with the tag.
            log.debug(f"[APP] Tail was EPISODIC tag only — stripped: '{_tail_buffer.strip()}'")
        _tail_buffer = ""

    # ── Path C: check Postgres checkpoint for pending interrupt (PRIMARY) ──
    # When interrupt_on fires, LangGraph intercepts before on_tool_start.
    # The stream closes with no events. interrupt_fired stays False.
    # We read the checkpoint to find the pending interrupt.
    if not interrupt_fired:
        try:
            state = await agent.aget_state(config)
            tasks = state.tasks if hasattr(state, "tasks") else []
            for task in tasks:
                task_interrupts = getattr(task, "interrupts", [])
                if task_interrupts:
                    iv   = task_interrupts[0]
                    val  = iv.value if hasattr(iv, "value") else iv
                    args = _extract_interrupt_args(val)

                    question = args.get("question", "Please clarify:")
                    options  = args.get("options", [])
                    allow_ft = args.get("allow_freetext", True)

                    log.info(
                        f"[APP] Path C — interrupt in state"
                        f"  question='{question[:60]}'"
                        f"  options={options}"
                    )
                    yield {
                        "type":           "interrupt",
                        "question":       question,
                        "options":        options,
                        "allow_freetext": allow_ft,
                    }
                    return
        except Exception as e:
            log.warning(f"[APP] State interrupt check failed: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)