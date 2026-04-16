"""
app.py — VS AgentCore Platform · Chainlit UI (SSE Streaming)
=============================================================
Connects to Platform FastAPI via SSE and streams tokens in real time.

Run locally:
    AGENT_API_URL=http://localhost:8000 chainlit run app.py --port 8501

Environment:
    AGENT_API_URL  — Platform FastAPI URL  (default: http://localhost:8000)
    AGENT_API_KEY  — X-API-Key header      (default: local-dev-key)
    AGENT_DOMAIN   — domain                (default: pharma)
"""

import json
import logging
import os
import re
import uuid

import chainlit as cl
import httpx

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("AGENT_API_KEY", "local-dev-key")
DOMAIN  = os.environ.get("AGENT_DOMAIN",  "pharma")
AGENT   = "clinical-trial"

HEADERS = {
    "X-API-Key":    API_KEY,
    "Content-Type": "application/json",
    "Accept":       "text/event-stream",
}

STARTERS = [
    "What are the Phase 3 efficacy results for Pfizer BNT162b2?",
    "Tell me about the COVID vaccine trial",
    "Is mRNA-1273 safe for patients with heart failure?",
    "Which trials study remdesivir for COVID-19?",
    "What are the primary outcomes for the Moderna vaccine trial NCT04470427?",
    "Who sponsors the Hepatitis B TAF trial?",
]


# ── Answer cleanup ────────────────────────────────────────────────────────

def _clean(answer: str) -> str:
    """Strip internal tags before showing answer to user."""
    # EPISODIC tag
    answer = re.sub(r'\n?EPISODIC:\s*(YES|NO)\s*', '', answer, flags=re.IGNORECASE)
    # Duplicate disclaimer (footer adds it back)
    answer = re.sub(
        r'\n?This information is for research purposes only'
        r' and does not constitute medical advice\.?\s*',
        '',
        answer,
        flags=re.IGNORECASE,
    )
    # Guardrail reason if leaked
    answer = re.sub(
        r'\n?\[Reason logged for review:.*?\]\s*',
        '',
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return answer.strip()


# ── Session helpers ───────────────────────────────────────────────────────

def get_thread_id() -> str:
    tid = cl.user_session.get("thread_id")
    if not tid:
        tid = str(uuid.uuid4())  # full UUID — AgentCore runtimeSessionId requires min 33 chars
        cl.user_session.set("thread_id", tid)
    return tid

def set_interrupted(val: bool):
    cl.user_session.set("interrupted", val)

def is_interrupted() -> bool:
    return bool(cl.user_session.get("interrupted", False))


# ── Lifecycle ─────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_start():
    thread_id = get_thread_id()
    set_interrupted(False)

    actions = [
        cl.Action(
            name="starter", value=q, label=q,
            description=q, payload={"value": q}
        )
        for q in STARTERS
    ]

    await cl.Message(
        content=(
            "## ⚕ Clinical Trial Research Assistant\n\n"
            "Explore clinical trial data, drug efficacy, safety profiles, "
            "and biomedical knowledge graphs powered by Pinecone + Neo4j on AWS.\n\n"
            f"**Session:** `{thread_id}`  ·  **Domain:** `{DOMAIN}`\n\n"
            "---\n\n**Try one of these questions:**"
        ),
        actions=actions,
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = get_thread_id()
    if is_interrupted():
        await _resume(thread_id, message.content)
    else:
        await _chat(thread_id, message.content)


@cl.action_callback("starter")
async def on_starter(action: cl.Action):
    await action.remove()
    await _chat(get_thread_id(), action.payload["value"])


@cl.action_callback("hitl_option")
async def on_hitl_option(action: cl.Action):
    await action.remove()
    selected = action.payload["value"]
    await cl.Message(
        content=f"✅ Selected: **{selected}**",
        author="You",
    ).send()
    await _resume(get_thread_id(), selected)


# ── SSE stream processing ─────────────────────────────────────────────────

async def _process_sse_stream(url: str, payload: dict):
    msg     = cl.Message(content="", author="Clinical Trial Assistant")
    answer  = ""
    latency = 0
    started = False

    async with cl.Step(name="🔍 Searching knowledge base", show_input=False) as step:
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                async with client.stream(
                    "POST", url, headers=HEADERS, json=payload
                ) as resp:
                    resp.raise_for_status()

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        raw = line[6:].strip()
                        if not raw or raw == "[DONE]":
                            continue

                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        etype = event.get("type", "")

                        # ── HITL interrupt (typed) ────────────────────────
                        if etype == "interrupt" or event.get("interrupted"):
                            step.output = "Awaiting clarification"
                            set_interrupted(True)

                            # Payload comes either as top-level keys (AgentCore
                            # passthrough) or nested under interrupt_payload
                            ip       = event.get("interrupt_payload") or event
                            question = ip.get("question", "Please clarify:")
                            options  = ip.get("options") or []
                            allow_ft = ip.get("allow_freetext", True)

                            options_md = "\n".join(
                                f"> **{i+1}.** {opt}"
                                for i, opt in enumerate(options)
                            )
                            actions = [
                                cl.Action(
                                    name="hitl_option",
                                    label=f"  {i+1}. {opt}  ",
                                    description=opt,
                                    payload={"value": opt},
                                )
                                for i, opt in enumerate(options)
                            ]
                            hint = "\n\n_Or type a custom answer below._" if allow_ft else ""
                            await cl.Message(
                                content=(
                                    f"### 🔍 Clarification needed\n\n"
                                    f"**{question}**\n\n"
                                    f"{options_md}{hint}\n\n"
                                    f"**Click an option below or type your own:**"
                                ),
                                actions=actions,
                            ).send()
                            return

                        # ── Error (typed) ─────────────────────────────────
                        if etype == "error":
                            await cl.Message(
                                content=f"❌ {event.get('message', 'Unknown error')}"
                            ).send()
                            return

                        # ── Done / final answer (typed OR single-blob) ────
                        if etype == "done":
                            latency = event.get("latency_ms", 0)
                            step.output = f"Latency: {latency:.0f}ms"
                            continue

                        # ── Single-blob response: {"answer": "...", "interrupted": false}
                        if "answer" in event and not event.get("interrupted"):
                            token = event["answer"]
                            latency = event.get("latency_ms", 0)
                            step.output = f"Latency: {latency:.0f}ms"
                            if token:
                                if not started:
                                    await msg.send()
                                    started = True
                                answer += token
                                await msg.stream_token(token)
                            continue

                        # ── Tool indicators (typed) ───────────────────────
                        if etype == "tool_start":
                            step.name = f"🔍 Using {event.get('name', 'tool')}..."
                            continue

                        if etype == "tool_end":
                            step.name = "🔍 Processing results..."
                            continue

                        # ── Token (typed OR AgentCore native {"result": "..."}) ──
                        # This MUST come last — it's the catch-all for actual content
                        token = (
                            event.get("content")      # typed:   {"type": "token", "content": "..."}
                            or event.get("result")    # AgentCore native: {"result": "..."}
                            or event.get("token")     # alternative key
                            or ""
                        )
                        if token:
                            if not started:
                                await msg.send()
                                started = True
                            answer += token
                            await msg.stream_token(token)

        except httpx.ConnectError:
            await cl.Message(
                content=(
                    "❌ **Cannot connect to the agent API.**\n\n"
                    f"Make sure the Platform is running at `{API_URL}`"
                )
            ).send()
            return
        except Exception as exc:
            log.exception("SSE stream error")
            await cl.Message(content=f"❌ Stream error: {exc}").send()
            return

    # ── Finalise — ALWAYS reached unless early return above ──────────────
    if started and answer:
        set_interrupted(False)
        cleaned = _clean(answer)
        footer  = (
            f"\n\n---\n"
            f"*⏱ {latency/1000:.1f}s · "
            f"This information is for research purposes only "
            f"and does not constitute medical advice.*"
        )
        msg.content = cleaned + footer
        await msg.update()
    elif not started:
        # Stream completed but nothing was rendered — surface it instead of silent hang
        await cl.Message(
            content="⚠️ No response received. Check CloudWatch logs for the agent error."
        ).send()


# ── API calls ─────────────────────────────────────────────────────────────

async def _chat(thread_id: str, text: str):
    await _process_sse_stream(
        url=f"{API_URL}/api/v1/{AGENT}/chat",
        payload={"message": text, "thread_id": thread_id, "domain": DOMAIN},
    )


async def _resume(thread_id: str, user_answer: str):
    await _process_sse_stream(
        url=f"{API_URL}/api/v1/{AGENT}/resume",
        payload={"thread_id": thread_id, "user_answer": user_answer, "domain": DOMAIN},
    )