"""
main.py — VS AgentCore Platform (FastAPI)
==========================================
SSE proxy between Chainlit UI and AgentCore Runtime.

SSE FLOW:
  Chainlit UI
    POST /api/v1/clinical-trial/chat  (with X-API-Key)
    ← StreamingResponse (text/event-stream)
      data: {"type": "token", "content": "The NCI-MATCH..."}\n\n
      data: {"type": "tool_start", "name": "search_tool"}\n\n
      data: {"type": "interrupt", "question": "...", "options": [...]}\n\n
      data: {"type": "done", "latency_ms": 12345}\n\n

  For HITL resume:
    POST /api/v1/clinical-trial/resume  (with X-API-Key)
    ← same SSE stream from AgentCore

AGENTCORE INVOKE:
  boto3 bedrock-agentcore-runtime client
    invoke_agent_runtime(
      agentRuntimeArn=...,
      runtimeSessionId=thread_id,    ← maintains session context
      payload=json.dumps(payload),
    )
  Returns streaming response chunks → forwarded as SSE

ALB CONFIG:
  idle_timeout = 300s   ← needed for long SSE connections
  X-Accel-Buffering: no ← disables nginx buffering
"""

import json
import logging
import os
import time
import uuid

import boto3
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gateway.auth import verify_api_key
from gateway.rate_limiter import RateLimiter
from gateway.schemas import ChatRequest, ResumeRequest
from gateway.logging_mw import LoggingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="VS AgentCore Platform",
    description="Clinical Trial Research Assistant — Production Gateway",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict to ALB DNS in production
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# ── Config ────────────────────────────────────────────────────────────────
REGION       = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX   = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")
AGENT        = "clinical-trial"
rate_limiter = RateLimiter(requests_per_minute=30)


def _get_agent_runtime_arn() -> str:
    """Fetch AgentCore Runtime ARN from SSM."""
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(
        Name=f"{SSM_PREFIX}/agent_runtime_arn"
    )["Parameter"]["Value"]


# ── SSE streaming ─────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format dict as SSE line."""
    return f"data: {json.dumps(data)}\n\n"


async def _stream_from_agentcore(payload: dict, request_id: str):
    """
    Invoke AgentCore Runtime and forward SSE events to client.

    AgentCore streams SSE bytes → we decode and forward each event.
    The agent's async generator yields dicts → AgentCore converts to SSE.
    """
    t0 = time.perf_counter()
    try:
        client    = boto3.client("bedrock-agentcore-runtime", region_name=REGION)
        agent_arn = _get_agent_runtime_arn()

        log.info(
            f"[PLATFORM] Invoking AgentCore"
            f"  request_id={request_id}"
            f"  thread_id={payload.get('thread_id')}"
            f"  resume={payload.get('resume', False)}"
        )

        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=payload["thread_id"],  # maps to AgentCore session
            payload=json.dumps(payload).encode("utf-8"),
        )

        # Stream chunks from AgentCore SSE response
        event_stream = response.get("response", {})
        for chunk in event_stream:
            body = chunk.get("chunk", {}).get("bytes", b"")
            if not body:
                continue

            text = body.decode("utf-8")

            # AgentCore emits raw SSE lines: "data: {...}\n\n"
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    # Forward raw SSE line
                    yield line + "\n\n"
                else:
                    # Raw JSON (some AgentCore versions omit "data: " prefix)
                    try:
                        event = json.loads(line)
                        yield _sse(event)
                    except json.JSONDecodeError:
                        continue

    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.error(
            f"[PLATFORM] AgentCore error"
            f"  request_id={request_id}"
            f"  elapsed_ms={elapsed}"
            f"  error={exc}"
        )
        yield _sse({"type": "error", "message": str(exc)})


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "vs-agentcore-platform", "agent": AGENT}


@app.post(f"/api/v1/{AGENT}/chat")
async def chat(
    body: ChatRequest,
    request: Request,
    _: str = Depends(verify_api_key),
):
    """Send a new message — returns SSE stream."""
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")

    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")

    payload = {
        "message":   body.message,
        "thread_id": body.thread_id,
        "domain":    body.domain,
        "resume":    False,
    }

    return StreamingResponse(
        _stream_from_agentcore(payload, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",       # nginx: disable response buffering
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )


@app.post(f"/api/v1/{AGENT}/resume")
async def resume(
    body: ResumeRequest,
    request: Request,
    _: str = Depends(verify_api_key),
):
    """Resume after HITL — returns SSE stream."""
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")

    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    payload = {
        "message":     "",
        "thread_id":   body.thread_id,
        "domain":      body.domain,
        "resume":      True,
        "user_answer": body.user_answer,
    }

    return StreamingResponse(
        _stream_from_agentcore(payload, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )
