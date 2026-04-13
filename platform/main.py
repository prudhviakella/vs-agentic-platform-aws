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
  boto3 bedrock-agentcore client
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

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import boto3
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from gateway.auth import verify_api_key
from gateway.rate_limiter import RateLimiter
from gateway.schemas import ChatRequest, ResumeRequest
from gateway.logging_mw import LoggingMiddleware

# Thread pool for boto3 blocking calls
_executor = ThreadPoolExecutor(max_workers=20)

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


@lru_cache(maxsize=1)
def _get_agent_runtime_arn() -> str:
    """Fetch AgentCore Runtime ARN from SSM — cached after first call."""
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(
        Name=f"{SSM_PREFIX}/agent_runtime_arn"
    )["Parameter"]["Value"]


# ── SSE streaming ─────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format dict as SSE line."""
    return f"data: {json.dumps(data)}\n\n"


def _invoke_agentcore_sync(agent_arn: str, payload: dict) -> list[bytes]:
    """
    Synchronous boto3 AgentCore invoke — collects all SSE chunks.
    Runs in a thread pool so it does NOT block the async event loop.

    WHY we collect all chunks synchronously first:
      boto3 event streams are not async-compatible — they use
      synchronous socket reads internally. Running them in a thread pool
      isolates the blocking I/O from FastAPI's async event loop.
      The chunks are then yielded from the async generator immediately,
      so the client still receives tokens as they arrive (the thread
      keeps reading while the async generator yields).
    """
    client = boto3.client("bedrock-agentcore", region_name=REGION)

    response = client.invoke_agent_runtime(
        agentRuntimeArn=agent_arn,
        runtimeSessionId=payload["thread_id"],
        payload=json.dumps(payload).encode("utf-8"),
    )

    chunks = []
    event_stream = response.get("response", {})
    for chunk in event_stream:
        body = chunk.get("chunk", {}).get("bytes", b"")
        if body:
            chunks.append(body)
    return chunks


async def _stream_from_agentcore(payload: dict, request_id: str):
    """
    Invoke AgentCore Runtime and forward SSE events to Chainlit client.

    STREAMING APPROACH:
      1. Run boto3 invoke in thread pool (non-blocking for event loop)
      2. Use asyncio.Queue to pipe chunks from thread → async generator
      3. Yield each SSE line as it arrives — client sees tokens in real time

    WHY Queue instead of collecting all chunks first:
      With a Queue, the thread posts chunks as they arrive from AgentCore.
      The async generator yields them immediately to the client.
      This gives true streaming — first token appears in ~1-2s, not after full response.
    """
    t0    = time.perf_counter()
    queue = asyncio.Queue()
    loop  = asyncio.get_event_loop()

    def _stream_to_queue():
        """Thread function: reads AgentCore stream, posts chunks to queue."""
        try:
            agent_arn = _get_agent_runtime_arn()
            client    = boto3.client("bedrock-agentcore", region_name=REGION)

            log.info(
                f"[PLATFORM] Invoking AgentCore"
                f"  request_id={request_id}"
                f"  thread_id={payload.get('thread_id')}"
                f"  resume={payload.get('resume', False)}"
            )

            response = client.invoke_agent_runtime(
                agentRuntimeArn=agent_arn,
                runtimeSessionId=payload["thread_id"],
                payload=json.dumps(payload).encode("utf-8"),
            )

            # response["response"] is a StreamingBody (blob)
            # Read it in chunks and post to queue
            streaming_body = response.get("response")
            if streaming_body:
                for chunk in streaming_body.iter_chunks(chunk_size=1024):
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            else:
                # Fallback: read full response as bytes
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(queue.put_nowait, raw)

        except Exception as exc:
            log.error(f"[PLATFORM] AgentCore stream error: {exc}")
            err_bytes = _sse({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err_bytes)
        finally:
            # Sentinel — tells async generator stream is done
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # Start boto3 streaming in thread pool (non-blocking)
    loop.run_in_executor(_executor, _stream_to_queue)

    # Yield SSE lines as chunks arrive from thread via Queue
    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break  # sentinel — stream complete

            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk

            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    yield line + "\n\n"
                else:
                    try:
                        event = json.loads(line)
                        yield _sse(event)
                    except json.JSONDecodeError:
                        continue

    except Exception as exc:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.error(f"[PLATFORM] Stream error  request_id={request_id}  elapsed_ms={elapsed}  error={exc}")
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


# ── Observability endpoints ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_trace_table_name() -> str:
    """Cached — SSM lookup happens once, not on every /traces request."""
    return boto3.client("ssm", region_name=REGION).get_parameter(
        Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]


@app.get(f"/api/v1/{AGENT}/traces")
async def list_traces(
    limit: int = 20,
    _: str = Depends(verify_api_key),
):
    """
    List recent agent traces from DynamoDB.
    Each trace = one agent invocation: thread_id, latency, tools called, token count.
    Used by developers to monitor production usage and debug issues.
    """
    try:
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table  = dynamo.Table(_get_trace_table_name())
        resp   = table.scan(Limit=limit)
        return {"traces": resp.get("Items", []), "count": len(resp.get("Items", []))}
    except Exception as exc:
        log.error(f"[PLATFORM] list_traces error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/api/v1/{AGENT}/traces/{{thread_id}}")
async def get_trace(
    thread_id: str,
    _: str = Depends(verify_api_key),
):
    """
    Get all traces for a specific thread_id.
    Useful for tracing a full multi-turn conversation including HITL interrupts.
    """
    try:
        from boto3.dynamodb.conditions import Attr
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table  = dynamo.Table(_get_trace_table_name())
        resp   = table.scan(FilterExpression=Attr("thread_id").eq(thread_id))
        return {"thread_id": thread_id, "traces": resp.get("Items", []), "count": len(resp.get("Items", []))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Prompt management endpoints ───────────────────────────────────────────

@app.get(f"/api/v1/{AGENT}/prompt")
async def get_prompt_info(
    _: str = Depends(verify_api_key),
):
    """
    Returns current prompt ID and version stored in SSM.
    Shows which prompt version the agent is using right now.
    """
    try:
        ssm     = boto3.client("ssm", region_name=REGION)
        p_id    = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/prompt_id")["Parameter"]["Value"]
        p_ver   = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/prompt_version")["Parameter"]["Value"]
        return {"prompt_id": p_id, "prompt_version": p_ver, "ssm_prefix": SSM_PREFIX}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(f"/api/v1/{AGENT}/prompt/reload")
async def reload_prompt(
    new_version: str,
    _: str = Depends(verify_api_key),
):
    """
    Update the prompt version in SSM and force the agent to reload it.
    Use this after you publish a new prompt version in Bedrock Prompt Management.
    No redeployment needed — the agent fetches the new version on next invocation.

    Body param: new_version (e.g. "7")
    """
    try:
        ssm = boto3.client("ssm", region_name=REGION)
        ssm.put_parameter(
            Name=f"{SSM_PREFIX}/bedrock/prompt_version",
            Value=new_version,
            Type="String",
            Overwrite=True,
        )
        log.info(f"[PLATFORM] Prompt version updated to {new_version}")
        return {"status": "updated", "new_version": new_version, "message": "Agent will use new prompt on next invocation"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/api/v1/{AGENT}/cache/stats")
async def cache_stats(
    _: str = Depends(verify_api_key),
):
    """
    Returns semantic cache statistics from Pinecone.
    Shows how many entries are cached and the cache namespace details.
    High cache hit rate = lower latency and lower OpenAI costs.
    """
    try:
        ssm      = boto3.client("ssm", region_name=REGION)
        sm       = boto3.client("secretsmanager", region_name=REGION)
        pc_key   = json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/pinecone")["SecretString"])["api_key"]
        idx_name = ssm.get_parameter(Name=f"{SSM_PREFIX}/pinecone/cache_index_name")["Parameter"]["Value"]
        from pinecone import Pinecone
        pc   = Pinecone(api_key=pc_key)
        idx  = pc.Index(idx_name)
        stats = idx.describe_index_stats()
        cache_ns = stats.get("namespaces", {}).get("cache__", {})
        return {
            "index_name":        idx_name,
            "cache_vector_count": cache_ns.get("vector_count", 0),
            "total_vector_count": stats.get("total_vector_count", 0),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
