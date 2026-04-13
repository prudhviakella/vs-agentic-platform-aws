"""
tracer.py — TracerMiddleware
=============================
Logs every agent invocation to DynamoDB.

Each row in the trace table represents one full request lifecycle:
  - thread_id, domain, latency_ms
  - Whether it was a cache hit or full LLM call
  - Error (if any)

The /traces and /traces/{thread_id} API endpoints read from this table.

DynamoDB table name is fetched from SSM at startup (cached with lru_cache).
"""

import logging
import time
import uuid

import boto3

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)


class TracerMiddleware(BaseAgentMiddleware):
    """
    First middleware in (ingress), last out (egress).
    Records start time on ingress, writes DynamoDB row on egress.
    """

    def __init__(self, dynamodb_table_name: str, region: str = "us-east-1"):
        self._table_name = dynamodb_table_name
        self._region     = region

    async def before_agent(self, state: dict) -> None:
        # Generate a unique trace ID and record start time
        # These are stored on state so after_agent can read them
        state["trace_id"]    = str(uuid.uuid4())
        state["trace_start"] = time.perf_counter()

        log.info(
            f"[TRACER] start"
            f"  trace={state['trace_id'][:8]}"
            f"  thread={state['config'].get('configurable', {}).get('thread_id', '?')}"
        )

    async def after_agent(self, state: dict) -> None:
        elapsed_ms = round(
            (time.perf_counter() - state.get("trace_start", time.perf_counter())) * 1000, 2
        )

        try:
            dynamo = boto3.resource("dynamodb", region_name=self._region)
            table  = dynamo.Table(self._table_name)
            table.put_item(Item={
                "trace_id":   state.get("trace_id", str(uuid.uuid4())),
                "thread_id":  state["config"].get("configurable", {}).get("thread_id", ""),
                "domain":     state["config"].get("configurable", {}).get("domain", "pharma"),
                "latency_ms": str(elapsed_ms),
                "cache_hit":  str(state.get("cached") is not None),
                "blocked":    str(state.get("blocked", False)),
                "error":      state.get("error", ""),
                "timestamp":  str(int(time.time())),
            })
            log.info(f"[TRACER] wrote trace  latency_ms={elapsed_ms}")
        except Exception as exc:
            # Never fail the request because of a trace write error
            log.warning(f"[TRACER] DynamoDB write failed: {exc}")
