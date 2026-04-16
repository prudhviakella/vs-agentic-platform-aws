"""
tracer.py — TracerMiddleware
==============================
Cross-cutting observability — FIRST in stack so even rejected requests are logged.

COMPLETE TRACE SCHEMA:
─────────────────────────────────────────────────────────────────────────────────
  Identity:
    run_id, user_id, session_id, domain, ts, observability

  Request:
    question          — real user question (skips summarization injections)
    answer            — final LLM response text
    answer_length     — len(answer) chars — proxy for response completeness
    is_resume         — True if this was a HITL /resume call
    prompt_version    — Bedrock prompt version that served this request

  Performance:
    elapsed_ms        — total request time (before_agent → after_agent)
    llm_timings       — [{turn, elapsed_ms, input_tokens, output_tokens}] per LLM call
    cache_hit         — True if semantic cache short-circuited the agent
    cache_namespace   — Pinecone namespace that served the cache hit

  Tokens + Cost:
    input_tokens      — total input tokens across all LLM calls
    output_tokens     — total output tokens across all LLM calls
    total_tokens      — input + output
    token_cost_usd    — approximate cost at GPT-4o pricing

  Tools:
    tools             — ordered list of tool names called
    tool_count        — total number of tool calls
    tool_details      — [{name, args_summary, result_summary, is_error}] (capped at 5)
    llm_turns         — number of LLM invocations

  HITL:
    hitl_fired        — True if clarification was requested
    hitl_question     — the clarifying question asked
    hitl_options      — list of options presented to user
    hitl_user_answer  — what the user actually selected (from resume call)

  Guardrails:
    guardrail_passed       — True if output passed all guardrail checks
    guardrail_blocked      — True if output was blocked/replaced
    faithfulness_score     — 0.0–1.0 from OutputGuardrailMiddleware layer 2
    consistency_score      — 0.0–1.0 from OutputGuardrailMiddleware layer 3

  Memory:
    episodic_hits     — number of past memories injected (0 if none found)
    episodic_stored   — True if this Q&A was stored to episodic memory

  Errors:
    errors            — list of tool error messages
    has_errors        — True if any tool failed

INTER-MIDDLEWARE COMMUNICATION:
  Other middleware annotate traces by calling the class-level method:
    TracerMiddleware.update_trace(run_id, {"guardrail_passed": True, ...})

  This avoids state schema changes and keeps each middleware independent.
  Each middleware imports TracerMiddleware and calls update_trace in after_agent
  or after_model with the data it produces.

  OutputGuardrailMiddleware calls:
    TracerMiddleware.update_trace(run_id, {
        "guardrail_passed":    True,
        "guardrail_blocked":   False,
        "faithfulness_score":  0.92,
        "consistency_score":   1.00,
    })

  EpisodicMemoryMiddleware calls:
    TracerMiddleware.update_trace(run_id, {
        "episodic_hits":   3,
        "episodic_stored": True,
    })

  SemanticCacheMiddleware calls:
    TracerMiddleware.update_trace(run_id, {
        "cache_hit":       True,
        "cache_namespace": "cache_pharma",
    })
"""

import logging
import os
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from langchain.agents.middleware import AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

MAX_TRACES_CACHE      = 1_000
MAX_TOOL_RESULTS      = 5
MAX_TOOL_ARG_CHARS    = 200
MAX_TOOL_RESULT_CHARS = 200

_SUMMARY_PREFIX = "Here is a summary of the conversation to date"
_HITL_PREFIX    = "[HITL Answer]:"

# GPT-4o pricing (per 1M tokens) — update if pricing changes
_GPT4O_INPUT_COST_PER_1M  = 2.50
_GPT4O_OUTPUT_COST_PER_1M = 10.00

# ── Class-level annotation registry ──────────────────────────────────────────
# Used by other middleware to annotate traces without coupling to instance state.
# Thread-safe: dict operations on CPython are GIL-protected for simple gets/sets.
# Key: run_id, Value: dict of extra fields to merge into the trace.
_pending_annotations: dict[str, dict] = {}
_annotations_lock = threading.Lock()


class TracerMiddleware(BaseAgentMiddleware):
    """
    Rich observability middleware.

    Hooks:
      before_agent  — request start, stash context
      after_agent   — extract trace, merge annotations, persist
      before_model  — per-LLM-call timing start
      after_model   — per-LLM-call timing end + token extraction
    """

    # ── Class-level annotation API ────────────────────────────────────────────
    # Called by other middleware (OutputGuardrail, EpisodicMemory, SemanticCache)
    # to annotate the current trace with their data.

    @classmethod
    def update_trace(cls, run_id: str, data: dict) -> None:
        """
        Annotate a trace with additional fields from another middleware.

        Thread-safe. Can be called from any thread (including background
        daemon threads used by EpisodicMemoryMiddleware._store_sync).

        Args:
            run_id: The run identifier — same value returned by _get_run_id().
            data:   Dict of fields to merge into the trace. Later calls for
                    the same run_id merge (not replace) existing annotations.

        Example (in OutputGuardrailMiddleware.after_agent):
            from core.middleware.tracer import TracerMiddleware
            TracerMiddleware.update_trace(run_id, {
                "guardrail_passed":   True,
                "faithfulness_score": 0.92,
                "consistency_score":  1.00,
            })
        """
        with _annotations_lock:
            if run_id not in _pending_annotations:
                _pending_annotations[run_id] = {}
            _pending_annotations[run_id].update(data)

    @classmethod
    def _pop_annotations(cls, run_id: str) -> dict:
        """Read and remove annotations for a run_id. Called by after_agent."""
        with _annotations_lock:
            return _pending_annotations.pop(run_id, {})

    def __init__(
        self,
        dynamodb_table_name: Optional[str] = None,
        ttl_days:            int            = 30,
        aws_region:          str            = "us-east-1",
    ):
        super().__init__()

        self._t0:          dict[str, float] = {}
        self._ctx:         dict[str, dict]  = {}
        self._model_t0:    dict[str, list]  = {}
        self._llm_timings: dict[str, list]  = {}

        self._traces: OrderedDict[str, dict] = OrderedDict()

        self._table_name = dynamodb_table_name
        self._ttl_days   = ttl_days
        self._aws_region = aws_region

        self._ddb_table       = None
        self._ddb_table_ready = False
        self._ddb_table_lock  = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tracer-ddb")

        # Read prompt version once at cold start — doesn't change during container lifetime
        self._prompt_version = self._read_prompt_version()

        if not dynamodb_table_name:
            log.warning("[TRACER] No DynamoDB table configured — traces stored in-memory only.")

    @staticmethod
    def _read_prompt_version() -> str:
        """
        Read current Bedrock prompt version from SSM at cold start.

        WHY read here and not in _extract_from_state:
          SSM is a network call. Reading once in __init__ (cold start) costs
          ~50ms once. Reading on every trace would add 50ms to every request.
          The prompt version doesn't change during a container's lifetime —
          it's fixed at cold start when build_system_prompt() fetches the template.
        """
        try:
            import boto3
            region = os.environ.get("AWS_REGION", "us-east-1")
            ssm    = boto3.client("ssm", region_name=region)
            return ssm.get_parameter(
                Name="/clinical-trial-agent/prod/bedrock/prompt_version"
            )["Parameter"]["Value"]
        except Exception:
            return "unknown"

    # ─────────────────────────────────────────────────────── lifecycle hooks ──

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id = self._get_run_id(runtime)
        self._t0[run_id]          = time.time()
        self._llm_timings[run_id] = []

        ctx = getattr(runtime, "context", None) or {}
        self._ctx[run_id] = {
            "user_id":    ctx.get("user_id",    "anonymous"),
            "session_id": ctx.get("session_id", "unknown"),
            "domain":     ctx.get("domain",     "general"),
        }
        log.info(f"[TRACER] T=0ms  request_received  run_id={run_id}")
        return None

    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Stash LLM call start time for per-turn timing."""
        run_id = self._get_run_id(runtime)
        if run_id not in self._model_t0:
            self._model_t0[run_id] = []
        self._model_t0[run_id].append(time.time())
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Record LLM call duration and token usage.

        usage_metadata is populated by LangChain from the model's response.
        OpenAI always returns token counts. If missing, defaults to 0.
        """
        run_id = self._get_run_id(runtime)
        starts = self._model_t0.get(run_id, [])
        if not starts:
            return None

        start      = starts.pop()
        elapsed_ms = round((time.time() - start) * 1_000, 2)
        turn_idx   = len(self._llm_timings.get(run_id, []))

        input_tokens = output_tokens = 0
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage):
                meta = getattr(msg, "usage_metadata", None) or {}
                input_tokens  = meta.get("input_tokens",  0)
                output_tokens = meta.get("output_tokens", 0)
                break

        if run_id not in self._llm_timings:
            self._llm_timings[run_id] = []

        self._llm_timings[run_id].append({
            "turn":          turn_idx + 1,
            "elapsed_ms":    elapsed_ms,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        })
        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Extract full trace, merge all annotations, persist."""
        run_id  = self._get_run_id(runtime)
        elapsed = (time.time() - self._t0.pop(run_id, 0.0)) * 1_000

        ctx         = self._ctx.pop(run_id, {})
        llm_timings = self._llm_timings.pop(run_id, [])
        self._model_t0.pop(run_id, None)

        # Extract base trace from state
        trace = self._extract_from_state(run_id, elapsed, state.get("messages", []))

        # ── Merge annotations from other middleware ───────────────────────
        # OutputGuardrailMiddleware, EpisodicMemoryMiddleware, SemanticCacheMiddleware
        # all call TracerMiddleware.update_trace() with their data.
        # We collect and merge those annotations here.
        annotations = self._pop_annotations(run_id)
        trace.update(annotations)

        # ── Ensure guardrail fields have safe defaults ────────────────────
        # If OutputGuardrailMiddleware didn't run (e.g. cache hit), these
        # fields would be missing from the trace. Set safe defaults so
        # DynamoDB always has a consistent schema.
        trace.setdefault("guardrail_passed",    None)   # None = not checked (cache hit)
        trace.setdefault("guardrail_blocked",   False)
        trace.setdefault("faithfulness_score",  None)
        trace.setdefault("consistency_score",   None)
        trace.setdefault("episodic_hits",       0)
        trace.setdefault("episodic_stored",     False)
        trace.setdefault("cache_namespace",     None)

        # ── LLM timing and token aggregation ─────────────────────────────
        trace["llm_timings"] = llm_timings

        total_input  = sum(t.get("input_tokens",  0) for t in llm_timings)
        total_output = sum(t.get("output_tokens", 0) for t in llm_timings)
        trace["input_tokens"]   = total_input
        trace["output_tokens"]  = total_output
        trace["total_tokens"]   = total_input + total_output
        trace["token_cost_usd"] = round(
            (total_input  / 1_000_000) * _GPT4O_INPUT_COST_PER_1M +
            (total_output / 1_000_000) * _GPT4O_OUTPUT_COST_PER_1M,
            6
        )

        # ── Identity and prompt version ───────────────────────────────────
        trace["user_id"]        = ctx.get("user_id",    "anonymous")
        trace["session_id"]     = ctx.get("session_id", "unknown")
        trace["domain"]         = ctx.get("domain",     "general")
        trace["prompt_version"] = self._prompt_version

        # ── Observability service ─────────────────────────────────────────
        try:
            from langsmith import get_current_run_tree
            run_tree = get_current_run_tree()
            if run_tree:
                trace["observability"] = "langsmith"
                trace["run_tree_url"]  = run_tree.url
        except (ImportError, Exception):
            pass

        if "observability" not in trace:
            try:
                import mlflow
                if mlflow.get_active_run():
                    mlflow.log_metrics({
                        "agent_latency_ms": elapsed,
                        "total_tokens":     trace["total_tokens"],
                        "token_cost_usd":   trace["token_cost_usd"],
                    })
                    trace["observability"] = "mlflow"
            except (ImportError, Exception):
                pass

        if "observability" not in trace:
            trace["observability"] = "local"

        log.info(
            f"[TRACER] T={elapsed:.0f}ms  run_complete"
            f"  run_id={run_id}"
            f"  user={trace['user_id']}"
            f"  prompt_v={trace['prompt_version']}"
            f"  tools={trace.get('tool_count')}"
            f"  tokens={trace.get('total_tokens')}"
            f"  cost=${trace.get('token_cost_usd')}"
            f"  hitl={trace.get('hitl_fired')}"
            f"  guardrail={trace.get('guardrail_passed')}"
            f"  episodic_hits={trace.get('episodic_hits')}"
            f"  cache={trace.get('cache_hit')}"
        )

        self._cache_trace(run_id, trace)
        self._persist_async(trace)
        return None

    # ─────────────────────────────────────────────────────── trace cache ──────

    def _cache_trace(self, run_id: str, trace: dict) -> None:
        self._traces[run_id] = trace
        self._traces.move_to_end(run_id)
        while len(self._traces) > MAX_TRACES_CACHE:
            self._traces.popitem(last=False)

    # ──────────────────────────────────────────────────── DynamoDB ───────────

    def _get_table(self) -> Any | None:
        if not self._table_name:
            return None
        if self._ddb_table_ready:
            return self._ddb_table
        with self._ddb_table_lock:
            if not self._ddb_table_ready:
                from core.aws import init_trace_table
                self._ddb_table = init_trace_table(
                    table_name = self._table_name,
                    ttl_days   = self._ttl_days,
                    region     = self._aws_region,
                )
                self._ddb_table_ready = True
        return self._ddb_table

    def _persist_async(self, trace: dict) -> None:
        if not self._table_name:
            return
        self._executor.submit(self._write_trace, dict(trace))

    def _write_trace(self, trace: dict) -> None:
        try:
            table = self._get_table()
            if table is None:
                return
            from core.aws import put_trace
            put_trace(table=table, trace=trace, ttl_days=self._ttl_days)
        except Exception as exc:
            log.error(f"[TRACER] DynamoDB write failed  run_id={trace.get('run_id')}  error={exc}")

    # ──────────────────────────────────────────────────── state extraction ────

    @staticmethod
    def _extract_from_state(run_id: str, elapsed_ms: float, messages: list) -> dict:
        """
        Walk state["messages"] and produce the base trace dict.

        Annotations from other middleware (guardrail scores, episodic hits,
        cache namespace) are merged in after_agent via update_trace().
        """
        question       = ""
        answer         = ""
        is_resume      = False
        hitl_user_answer = ""
        tools          = []
        tool_details   = []
        llm_turns      = 0
        errors         = []

        hitl_fired    = False
        hitl_question = ""
        hitl_options  = []

        # Build tool result lookup by tool_call_id
        tool_results_by_id: dict[str, dict] = {}
        for msg in messages:
            if getattr(msg, "type", None) == "tool":
                tc_id    = getattr(msg, "tool_call_id", None)
                content  = str(getattr(msg, "content", ""))
                is_error = getattr(msg, "status", "") == "error"
                if tc_id:
                    tool_results_by_id[tc_id] = {
                        "content":  content[:MAX_TOOL_RESULT_CHARS],
                        "is_error": is_error,
                    }
                if is_error:
                    errors.append(f"{getattr(msg, 'name', 'unknown')}: {content[:100]}")

        # Walk backwards to find real user question (skip summarization injections)
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                content = str(msg.content) if msg.content else ""
                if content.startswith(_SUMMARY_PREFIX):
                    continue
                question = content

                # Detect resume calls and extract what the user answered
                if content.startswith(_HITL_PREFIX):
                    is_resume = True
                    # "[HITL Answer]: NCI-MATCH. Now search and answer."
                    # Extract the answer between ": " and ". Now search"
                    try:
                        after_prefix = content[len(_HITL_PREFIX):].strip()
                        hitl_user_answer = after_prefix.split(". Now search")[0].strip()
                    except Exception:
                        hitl_user_answer = content[len(_HITL_PREFIX):].strip()[:200]
                break

        # Walk forwards: collect LLM turns, tool calls, tool details, final answer
        for msg in messages:
            if isinstance(msg, AIMessage):
                llm_turns += 1

                for tc in getattr(msg, "tool_calls", []) or []:
                    tc_name = tc.get("name", "unknown")
                    tc_args = tc.get("args", {})
                    tc_id   = tc.get("id")

                    tools.append(tc_name)

                    # Extract HITL details from ask_user_input call args
                    if "ask_user_input" in tc_name:
                        hitl_fired    = True
                        hitl_question = tc_args.get("question", "")
                        hitl_options  = tc_args.get("options",  [])

                    # Build rich tool detail record
                    if len(tool_details) < MAX_TOOL_RESULTS:
                        result = tool_results_by_id.get(tc_id, {})
                        tool_details.append({
                            "name":           tc_name,
                            "args_summary":   _summarise_args(tc_name, tc_args),
                            "result_summary": result.get("content", ""),
                            "is_error":       result.get("is_error", False),
                        })

                if msg.content:
                    answer = str(msg.content)

        cache_hit = bool(answer) and len(tools) == 0

        return {
            "run_id":           run_id,
            "elapsed_ms":       round(elapsed_ms, 2),
            "ts":               time.time(),

            # Request
            "question":         question,
            "answer":           answer,
            "answer_length":    len(answer),
            "is_resume":        is_resume,
            "hitl_user_answer": hitl_user_answer,

            # Tools
            "tools":            tools,
            "tool_count":       len(tools),
            "tool_details":     tool_details,
            "llm_turns":        llm_turns,

            # HITL
            "hitl_fired":       hitl_fired,
            "hitl_question":    hitl_question,
            "hitl_options":     hitl_options,

            # Cache
            "cache_hit":        cache_hit,

            # Errors
            "errors":           errors,
            "has_errors":       len(errors) > 0,

            # Fields merged from annotations in after_agent:
            # user_id, session_id, domain, prompt_version
            # guardrail_passed, guardrail_blocked, faithfulness_score, consistency_score
            # episodic_hits, episodic_stored
            # cache_namespace
            # input_tokens, output_tokens, total_tokens, token_cost_usd
            # llm_timings
        }

    # ─────────────────────────────────────────────────────── public API ──────

    def get_trace(self, run_id: str) -> Optional[dict]:
        """Return in-memory cached trace. None if evicted by LRU cap."""
        return self._traces.get(run_id)

    def get_trace_from_dynamodb(self, run_id: str) -> Optional[dict]:
        """Fetch trace from DynamoDB. Admin/debug only."""
        if not self._table_name:
            log.warning("[TRACER] DynamoDB not configured.")
            return None
        from core.aws import get_trace_item
        return get_trace_item(
            table_name = self._table_name,
            run_id     = run_id,
            region     = self._aws_region,
        )


# ── Module-level helper ───────────────────────────────────────────────────────

def _summarise_args(tool_name: str, args: dict) -> str:
    """Produce a short storage-friendly summary of tool call args."""
    if not args:
        return ""
    if "search" in tool_name:
        return str(args.get("query", ""))[:MAX_TOOL_ARG_CHARS]
    if "graph" in tool_name:
        cypher = str(args.get("cypher", ""))
        return cypher[:MAX_TOOL_ARG_CHARS] + ("..." if len(cypher) > MAX_TOOL_ARG_CHARS else "")
    if "ask_user_input" in tool_name:
        q = str(args.get("question", ""))[:100]
        n = len(args.get("options", []))
        return f"Q: {q} | options: {n}"
    if "summariser" in tool_name:
        q = str(args.get("query", ""))[:80]
        n = len(args.get("chunks", []))
        return f"query: {q} | chunks: {n}"
    return str(args)[:MAX_TOOL_ARG_CHARS]