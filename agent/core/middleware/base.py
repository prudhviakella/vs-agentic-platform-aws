"""
base.py — BaseAgentMiddleware
==============================

Shared base class for all middleware in this package.

WHY THIS FILE EXISTS:
  _get_run_id() was originally duplicated across TracerMiddleware,
  SemanticCacheMiddleware, and ActionGuardrailMiddleware. Each copy was
  slightly different, creating subtle bugs where different middleware layers
  used different run IDs for the same request — breaking the before_agent →
  after_agent bridge that TracerMiddleware relies on to compute latency.

  Extracting it here gives every middleware a single, consistent implementation.
  Fix once → fixed everywhere.

WHY MIDDLEWARE NEEDS A RUN ID AT ALL:
  LangChain middleware has two hooks:
    before_agent(state, runtime) — called before LLM runs
    after_agent(state, runtime)  — called after LLM produces output

  These are two SEPARATE method calls. Any state you need to pass from
  before_agent to after_agent must be stored somewhere keyed by a stable ID.

  TracerMiddleware stores the wall-clock start time in self._t0[run_id]
  in before_agent, then reads self._t0.pop(run_id) in after_agent to compute
  elapsed time. If the two calls produce different run IDs, _t0.pop() finds
  nothing and latency tracking silently breaks.

  The run_id must be:
    1. Stable: same value in both before_agent and after_agent for the same request
    2. Unique: different values for different concurrent requests
    3. Deterministic: derived from request context, not generated randomly on each call

WHY NOT USE A RANDOM UUID ON EVERY CALL:
  BUG (original code): return uuid.uuid4().hex[:8]
    → before_agent generates UUID "a1b2c3d4"
    → after_agent generates UUID "e5f6g7h8" (different!)
    → _t0.pop("e5f6g7h8") finds nothing
    → elapsed = time.time() - 0.0 = unix timestamp in ms (obviously wrong)
    → TracerMiddleware.after_agent silently produces a broken trace

  FIX: generate the fallback UUID ONCE in __init__ and reuse it for both calls.
  The same instance handles both hooks for the same request, so self._fallback_run_id
  is always the same value within one middleware instance's lifecycle.

HOW _get_run_id PRIORITY WORKS IN PRACTICE:
  In production (AgentCore + app.py):
    agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": "pharma"}
    agent.astream_events(..., context=agent_context)
    → runtime.context["session_id"] = thread_id ← always set
    → Priority 1 fires every time, UUID fallback never reached

  In local testing (test scripts, notebooks):
    config = {"configurable": {"thread_id": t, ...}}
    → context= kwarg may not be set
    → runtime.context is None or empty
    → Priority 3 fires → fallback UUID is used
    → before/after still bridge correctly because same instance, same UUID

  The WARNING log tells you when priority 3 fires — it should never appear
  in production CloudWatch logs. If it does, context= kwarg is missing from
  the astream_events() call in app.py.
"""

import uuid
import logging

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

log = logging.getLogger(__name__)


class BaseAgentMiddleware(AgentMiddleware):
    """
    Base class for all agent middleware in this project.

    Provides _get_run_id() — a stable request identifier that bridges
    before_agent and after_agent calls for the same request. All middleware
    that needs to carry state across the two hooks should inherit from this
    class and use _get_run_id() as the key.

    Usage:
        class MyMiddleware(BaseAgentMiddleware):
            def before_agent(self, state, runtime):
                run_id = self._get_run_id(runtime)
                self._start_times[run_id] = time.time()

            def after_agent(self, state, runtime):
                run_id = self._get_run_id(runtime)
                elapsed = time.time() - self._start_times.pop(run_id, 0.0)
    """

    def __init__(self):
        super().__init__()

        # Stable fallback run ID for the rare case where runtime.context
        # is missing both session_id and user_id (typically in local tests).
        #
        # WHY generate in __init__ instead of on each _get_run_id call:
        #   before_agent and after_agent are separate method calls.
        #   If we called uuid.uuid4() inside _get_run_id, each call would
        #   produce a different UUID — the before/after bridge would break.
        #   Generating once in __init__ guarantees both calls return the
        #   same value for the same middleware instance.
        #
        # hex[:8] gives 8 hex chars (32 bits of randomness) — enough to be
        # unique within a single container's lifetime without being verbose
        # in log messages.
        self._fallback_run_id: str = uuid.uuid4().hex[:8]

    def _get_run_id(self, runtime: Runtime) -> str:
        """
        Extract a stable identifier to bridge before_agent → after_agent.

        In LangChain 1.0, Runtime exposes only: context, store, stream_writer,
        merge, override, previous. There is NO run_id or thread_id attribute —
        the only way to get a per-request identifier is from runtime.context,
        which is populated by the context= kwarg passed to astream_events().

        In app.py, context is set as:
          agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": "pharma"}
          agent.astream_events(input_data, config=config, context=agent_context)

        Priority:
          1. runtime.context["session_id"]
             Best option — set to thread_id (the AgentCore runtimeSessionId).
             Unique per user session and stable for the session's lifetime.
             Always set in production via app.py.

          2. runtime.context["user_id"]
             Second best — set to thread_id in production (same value as session_id).
             Used as fallback if session_id is somehow missing.

          3. self._fallback_run_id (UUID generated in __init__)
             Last resort — used in local testing where context= kwarg is not set.
             Stable within one instance (same value for before AND after).
             Should NEVER fire in production — a WARNING is logged if it does.

        Args:
            runtime: The LangChain Runtime object injected into each middleware hook.

        Returns:
            A string identifier that is stable across before_agent and after_agent
            for the same request.
        """
        # getattr guard: runtime.context may not exist in some test environments
        # where a mock Runtime object is used without all attributes set.
        # The `or {}` converts None to an empty dict so .get() calls work safely.
        ctx = getattr(runtime, "context", None) or {}

        # Priority 1: session_id — always set in production via app.py
        session_id = ctx.get("session_id")
        if session_id:
            return str(session_id)

        # Priority 2: user_id — fallback if session_id is missing
        user_id = ctx.get("user_id")
        if user_id:
            return str(user_id)

        # Priority 3: stable fallback UUID — local testing only
        # This WARNING should NEVER appear in production CloudWatch logs.
        # If it does: check that context= kwarg is set in app.py's astream_events() call.
        log.warning(
            "[BASE_MW] session_id and user_id both missing from runtime.context — "
            "falling back to instance UUID. before_agent/after_agent bridges will "
            "work (same instance, same UUID) but traces will not be user-scoped. "
            "Ensure context= is passed to agent.astream_events() with session_id set."
        )
        return self._fallback_run_id