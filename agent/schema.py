"""
schema.py — Runtime Context Schema
====================================

This file defines AgentContext — the per-request context object that flows
through every layer of the middleware stack and is accessible in the system prompt.

WHAT IS "CONTEXT" IN LANGCHAIN 1.0?
  LangChain agents have two kinds of per-request data:

    1. STATE (config["configurable"]) — LangGraph's built-in mechanism.
       Stores: thread_id, user_id, session_id, domain.
       Used by: LangGraph checkpointer (thread_id), LangChain internals.
       Accessed via: config["configurable"]["user_id"]

    2. CONTEXT (context= kwarg) — LangChain 1.0's middleware mechanism.
       Stores: the same user_id, session_id, domain fields.
       Used by: middleware hooks (before_agent, after_agent).
       Accessed via: runtime.context["user_id"]

  WHY DO WE NEED BOTH?
    They serve different purposes in LangChain's architecture.
    LangGraph reads thread_id from config["configurable"] for checkpointing.
    Middleware reads user_id from runtime.context for episodic memory namespacing.
    Without both, either LangGraph breaks (no thread_id in configurable) or
    middleware breaks (no user_id in context).

    In app.py we set BOTH:
      config = {"configurable": {"thread_id": t, "user_id": t, "session_id": t}}
      agent_context = {"user_id": t, "session_id": t, "domain": "pharma"}
      agent.astream_events(input_data, config=config, context=agent_context)

WHY TypedDict AND NOT dataclass OR Pydantic BaseModel?
  LangChain 1.0 documentation explicitly states:
    "Custom context schemas must be TypedDict types.
     Pydantic models and dataclasses are no longer supported."

  This is a hard API constraint — if you pass a Pydantic model or dataclass
  as context_schema to create_agent(), LangChain raises a TypeError at build time.

  TypedDict gives us:
    - Static type checking via mypy / Pylance (IDE autocomplete on context fields)
    - Runtime validation via LangChain (it validates the context= kwarg structure)
    - Zero runtime overhead (TypedDict is erased at runtime — it's just a dict)

  Pydantic BaseModel was supported in older LangChain versions but was removed
  because the serialisation overhead added ~2ms per middleware call, which
  compounds across 9 middleware layers.

total=False — WHY ALL KEYS OPTIONAL?
  total=False makes every key in the TypedDict optional (as if every field
  had Optional[] type). This means:
    - Tests that don't set up a full context don't crash
    - Callers that only set user_id don't need to provide session_id and domain
    - Middleware that calls runtime.context.get("domain", "general") gets a
      safe fallback instead of a KeyError

  The trade-off: without total=True, mypy won't catch missing required fields
  at static analysis time. We accept this trade-off because the middleware
  code defensively uses .get() with defaults everywhere.

HOW MIDDLEWARE READS THIS CONTEXT:
  Every middleware class receives a 'runtime' object in before_agent() and
  after_agent(). The context fields are accessed as:

    runtime.context["user_id"]          # raises KeyError if missing
    runtime.context.get("user_id", "")  # safe with default

  Example from EpisodicMemoryMiddleware:
    user_id = runtime.context.get("user_id") or config["configurable"].get("user_id")

  The OR fallback pattern is defensive — it tries runtime.context first (the
  LangChain 1.0 mechanism) and falls back to config["configurable"] (the
  LangGraph mechanism) so the middleware works even if one path is misconfigured.

HOW @dynamic_prompt READS THIS CONTEXT:
  If using Bedrock's @dynamic_prompt decorator (for per-request prompt variation),
  the context is accessible via:
    request.runtime.context["domain"]
  This lets you vary the system prompt based on the domain without rebuilding
  the agent — useful for multi-tenant deployments where different customers
  get different prompt framing.
"""

from typing import TypedDict


class AgentContext(TypedDict, total=False):
    """
    Per-request runtime configuration injected via the context= kwarg.

    All fields are optional (total=False) — middleware uses .get() with
    defaults to handle missing fields gracefully.

    Passed in app.py as:
      agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": "pharma"}
      agent.astream_events(input_data, config=config, context=agent_context)

    Read in middleware as:
      runtime.context.get("user_id", "anonymous")
      runtime.context.get("domain", "general")
    """

    user_id: str
    # WHO is making this request.
    # Used by EpisodicMemoryMiddleware to namespace Pinecone vectors per user:
    #   namespace = f"episodic__{user_id}"
    # Used by TracerMiddleware to tag DynamoDB trace records:
    #   {"user_id": user_id, "timestamp": ..., "tools_called": [...]}
    # In production: set to thread_id (which is the AgentCore runtimeSessionId).
    # In tests: set to a fixed string like "test-user-001".

    session_id: str
    # WHICH conversation session this request belongs to.
    # Used by BaseAgentMiddleware._get_run_id() to generate a stable run ID
    # for the duration of the session — ensures before_agent() and after_agent()
    # hooks on the same request share the same run_id for trace correlation.
    # In production: set to thread_id (same as user_id for single-user sessions).
    # In multi-user deployments: session_id would differ from user_id to allow
    # one user to have multiple concurrent sessions.

    domain: str
    # WHAT domain this agent is operating in.
    # Values: "pharma" | "general"
    # Used by SemanticCacheMiddleware to select cache namespace:
    #   namespace = f"cache_{domain}"   → "cache_pharma" or "cache_general"
    # Used by SemanticCacheWithRules to apply domain-specific cache thresholds:
    #   pharma: 0.97 similarity required (strict — clinical queries are specific)
    #   general: 0.88 similarity required (lenient — broader topic queries)
    # Note: domain is also baked into the system prompt at agent creation time
    # via build_system_prompt(domain) in prompt.py. The value here and the value
    # passed to build_agent() should always match.