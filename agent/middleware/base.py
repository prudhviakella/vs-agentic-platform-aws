"""
base.py — BaseAgentMiddleware
==============================
All middleware classes inherit from this.

ONION PATTERN:
  Request comes in →  before_agent runs top → bottom through the stack
  Agent generates →  after_agent runs bottom → top through the stack

  Stack order: [Tracer, PII, ContentFilter, Cache, Episodic, HITL, Guardrail]

  before_agent:  Tracer → PII → ContentFilter → Cache → Episodic → HITL → Guardrail
  after_agent:   Guardrail → HITL → Episodic → Cache → ContentFilter → PII → Tracer

STATE DICT:
  Each middleware receives a mutable `state` dict and can read/write it.
  This is how middleware communicate with each other without tight coupling.

  state keys:
    messages       — current message list (can be modified by before_agent)
    config         — LangGraph config dict (thread_id, domain, etc.)
    cached         — str: set by SemanticCacheMiddleware if cache hit found
                     MiddlewareAgent short-circuits the LLM call when this is set
    response       — str: set by MiddlewareAgent after agent completes
    episodic_ctx   — str: relevant past context injected by EpisodicMemoryMiddleware
    pii_scrubbed   — bool: set by DomainPIIMiddleware after scrubbing
    blocked        — bool: set by ContentFilterMiddleware to block the request
    block_reason   — str: why the request was blocked
"""


class BaseAgentMiddleware:
    """
    Abstract base class for all agent middleware.

    Subclasses override before_agent and/or after_agent.
    Both methods receive the same mutable state dict — modifying it
    affects every subsequent middleware in the chain.
    """

    async def before_agent(self, state: dict) -> None:
        """
        Called BEFORE the agent processes the request.
        Runs in order: first middleware in the stack runs first.

        Use for: input validation, PII scrubbing, cache lookup,
                 context injection, content filtering.
        """
        pass

    async def after_agent(self, state: dict) -> None:
        """
        Called AFTER the agent generates a response.
        Runs in REVERSE order: last middleware in the stack runs first.

        Use for: output guardrails, episodic memory storage,
                 cache population, trace logging.
        """
        pass
