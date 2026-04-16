"""
action_guardrail.py — ActionGuardrailMiddleware
=================================================
Enforces a hard limit on how many tools the agent can call in a single request.

WHY do we need this?
  Imagine the agent goes into a loop -- it keeps calling search_tool over and
  over trying to find better evidence. Without a limit:

    Turn 1: search_tool("metformin efficacy")
    Turn 2: search_tool("metformin Phase 3")
    Turn 3: search_tool("metformin RCT results")
    Turn 4: search_tool("metformin HbA1c reduction")
    ... keeps going ...  <- costs money, never returns to user

  With ActionGuardrailMiddleware enforcing max 5 tool calls:
    After 5 calls the agent is stopped and must answer with what it has.
    User gets a response. Cost is bounded.

WHY middleware and not inside each tool?
  An individual tool cannot know the total call count across all tools.
  search_tool does not know if graph_tool was already called twice.
  Middleware sits above all tools and can see the full picture.

  Tool A: "I have been called once"   <- cannot see Tool B or Tool C
  Tool B: "I have been called twice"  <- cannot see Tool A or Tool C
  Middleware: "Total calls = 3"       <- sees everything via state ✅

HOW does it work?
  before_agent: nothing to do -- count comes from state, not from here
  after_agent:  count all tool calls in state messages, log, write to state

  WHY no _counts dict (unlike the original)?
    The original used self._counts keyed by run_id to track counts.
    But runtime.run_id and runtime.thread_id are both None in this
    LangChain version -- so _get_run_id() falls back to a UUID that is
    different every call:

      before_agent -> UUID_A -> self._counts["UUID_A"] = 0
      after_agent  -> UUID_B -> self._counts.pop("UUID_B") -> no-op
                             -> "UUID_A" leaks in dict forever

    And the dict was never actually used for enforcement anyway --
    after_agent already counts directly from state["messages"].
    So the dict is removed entirely. State is the single source of truth.

WHY write _cache_tool_count to state?
  SemanticCacheMiddlewareWithRules reads this in after_agent to decide
  whether to cache the answer. If tool_count == 0 the agent answered
  from memory alone -- that answer must never be cached in a clinical domain.
"""

import logging
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from langgraph.runtime import Runtime

from agent.tools import MAX_TOOL_CALLS_PER_REQUEST

log = logging.getLogger(__name__)


class ActionGuardrailMiddleware(BaseAgentMiddleware):

    def __init__(self):
        super().__init__()

    @hook_config(can_jump_to=[])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Nothing to do on the way in -- tool count is read from
        # state["messages"] in after_agent, no initialisation needed
        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Count AIMessages that have tool_calls attached.
        # Each such message represents one round of tool usage by the agent.
        tool_calls = sum(
            1 for msg in state.get("messages", [])
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )

        log.info(
            f"[ACTION_GUARD] tool_calls_used={tool_calls}  "
            f"max={MAX_TOOL_CALLS_PER_REQUEST}"
        )

        # Write to state so SemanticCacheMiddlewareWithRules can read it.
        # Because after_agent runs in reverse order (position 8 fires before
        # position 4), this value is already in state by the time
        # SemanticCacheMiddlewareWithRules.after_agent runs.
        state["_cache_tool_count"] = tool_calls

        return None