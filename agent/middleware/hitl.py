"""
hitl.py — HumanInTheLoopMiddleware
=====================================
Configures which tool calls should trigger a HITL interrupt.

HOW HITL ACTUALLY WORKS:
  The interrupt does NOT happen in this middleware's before_agent/after_agent.
  It happens inside agent.py's _stream_events() when LangGraph fires an
  on_tool_start event for the ask_user_input tool.

  This middleware's role is:
    1. Declare which tool names should interrupt (interrupt_on dict)
    2. Provide configuration that the MiddlewareAgent can read

  The actual graph pause, state serialisation to Postgres, and resume via
  Command(resume=...) are all handled by LangGraph's checkpointer + agent.py.

WHY IS THIS A MIDDLEWARE LAYER AT ALL?
  Having it in the stack makes it explicit and configurable. You can
  easily add or remove tools that trigger interrupts without touching
  agent.py. It also serves as documentation: when students read the stack,
  they see "HITL is enabled" immediately.
"""

import logging
from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)


class HumanInTheLoopMiddleware(BaseAgentMiddleware):
    """
    Declares which tool calls should pause the agent and wait for human input.

    interrupt_on: dict mapping tool names to True/False
    Example: {"ask_user_input": True}
    """

    def __init__(self, interrupt_on: dict):
        self.interrupt_on = interrupt_on
        log.info(f"[HITL] Configured  interrupt_on={list(k for k,v in interrupt_on.items() if v)}")

    async def before_agent(self, state: dict) -> None:
        # Store interrupt config in state so MiddlewareAgent can expose it
        state["hitl_interrupt_on"] = self.interrupt_on

    async def after_agent(self, state: dict) -> None:
        # After a resume: clear the interrupt flag
        state.pop("hitl_interrupt_on", None)
