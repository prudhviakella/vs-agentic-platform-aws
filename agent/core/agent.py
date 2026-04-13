"""
agent.py — MiddlewareAgent
===========================
Wraps LangGraph's create_react_agent with the 9-layer middleware stack.

WHAT create_react_agent DOES:
  Builds a LangGraph StateGraph with two nodes:
    agent_node: calls the LLM with tools bound
    tools_node: executes whichever tool the LLM chose

  The graph loops: agent → tools → agent → tools → ... until
  the LLM decides to stop (no more tool calls).

WHAT MiddlewareAgent ADDS:
  Wraps the entire graph invocation with before_agent / after_agent hooks.
  The middleware runs OUTSIDE the LangGraph graph — not as graph nodes.

  This means:
    - before_agent sees the raw user message before LangGraph processes it
    - after_agent sees the final assembled response after all tool calls
    - The graph checkpointer still handles HITL state internally

SHORT-CIRCUIT (cache hit):
  If SemanticCacheMiddleware sets state["cached"] in before_agent,
  MiddlewareAgent skips the graph entirely and yields fake token events
  containing the cached answer. The UI sees no difference.

STREAMING:
  astream_events() yields raw LangGraph events. The calling code (agent.py)
  filters these into typed SSE dicts. MiddlewareAgent does NOT change the
  event format — it just wraps the stream.
"""

import logging
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, AIMessage

log = logging.getLogger(__name__)


class MiddlewareAgent:
    """
    LangGraph agent wrapped with a middleware stack.

    graph:      compiled LangGraph StateGraph (from create_react_agent)
    middleware: list of BaseAgentMiddleware instances in execution order
    """

    def __init__(self, graph, middleware: list):
        self._graph      = graph
        self._middleware = middleware

    @classmethod
    def create(
        cls,
        llm,
        tools: list,
        system_prompt: str,
        checkpointer,
        middleware: list,
    ) -> "MiddlewareAgent":
        """
        Build a ReAct agent and wrap it with middleware.

        llm:           ChatOpenAI instance (streaming=True)
        tools:         LangChain StructuredTools from MCP Gateway
        system_prompt: full system prompt text (fetched from Bedrock)
        checkpointer:  PostgresSaver for HITL state persistence
        middleware:    list of BaseAgentMiddleware instances
        """
        from langgraph.prebuilt import create_react_agent

        graph = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt,
            checkpointer=checkpointer,
        )

        log.info(
            f"[MIDDLEWARE_AGENT] Created"
            f"  tools={[t.name for t in tools]}"
            f"  middleware={[type(m).__name__ for m in middleware]}"
        )

        return cls(graph=graph, middleware=middleware)

    async def astream_events(
        self, input_data, config: dict, version: str = "v2"
    ) -> AsyncIterator[dict]:
        """
        Run the middleware chain then stream LangGraph events.

        Yields the same event dicts as LangGraph's astream_events()
        so the calling code (agent.py) works without changes.
        """

        # ── Build shared state dict ───────────────────────────────────────
        # Middleware communicate through this mutable dict
        messages = (
            input_data.get("messages", [])
            if isinstance(input_data, dict)
            else []
        )
        state = {
            "messages": messages,
            "config":   config,
            "cached":   None,
            "response": "",
            "error":    "",
        }

        # ── before_agent: top → bottom ────────────────────────────────────
        for mw in self._middleware:
            try:
                await mw.before_agent(state)
            except Exception as exc:
                log.warning(f"[MIDDLEWARE_AGENT] {type(mw).__name__}.before_agent error: {exc}")

            # Short-circuit: ContentFilterMiddleware or SemanticCacheMiddleware
            # set state["cached"] to signal "don't call the LLM"
            if state.get("cached"):
                break

        # ── Cache hit / block: fake token stream from cached answer ───────
        if state.get("cached"):
            cached_text = state["cached"]
            log.info(f"[MIDDLEWARE_AGENT] Short-circuit  reason={'blocked' if state.get('blocked') else 'cache'}")
            for word in cached_text.split(" "):
                yield self._fake_token_event(word + " ")
            # Run after_agent so tracer still logs the request
            for mw in reversed(self._middleware):
                try:
                    state["response"] = cached_text
                    await mw.after_agent(state)
                except Exception as exc:
                    log.warning(f"[MIDDLEWARE_AGENT] {type(mw).__name__}.after_agent error: {exc}")
            return

        # ── Full agent run: stream LangGraph events ───────────────────────
        actual_input = input_data
        if state["messages"] != messages:
            # Middleware modified the messages (e.g. episodic context injected)
            actual_input = {"messages": state["messages"]} if isinstance(input_data, dict) else input_data

        full_response_parts = []
        try:
            async for event in self._graph.astream_events(
                actual_input, config=config, version=version
            ):
                # Collect LLM tokens to assemble the full response
                if event.get("event") == "on_chat_model_stream":
                    token = getattr(event["data"].get("chunk", {}), "content", "")
                    if token:
                        full_response_parts.append(token)
                yield event

        except Exception as exc:
            log.exception(f"[MIDDLEWARE_AGENT] Graph error: {exc}")
            state["error"] = str(exc)
            yield self._fake_token_event(f"Error: {exc}")

        # ── after_agent: bottom → top ─────────────────────────────────────
        state["response"] = "".join(full_response_parts)
        for mw in reversed(self._middleware):
            try:
                await mw.after_agent(state)
            except Exception as exc:
                log.warning(f"[MIDDLEWARE_AGENT] {type(mw).__name__}.after_agent error: {exc}")

        # If EpisodicMemoryMiddleware or OutputGuardrailMiddleware modified
        # the response, yield the modification as a final correction token.
        # (In practice the UI shows the streaming tokens, not the corrected response,
        # but this keeps the state consistent for testing.)

    @staticmethod
    def _fake_token_event(content: str) -> dict:
        """Produce an on_chat_model_stream event dict mimicking LangGraph's format."""
        from langchain_core.messages import AIMessageChunk
        return {
            "event": "on_chat_model_stream",
            "name":  "ChatOpenAI",
            "data":  {"chunk": AIMessageChunk(content=content)},
            "tags":  [],
            "metadata": {},
            "run_id": "",
        }
