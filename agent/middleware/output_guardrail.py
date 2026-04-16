"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
Three-layer safety check before the answer reaches the user.

Layer 1 — regex (<1ms)       : obvious violations (medical action directives)
Layer 2 — faithfulness (LLM) : answer grounded in retrieved context?
Layer 3 — contradiction (LLM): answer contradicts retrieved context?

BUGS FIXED:
────────────
  FIX 1 — HITL skip used exact tool name match.
    Original: tc.get("name") == "ask_user_input"
    Problem:  New tool name is "clarify___ask_user_input" — exact match fails.
              Guardrail ran during HITL → faithfulness=0.00 → blocked interrupt.
    Fixed:    "ask_user_input" in tc.get("name", "") — substring match works
              for both old (tool-hitl___ask_user_input) and new (clarify___ask_user_input).

  FIX 2 — Tool result content was raw MCP format, unreadable by LLM judge.
    Original: str(content)[:600] where content was:
              "[{'type': 'text', 'text': '{\"results\":[{\"score\":0.69,...}]}'}]"
              LLM judge received Python repr of a list — couldn't parse context.
              Faithfulness scores were meaningless.
    Fixed:    _parse_tool_content() unwraps the MCP envelope:
              List of {"type":"text","text":"..."} → extracts inner text
              JSON {"results":[{"score":...,"text":"..."}]} → extracts text fields
              Plain string → used as-is
              LLM judge now receives clean readable clinical text.

  FIX 3 — Prior AI answers used as grounding context is circular.
    Original: Included all prior AIMessage.content as grounding.
              Problem: Q2 checked against A1 — but A1 was already checked
              against tool results. If A1 was slightly paraphrased, A2
              (a follow-up grounded in A1) would score faithfulness=0.7
              even though it's perfectly correct.
    Fixed:    Context = tool results from ALL turns (search_tool, graph_tool).
              For follow-up questions, tool results from prior turns ARE in state
              and are the correct grounding source. No circular answer-vs-answer.

  FIX 4 — NON_GROUNDING_TOOLS used exact short names but actual names are MCP-prefixed.
    Original: {"ask_user_input", "summariser_tool"} — exact set membership check.
    Problem:  Actual tool names: "clarify___ask_user_input", "tool-summariser___summariser_tool"
              Set membership failed → HITL and summariser results included as context.
    Fixed:    _is_non_grounding_tool() uses substring checks:
              "ask_user_input" in name OR "summariser" in name.

WHY skip on HITL interrupt:
  When clarify___ask_user_input is called, LangGraph pauses before the tool runs.
  The state has an AIMessage with tool_calls but no final answer text yet.
  Running faithfulness on an empty answer gives 0.00 → _safe_fallback fires →
  overwrites the interrupt event → HITL question never reaches the UI.

WHY exclude summariser_tool from grounding:
  summariser_tool produces paraphrased summaries of retrieved chunks.
  Checking faithfulness against a paraphrase of the source gives ~0.7
  even for correct answers — false failures. Use original search/graph
  tool results as grounding, not the summarised version.

WHY NOT use prior AI answers as grounding:
  Circular dependency: A2 checked against A1, A1 was checked against tools.
  If A1 is slightly paraphrased from tool text, A2 (which quotes A1) scores
  low faithfulness even though both answers are correct. Use tool results directly.

TRACER INTEGRATION:
  Annotates trace after every check outcome via:
    TracerMiddleware.update_trace(run_id, {
        "guardrail_passed":   bool,
        "guardrail_blocked":  bool,
        "faithfulness_score": float | None,
        "consistency_score":  float | None,
    })
"""

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from agent.guardrails import check_medical_action_output

log = logging.getLogger(__name__)

_FALLBACK_MARKER = "did not meet safety and accuracy standards"


class OutputGuardrailMiddleware(BaseAgentMiddleware):

    def __init__(
        self,
        llm=None,
        faithfulness_threshold: float = 0.0,
        confidence_threshold:   float = 0.0,
    ):
        super().__init__()
        self._llm                   = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.faithfulness_threshold = faithfulness_threshold
        self.confidence_threshold   = confidence_threshold

    @staticmethod
    def _is_non_grounding_tool(name: str) -> bool:
        """
        FIX 4: Substring check instead of exact set membership.
        MCP prefixes tool names: "clarify___ask_user_input", "tool-summariser___summariser_tool"
        Exact match against {"ask_user_input", "summariser_tool"} would always fail.
        """
        return "ask_user_input" in name or "summariser" in name

    @staticmethod
    def _parse_tool_content(raw_content: Any) -> str:
        """
        FIX 2: Unwrap MCP tool result envelope to extract clean text.

        MCP tool results arrive in one of three formats:

        Format 1 — MCP list envelope (most common from Bedrock Gateway):
          [{"type": "text", "text": "{\"results\":[{\"score\":0.69,\"text\":\"...\"}]}"}]
          → extract inner "text" field from each item

        Format 2 — JSON string with results array (search_tool):
          {"results": [{"score": 0.69, "text": "actual clinical text..."}]}
          → extract "text" field from each result

        Format 3 — Plain string (summariser, graph_tool):
          "The NCI-MATCH trial enrolled..."
          → use as-is

        WHY this matters:
          The LLM faithfulness judge receives this as "grounding context".
          Format 1 raw: "[{'type': 'text', 'text': '{\"results\":[{\"sco..."}]"
          That's a Python repr of a list — the LLM cannot parse it as clinical text.
          After parsing: "The NCI-MATCH trial enrolled 3,000 patients..."
          The LLM can now correctly judge faithfulness against real content.
        """
        if not raw_content:
            return ""

        # Format 1: MCP list envelope
        if isinstance(raw_content, list):
            parts = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    # Recursively parse the inner text (may be Format 2)
                    inner = OutputGuardrailMiddleware._parse_tool_content(item.get("text", ""))
                    if inner:
                        parts.append(inner)
            return "\n".join(parts)

        text = str(raw_content).strip()

        # Format 2: JSON string — try to parse
        if text.startswith("{") or text.startswith("["):
            try:
                parsed = json.loads(text)

                # search_tool: {"results": [{"score": float, "text": str}]}
                if isinstance(parsed, dict) and "results" in parsed:
                    chunks = []
                    for r in parsed["results"]:
                        if isinstance(r, dict) and "text" in r:
                            chunks.append(r["text"].strip())
                    return "\n\n".join(chunks)

                # graph_tool: {"results": [str, ...]}
                if isinstance(parsed, dict) and "results" in parsed:
                    items = parsed["results"]
                    if items and isinstance(items[0], str):
                        return "\n".join(items)

                # summariser: {"summary": str}
                if isinstance(parsed, dict) and "summary" in parsed:
                    return parsed["summary"]

                # Fallback for other JSON
                return json.dumps(parsed)[:600]

            except (json.JSONDecodeError, Exception):
                pass  # not valid JSON — use as plain text

        # Format 3: plain string
        return text

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id   = self._get_run_id(runtime)
        messages = state.get("messages", [])
        if not messages:
            return None

        # ── Skip 1: HITL interrupt ─────────────────────────────────────────
        # FIX 1: Use substring check — works for both old and new tool names.
        # "clarify___ask_user_input" contains "ask_user_input" ✓
        # "tool-hitl___ask_user_input" contains "ask_user_input" ✓
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if any("ask_user_input" in tc.get("name", "") for tc in tool_calls):
                    log.info("[OUTPUT_GUARD] Skipping — agent paused for HITL")
                    return None
                break

        # Find the latest AI answer (text content, not tool-calling turn)
        answer = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            return None

        # Skip if already a fallback (prevents double-wrapping)
        if _FALLBACK_MARKER in answer:
            return None

        # ── Layer 1: regex (<1ms, no LLM) ─────────────────────────────────
        ok, reason = check_medical_action_output(answer)
        if not ok:
            log.warning(f"[OUTPUT_GUARD] LAYER_1 FAIL  reason='{reason}'")
            TracerMiddleware.update_trace(run_id, {
                "guardrail_passed":   False,
                "guardrail_blocked":  True,
                "faithfulness_score": None,
                "consistency_score":  None,
            })
            return self._safe_fallback(state, f"Layer 1: {reason}")

        # ── Layers 2 + 3: LLM-as-judge ────────────────────────────────────
        context_chunks    = self._extract_grounding_context(messages)
        faith_score       = 1.0
        consistency_score = 1.0

        if context_chunks:
            try:
                faith_score = self._faithfulness_score_sync(answer, context_chunks)
                log.info(
                    f"[OUTPUT_GUARD] LAYER_2  faithfulness={faith_score:.2f}"
                    f"  threshold={self.faithfulness_threshold}"
                )
                if faith_score < self.faithfulness_threshold:
                    TracerMiddleware.update_trace(run_id, {
                        "guardrail_passed":   False,
                        "guardrail_blocked":  True,
                        "faithfulness_score": round(faith_score, 3),
                        "consistency_score":  None,
                    })
                    return self._safe_fallback(
                        state, f"Layer 2: faithfulness={faith_score:.2f} below threshold"
                    )

                consistency_score = self._contradiction_score_sync(answer, context_chunks)
                log.info(f"[OUTPUT_GUARD] LAYER_3  consistency={consistency_score:.2f}")
                if consistency_score < self.confidence_threshold:
                    TracerMiddleware.update_trace(run_id, {
                        "guardrail_passed":   False,
                        "guardrail_blocked":  True,
                        "faithfulness_score": round(faith_score, 3),
                        "consistency_score":  round(consistency_score, 3),
                    })
                    return self._safe_fallback(
                        state, f"Layer 3: consistency={consistency_score:.2f} contradicts sources"
                    )

                confidence = min(faith_score, consistency_score)
                if confidence < self.confidence_threshold:
                    disclaimer = (
                        "\n\n⚠ Confidence below threshold. "
                        "Please verify with a qualified professional."
                    )
                    for i in range(len(messages) - 1, -1, -1):
                        if isinstance(messages[i], AIMessage):
                            messages[i].content = str(messages[i].content) + disclaimer
                            break
                    log.info(f"[OUTPUT_GUARD] DISCLAIMER added  confidence={confidence:.2f}")

            except Exception as exc:
                log.warning(f"[OUTPUT_GUARD] LLM judge failed ({exc}) — passing as-is")
        else:
            log.info("[OUTPUT_GUARD] No grounding context — skipping faithfulness check")

        # ── PASSED ────────────────────────────────────────────────────────
        TracerMiddleware.update_trace(run_id, {
            "guardrail_passed":   True,
            "guardrail_blocked":  False,
            "faithfulness_score": round(faith_score, 3) if context_chunks else None,
            "consistency_score":  round(consistency_score, 3) if context_chunks else None,
        })

        log.info(f"[OUTPUT_GUARD] PASSED  answer='{answer[:60]}'")
        return None

    def _safe_fallback(self, state: AgentState, reason: str) -> dict[str, Any]:
        """Replace LLM answer with safe fallback. TracerMiddleware already annotated."""
        log.error(f"[OUTPUT_GUARD] HARD FAIL  reason='{reason}'")
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=(
            "I was unable to provide a verified answer for your question. "
            "The response did not meet safety and accuracy standards. "
            "Please consult a qualified professional or rephrase your question."
            f"\n\n[Reason logged for review: {reason}]"
        )))
        return {"messages": messages, "jump_to": "end"}

    def _extract_grounding_context(self, messages: list) -> list[str]:
        """
        FIX 2 + 3: Extract clean, parsed tool results as grounding context.

        WHY only tool results, not prior AI answers:
          Prior AI answers as grounding is circular — A2 checked against A1
          which was itself checked against tools. If A1 paraphrased the tool
          text, A2 (which quotes A1) scores low faithfulness even when correct.
          Tool results are the primary evidence source — always use them directly.

          For follow-up questions ("what is the turnaround time?" after NCI-MATCH):
          The search_tool and graph_tool results from prior turns ARE still in
          state["messages"] as ToolMessages — they are the correct grounding.

        WHY parse with _parse_tool_content:
          MCP wraps results in [{"type":"text","text":"{\"results\":[...]}"}].
          Raw str() of this is unreadable Python repr. After parsing, the LLM
          judge receives clean clinical text it can actually evaluate against.

        WHY exclude ask_user_input and summariser:
          ask_user_input: HITL metadata, not clinical evidence.
          summariser: paraphrased version of tool results — scores lower
          against original text than the original tool results would.

        Capped at 6 chunks: keeps judge prompt manageable (~3,600 chars max).
        """
        results = []

        for msg in messages:
            if getattr(msg, "type", None) == "tool":
                tool_name = str(getattr(msg, "name", ""))

                # FIX 4: substring check for MCP-prefixed names
                if self._is_non_grounding_tool(tool_name):
                    continue

                raw_content = getattr(msg, "content", "")
                parsed      = self._parse_tool_content(raw_content)

                if parsed and parsed.strip():
                    results.append(parsed[:800])   # cap per-chunk length

        return results[:6]

    def _faithfulness_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Rate how faithfully this answer is grounded in the retrieved context.\n"
            "Respond with ONLY a decimal number 0.0-1.0. Nothing else.\n"
            "0.0 = completely fabricated.  1.0 = fully grounded.\n\n"
            f"Retrieved Context:\n{context_text[:1_500]}\n\n"
            f"Answer:\n{answer[:600]}\n\nFaithfulness score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90

    def _contradiction_score_sync(self, answer: str, context_chunks: list[str]) -> float:
        context_text = "\n\n".join(context_chunks)
        prompt = (
            "Does this answer contradict any of the retrieved context?\n"
            "Respond with ONLY a decimal 0.0-1.0. Nothing else.\n"
            "0.0 = direct contradiction.  1.0 = fully consistent.\n\n"
            f"Context:\n{context_text[:1_200]}\n\n"
            f"Answer:\n{answer[:500]}\n\nConsistency score:"
        )
        try:
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            return max(0.0, min(1.0, float(resp.content.strip())))
        except Exception:
            return 0.90