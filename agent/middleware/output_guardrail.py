"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
Checks whether the agent's response is faithful to the retrieved evidence.

WHY THIS EXISTS:
  LLMs can "hallucinate" — generate confident-sounding text that isn't
  supported by the retrieved evidence. In a clinical context, this is
  dangerous. A doctor reading an AI-generated response might act on
  incorrect information.

  This middleware asks a second, cheaper LLM (GPT-4o-mini) to judge whether
  the response is grounded in the evidence the agent retrieved.

HOW IT WORKS:
  1. after_agent: take the agent's response + the retrieved chunks
  2. Ask GPT-4o-mini: "Rate faithfulness 0-1 — is this answer supported by these sources?"
  3. If faithfulness < threshold → replace response with a safe refusal message
  4. Log the reason for review

THRESHOLDS:
  faithfulness_threshold:  0.70 (currently disabled at 0.0 — re-enable when index is large enough)
  confidence_threshold:    0.70 (overall confidence)

NOTE:
  Disabled (threshold=0.0) until the Pinecone index has enough data for
  the judge to reliably distinguish supported from unsupported claims.
  Once you have 10k+ chunks, re-enable with faithfulness_threshold=0.70.
"""

import json
import logging

from agent.middleware.base import BaseAgentMiddleware

log = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are a faithfulness judge for a clinical research assistant.

RETRIEVED EVIDENCE:
{evidence}

AGENT RESPONSE:
{response}

Rate the faithfulness of the response to the retrieved evidence.
Faithfulness = 1.0 means every claim is directly supported by the evidence.
Faithfulness = 0.0 means the response contains significant unsupported claims.

Respond with ONLY a JSON object:
{{"faithfulness": 0.85, "reason": "brief reason"}}"""

_REFUSAL = (
    "I was unable to provide a confident answer based on the available clinical trial data. "
    "The retrieved evidence was insufficient to fully support a response. "
    "Please consult primary literature or a qualified researcher for this query."
)


class OutputGuardrailMiddleware(BaseAgentMiddleware):
    """
    GPT-4o-mini faithfulness judge run on every agent response.

    llm: ChatOpenAI(model="gpt-4o-mini") instance
    faithfulness_threshold: min acceptable faithfulness score (0.0 = disabled)
    confidence_threshold: min overall confidence score (0.0 = disabled)
    """

    def __init__(self, llm, faithfulness_threshold: float = 0.0, confidence_threshold: float = 0.0):
        self._llm                    = llm
        self._faithfulness_threshold = faithfulness_threshold
        self._confidence_threshold   = confidence_threshold

        if faithfulness_threshold == 0.0:
            log.info("[GUARDRAIL] Disabled (threshold=0.0) — re-enable at 0.70")
        else:
            log.info(f"[GUARDRAIL] Enabled  faithfulness≥{faithfulness_threshold}")

    async def after_agent(self, state: dict) -> None:
        # Skip if disabled, blocked, or cache hit
        if self._faithfulness_threshold == 0.0:
            return
        if state.get("blocked") or state.get("cached"):
            return

        response = state.get("response", "")
        evidence = state.get("retrieved_chunks", "")  # set by search_tool callback

        if not response or not evidence:
            return

        try:
            prompt   = _JUDGE_PROMPT.format(
                evidence=str(evidence)[:3000],
                response=response[:2000],
            )
            result   = await self._llm.ainvoke(prompt)
            verdict  = json.loads(result.content.strip())
            score    = float(verdict.get("faithfulness", 1.0))
            reason   = verdict.get("reason", "")

            log.info(f"[GUARDRAIL] faithfulness={score:.2f}  reason='{reason[:80]}'")

            if score < self._faithfulness_threshold:
                log.warning(f"[GUARDRAIL] Response blocked  score={score:.2f}  reason={reason}")
                state["response"] = _REFUSAL
                state["guardrail_blocked"] = True
                state["guardrail_reason"]  = reason

        except Exception as exc:
            # Guardrail failure must never block a response
            log.warning(f"[GUARDRAIL] Judge error (bypassing): {exc}")
