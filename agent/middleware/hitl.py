"""
hitl.py — SingleClarificationHITLMiddleware
=============================================

Human-in-the-Loop middleware that forces the agent to ask one clarifying
question before answering broad queries.

WHY THIS FILE EXISTS — THE CORE PROBLEM:
  The system prompt (Bedrock v8) instructs the LLM:
    "If query contains trigger words → MUST call tool-hitl___ask_user_input"
  This does NOT reliably work. GPT-4o has been trained on billions of examples
  of being helpful and answering questions directly. That base training
  overrides a single rule in the system prompt when the LLM has enough
  context to produce what it judges to be a useful answer.

  Result: "show me cancer trials" → LLM searches and answers directly,
  ignoring the HITL rule entirely. We tested 8 prompt versions and none
  reliably triggered HITL without code enforcement.

  Solution: CODE ENFORCEMENT in before_agent().
  Inject a SystemMessage as the LAST message in the context window just
  before the LLM runs. The LLM prioritises the most recent message in
  context — a MANDATORY OVERRIDE SystemMessage right before its input
  is far harder to ignore than a rule buried in the system prompt.

WHY BEFORE_AGENT NOT INTERRUPT_ON ALONE:
  interrupt_on={"tool-hitl___ask_user_input": True} handles the PAUSE
  mechanism — when the LLM calls the HITL tool, LangGraph intercepts before
  the tool runs and stores the question/options in the Postgres checkpoint.
  app.py then reads the checkpoint and yields the interrupt event.

  But interrupt_on says nothing about WHETHER the LLM calls the tool.
  If the LLM skips the HITL call and answers directly, interrupt_on never fires.
  before_agent() is the gate that ensures the LLM CALLS the tool.

  The two mechanisms are complementary:
    before_agent()     → forces the LLM to call ask_user_input
    interrupt_on       → intercepts that call and pauses the graph
    app.py Path C      → reads question/options from state after pause

SINGLE CLARIFICATION GUARANTEE:
  The "Single" in SingleClarificationHITLMiddleware is a hard constraint:
  ONE clarification per conversation, never more.

  WHY: asking for clarification twice in one conversation is a bad UX.
  The user answered once — asking again signals the agent didn't listen.
  The already_clarified check (presence of a ToolMessage from ask_user_input)
  prevents re-entry and instead injects a "you already asked, now answer" instruction.

INTERRUPT FLOW END-TO-END:
  1. User: "show me cancer trials"
  2. before_agent(): _is_broad_query() matches "trials" → inject MANDATORY OVERRIDE
  3. LLM calls tool-search___search_tool (step 1 of the override instruction)
  4. LLM calls tool-hitl___ask_user_input with question + 5 options from search results
  5. interrupt_on fires → LangGraph pauses graph, stores interrupt in Postgres checkpoint
  6. astream_events loop ends (no more events)
  7. app.py Path C: agent.aget_state() reads state.tasks[0].interrupts[0].value
  8. interrupt event yielded: {"type":"interrupt","question":"...","options":[...]}
  9. Platform SSE → UI renders clarification card with option buttons
  10. User clicks "NCI-MATCH trial" → POST /resume with user_answer
  11. handler() repairs dangling tool call, injects "[HITL Answer]: NCI-MATCH..."
  12. before_agent(): already_clarified=True → inject "now search and answer" instruction
  13. LLM searches for NCI-MATCH specifically → streams final answer
"""

from typing import Any
from langchain.agents.middleware import HumanInTheLoopMiddleware, AgentState, hook_config
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime

# ── Broad query trigger words ──────────────────────────────────────────────
# A query is "broad" if it contains any of these words AND lacks a specific
# identifier (NCT ID, drug name, or a long detailed question).
#
# WHY a word set rather than LLM classification:
#   LLM classification would cost ~500ms and an extra API call per request.
#   This set runs in < 1ms. False positive rate is acceptable — we'd rather
#   ask one unnecessary clarification than skip HITL on a genuinely ambiguous query.
#
# HOW to extend this set:
#   Add words that represent broad category requests in your domain.
#   Avoid words that appear in specific queries (e.g. "efficacy" is specific
#   enough that a query containing it probably doesn't need clarification).
BROAD_TRIGGER_WORDS = {
    "trials", "studies", "drugs", "show", "list", "examples",
    "what are", "give me", "any", "some", "cancer", "research",
    "treatment", "therapy", "data", "results", "information",
}


def _is_broad_query(messages: list) -> bool:
    """
    Detect broad/ambiguous queries that require HITL clarification.

    A query is broad if it:
      1. Contains at least one BROAD_TRIGGER_WORD (suggests a category request)
      2. AND lacks any specific identifier (suggests the user hasn't narrowed down)

    Specific identifiers (any one disqualifies "broad"):
      - "nct" in text    → NCT trial ID (e.g. "NCT04470427") — already specific
      - len > 8 words    → long question usually contains enough context
      - uppercase word   → drug name or abbreviation (e.g. "BNT162b2", "FOLFOX")

    WHY check the LAST HumanMessage (reversed search):
      In a multi-turn conversation, we want to check the CURRENT question —
      not an earlier turn. Walking backwards finds the most recent HumanMessage
      first, which is the current user input.

    WHY return on first HumanMessage found (not check all):
      Only the current question needs HITL gating. Past questions have already
      been handled. Checking all HumanMessages could trigger HITL on a resumed
      conversation that already completed its clarification.

    Args:
        messages: state["messages"] — full conversation history.

    Returns:
        True if the current question is broad and needs clarification.
        False if specific enough to answer directly, or no HumanMessage found.
    """
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            # Safe content extraction — msg.content may be a list of content
            # blocks (OpenAI multimodal format) or a plain string.
            text  = msg.content.lower() if isinstance(msg.content, str) else ""
            words = set(text.split())

            # CONDITION 1: contains a broad trigger word
            has_trigger = bool(words & BROAD_TRIGGER_WORDS)

            # CONDITION 2: lacks any specific identifier
            has_specific = any([
                "nct" in text,                                           # NCT trial ID
                len(text.split()) > 8,                                   # detailed question
                any(c.isupper() and len(c) > 3 for c in text.split()),  # drug name / abbreviation
            ])

            # Broad = trigger present AND no specific identifier
            return has_trigger and not has_specific

    return False   # no HumanMessage found — not broad (safe default)


class SingleClarificationHITLMiddleware(HumanInTheLoopMiddleware):
    """
    Middleware that enforces a single HITL clarification for broad queries.

    Inherits from HumanInTheLoopMiddleware to get interrupt_on support.
    interrupt_on={"tool-hitl___ask_user_input": True} is passed at construction
    time in middleware/__init__.py. It tells LangGraph to pause graph execution
    when the LLM calls ask_user_input, storing the interrupt in the Postgres
    checkpoint before the tool actually runs.

    This class adds before_agent() on top:
      - Detects broad queries via _is_broad_query()
      - Injects a MANDATORY OVERRIDE SystemMessage forcing the LLM to call HITL
      - Prevents second HITL calls after the user has already answered once
    """

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Gate broad queries behind HITL and prevent repeated clarification.

        Called before every LLM run in the ReAct loop. Runs on EVERY agent
        node execution — including the resume turn after the user answers.
        The already_clarified check handles the resume case correctly.

        Two paths:

        PATH 1 — RESUME (user already answered HITL):
          Detects a ToolMessage from ask_user_input in message history.
          Injects a "you already asked, now search and answer" SystemMessage.
          This prevents the LLM from asking for clarification a second time,
          which it might otherwise do because it sees "[HITL Answer]: X" in
          history and thinks another clarification is appropriate.

        PATH 2 — NEW BROAD QUERY (first turn):
          _is_broad_query() returns True for "show me cancer trials" type queries.
          Injects MANDATORY OVERRIDE SystemMessage as the last message before
          the LLM runs — the LLM prioritises the most recent context message.
          Forces the LLM to: (1) search first, (2) then call ask_user_input.

        PATH 3 — SPECIFIC QUERY (no injection):
          _is_broad_query() returns False for "What are the Phase 3 results for
          BNT162b2?" — specific enough to answer directly.
          Returns None → no state modification → LLM runs normally.

        Returns:
          dict with "messages" key → state update injecting SystemMessage
          None → no modification, proceed normally
        """
        messages = state.get("messages", [])

        # ── PATH 1: RESUME — prevent second HITL call ────────────────────
        # Check if a ToolMessage from ask_user_input exists in history.
        # A ToolMessage with name containing "ask_user_input" means the LLM
        # already called the HITL tool and the user already answered.
        # (ToolMessages are what LangGraph stores as tool execution results.)
        #
        # WHY check msg.type == "tool" AND name contains "ask_user_input":
        #   msg.type == "tool" narrows to ToolMessages only (not AIMessages
        #   or HumanMessages). The name check ensures it's specifically the
        #   HITL tool and not search_tool or graph_tool results.
        already_clarified = any(
            hasattr(msg, "type") and msg.type == "tool"
            and "ask_user_input" in str(getattr(msg, "name", ""))
            for msg in messages
        )

        if already_clarified:
            # Inject a hard instruction to search and answer directly.
            # Without this, the LLM on the resume turn sees the user's answer
            # "[HITL Answer]: NCI-MATCH. Now search and answer." and might
            # think it should ask another clarifying question.
            # This SystemMessage overrides that tendency.
            return {
                "messages": [SystemMessage(content=(
                    "INSTRUCTION: You have already asked one clarifying question "
                    "and received the user's answer. "
                    "You MUST NOT call tool-hitl___ask_user_input again. "
                    "Call tool-search___search_tool NOW to retrieve evidence "
                    "and answer the user's question directly."
                ))]
            }

        # ── PATH 2: BROAD QUERY — force HITL via code enforcement ─────────
        # _is_broad_query() detects category-level queries like "show me cancer trials"
        # that need narrowing before the agent can give a useful answer.
        #
        # WHY code enforcement instead of prompt instructions alone:
        #   We tested 8 prompt versions (v1–v8) with increasingly strict rules.
        #   None reliably forced the LLM to call ask_user_input when it had
        #   enough context to produce an answer. GPT-4o's base training to be
        #   helpful overrides prompt rules when the LLM judges it can answer.
        #   The only reliable approach: inject a SystemMessage as THE LAST
        #   message in context right before the LLM runs. The LLM always
        #   processes context in order and weights recent messages more heavily.
        #   A MANDATORY OVERRIDE as the last thing it reads is very hard to ignore.
        #
        # WHY inject a 2-step instruction (search THEN ask_user_input):
        #   Step 1 (search first): the HITL options must come from real search
        #   results, not the LLM's training knowledge. If the LLM calls
        #   ask_user_input without searching, it generates options from memory —
        #   hallucinated or outdated trial names. Forcing search first ensures
        #   options are grounded in the actual Pinecone knowledge base.
        #   Step 2 (then HITL): after search returns, the LLM has real options
        #   to present to the user.
        if _is_broad_query(messages):
            return {
                "messages": [SystemMessage(content=(
                    "MANDATORY OVERRIDE: The user's query is broad and ambiguous. "
                    "You MUST follow these steps EXACTLY:\n"
                    "1. Call tool-search___search_tool with the user's query.\n"
                    "2. After results return, call tool-hitl___ask_user_input "
                    "with 3-5 specific options derived from the search results.\n"
                    "DO NOT answer directly. DO NOT skip tool-hitl___ask_user_input. "
                    "Answering directly is a CRITICAL FAILURE."
                ))]
            }

        # ── PATH 3: SPECIFIC QUERY — no injection ─────────────────────────
        # Query is specific enough to answer directly. Return None so execution
        # continues to the LLM without any state modification.
        return None