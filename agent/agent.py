"""
agent.py — Agent Assembly
==========================

This file is the WIRING LAYER. It doesn't contain any business logic —
its only job is to instantiate all the components and assemble them into
a compiled LangChain agent ready to handle requests.

Think of it like a car factory assembly line:
  - The parts (LLM, tools, middleware, checkpointer, store) are built elsewhere
  - build_agent() is the assembly step that bolts them together correctly

SEVEN PILLARS:
  01 Tools         → agent/tools/
                     In production: MCP tools discovered from Bedrock Gateway
                     Locally: ALL_TOOLS (in-process Python functions)

  02 Middleware    → agent/middleware/        (9 layers, applied in order)
                     Each layer wraps the agent like an onion — before_agent()
                     runs outermost-first, after_agent() runs innermost-first

  03 System Prompt → agent/prompt.py
                     Fetched from Bedrock Prompt Management at cold start.
                     Versioned — changing BEDROCK_PROMPT_VERSION SSM param
                     takes effect on next cold start without a redeploy.

  04 Schema        → agent/schema.py
                     AgentContext: Pydantic model defining the per-request
                     context fields (user_id, session_id, domain). LangChain
                     validates the context= kwarg against this at runtime.

  05 Checkpointer  → PostgresSaver (prod) / MemorySaver (local)
                     Persists LangGraph state between calls for the same
                     thread_id. Required for HITL — the interrupted graph
                     state must survive across the pause and the /resume call.

  06 Store         → PineconeStore
                     Semantic vector store used by EpisodicMemoryMiddleware
                     to retrieve and save conversation memories per user.

  07 Cache         → SemanticCache (Pinecone)
                     Caches LLM responses by semantic similarity of the input.
                     Pharma domain uses stricter threshold (0.97) because
                     clinical queries need near-exact matches to be safe.

WHY build_agent() IS ASYNC:
  Two things require async setup:
    1. psycopg.AsyncConnection.connect() — creates the async Postgres connection
    2. checkpointer.setup() — runs CREATE TABLE IF NOT EXISTS migrations for
       LangGraph's checkpoint tables (checkpoint, checkpoint_blobs, checkpoint_writes)
  If we used synchronous code here, these would block the event loop.

CALLING CONVENTIONS:
  From app.py (production):  await build_agent(domain="pharma", use_postgres=True, tools=mcp_tools)
  From tests (local):        await build_agent(domain="pharma", use_postgres=False)
  tools=None → falls back to ALL_TOOLS (in-process test functions)
"""

import logging
from typing import Any

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core import aws
from core.cache import SemanticCache
from core.pinecone_store import PineconeStore
from agent.middleware import build_stack
from agent.prompt import build_system_prompt
from agent.schema import AgentContext

log = logging.getLogger(__name__)


async def build_agent(domain: str = "general", use_postgres: bool = False, tools=None) -> Any:
    """
    Assemble and return a compiled LangChain agent with all pillars wired together.

    Args:
        domain:       Controls semantic cache threshold and the {domain_frame}
                      variable injected into the system prompt. Use "pharma"
                      for clinical trial queries.
        use_postgres: True in production (AgentCore) to persist HITL state across
                      the interrupt/resume cycle. False in local testing to avoid
                      needing a running Postgres instance.
        tools:        List of LangChain StructuredTool objects. In production these
                      come from get_mcp_tools() in app.py (MCP Gateway discovery).
                      Pass None to fall back to ALL_TOOLS for local development.

    Returns:
        Compiled LangChain agent (CompiledStateGraph) ready for astream_events().

    Raises:
        psycopg.OperationalError: If use_postgres=True and Postgres is unreachable.
        pinecone.exceptions.PineconeException: If Pinecone index doesn't exist.
        botocore.exceptions.ClientError: If Secrets Manager / SSM access fails.
    """

    # ── Pillar 06: Embedder ────────────────────────────────────────────────
    # text-embedding-3-small is used for:
    #   a) Semantic cache lookups (SemanticCache) — is this query semantically
    #      similar enough to a cached query to return the cached answer?
    #   b) Episodic memory search (EpisodicMemoryMiddleware) — retrieve relevant
    #      past conversations for this user before the LLM runs
    #
    # WHY text-embedding-3-small and not text-embedding-3-large?
    #   Speed and cost. We call the embeddings API on every request (cache lookup
    #   + episodic search). The small model is 5x cheaper and 2x faster with
    #   only marginal quality loss for the 512-dim clinical trial text.
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # ── Pillar 06: PineconeStore (vector store for episodic memory) ────────
    # init_pinecone_index() reads the Pinecone API key from SSM and returns
    # a connected pinecone.Index object pointing at the "clinical-agent" index.
    # PineconeStore wraps this Index with upsert/query helpers and namespace
    # support so each middleware layer can use its own namespace.
    #
    # NOTE: This same Pinecone index serves double duty:
    #   namespace="episodic__{user_id}"  → episodic memory per user
    #   namespace="cache_{domain}"       → semantic cache results
    # Using one index with multiple namespaces is more cost-effective than
    # separate indexes (Pinecone charges per index on starter plans).
    pinecone_index = aws.init_pinecone_index()
    store          = PineconeStore(index=pinecone_index, embedder=embedder)

    # ── Pillar 07: SemanticCache ───────────────────────────────────────────
    # SemanticCache intercepts queries BEFORE they reach the LLM.
    # It embeds the user's query and checks if a similar query was answered
    # recently. If the cosine similarity exceeds the threshold, it returns
    # the cached answer immediately — saving ~3 seconds and OpenAI costs.
    #
    # WHY different thresholds per domain?
    #   pharma: 0.97 — clinical queries like "NCT04470427 phase 2 results"
    #           are highly specific. "NCT04470427" vs "NCT04470428" are
    #           completely different trials. High threshold prevents wrong
    #           cached answers being served for similar-sounding but different
    #           queries. Patient safety > cost savings.
    #
    #   general: 0.88 — broader domain, semantic similarity is a safe proxy
    #            for query equivalence. "What is metformin?" ≈ "Tell me about
    #            metformin" at 0.88 similarity — safe to serve cached answer.
    cache = SemanticCache(
        index                = pinecone_index,
        embedder             = embedder,
        similarity_threshold = 0.97 if domain == "pharma" else 0.88,
        namespace            = f"cache_{domain}",
    )

    # ── Pillar 05: Checkpointer ────────────────────────────────────────────
    # The checkpointer is LangGraph's state persistence layer. It serialises
    # the full graph state (all messages, tool call history, middleware state)
    # into a storage backend keyed by (thread_id, checkpoint_id).
    #
    # WHY Postgres in production and MemorySaver locally?
    #
    #   HITL requires the graph state to survive the interrupt/resume gap:
    #     T=0:  User asks "show me cancer trials"
    #     T=1:  Agent calls tool-hitl___ask_user_input, graph PAUSES
    #     T=2:  interrupt event streamed to UI (question + options displayed)
    #     T=60: User clicks "NCI-MATCH" option (could be 60 seconds later!)
    #     T=61: /resume call arrives with user_answer="NCI-MATCH"
    #     T=62: Agent loads paused state from Postgres, repairs dangling tool
    #           call, injects answer, and continues the graph
    #
    #   If we used MemorySaver in production:
    #     - State lives in the container's memory
    #     - AgentCore creates a FRESH container for each runtimeSessionId invocation
    #     - The resume call gets a brand new container with empty MemorySaver
    #     - LangGraph can't find thread_id in the empty MemorySaver → crash
    #
    #   With Postgres:
    #     - State is persisted in RDS between calls
    #     - The resume container reads the paused state by thread_id from Postgres
    #     - Graph resumes exactly where it paused — no state loss
    #
    #   MemorySaver is fine locally because our local tests run chat + resume
    #   in the same process with the same MemorySaver instance.
    if use_postgres:
        import psycopg
        # init_postgres_url() reads the RDS credentials from Secrets Manager
        # and builds a postgresql+psycopg://... connection string
        conn = await psycopg.AsyncConnection.connect(
            aws.init_postgres_url(),
            autocommit=True,   # Required by AsyncPostgresSaver — it manages its own transactions
        )
        checkpointer = AsyncPostgresSaver(conn)
        # setup() creates the checkpoint tables if they don't exist:
        #   checkpoint        — main state snapshot per (thread_id, checkpoint_id)
        #   checkpoint_blobs  — large binary values chunked for storage efficiency
        #   checkpoint_writes — pending writes not yet committed to a snapshot
        # Safe to call on every cold start — uses CREATE TABLE IF NOT EXISTS
        await checkpointer.setup()
        checkpointer_label = "postgres"
    else:
        # MemorySaver stores state in a Python dict — no persistence across restarts
        checkpointer       = MemorySaver()
        checkpointer_label = "memory"

    # ── Safety LLM for OutputGuardrailMiddleware ───────────────────────────
    # gpt-4o-mini is used by OutputGuardrailMiddleware for two checks:
    #   LAYER_2: Faithfulness — does the answer contradict the retrieved evidence?
    #   LAYER_3: Consistency  — does the answer contradict prior answers in this thread?
    #
    # WHY a separate smaller model for safety checks?
    #   Using gpt-4o for safety checks would double our token costs.
    #   gpt-4o-mini is sufficient for binary yes/no judgements:
    #   "Does this answer contradict this evidence? Respond YES or NO."
    #   Temperature=0 ensures deterministic safety decisions.
    safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ── Pillar 01: Tools ──────────────────────────────────────────────────
    # In production (AgentCore), tools are passed in from app.py after being
    # discovered from the Bedrock MCP Gateway. They are LangChain StructuredTool
    # wrappers around the MCP protocol calls, with names like:
    #   tool-search___search_tool, tool-graph___graph_tool, etc.
    #
    # Locally (tools=None), we fall back to ALL_TOOLS which are regular
    # Python functions decorated with @tool. This lets you run and test
    # the agent without deploying Lambda functions or the MCP Gateway.
    #
    # WHY pass tools in from app.py instead of building them here?
    #   MCP tool discovery (get_mcp_tools()) requires an active HTTPS connection
    #   to the Bedrock Gateway and AWS SigV4 auth. This is only available in
    #   the production environment. Keeping it in app.py (the entrypoint) makes
    #   the agent assembly testable in isolation — you can call build_agent()
    #   with mock tools without needing live AWS credentials.
    if tools is None:
        from agent.tools import ALL_TOOLS
        tools = ALL_TOOLS

    # ── Pillar 02+03+04: create_agent ─────────────────────────────────────
    # create_agent() is LangChain's high-level factory that compiles a
    # ReAct (Reasoning + Acting) LangGraph agent.
    #
    # Under the hood it builds a StateGraph with these nodes:
    #   agent  → calls the LLM with messages + system prompt + tool definitions
    #   tools  → executes the tool chosen by the LLM
    # And these edges:
    #   agent → tools (if LLM chose a tool)
    #   agent → END   (if LLM produced a final answer)
    #   tools → agent (always — after tool result, go back to LLM)
    #
    # HOW create_agent() APPLIES MIDDLEWARE:
    #   The middleware list from build_stack() wraps the agent node.
    #   Before each agent node execution:
    #     middleware[0].before_agent() → ... → middleware[8].before_agent()
    #   After each agent node execution:
    #     middleware[8].after_agent() → ... → middleware[0].after_agent()
    #   The outermost middleware (index 0) has the final say on what reaches
    #   the LLM (before) and what reaches the user (after).
    #
    # MIDDLEWARE ORDER (from build_stack() in agent/middleware/__init__.py):
    #   0. TracerMiddleware          — records start/end timestamps and tool usage
    #   1. PIIMiddleware             — strips phone numbers, SSNs, emails from input
    #   2. ContentFilterMiddleware   — blocks off-topic or harmful queries
    #   3. SemanticCacheMiddleware   — short-circuits to cached answer if available
    #   4. EpisodicMemoryMiddleware  — injects relevant past conversation context
    #   5. SummarizationMiddleware   — summarises long conversation history
    #   6. SingleClarificationHITL  — code-enforces HITL for broad queries
    #   7. OutputGuardrailMiddleware — faithfulness + consistency checks on output
    #   8. TracerMiddleware (inner)  — records per-tool-call timings
    #
    # context_schema=AgentContext:
    #   Tells LangChain to validate the context= kwarg against the AgentContext
    #   Pydantic model on every astream_events() call. Fields: user_id, session_id,
    #   domain. Without this, middleware that reads runtime.context["session_id"]
    #   would get a raw unvalidated dict and could silently fail on missing keys.
    agent = create_agent(
        model="gpt-4o",                        # Primary reasoning LLM — streaming enabled
        tools=tools,                           # MCP tools (prod) or ALL_TOOLS (local)
        system_prompt=build_system_prompt(domain),  # Versioned Bedrock prompt
        middleware=build_stack(
            domain     = domain,
            store      = store,        # Shared PineconeStore for episodic memory
            safety_llm = safety_llm,   # gpt-4o-mini for output safety checks
            cache      = cache,        # SemanticCache for response caching
        ),
        store            = store,           # Passed to agent for direct access if needed
        checkpointer     = checkpointer,    # Postgres (prod) or Memory (local)
        context_schema   = AgentContext,    # Validates context= kwarg structure
    )

    log.info(
        f"[AGENT] Built  domain={domain}"
        f"  tools={[t.name for t in tools]}"
        f"  checkpointer={checkpointer_label}"
    )
    return agent