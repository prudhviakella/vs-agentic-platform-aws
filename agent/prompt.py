"""
prompt.py — System Prompt Builder
===================================

This file has one job: fetch the versioned system prompt from AWS Bedrock
Prompt Management and substitute the two runtime variables before returning
it to build_agent() as a plain string.

WHY BEDROCK PROMPT MANAGEMENT INSTEAD OF A HARDCODED STRING?
  The fundamental principle: prompts are CONTENT, not CODE.

  If the system prompt lived in this file (or in a .txt file in the repo),
  every prompt change would require:
    1. Edit the string in code
    2. Commit, push, code review
    3. Build a new Docker image (~40 seconds)
    4. Push to ECR (~3 minutes on arm64)
    5. Update the AgentCore runtime and wait for READY (~5 minutes)
    Total: ~10 minutes per prompt iteration

  With Bedrock Prompt Management:
    1. Edit the template in the Bedrock console (no code change)
    2. Click "Publish version" (creates version N+1 with the old version intact)
    3. Update one SSM parameter: /clinical-trial-agent/prod/bedrock/prompt_version
    4. The next AgentCore cold start loads the new version automatically
    Total: ~30 seconds, zero deployment

  ADDITIONAL BENEFITS:
    - Version history: every published version is preserved, full audit trail
    - Instant rollback: change SSM param back to version N-1, done in 30 seconds
    - A/B testing: run two agent pools pointing at different prompt versions
    - Non-developer access: clinical writers can improve prompts without code access
    - Zero downtime: the prompt change takes effect on the next cold start with
      no interruption to in-flight requests on warm containers

HOW BEDROCK PROMPT MANAGEMENT WORKS:
  aws.get_bedrock_prompt("clinical-trial-agent") does two things:
    1. Calls SSM to read /clinical-trial-agent/prod/bedrock/prompt_id (e.g. "YEVDY4MYU6")
       and /clinical-trial-agent/prod/bedrock/prompt_version (e.g. "8")
       → SSM is read on EVERY call so a version update takes effect without restart
    2. Calls bedrock.get_prompt(promptIdentifier=id, promptVersion=version)
       → The Bedrock API returns the template string
       → The template fetch is cached by (prompt_id, prompt_version) via lru_cache
          so repeated calls within the same container lifetime are free

  The template uses {{variable}} double-brace syntax (Bedrock's native format).
  We replace those placeholders here with .replace() before returning.

WHY TWO PLACEHOLDERS AND NOT MORE?
  Only values that are FIXED AT AGENT CREATION TIME belong in the system prompt.
  build_agent() is called once per cold start — domain and MAX_TOOL_CALLS don't
  change during the container's lifetime, so they're safe to bake into the string.

  Values that CHANGE PER REQUEST are NOT injected here:
    - Episodic memory context: changes per user per request
      → EpisodicMemoryMiddleware injects a SystemMessage into the graph state
        before each model call via before_agent()
    - Conversation history: managed by LangGraph's Postgres checkpointer
    - User's query: arrives as a HumanMessage, not in the system prompt

  Mixing per-request dynamic content into the system prompt would mean
  rebuilding the system prompt string on every request — slower and harder
  to cache.

PLACEHOLDER REFERENCE:
  {{domain_frame}}     Injected by build_system_prompt(domain)
                       Adds domain-specific framing before the rest of the prompt.
                       "pharma" adds clinical disclaimer and faithfulness rule.
                       "general" adds a lighter evidence-citing instruction.
                       The Bedrock template starts with {{domain_frame}} so this
                       framing appears before any other instruction.

  {{max_tool_calls}}   Injected from tools/__init__.py MAX_TOOL_CALLS_PER_REQUEST
                       Tells the LLM the hard cap on tool calls per request.
                       Keeping this in SSM/config (not hardcoded in the template)
                       lets us tune it without editing the prompt.
                       Currently: 15 calls per request.
"""

import logging
from core import aws
from agent.tools import MAX_TOOL_CALLS_PER_REQUEST

log = logging.getLogger(__name__)

# The app name maps to the SSM parameter path prefix:
#   /clinical-trial-agent/prod/bedrock/prompt_id
#   /clinical-trial-agent/prod/bedrock/prompt_version
# This must match the SSM paths written by deploy.sh step_secrets().
_APP_NAME = "clinical-trial-agent"


def build_system_prompt(domain: str) -> str:
    """
    Fetch the Bedrock prompt template and substitute {{domain_frame}} and
    {{max_tool_calls}}. Returns a ready-to-use system prompt string.

    Called ONCE per cold start in build_agent(). Domain is fixed per agent
    instance — there is no per-request prompt rebuild.

    Args:
        domain: "pharma" for clinical trial queries (strict evidence rules,
                clinical disclaimers). Any other value gets the general framing
                (lighter instructions, same evidence requirement).

    Returns:
        Complete system prompt string with all placeholders substituted.
        Ready to pass directly to create_agent(system_prompt=...).

    Raises:
        botocore.exceptions.ClientError: If SSM or Bedrock calls fail
          (missing IAM permissions, wrong prompt ID, wrong version number).
          These will crash the cold start — check CloudWatch logs for details.
    """

    # ── Build domain_frame ─────────────────────────────────────────────────
    # domain_frame is a short paragraph prepended to the prompt that sets
    # the operational context for the LLM. It answers: "what kind of agent
    # am I and what are my core constraints?"
    #
    # WHY pharma gets a stricter frame than general:
    #   Clinical trial information can directly influence medical decisions.
    #   "Never provide direct treatment recommendations" and
    #   "Faithfulness to retrieved context is non-negotiable" are hard rules
    #   that prevent the LLM from combining its training knowledge with retrieved
    #   data to produce confident-sounding but unsupported medical claims.
    #
    #   The general frame omits these because a general research assistant has
    #   more latitude to synthesise and infer — the stakes are lower.
    if domain == "pharma":
        domain_frame = (
            "You are operating in a PHARMA / CLINICAL TRIAL domain. "
            "All answers must be evidence-based, cite retrieved sources, and include "
            "appropriate clinical disclaimers. Never provide direct treatment "
            "recommendations. Faithfulness to retrieved context is non-negotiable."
        )
    else:
        domain_frame = (
            "You are a knowledgeable research assistant. "
            "Always retrieve evidence before answering. Cite sources. Be precise."
        )

    # ── Fetch versioned template from Bedrock ──────────────────────────────
    # aws.get_bedrock_prompt() handles the two-step fetch:
    #   Step 1: Read prompt_id + prompt_version from SSM (fresh on every call —
    #           no cache on SSM reads so a version update is picked up at next
    #           cold start without any code change)
    #   Step 2: Fetch the template from Bedrock (cached by (prompt_id, version)
    #           via lru_cache — a warm container always uses the same template
    #           it loaded at cold start, which is the correct behaviour)
    #
    # This design means:
    #   - Prompt version changes: effective on NEXT cold start (not immediately)
    #   - Template content changes within the same version: NOT picked up
    #     (you must publish a new version to change the content)
    #   - Container crash + restart: picks up whatever version SSM points to
    template = aws.get_bedrock_prompt(_APP_NAME)

    # ── Substitute placeholders ────────────────────────────────────────────
    # Simple str.replace() rather than Python's str.format() or Template
    # because Bedrock's native placeholder syntax uses {{double braces}}.
    # str.format() uses {single braces} and would require escaping every
    # literal brace in the template (of which there are many in Cypher
    # query examples). .replace() is explicit and has no escaping rules.
    return (
        template
        .replace("{{domain_frame}}",   domain_frame)
        .replace("{{max_tool_calls}}", str(MAX_TOOL_CALLS_PER_REQUEST))
    )