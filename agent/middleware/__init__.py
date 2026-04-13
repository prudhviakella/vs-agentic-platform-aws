"""
middleware/__init__.py
======================
Builds the 9-layer middleware stack for the clinical trial agent.

All middleware classes live in this package — no external dependencies
on vs-agentic-platform. Everything is self-contained.

LAYER ORDER (top = first in, last out):
  1. TracerMiddleware         — logs latency + metadata to DynamoDB
  2. DomainPIIMiddleware      — scrubs patient names, NHS numbers, DOBs
  3. ContentFilterMiddleware  — blocks off-topic queries before LLM is called
  4. SemanticCacheMiddleware  — returns cached answer if similarity > 0.97
  5. EpisodicMemoryMiddleware — injects relevant past context from Pinecone
  6. HumanInTheLoopMiddleware — marks ask_user_input as an interrupt tool
  7. OutputGuardrailMiddleware— checks response faithfulness (disabled until index ≥10k)
"""

import logging
import os

import boto3

log = logging.getLogger(__name__)


def build_middleware_stack(cache, store, safety_llm) -> list:
    """
    Instantiate and return the 7-layer middleware stack.

    cache:      SemanticCache  — Pinecone cache__ namespace wrapper
    store:      PineconeStore  — Pinecone episodic__ namespace wrapper
    safety_llm: ChatOpenAI(model="gpt-4o-mini") — faithfulness judge
    """
    REGION     = os.environ.get("AWS_REGION", "us-east-1")
    SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")

    dynamo_table = boto3.client("ssm", region_name=REGION).get_parameter(
        Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]

    from agent.middleware.tracer          import TracerMiddleware
    from agent.middleware.pii             import DomainPIIMiddleware
    from agent.middleware.content_filter  import ContentFilterMiddleware
    from agent.middleware.semantic_cache  import SemanticCacheMiddleware
    from agent.middleware.episodic_memory import EpisodicMemoryMiddleware
    from agent.middleware.hitl            import HumanInTheLoopMiddleware
    from agent.middleware.output_guardrail import OutputGuardrailMiddleware

    stack = [
        TracerMiddleware(dynamodb_table_name=dynamo_table, region=REGION),
        DomainPIIMiddleware(),
        ContentFilterMiddleware(),
        SemanticCacheMiddleware(cache=cache),
        EpisodicMemoryMiddleware(store=store),
        HumanInTheLoopMiddleware(interrupt_on={"ask_user_input": True}),
        OutputGuardrailMiddleware(
            llm=safety_llm,
            faithfulness_threshold=0.0,   # re-enable at 0.70 when index ≥ 10k chunks
            confidence_threshold=0.0,
        ),
    ]

    log.info(f"[MIDDLEWARE] Stack ready  layers={len(stack)}")
    return stack
