"""
middleware/__init__.py
======================
Same 9-layer middleware stack as local vs-agentic-platform.
Imports from vs-agent-core and clinical_trial_agent packages
which are installed via requirements.txt in the agent container.
"""

import logging
import os

log = logging.getLogger(__name__)


def build_middleware_stack(cache, store, safety_llm) -> list:
    """
    Build the middleware stack.
    All middleware classes are imported from the local packages
    that are bundled in the agent container/deployment.
    """
    import boto3
    REGION     = os.environ.get("AWS_REGION", "us-east-1")
    SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")

    dynamo_table = boto3.client("ssm", region_name=REGION).get_parameter(
        Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]

    from core.middleware.tracer import TracerMiddleware
    from agent.middleware.pii import DomainPIIMiddleware
    from agent.middleware.content_filter import ContentFilterMiddleware
    from core.middleware.semantic_cache import SemanticCacheMiddleware
    from core.middleware.episodic_memory import EpisodicMemoryMiddleware
    from agent.middleware.hitl import HumanInTheLoopMiddleware
    from agent.middleware.output_guardrail import OutputGuardrailMiddleware

    stack = [
        TracerMiddleware(dynamodb_table_name=dynamo_table),
        DomainPIIMiddleware(),
        ContentFilterMiddleware(),
        SemanticCacheMiddleware(cache=cache),
        EpisodicMemoryMiddleware(store=store),
        HumanInTheLoopMiddleware(interrupt_on={"ask_user_input": True}),
        OutputGuardrailMiddleware(
            llm=safety_llm,
            faithfulness_threshold=0.0,  # re-enable at 0.70 when index has enough data
            confidence_threshold=0.0,
        ),
    ]

    log.info(f"[MIDDLEWARE] Stack built  layers={len(stack)}")
    return stack
