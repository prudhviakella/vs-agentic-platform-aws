"""prompt.py — Bedrock Prompt Management (same prompt as local)"""
import logging, os
from functools import lru_cache
import boto3

log       = logging.getLogger(__name__)
REGION    = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


def _ssm(key): 
    return boto3.client("ssm", region_name=REGION).get_parameter(Name=key)["Parameter"]["Value"]


@lru_cache(maxsize=4)
def _fetch(prompt_id, version):
    log.info(f"[PROMPT] Fetching id={prompt_id} version={version}")
    client = boto3.client("bedrock-agent", region_name=REGION)
    resp   = client.get_prompt(promptIdentifier=prompt_id, promptVersion=version)
    return resp["variants"][0]["templateConfiguration"]["text"]["text"]


def build_system_prompt(domain: str = "pharma") -> str:
    prompt_id = _ssm(f"{SSM_PREFIX}/bedrock/prompt_id")
    version   = _ssm(f"{SSM_PREFIX}/bedrock/prompt_version")
    template  = _fetch(prompt_id, version)

    domain_frame = (
        "You are operating in a pharmaceutical/clinical research context. "
        "Prioritise regulatory accuracy, safety data, and evidence-based responses."
        if domain == "pharma"
        else "You are in a general biomedical research context."
    )

    return template.replace("{{domain_frame}}", domain_frame) \
                   .replace("{{max_tool_calls}}", "10")
