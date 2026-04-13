"""
auth.py — API Key authentication for Platform gateway.
Reads key from Secrets Manager on cold start, cached for warm requests.
Falls back to PLATFORM_API_KEY env var for local testing.
"""
import json
import logging
import os
from functools import lru_cache

import boto3
from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


@lru_cache(maxsize=1)
def _get_api_key() -> str:
    secret_id = os.environ.get("PLATFORM_API_KEY_SECRET_ID", f"{SSM_PREFIX}/platform_api_key")
    try:
        sm  = boto3.client("secretsmanager", region_name=REGION)
        raw = sm.get_secret_value(SecretId=secret_id)["SecretString"]
        return json.loads(raw)["api_key"]
    except Exception as exc:
        log.warning(f"[AUTH] Secrets Manager failed ({exc}) — using env var")
        return os.environ.get("PLATFORM_API_KEY", "local-dev-key")


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key != _get_api_key():
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key
