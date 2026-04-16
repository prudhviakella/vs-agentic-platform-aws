"""
aws.py — AWS Credential Helpers
================================
Centralises all AWS SDK calls so the rest of the codebase has zero boto3
imports. agent.py, cache.py, and pinecone_store.py stay cloud-agnostic —
they receive initialised clients, not raw credentials.

All credentials are read from SSM Parameter Store or Secrets Manager.
ENV is hardcoded to "dev" — no APP_ENV or local-mode branching.
Students run against real AWS services from day one.

SSM parameter paths:
  /clinical-agent/dev/pinecone/api_key
  /clinical-agent/dev/pinecone/index_name
  /clinical-agent/dev/dynamodb/trace_table_name
  /clinical-agent/dev/platform/api_key
  /clinical-trial-agent/dev/bedrock/prompt_id
  /clinical-trial-agent/dev/bedrock/prompt_version

Secrets Manager:
  clinical-agent/dev/postgres
    {"host":..., "port":..., "dbname":..., "username":..., "password":...}
"""

import json
import logging
import time
from functools import lru_cache
from typing import Any

log = logging.getLogger(__name__)

# Fixed environment — all SSM paths use /dev/ prefix.
# Change this one constant to switch environments.
import os
ENV = os.environ.get("AGENT_ENV", "dev")


# ── Boto3 clients ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _ssm() -> Any:
    import boto3
    return boto3.client("ssm", region_name="us-east-1")


@lru_cache(maxsize=None)
def _secretsmanager() -> Any:
    import boto3
    return boto3.client("secretsmanager", region_name="us-east-1")


@lru_cache(maxsize=None)
def _bedrock() -> Any:
    import boto3
    return boto3.client("bedrock-agent", region_name="us-east-1")


@lru_cache(maxsize=None)
def _dynamodb(region: str = "us-east-1") -> Any:
    import boto3
    return boto3.resource("dynamodb", region_name=region)


# ── SSM ────────────────────────────────────────────────────────────────────────

def get_ssm_parameter(name: str, with_decryption: bool = True) -> str:
    resp = _ssm().get_parameter(Name=name, WithDecryption=with_decryption)
    return resp["Parameter"]["Value"]


# ── Secrets Manager ────────────────────────────────────────────────────────────

def get_secret_json(secret_name: str) -> dict:
    resp       = _secretsmanager().get_secret_value(SecretId=secret_name)
    secret_str = resp.get("SecretString") or resp.get("SecretBinary", b"").decode()
    return json.loads(secret_str)


# ── Pinecone ───────────────────────────────────────────────────────────────────

def init_pinecone_index() -> Any:
    from pinecone import Pinecone
    api_key    = get_ssm_parameter(f"/clinical-agent/{ENV}/pinecone/api_key")
    index_name = get_ssm_parameter(f"/clinical-agent/{ENV}/pinecone/index_name", with_decryption=False)
    log.info(f"[AWS] Pinecone index='{index_name}'  env={ENV}")
    return Pinecone(api_key=api_key).Index(index_name)


# ── Postgres ───────────────────────────────────────────────────────────────────

def init_postgres_url() -> str:
    secret = get_secret_json(f"clinical-agent/{ENV}/postgres")
    from urllib.parse import quote_plus
    return (
        f"postgresql://{secret['username']}:{quote_plus(secret['password'])}"
        f"@{secret['host']}:{secret.get('port', '5432')}/{secret['dbname']}"
    )


# ── DynamoDB ───────────────────────────────────────────────────────────────────

def get_trace_table_name(app_name: str = "clinical-agent") -> str:
    return get_ssm_parameter(
        f"/{app_name}/{ENV}/dynamodb/trace_table_name",
        with_decryption=False,
    )


def init_trace_table(table_name: str, ttl_days: int = 30, region: str = "us-east-1") -> Any:
    from botocore.exceptions import ClientError
    ddb   = _dynamodb(region)
    table = ddb.Table(table_name)
    try:
        table.load()
        log.info(f"[AWS] DynamoDB table '{table_name}' found  region={region}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        table = ddb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        if ttl_days > 0:
            ddb.meta.client.update_time_to_live(
                TableName=table_name,
                TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
            )
        log.info(f"[AWS] DynamoDB table '{table_name}' created")
    return table


def put_trace(table: Any, trace: dict, ttl_days: int = 30) -> None:
    from decimal import Decimal
    def _to_decimal(v):
        if isinstance(v, float): return Decimal(str(round(v, 4)))
        if isinstance(v, dict):  return {k: _to_decimal(val) for k, val in v.items()}
        if isinstance(v, list):  return [_to_decimal(i) for i in v]
        return v
    try:
        item = _to_decimal(dict(trace))
        item["run_id"] = str(trace.get("run_id", "unknown"))
        item["ts"]     = Decimal(str(round(time.time(), 3)))
        if ttl_days > 0:
            item["expires_at"] = int(time.time()) + ttl_days * 86_400
        item = {k: v for k, v in item.items() if v is not None}
        table.put_item(Item=item)
    except Exception as exc:
        log.error(f"[AWS] DynamoDB put_trace failed: {exc}")


def get_trace_item(table_name: str, run_id: str, region: str = "us-east-1") -> dict | None:
    try:
        result = _dynamodb(region).Table(table_name).get_item(Key={"run_id": run_id})
        return result.get("Item")
    except Exception as exc:
        log.error(f"[AWS] DynamoDB get_trace failed: {exc}")
        return None


# ── Bedrock Prompt Management ──────────────────────────────────────────────────

@lru_cache(maxsize=2)
def _fetch_prompt_template(prompt_id: str, prompt_version: str) -> str:
    """
    Fetch and cache a prompt template from Bedrock Prompt Management.
    lru_cache keeps the last two (id, version) pairs warm.
    A version bump in SSM causes a cache miss and a fresh Bedrock fetch.
    """
    log.info(f"[AWS] Fetching Bedrock prompt  id={prompt_id}  version={prompt_version}")
    resp     = _bedrock().get_prompt(promptIdentifier=prompt_id, promptVersion=prompt_version)
    variants = resp.get("variants", [])
    if not variants:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no variants")
    template = variants[0].get("templateConfiguration", {}).get("text", {}).get("text", "")
    if not template:
        raise ValueError(f"Bedrock prompt '{prompt_id}' v{prompt_version} has no text template")
    return template


def get_bedrock_prompt(app_name: str = "clinical-trial-agent") -> str:
    """
    Read prompt_id + prompt_version from SSM then fetch the template from Bedrock.
    SSM is read on every call so a version update takes effect without a restart.
    The template itself is cached by (prompt_id, prompt_version).
    """
    prompt_id      = get_ssm_parameter(f"/{app_name}/{ENV}/bedrock/prompt_id",      with_decryption=False)
    prompt_version = get_ssm_parameter(f"/{app_name}/{ENV}/bedrock/prompt_version",  with_decryption=False)
    return _fetch_prompt_template(prompt_id, prompt_version)