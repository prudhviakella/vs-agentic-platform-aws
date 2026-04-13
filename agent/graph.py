"""
graph.py — LangGraph Agent Builder
====================================
Same logic as local vs-agentic-platform but uses:
  - MCP Gateway tools (instead of in-process functions)
  - RDS Postgres checkpointer (instead of local Postgres)
  - Secrets from Secrets Manager (instead of env vars)
"""

import json
import logging
import os
from functools import lru_cache

import boto3
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver

from agent.prompt import build_system_prompt
from agent.middleware import build_middleware_stack
from agent.core.cache import SemanticCache
from agent.core.pinecone_store import PineconeStore
from agent.core.agent import MiddlewareAgent

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


@lru_cache(maxsize=1)
def _get_openai_key() -> str:
    sm = boto3.client("secretsmanager", region_name=REGION)
    return json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"])["api_key"]


@lru_cache(maxsize=1)
def _get_postgres_conn() -> str:
    """
    Build Postgres connection URL from individual secret fields.

    The secret is stored as:
        {"username": "postgres", "password": "...", "host": "...", "port": "5432", "dbname": "clinical_agent"}

    NOT as a single "connection_string" field — reading that key causes a
    KeyError at runtime. We build the URL manually using quote_plus on the
    password to handle special characters safely.
    """
    from urllib.parse import quote_plus
    sm     = boto3.client("secretsmanager", region_name=REGION)
    secret = json.loads(
        sm.get_secret_value(SecretId=f"{SSM_PREFIX}/postgres")["SecretString"]
    )
    return (
        f"postgresql://{secret['username']}:{quote_plus(secret['password'])}"
        f"@{secret['host']}:{secret.get('port', '5432')}/{secret['dbname']}"
    )


@lru_cache(maxsize=1)
def _get_pinecone_key() -> str:
    sm = boto3.client("secretsmanager", region_name=REGION)
    return json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/pinecone")["SecretString"])["api_key"]


@lru_cache(maxsize=1)
def _get_pinecone_index_name() -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=f"{SSM_PREFIX}/pinecone/cache_index_name")["Parameter"]["Value"]


async def build_agent(tools: list):
    """
    Build the LangGraph clinical trial agent for AgentCore.

    Args:
        tools: list of LangChain StructuredTools from MCP Gateway

    Returns:
        Compiled LangGraph agent with Postgres checkpointer + middleware stack
    """
    log.info("[GRAPH] Building agent")

    openai_key = _get_openai_key()

    # ── LLM — streaming=True for token-level SSE ─────────────────────────
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=openai_key,
        streaming=True,
    )
    safety_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_key,
    )

    # ── System prompt from Bedrock Prompt Management ──────────────────────
    system_prompt = build_system_prompt(domain="pharma")

    # ── Postgres checkpointer (RDS) ───────────────────────────────────────
    postgres_conn = _get_postgres_conn()
    checkpointer  = PostgresSaver.from_conn_string(postgres_conn)
    checkpointer.setup()
    log.info("[GRAPH] Postgres checkpointer ready")

    # ── Pinecone for cache + episodic memory ──────────────────────────────
    from pinecone import Pinecone
    pc    = Pinecone(api_key=_get_pinecone_key())
    index = pc.Index(_get_pinecone_index_name())

    # Pass the OpenAI async client to cache and store so they can embed queries
    from openai import AsyncOpenAI
    oai_async = AsyncOpenAI(api_key=openai_key)

    cache = SemanticCache(index=index, domain="pharma")
    cache.set_openai_client(oai_async)

    store = PineconeStore(index=index)
    store.set_openai_client(oai_async)

    # ── Middleware stack (same 9 layers as local) ─────────────────────────
    middleware = build_middleware_stack(
        cache=cache,
        store=store,
        safety_llm=safety_llm,
    )

    # ── Build agent ───────────────────────────────────────────────────────
    agent = MiddlewareAgent.create(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        middleware=middleware,
    )

    log.info(f"[GRAPH] Agent ready  tools={[t.name for t in tools]}")
    return agent
