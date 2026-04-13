"""
handler.py — search_lambda
===========================
MCP Lambda tool: semantic search over clinical trials (Pinecone).

Registered in MCP Gateway as tool name: search_tool
Input:  {"query": "...", "top_k": 5}
Output: {"results": [...], "count": N, "source": "pinecone"}

Secrets in AWS Secrets Manager:
  /vs-agentcore/prod/pinecone  -> {"api_key": "..."}
  /vs-agentcore/prod/openai    -> {"api_key": "..."}

SSM Parameters:
  /vs-agentcore/prod/pinecone/clinical_trials_index  -> index name
"""

import json
import logging
import os
from functools import lru_cache

import boto3

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")

# ── Lazy singletons (warm Lambda reuse) ──────────────────────────────────
_pc_index   = None
_oai_client = None


@lru_cache(maxsize=1)
def _get_secrets() -> dict:
    sm = boto3.client("secretsmanager", region_name=REGION)
    pc_secret  = json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/pinecone")["SecretString"])
    oai_secret = json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"])
    ssm        = boto3.client("ssm", region_name=REGION)
    index_name = ssm.get_parameter(Name=f"{SSM_PREFIX}/pinecone/clinical_trials_index")["Parameter"]["Value"]
    return {
        "pinecone_api_key": pc_secret["api_key"],
        "openai_api_key":   oai_secret["api_key"],
        "index_name":       index_name,
    }


def _get_pinecone_index():
    global _pc_index
    if _pc_index is None:
        from pinecone import Pinecone
        secrets   = _get_secrets()
        pc        = Pinecone(api_key=secrets["pinecone_api_key"])
        _pc_index = pc.Index(secrets["index_name"])
        log.info(f"[SEARCH] Connected to Pinecone index '{secrets['index_name']}'")
    return _pc_index


def _get_openai():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        _oai_client = OpenAI(api_key=_get_secrets()["openai_api_key"])
    return _oai_client


def _embed(text: str) -> list[float]:
    resp = _get_openai().embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def _search(query: str, top_k: int = 5) -> list[dict]:
    vector = _embed(query)
    index  = _get_pinecone_index()
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace="clinical-trials",
        include_metadata=True,
    )
    chunks = []
    for match in result.matches:
        meta = match.metadata or {}
        chunks.append({
            "score":      round(match.score, 4),
            "text":       meta.get("text", ""),
            "source":     meta.get("source", ""),
            "trial_id":   meta.get("trial_id", ""),
            "breadcrumb": meta.get("breadcrumb", ""),
        })
    return chunks


# ── Lambda handler ────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    MCP tool handler — called by AgentCore Gateway.
    event = {"query": "...", "top_k": 5}
    """
    log.info(f"[SEARCH] event={json.dumps(event)[:150]}")

    query = event.get("query", "").strip()
    top_k = int(event.get("top_k", 5))

    if not query:
        return {"results": [], "error": "query is required"}

    try:
        results = _search(query, top_k)
        log.info(f"[SEARCH] {len(results)} chunks  query='{query[:60]}'")
        return {
            "results": results,
            "count":   len(results),
            "source":  "pinecone/clinical-trials-index",
        }
    except Exception as exc:
        log.exception(f"[SEARCH] Error: {exc}")
        return {"results": [], "error": str(exc)}
