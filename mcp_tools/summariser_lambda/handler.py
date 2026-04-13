"""
handler.py — summariser_lambda
================================
MCP Lambda tool: summarises multiple text chunks using GPT-4o-mini.

Registered in MCP Gateway as tool name: summariser_tool
Input:  {"chunks": ["text1", "text2", ...], "query": "original query"}
Output: {"summary": "...", "count": N}
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

_oai_client = None


@lru_cache(maxsize=1)
def _get_openai_key() -> str:
    sm = boto3.client("secretsmanager", region_name=REGION)
    return json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"])["api_key"]


def _get_openai():
    global _oai_client
    if _oai_client is None:
        from openai import OpenAI
        _oai_client = OpenAI(api_key=_get_openai_key())
    return _oai_client


def handler(event: dict, context) -> dict:
    """
    MCP tool handler.
    event = {"chunks": [...], "query": "..."}
    """
    chunks = event.get("chunks", [])
    query  = event.get("query", "")

    if not chunks:
        return {"summary": "", "error": "chunks is required"}

    combined = "\n\n---\n\n".join(str(c) for c in chunks[:10])

    prompt = (
        f"Synthesise the following clinical trial information to answer: {query}\n\n"
        f"Source chunks:\n{combined[:4000]}\n\n"
        "Provide a concise, accurate synthesis. Cite trial IDs where relevant."
    )

    try:
        oai  = _get_openai()
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
        )
        summary = resp.choices[0].message.content
        log.info(f"[SUMMARISER] Synthesised {len(chunks)} chunks")
        return {"summary": summary, "count": len(chunks)}
    except Exception as exc:
        log.exception(f"[SUMMARISER] Error: {exc}")
        return {"summary": "", "error": str(exc)}
