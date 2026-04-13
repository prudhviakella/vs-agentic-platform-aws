"""
handler.py — graph_lambda
==========================
MCP Lambda tool: Cypher query executor against Neo4j AuraDB.

Registered in MCP Gateway as tool name: graph_tool
Input:  {"cypher": "MATCH (t:Trial)-[:USES]->(d:Drug) WHERE ..."}
Output: {"results": [...], "count": N, "source": "neo4j"}

Secret: /vs-agentcore/prod/neo4j -> {"uri": "...", "user": "...", "password": "..."}

GRAPH SCHEMA:
  (Trial)-[:TARGETS]->(Disease)
  (Trial)-[:USES]->(Drug)
  (Trial)-[:SPONSORED_BY]->(Sponsor)
  (Trial)-[:CONDUCTED_IN]->(Country)
  (Trial)-[:MEASURES]->(Outcome)
  (Trial)-[:INCLUDES]->(PatientPopulation)
  (Trial)-[:ASSOCIATED_WITH]->(MeSHTerm)

WRITE OPERATIONS BLOCKED: DELETE, DETACH, CREATE, MERGE, SET, REMOVE, DROP
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

_FORBIDDEN = {"DELETE", "DETACH", "CREATE", "MERGE", "SET", "REMOVE", "DROP"}

# Lazy singleton
_driver = None


@lru_cache(maxsize=1)
def _get_neo4j_creds() -> dict:
    sm = boto3.client("secretsmanager", region_name=REGION)
    return json.loads(sm.get_secret_value(SecretId=f"{SSM_PREFIX}/neo4j")["SecretString"])


def _get_driver():
    global _driver
    if _driver is None:
        from neo4j import GraphDatabase
        creds   = _get_neo4j_creds()
        _driver = GraphDatabase.driver(
            creds["uri"],
            auth=(creds["user"], creds["password"]),
        )
        log.info(f"[GRAPH] Connected to Neo4j  uri={creds['uri']}")
    return _driver


def _run_query(cypher: str) -> list[dict]:
    with _get_driver().session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def _format_rows(rows: list[dict]) -> list[str]:
    """Format Neo4j rows as readable text for the LLM."""
    chunks = []
    for row in rows:
        parts = []
        for key, value in row.items():
            if value is not None and value != [] and value != "":
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value if v)
                parts.append(f"{key}: {value}")
        if parts:
            chunks.append("\n".join(parts))
    return chunks


# ── Lambda handler ────────────────────────────────────────────────────────

def handler(event: dict, context) -> dict:
    """
    MCP tool handler — executes LLM-generated Cypher against Neo4j.
    event = {"cypher": "MATCH ..."}
    """
    log.info(f"[GRAPH] event keys={list(event.keys())}")

    cypher = event.get("cypher", "").strip()
    if not cypher:
        return {"results": [], "error": "cypher is required"}

    # Block write operations
    upper = cypher.upper()
    for op in _FORBIDDEN:
        if op in upper:
            log.warning(f"[GRAPH] Blocked write op: {op}")
            return {"results": [], "error": f"Write operation '{op}' not permitted. Read-only queries only."}

    try:
        rows   = _run_query(cypher)
        chunks = _format_rows(rows)
        log.info(f"[GRAPH] {len(chunks)} results  cypher='{cypher[:60]}'")
        return {
            "results": chunks,
            "count":   len(chunks),
            "source":  "neo4j/clinical-trials",
        }
    except Exception as exc:
        log.exception(f"[GRAPH] Error: {exc}")
        return {"results": [], "error": str(exc)}
