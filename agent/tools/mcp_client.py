"""
mcp_client.py — MCP Gateway Tool Client (AWS_IAM auth)
========================================================
Connects to AgentCore MCP Gateway using AWS SigV4 signing.

Gateway uses --authorizer-type AWS_IAM so the agent signs requests
using its execution role credentials (available via boto3 on AgentCore).

TOOLS REGISTERED IN GATEWAY:
  search_tool     → search_lambda    (Pinecone)
  graph_tool      → graph_lambda     (Neo4j)
  ask_user_input  → hitl_lambda      (HITL)
  summariser_tool → summariser_lambda (GPT-4o-mini)

HOW CALLING WORKS:
  Agent (on AgentCore) has IAM execution role
  → Role has permission: bedrock-agentcore:InvokeGateway
  → Calls Gateway URL with SigV4 signed request
  → Gateway routes to correct Lambda
  → Lambda returns result
"""

import json
import logging
import os
from typing import Optional, List

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


def _get_gateway_url() -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=f"{SSM_PREFIX}/mcp/gateway_url")["Parameter"]["Value"]


# ── SigV4 signed HTTP call to MCP Gateway ────────────────────────────────

def _call_gateway(tool_name: str, tool_input: dict) -> str:
    """
    Call MCP Gateway tool with AWS SigV4 signing.
    The agent's execution role credentials are used automatically by boto3.
    """
    import urllib.request

    gateway_url = _get_gateway_url()
    url         = gateway_url.rstrip("/")

    body    = json.dumps({"name": tool_name, "input": tool_input}).encode()
    session = boto3.Session()
    creds   = session.get_credentials().get_frozen_credentials()

    # Build the request
    request = AWSRequest(
        method="POST",
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    # Sign with SigV4
    SigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(request)

    # Execute
    req = urllib.request.Request(
        url,
        data=body,
        headers=dict(request.headers),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode())

    log.info(f"[MCP] {tool_name} → {str(result)[:100]}")
    return json.dumps(result)


# ── Tool input schemas ────────────────────────────────────────────────────

class SearchInput(BaseModel):
    query: str = Field(description="Search query for clinical trial evidence")
    top_k: int = Field(default=5, description="Number of results (default 5, max 10)")


class GraphInput(BaseModel):
    cypher: str = Field(
        description=(
            "Read-only Cypher query against the Neo4j clinical trials graph.\n\n"
            "SCHEMA:\n"
            "  (Trial)-[:TARGETS]->(Disease)\n"
            "  (Trial)-[:USES]->(Drug)\n"
            "  (Trial)-[:SPONSORED_BY]->(Sponsor)\n"
            "  (Trial)-[:CONDUCTED_IN]->(Country)\n"
            "  (Trial)-[:MEASURES]->(Outcome {type: 'primary'|'secondary'})\n"
            "  (Trial)-[:INCLUDES]->(PatientPopulation)\n"
            "  (Trial)-[:ASSOCIATED_WITH]->(MeSHTerm)\n\n"
            "PROPERTIES:\n"
            "  Trial: nctId, briefTitle, phase, overallStatus, enrollmentCount\n"
            "  Drug: name, type\n"
            "  PatientPopulation: minimumAge, maximumAge, gender, eligibilityCriteria\n\n"
            "RULES: Always toLower() for strings. Always LIMIT 10. No writes."
        )
    )


class HitlInput(BaseModel):
    user_answer:    str                  = Field(default="", description="Leave empty on initial call. Injected on resume.")
    question:       Optional[str]        = Field(default=None, description="Clarifying question to ask the user")
    options:        Optional[List[str]]  = Field(default=None, description="Options derived from search results only")
    allow_freetext: bool                 = Field(default=True, description="Allow typed custom answer")


class SummariserInput(BaseModel):
    chunks: List[str] = Field(description="List of text chunks to synthesise")
    query:  str       = Field(description="Original query to focus the synthesis")


# ── Tool functions ────────────────────────────────────────────────────────

def _search(query: str, top_k: int = 5) -> str:
    return _call_gateway("search_tool", {"query": query, "top_k": top_k})

def _graph(cypher: str) -> str:
    return _call_gateway("graph_tool", {"cypher": cypher})

def _hitl(user_answer: str = "", question: str = None,
          options: list = None, allow_freetext: bool = True) -> str:
    return _call_gateway("ask_user_input", {
        "user_answer": user_answer, "question": question,
        "options": options or [], "allow_freetext": allow_freetext,
    })

def _summarise(chunks: list, query: str = "") -> str:
    return _call_gateway("summariser_tool", {"chunks": chunks, "query": query})


# ── Public API ────────────────────────────────────────────────────────────

async def get_mcp_tools() -> list:
    """Build LangChain tools backed by MCP Gateway Lambda functions."""
    gateway_url = _get_gateway_url()
    log.info(f"[MCP] Gateway URL: {gateway_url}")

    return [
        StructuredTool.from_function(
            func=_search,
            name="search_tool",
            description=(
                "Semantic search over the clinical trials knowledge base (Pinecone). "
                "Use this FIRST for any query about trial results, efficacy, safety, or evidence."
            ),
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=_graph,
            name="graph_tool",
            description=(
                "Execute a read-only Cypher query against the Neo4j clinical trials graph. "
                "Use for: drug-disease relationships, contraindications, sponsors, "
                "patient eligibility, trial locations. "
                "For patient-specific safety questions, call BOTH search_tool AND graph_tool."
            ),
            args_schema=GraphInput,
        ),
        StructuredTool.from_function(
            func=_hitl,
            name="ask_user_input",
            description=(
                "Ask the user a clarifying question when their request is ambiguous. "
                "MANDATORY: Always call search_tool FIRST. "
                "Generate options ONLY from search results — never from training knowledge. "
                "Call this at most ONCE per conversation."
            ),
            args_schema=HitlInput,
        ),
        StructuredTool.from_function(
            func=_summarise,
            name="summariser_tool",
            description=(
                "Synthesise multiple retrieved text chunks into a concise summary. "
                "Use when search_tool returns 3+ chunks on the same topic."
            ),
            args_schema=SummariserInput,
        ),
    ]
