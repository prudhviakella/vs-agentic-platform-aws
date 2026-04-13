"""
mcp_client.py — MCP Gateway Tool Client
=========================================

WHAT IS STREAMABLE HTTP?
─────────────────────────
Traditional HTTP is request-response: client sends, server replies, connection closes.

Streamable HTTP is different — the server can stream data back in chunks over
a single long-lived HTTP connection. Think of it like a tap left open:

    Client  ──── POST /mcp ────►  Gateway
    Client  ◄─── chunk 1 ────────  (tool result part 1)
    Client  ◄─── chunk 2 ────────  (tool result part 2)
    Client  ◄─── done ───────────  (connection closes)

MCP (Model Context Protocol) uses Streamable HTTP as its transport layer
for version 2025-03-26 onwards. This is why we use transport="streamable_http"
when connecting to the AgentCore MCP Gateway.

If you ever see a 404 or "method not allowed" error, change transport to "sse"
(Server-Sent Events — the older MCP transport used before 2025-03-26).


WHAT IS AWS SIGV4?
───────────────────
Every AWS API call must prove the caller's identity. AWS uses a signing standard
called Signature Version 4 (SigV4) for this.

How it works:

    Step 1 — Take the request (URL + headers + body)
    Step 2 — Hash the body with SHA-256
    Step 3 — Build a canonical string from method + URL + headers + hash
    Step 4 — Sign that string using your AWS secret key
    Step 5 — Add the signature as an Authorization header

    Client ──── POST /mcp ────────────────────────────────────► Gateway
                 Authorization: AWS4-HMAC-SHA256
                   Credential=AKIAIOSFODNN7/20260413/us-east-1/
                              bedrock-agentcore/aws4_request,
                   SignedHeaders=content-type;host;x-amz-date,
                   Signature=abc123...

    Gateway verifies the signature using your IAM role → allows or denies.

On AgentCore Runtime, the agent has an IAM execution role. boto3 picks up
those temporary credentials automatically (same mechanism as Lambda/ECS).
The AwsSigV4 class below signs every outbound HTTP request with those credentials
before it hits the MCP Gateway.

The gateway was created with --authorizer-type AWS_IAM, so it REJECTS any
request that is not correctly signed. Without this class, every tool call
returns 403 Forbidden.


HOW A TOOL CALL FLOWS END TO END:
───────────────────────────────────
  LangGraph agent decides to call search_tool("cancer trial phase 3")
        │
        ▼
  AwsSigV4.auth_flow()  ←── signs the HTTP request with execution role creds
        │
        ▼
  POST https://clinical-trial-mcp-xxx.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp
  Authorization: AWS4-HMAC-SHA256 Credential=...Signature=...
        │
        ▼
  AgentCore MCP Gateway  ←── verifies SigV4 signature, routes to Lambda
        │
        ▼
  search_lambda  ←── embeds query, queries Pinecone, returns top 5 chunks
        │
        ▼
  Tool result returned to agent as a ToolMessage


NOTE ON SESSION LIFECYCLE:
───────────────────────────
Each tool call opens a FRESH MCP session to the gateway (verified from
langchain-mcp-adapters source). This means:
  - No persistent connection to manage
  - AwsSigV4 runs on every call (always fresh, valid signature)
  - The context manager in get_mcp_tools() is only for tool DISCOVERY
    (fetching names, descriptions, schemas) — not for tool execution
"""

import logging
import os

import boto3
import httpx
from botocore.auth import SigV4Auth as BotoSigV4Auth
from botocore.awsrequest import AWSRequest
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


class AwsSigV4(httpx.Auth):
    """
    Custom httpx authentication class that signs every HTTP request with AWS SigV4.

    httpx calls auth_flow() before sending each request.
    We build an AWSRequest (botocore's signing abstraction), sign it with the
    agent's execution role credentials, then copy the signed headers back onto
    the original httpx request.

    boto3 finds credentials automatically from the AgentCore execution role
    — same mechanism used by Lambda, ECS, and EC2 instance metadata.

    httpx.Auth.async_auth_flow() (used by the async MCP client) calls auth_flow()
    by default, so no async override is needed here.
    """

    def auth_flow(self, request: httpx.Request):
        # Get temporary credentials from the AgentCore execution role
        creds = boto3.Session().get_credentials().get_frozen_credentials()

        # Build a botocore AWSRequest — needed for SigV4 signing
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=dict(request.headers),
        )

        # Sign: adds Authorization, x-amz-date, x-amz-security-token headers
        # Service name "bedrock-agentcore" must match the gateway's expected service
        BotoSigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(aws_req)

        # Copy signed headers back onto the httpx request
        for key, value in aws_req.headers.items():
            request.headers[key] = value

        yield request  # httpx sends the now-signed request


async def get_mcp_tools() -> list:
    """
    Connect to the AgentCore MCP Gateway and return all registered tools.

    MultiServerMCPClient handles the MCP protocol:
      1. Opens a Streamable HTTP connection to the gateway
      2. Calls tools/list — gateway returns names, descriptions, input schemas
         for all 4 registered targets (search, graph, hitl, summariser)
      3. Converts each MCP tool definition into a LangChain StructuredTool
      4. Closes the discovery session

    Each SUBSEQUENT tool call (when the agent runs) opens its own fresh
    MCP session — this context manager is only for the initial tool discovery.
    """
    ssm         = boto3.client("ssm", region_name=REGION)
    gateway_url = ssm.get_parameter(
        Name=f"{SSM_PREFIX}/mcp/gateway_url"
    )["Parameter"]["Value"]

    log.info(f"[MCP] Connecting to gateway: {gateway_url}")

    async with MultiServerMCPClient({
        "clinical-trial-tools": {
            # Streamable HTTP = MCP over long-lived HTTP (version 2025-03-26)
            # If this fails with 404, change to "sse" (older transport)
            "transport": "streamable_http",
            "url":       gateway_url,
            # AwsSigV4 signs every HTTP request before it leaves the container
            "auth":      AwsSigV4(),
        }
    }) as client:
        tools = await client.get_tools()

    log.info(f"[MCP] Tools discovered: {[t.name for t in tools]}")
    return tools
