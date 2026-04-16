"""
mcp_client.py — MCP Gateway Tool Client
=========================================

This file handles two things:
  1. Signing every HTTP request to the Bedrock MCP Gateway with AWS SigV4
  2. Discovering the 4 tool definitions from the gateway at agent cold start

If you've ever wondered "how does the LangGraph agent actually call a Lambda
function?", this file is the answer. The full call path is:

  LangGraph agent
    → LangChain StructuredTool (built from MCP tool definition)
      → httpx HTTP request (signed by AwsSigV4)
        → Bedrock MCP Gateway (verifies SigV4, routes by tool name)
          → Lambda function (search/graph/hitl/summariser)
            → Tool result returned as ToolMessage
              → Back into LangGraph agent context


WHAT IS MCP (MODEL CONTEXT PROTOCOL)?
───────────────────────────────────────
MCP is an open protocol (originally by Anthropic) that standardises how AI
models discover and call external tools. Think of it as a USB standard for
AI tools — any tool that speaks MCP can connect to any MCP-compatible AI host.

Before MCP:
  Each AI framework invented its own tool-calling format.
  LangChain tools looked different from OpenAI function calling which looked
  different from Anthropic's tool use. Building a tool for one framework
  meant rewriting it for another.

With MCP:
  Tools expose a standard JSON schema via tools/list.
  AI hosts (like AgentCore) discover tools via tools/list and call them via
  tools/call. One Lambda implementation works with any MCP-compatible AI.

MCP TRANSPORT: STREAMABLE HTTP vs SSE
───────────────────────────────────────
MCP supports two transport layers:

  SSE (Server-Sent Events) — the original transport (pre-2025-03-26):
    Uses two separate HTTP connections:
      - One long-lived GET for the server→client event stream
      - One POST per tool call for client→server messages
    Still works but more complex connection management.

  Streamable HTTP — the current transport (2025-03-26 onwards):
    A single POST connection handles both directions.
    The client sends a JSON-RPC request in the POST body.
    The server streams the response back over the same connection.

    Like a tap left open — one connection, bidirectional:
      Client  ──── POST /mcp ────►  Gateway
      Client  ◄─── chunk 1 ──────  (partial tool result)
      Client  ◄─── chunk 2 ──────  (more data)
      Client  ◄─── done ──────────  (connection closes)

    WHY use Streamable HTTP?
      Simpler connection management — no need to maintain a separate
      long-lived GET connection. Each tool call is a self-contained POST.

  If you ever see 404 or "method not allowed" from the gateway, check which
  transport the gateway version supports. Downgrade to transport="sse" if
  the gateway is running an older MCP version.

WHAT IS AWS SIGV4 AND WHY DO WE NEED IT?
───────────────────────────────────────────
Every AWS API must verify the caller's identity before processing a request.
AWS uses a signing standard called Signature Version 4 (SigV4) for this.

WHY NOT JUST PASS AN API KEY?
  API keys are static secrets — if leaked, they're valid forever until rotated.
  SigV4 signatures are time-bound (valid for ~15 minutes) and tied to IAM roles.
  Even if a signature is intercepted, it can't be replayed later.
  IAM roles also give you fine-grained permissions (this role can only invoke
  the gateway) and full audit logs in CloudTrail.

HOW SIGV4 WORKS (the 5-step signing algorithm):
  Step 1 — Canonicalize the request:
            method + URL path + query string + headers + SHA-256(body)
            → a deterministic "canonical request" string

  Step 2 — Build the string to sign:
            "AWS4-HMAC-SHA256" + datetime + scope + SHA-256(canonical_request)
            scope = "20260413/us-east-1/bedrock-agentcore/aws4_request"

  Step 3 — Derive the signing key:
            HMAC(HMAC(HMAC(HMAC("AWS4" + secret_key, date), region), service), "aws4_request")
            Each nested HMAC narrows the key's scope to a specific date/region/service.

  Step 4 — Compute the signature:
            HMAC(signing_key, string_to_sign)  → hex string

  Step 5 — Add the Authorization header:
            Authorization: AWS4-HMAC-SHA256
              Credential=AKIAIOSFODNN7/20260413/us-east-1/bedrock-agentcore/aws4_request,
              SignedHeaders=content-type;host;x-amz-date,
              Signature=a4c5d6e7f8...

  The gateway receives this header, derives the same signature using the
  IAM role's credentials, and compares. If they match → 200 OK. If not → 403.

WHERE DO THE CREDENTIALS COME FROM IN AGENTCORE?
  AgentCore injects temporary IAM credentials into the container at runtime
  via the execution role (vs-agentcore-agent-role). boto3 picks them up
  automatically from the environment variables AWS_ACCESS_KEY_ID,
  AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN — the same mechanism used by
  Lambda, ECS, and EC2 instance profiles. We never hardcode credentials.

FULL END-TO-END TOOL CALL FLOW:
─────────────────────────────────

  1. LangGraph agent decides to call tool-search___search_tool("cancer trials")

  2. LangChain calls the StructuredTool.invoke() — which calls httpx to POST:
       POST https://vs-agentcore-mcp-xxx.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp
       Body: {"jsonrpc":"2.0","method":"tools/call","params":{"name":"tool-search___search_tool","arguments":{"query":"cancer trials","top_k":5}}}

  3. AwsSigV4.auth_flow() intercepts the httpx request BEFORE it's sent:
       - Gets execution role credentials from boto3.Session()
       - Builds AWSRequest (botocore's signing abstraction)
       - BotoSigV4Auth adds Authorization, x-amz-date, x-amz-security-token headers
       - Copies signed headers back onto the httpx request
       - Yields the signed request → httpx sends it

  4. Bedrock MCP Gateway receives the signed request:
       - Verifies SigV4 signature against the vs-agentcore-agent-role credentials
       - Routes the tool call to the vs-agentcore-search-tool Lambda (by tool name)
       - Lambda executes: embed query → Pinecone search → return top 5 chunks
       - Gateway streams the Lambda response back over the HTTP connection

  5. StructuredTool gets the result as a dict → LangChain wraps it as a ToolMessage
       → Goes back into LangGraph state → LLM sees the retrieved chunks in context

SESSION LIFECYCLE — WHY WE DON'T MAINTAIN A PERSISTENT CONNECTION:
─────────────────────────────────────────────────────────────────────
  Each tool call opens a FRESH MCP session (verified from langchain-mcp-adapters
  source code). This is intentional — Streamable HTTP is stateless at the
  session level. There is no "MCP session ID" to maintain between tool calls.

  Consequence: AwsSigV4 runs on EVERY tool call (fresh signature every time).
  This is correct — SigV4 signatures are short-lived and should not be reused.

  The context manager in get_mcp_tools() is only for the initial tool DISCOVERY
  at cold start — it fetches the tool schemas once and closes the session.
  When the agent actually RUNS the tools, each call opens its own fresh session.
"""

import logging
import os

import boto3
import httpx
from botocore.auth import SigV4Auth as BotoSigV4Auth
from botocore.awsrequest import AWSRequest
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)

# Read region and SSM prefix from environment (set by deploy.sh in AgentCore runtime env vars)
# Defaults allow local testing without environment variables set
REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore/prod")


class AwsSigV4(httpx.Auth):
    """
    httpx authentication class that signs every outbound HTTP request with AWS SigV4.

    HOW httpx.Auth WORKS:
      httpx calls auth_flow() as a generator before sending each request.
      The generator yields the (modified) request back to httpx.
      httpx then sends the yielded request and passes the response back
      into the generator via send(). We don't need the response here so
      we just yield once and return.

    WHY inherit from httpx.Auth and not just sign in a function?
      The MultiServerMCPClient accepts an auth= parameter that must be
      an httpx.Auth instance. This plugs our SigV4 signing seamlessly into
      httpx's request pipeline — every HTTP call (tools/list, tools/call,
      keepalive pings) is signed automatically without any extra code.

    WHY httpx.Auth.async_auth_flow() is NOT needed:
      httpx.Auth provides a default async_auth_flow() that wraps the
      synchronous auth_flow() generator. Since the MCP client is async,
      it calls async_auth_flow() — which calls our synchronous auth_flow()
      internally. boto3 credential fetching is fast enough (< 1ms from
      cached credentials) that the sync call doesn't block the event loop.
    """

    def auth_flow(self, request: httpx.Request):
        # ── Step 1: Get execution role credentials ────────────────────────
        # boto3.Session() reads credentials from environment variables
        # injected by AgentCore: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
        # AWS_SESSION_TOKEN. get_frozen_credentials() returns a snapshot
        # (FrozenCredentials) — thread-safe, no risk of credentials rotating
        # mid-signing. The temporary credentials are valid for ~1 hour and
        # are automatically refreshed by the AgentCore runtime.
        creds = boto3.Session().get_credentials().get_frozen_credentials()

        # ── Step 2: Build botocore AWSRequest ─────────────────────────────
        # botocore's AWSRequest is the signing abstraction — it's not an
        # actual HTTP request, just a data structure that SigV4Auth can sign.
        # We copy method, URL, body, and headers from the httpx request.
        aws_req = AWSRequest(
            method  = request.method,
            url     = str(request.url),
            data    = request.content,     # raw bytes body (JSON-RPC payload)
            headers = dict(request.headers),
        )

        # ── Step 3: Sign the request ──────────────────────────────────────
        # BotoSigV4Auth.add_auth() runs the full 5-step SigV4 algorithm and
        # adds three headers to aws_req:
        #   Authorization: AWS4-HMAC-SHA256 Credential=.../Signature=...
        #   x-amz-date: 20260413T103045Z  (timestamp in ISO 8601 format)
        #   x-amz-security-token: ...      (session token for temporary creds)
        #
        # Service name "bedrock-agentcore" must EXACTLY match the service name
        # embedded in the gateway's IAM resource ARN. Using "bedrock" or any
        # other service name produces a signature that the gateway rejects.
        BotoSigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(aws_req)

        # ── Step 4: Copy signed headers back to httpx request ─────────────
        # httpx's request object is immutable except for headers.
        # We can't replace the request, so we update its headers in-place
        # with the three new signed headers from botocore's AWSRequest.
        for key, value in aws_req.headers.items():
            request.headers[key] = value

        # ── Step 5: Yield the signed request ──────────────────────────────
        # Yielding tells httpx "this request is ready to send".
        # httpx sends it and (if we needed to) would send the response back
        # via generator.send(response). We don't need the response here.
        yield request


async def get_mcp_tools() -> list:
    """
    Connect to the Bedrock MCP Gateway and return the 4 tool definitions
    as LangChain StructuredTool objects ready for use with create_agent().

    WHAT THIS FUNCTION DOES:
      1. Reads the gateway URL from SSM Parameter Store
         (/vs-agentcore/prod/mcp/gateway_url)
         Written by deploy.sh step_gateway() after gateway creation.

      2. Creates a MultiServerMCPClient configured with:
         - transport="streamable_http" (MCP 2025-03-26 protocol)
         - auth=AwsSigV4() (SigV4 signing for every request)

      3. Calls client.get_tools() which:
         - Opens a Streamable HTTP connection to the gateway
         - Sends: {"jsonrpc":"2.0","method":"tools/list","params":{}}
         - Gateway responds with all 4 registered tool schemas:
             tool-search___search_tool   (search Lambda)
             tool-graph___graph_tool     (graph Lambda)
             tool-hitl___ask_user_input  (HITL Lambda)
             tool-summariser___summariser_tool (summariser Lambda)
         - MultiServerMCPClient converts each schema into a LangChain
           StructuredTool with name, description, and args_schema

      4. Returns the list of StructuredTool objects.
         These are passed to create_agent(tools=...) in agent.py.

    NOTE ON TOOL NAMING:
      The MCP gateway prefixes tool names with the target name:
        target "tool-search" + tool "search_tool" → "tool-search___search_tool"
      This is why interrupt_on and prompt instructions use the full prefixed name.
      The triple underscore (___) is MCP's separator between target and tool.

    WHY READ GATEWAY URL FROM SSM INSTEAD OF HARDCODING?
      The gateway URL contains a random ID generated at creation time:
        https://vs-agentcore-mcp-aefpwqegtq.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp
      It changes if the gateway is recreated (e.g. after destroy + redeploy).
      Reading from SSM means the container always gets the current URL without
      a code change or image rebuild.

    WHY IS THIS FUNCTION ASYNC?
      MultiServerMCPClient.get_tools() is async because it makes HTTP requests
      to the gateway. async/await lets other coroutines run while waiting for
      the network response — important in the AgentCore container where cold
      start initialisation runs multiple async operations concurrently.
    """
    # Read gateway URL from SSM — written by deploy.sh step_gateway()
    ssm         = boto3.client("ssm", region_name=REGION)
    gateway_url = ssm.get_parameter(
        Name=f"{SSM_PREFIX}/mcp/gateway_url"
    )["Parameter"]["Value"]

    log.info(f"[MCP] Connecting to gateway: {gateway_url}")

    # MultiServerMCPClient accepts a dict of server configs.
    # "clinical-trial-tools" is an arbitrary name for this server group —
    # it's used internally by the client to namespace the discovered tools.
    # transport="streamable_http" selects the MCP 2025-03-26 protocol.
    # auth=AwsSigV4() plugs in our SigV4 signing for every HTTP request.
    #
    # NOTE: langchain-mcp-adapters>=0.2 removed async context manager support.
    # Earlier versions used: async with MultiServerMCPClient(...) as client:
    # Current version: create client directly and call get_tools() directly.
    client = MultiServerMCPClient({
        "clinical-trial-tools": {
            "transport": "streamable_http",
            "url":       gateway_url,
            "auth":      AwsSigV4(),         # signs every HTTP request
        }
    })

    # get_tools() opens a fresh connection, calls tools/list, converts schemas
    # to StructuredTool objects, and closes the discovery session.
    # The returned tools are cached in _agent in app.py for the container lifetime.
    tools = await client.get_tools()

    log.info(f"[MCP] Tools discovered: {[t.name for t in tools]}")
    return tools