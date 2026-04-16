"""
tools/__init__.py
==================
In AWS, tools come from the MCP Gateway via app.py:
    mcp_tools = await get_mcp_tools()
    build_agent(tools=mcp_tools)

ALL_TOOLS is never used in AWS — it is only a fallback for local development
where the actual tool files (search.py, graph.py, etc.) exist in the repo.
Those files are NOT copied here to keep this repo self-contained.
"""

# Safety net: if build_agent(tools=None) is ever called in AWS,
# this prevents an ImportError from the empty module.
ALL_TOOLS = []
MAX_TOOL_CALLS_PER_REQUEST = 10