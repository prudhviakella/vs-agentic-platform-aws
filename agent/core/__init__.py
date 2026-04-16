"""
vs-agent-core — Vidya Sankalp Agent Core Package
==================================================
Domain-agnostic foundation shared across all VS agents.

Provides:
  core.aws              — AWS credential fetching (SSM, Secrets Manager, Bedrock)
  core.cache            — Pinecone-backed SemanticCache
  core.pinecone_store   — LangGraph BaseStore adapter for Pinecone
  core.middleware.base  — BaseAgentMiddleware
  core.middleware.tracer          — cross-cutting observability
  core.middleware.semantic_cache  — cache lookup/store hooks
  core.middleware.episodic_memory — per-user long-term memory hooks

No domain-specific logic lives here. PII rules, toxic patterns, guardrails,
tools, and prompts belong in the consuming agent package.
"""
