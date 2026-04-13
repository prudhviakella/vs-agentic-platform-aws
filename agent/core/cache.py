"""
cache.py — SemanticCache
=========================
Pinecone-backed semantic cache for agent responses.

Stores and retrieves Q+A pairs by vector similarity.
Used by SemanticCacheMiddleware.

NAMESPACE: cache__  (double underscore to distinguish from data vectors)
"""

import hashlib
import logging
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"
_CACHE_NAMESPACE = "cache__"


class SemanticCache:
    """
    Wraps a Pinecone index to provide semantic cache operations.

    index:  pinecone.Index instance
    domain: "pharma" or "general" (stored as metadata for filtering)
    """

    def __init__(self, index, domain: str = "pharma", openai_api_key: str = ""):
        self._index  = index
        self._domain = domain
        self._openai = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

    def set_openai_client(self, client: AsyncOpenAI):
        """Called by graph.py after initialising the OpenAI client."""
        self._openai = client

    async def _embed(self, text: str) -> list[float]:
        resp = await self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=text[:2000],   # trim very long queries
        )
        return resp.data[0].embedding

    async def lookup(self, query: str, threshold: float = 0.97) -> str | None:
        """
        Search for a cached answer to `query`.
        Returns the cached answer string if similarity >= threshold, else None.
        """
        if self._openai is None:
            return None
        try:
            vector = await self._embed(query)
            result = self._index.query(
                vector=vector,
                top_k=1,
                namespace=_CACHE_NAMESPACE,
                include_metadata=True,
                filter={"domain": {"$eq": self._domain}},
            )
            if result.matches and result.matches[0].score >= threshold:
                match = result.matches[0]
                log.debug(f"[CACHE] hit score={match.score:.4f}")
                return match.metadata.get("answer", "")
        except Exception as exc:
            log.warning(f"[CACHE] lookup error: {exc}")
        return None

    async def store(self, query: str, answer: str) -> None:
        """
        Store a Q+A pair in the cache.
        Uses SHA-256 of the query as the vector ID to avoid duplicates.
        """
        if self._openai is None:
            return
        try:
            vector   = await self._embed(query)
            cache_id = "cache__" + hashlib.sha256(query.encode()).hexdigest()[:16]
            self._index.upsert(
                vectors=[{
                    "id":       cache_id,
                    "values":   vector,
                    "metadata": {
                        "query":  query[:500],
                        "answer": answer[:2000],
                        "domain": self._domain,
                    },
                }],
                namespace=_CACHE_NAMESPACE,
            )
        except Exception as exc:
            log.warning(f"[CACHE] store error: {exc}")
