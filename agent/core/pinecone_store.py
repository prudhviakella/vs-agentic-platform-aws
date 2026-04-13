"""
pinecone_store.py — PineconeStore
===================================
Pinecone-backed store for episodic memory (past Q+A pairs).

Distinct from SemanticCache:
  Cache  → exact answer retrieval (similarity > 0.97)
  Store  → context injection    (similarity > 0.75, looser match)

NAMESPACE: episodic__
"""

import hashlib
import logging
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_EMBEDDING_MODEL  = "text-embedding-3-small"
_EPISODIC_NS      = "episodic__"


class PineconeStore:
    """
    Wraps a Pinecone index for episodic memory operations.
    Used by EpisodicMemoryMiddleware.

    index: pinecone.Index instance
    """

    def __init__(self, index, openai_api_key: str = ""):
        self._index  = index
        self._openai = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

    def set_openai_client(self, client: AsyncOpenAI):
        self._openai = client

    async def _embed(self, text: str) -> list[float]:
        resp = await self._openai.embeddings.create(
            model=_EMBEDDING_MODEL,
            input=text[:2000],
        )
        return resp.data[0].embedding

    async def search(self, query: str, top_k: int = 3, threshold: float = 0.75) -> list[str]:
        """
        Find past Q+A pairs relevant to the current query.
        Returns a list of formatted context strings.
        """
        if self._openai is None:
            return []
        try:
            vector = await self._embed(query)
            result = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=_EPISODIC_NS,
                include_metadata=True,
            )
            memories = []
            for match in result.matches:
                if match.score >= threshold:
                    q = match.metadata.get("question", "")
                    a = match.metadata.get("answer", "")
                    if q and a:
                        memories.append(f"Q: {q}\nA: {a[:500]}")
            log.debug(f"[EPISODIC] found {len(memories)} memories  score_threshold={threshold}")
            return memories
        except Exception as exc:
            log.warning(f"[EPISODIC] search error: {exc}")
            return []

    async def store(self, question: str, answer: str) -> None:
        """Store a Q+A pair in episodic memory."""
        if self._openai is None:
            return
        try:
            vector    = await self._embed(question)
            memory_id = "ep__" + hashlib.sha256(question.encode()).hexdigest()[:16]
            self._index.upsert(
                vectors=[{
                    "id":       memory_id,
                    "values":   vector,
                    "metadata": {
                        "question": question[:500],
                        "answer":   answer[:1000],
                    },
                }],
                namespace=_EPISODIC_NS,
            )
            log.debug(f"[EPISODIC] stored memory id={memory_id}")
        except Exception as exc:
            log.warning(f"[EPISODIC] store error: {exc}")
