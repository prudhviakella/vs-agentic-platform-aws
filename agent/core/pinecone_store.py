"""
pinecone_store.py — PineconeStore
==================================
Wraps Pinecone so EpisodicMemoryMiddleware can use it as a store without
knowing anything about Pinecone internals.

The middleware calls store.put() and store.search() -- standard LangGraph
BaseStore methods. We implement those methods using Pinecone underneath.

WHY wrap Pinecone in a BaseStore adapter?
  EpisodicMemoryMiddleware expects any object that has put() and search().
  Wrapping Pinecone means we can swap the storage backend (e.g. swap Pinecone
  for Redis or PostgreSQL) without touching any middleware code.

Namespace:
  LangGraph uses tuples  → ("episodic", "user_abc")
  Pinecone uses strings  → "episodic__user_abc"
  We join with "__" so the parts are always recoverable by splitting on "__".

Vector ID convention:
  Every vector in Pinecone needs a unique ID.
  We use  "episodic__user_abc__entry_id"  (namespace + key).
  Without the namespace prefix, two users with the same entry_id would
  silently overwrite each other's memories.

Metadata stored per vector:
  {
    "namespace":  "episodic__user_abc",
    "key":        "3f7a92bc",
    "text":       "Q: metformin dose for eGFR 25?\nA: 500mg...",
    "ts":         1714000000.0,      <- Unix timestamp for recency sorting
    "created_at": "2024-04-25T...",  <- ISO for LangGraph Item
    "updated_at": "2024-04-25T...",
  }

WHY embed text at write time (not at search time)?
  Episodic entries are written once and searched many times.
  Embedding at write time means each search pays only one embedding call
  (for the query). If we embedded at search time we would need to re-embed
  every stored entry on every search -- very expensive.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import (
    BaseStore, GetOp, Item, ListNamespacesOp,
    Op, PutOp, Result, SearchItem, SearchOp,
)

log = logging.getLogger(__name__)


def _ns(namespace: tuple) -> str:
    """("episodic", "user_abc")  ->  "episodic__user_abc" """
    return "__".join(namespace)


def _vid(namespace_str: str, key: str) -> str:
    """Globally unique Pinecone vector ID -- namespace prefix prevents collisions."""
    return f"{namespace_str}__{key}"


def _to_item(meta: dict, score: float = None) -> SearchItem:
    """Rebuild a LangGraph SearchItem from Pinecone vector metadata."""
    # These keys belong to the Item envelope, not the value payload
    envelope_keys = {"namespace", "key", "created_at", "updated_at"}
    return SearchItem(
        namespace=tuple(meta["namespace"].split("__")),
        key=meta["key"],
        value={k: v for k, v in meta.items() if k not in envelope_keys},
        created_at=datetime.fromisoformat(meta["created_at"]),
        updated_at=datetime.fromisoformat(meta["updated_at"]),
        score=score,
    )


class PineconeStore(BaseStore):
    """
    LangGraph BaseStore backed by Pinecone.

    EpisodicMemoryMiddleware writes memories with put() and reads them
    with search(). Both use the same instance so writes are immediately
    visible on the next turn.
    """

    def __init__(self, index: Any, embedder: OpenAIEmbeddings, top_k: int = 3):
        self._index    = index
        self._embedder = embedder
        self._top_k    = top_k

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    # ── LangGraph calls batch() for every get/put/search/delete ───────────
    # Implementing batch() is enough to satisfy the full BaseStore interface.

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Pinecone client is synchronous -- delegate to batch()
        return self.batch(ops)

    # ── Handlers ──────────────────────────────────────────────────────────

    def _get(self, op: GetOp) -> Item | None:
        """Fetch one item by namespace + key."""
        ns  = _ns(op.namespace)
        vid = _vid(ns, op.key)
        try:
            resp = self._index.fetch(ids=[vid], namespace=ns)
            vec  = resp.get("vectors", {}).get(vid)
            return _to_item(vec["metadata"]) if vec else None
        except Exception as e:
            log.warning(f"[STORE] get failed  id={vid}  err={e}")
            return None

    def _put(self, op: PutOp) -> None:
        """
        Upsert or delete a single item.
        value=None means delete. value=dict means upsert.

        We embed the "text" field at write time so future searches only need
        to embed the query -- not every stored entry on every search.
        """
        ns  = _ns(op.namespace)
        vid = _vid(ns, op.key)

        if op.value is None:
            try:
                self._index.delete(ids=[vid], namespace=ns)
                log.info(f"[STORE] deleted  id={vid}")
            except Exception as e:
                log.warning(f"[STORE] delete failed  id={vid}  err={e}")
            return

        try:
            vector = self._embed(op.value.get("text", op.key))
        except Exception as e:
            log.warning(f"[STORE] embed failed  id={vid}  err={e}")
            return

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "namespace":  ns,
            "key":        op.key,
            "created_at": now,
            "updated_at": now,
            **op.value,   # includes "text", "ts", and any other fields
        }

        try:
            self._index.upsert(
                vectors=[{"id": vid, "values": vector, "metadata": metadata}],
                namespace=ns,
            )
            log.info(f"[STORE] upserted  id={vid}  ns={ns}")
        except Exception as e:
            log.warning(f"[STORE] upsert failed  id={vid}  err={e}")

    def _search(self, op: SearchOp) -> list[SearchItem]:
        """
        Two modes:
          query provided -> semantic search (find most RELEVANT memories)
          query empty    -> recency fetch   (find most RECENT memories)

        EpisodicMemoryMiddleware always passes query=current_question so
        semantic search is the normal path. Recency fetch exists for admin
        or debug use cases.
        """
        ns = _ns(op.namespace_prefix)
        if op.query:
            return self._semantic_search(ns, op.query, op.limit)
        else:
            return self._recent(ns, op.limit)

    def _semantic_search(self, ns: str, query: str, limit: int) -> list[SearchItem]:
        """
        Embed the query and find the most similar episodic memories in Pinecone.
        This is the main search path -- relevance over recency.
        """
        try:
            resp = self._index.query(
                vector=self._embed(query),
                top_k=limit,
                namespace=ns,
                include_metadata=True,
            )
            items = [
                _to_item(m["metadata"], score=m.get("score"))
                for m in resp.get("matches", [])
            ]
            log.info(f"[STORE] semantic search  ns={ns}  hits={len(items)}")
            return items
        except Exception as e:
            log.warning(f"[STORE] semantic search failed  ns={ns}  err={e}")
            return []

    def _recent(self, ns: str, limit: int) -> list[SearchItem]:
        """
        Return the most recent entries by timestamp.

        Pinecone has no native "sort by recency" operation so we list all IDs,
        fetch their metadata, sort by the "ts" field in Python, and slice.
        This is fine because episodic memory per user is small (3-20 entries).
        """
        try:
            all_ids = [id for batch in self._index.list(namespace=ns) for id in batch]
            if not all_ids:
                return []

            vectors = self._index.fetch(ids=all_ids, namespace=ns).get("vectors", {})
            items   = [_to_item(v["metadata"]) for v in vectors.values()]
            items.sort(key=lambda i: i.value.get("ts", 0), reverse=True)

            log.info(f"[STORE] recent fetch  ns={ns}  hits={len(items[:limit])}")
            return items[:limit]
        except Exception as e:
            log.warning(f"[STORE] recent fetch failed  ns={ns}  err={e}")
            return []