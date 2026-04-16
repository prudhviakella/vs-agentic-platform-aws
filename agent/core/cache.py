"""
cache.py — SemanticCache
=========================
Simple semantic cache backed by Pinecone.

How it works:
  STORE  — embed the user question, upsert vector to Pinecone with the
           LLM answer stored in metadata. Set expires_at for TTL.
  LOOKUP — embed the incoming question, query Pinecone for the nearest
           vector. If similarity >= threshold and entry is not expired,
           return the cached answer. Agent call is skipped entirely.

One instance per domain (pharma / general) — different namespaces and
thresholds prevent cross-domain cache pollution.
"""

import hashlib
import logging
import time
from typing import Any, Optional

from langchain_openai import OpenAIEmbeddings

log = logging.getLogger(__name__)


class SemanticCache:
    """
    Args:
        index:                Connected Pinecone Index.
        embedder:             OpenAIEmbeddings instance.
        similarity_threshold: Cosine similarity floor for a cache HIT.
                              pharma=0.97 (strict), general=0.88 (relaxed).
        namespace:            Pinecone namespace — one per domain.
    """

    def __init__(
        self,
        index:                Any,
        embedder:             OpenAIEmbeddings,
        similarity_threshold: float = 0.88,
        namespace:            str   = "cache_general",
    ):
        self._index     = index
        self._embedder  = embedder
        self.threshold  = similarity_threshold
        self._namespace = namespace
        log.info(f"[CACHE] ready  namespace={namespace}  threshold={similarity_threshold}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(self, question: str, user_id: str) -> Optional[str]:
        """
        Check if a similar question is cached for this user.
        Returns the cached answer string on HIT, None on MISS.
        Filters by user_id and excludes expired entries via expires_at.
        """
        log.debug(f"[CACHE] lookup start  user={user_id}  namespace={self._namespace}  question='{question[:80]}'")

        try:
            # Step 1 — embed the incoming question
            t0     = time.perf_counter()
            vector = self._embedder.embed_query(question)
            log.debug(f"[CACHE] embed done  latency={int((time.perf_counter()-t0)*1000)}ms")

            # Step 2 — query Pinecone with user_id + TTL filter
            t1   = time.perf_counter()
            resp = self._index.query(
                vector=vector,
                top_k=1,
                namespace=self._namespace,
                include_metadata=True,
                filter={
                    "user_id":    {"$eq": user_id},
                    "expires_at": {"$gt": time.time()},
                },
            )
            log.debug(f"[CACHE] pinecone query done  latency={int((time.perf_counter()-t1)*1000)}ms")

            # Step 3 — evaluate the best match
            matches = resp.get("matches", [])
            if not matches:
                log.info(f"[CACHE] MISS  reason=no_matches  user={user_id}  namespace={self._namespace}")
                return None

            score  = float(matches[0].get("score", 0.0))
            answer = matches[0].get("metadata", {}).get("answer", "")

            log.debug(f"[CACHE] best match  score={score:.4f}  threshold={self.threshold}  has_answer={bool(answer)}")

            if score >= self.threshold and answer:
                log.info(
                    f"[CACHE] HIT  score={score:.4f}  threshold={self.threshold}"
                    f"  user={user_id}  namespace={self._namespace}"
                    f"  answer_len={len(answer)}"
                )
                return answer

            log.info(
                f"[CACHE] MISS  reason=below_threshold"
                f"  score={score:.4f}  threshold={self.threshold}"
                f"  user={user_id}  namespace={self._namespace}"
            )
            return None

        except Exception as exc:
            log.warning(f"[CACHE] lookup failed  user={user_id}  error={exc} — treating as MISS")
            return None

    def store(self, question: str, answer: str, user_id: str, ttl: int = 3_600) -> None:
        """
        Cache the question → answer pair for this user with a TTL.
        Vector ID is MD5(namespace + user_id + question) — deterministic so
        re-asking the same question overwrites the old entry instead of
        creating a duplicate.
        Called from a background thread — response already returned to the user.
        """
        log.debug(
            f"[CACHE] store start  user={user_id}  namespace={self._namespace}"
            f"  question='{question[:80]}'  answer_len={len(answer)}  ttl={ttl}s"
        )

        try:
            # Step 1 — embed the question
            t0        = time.perf_counter()
            vector    = self._embedder.embed_query(question)
            log.debug(f"[CACHE] embed done  latency={int((time.perf_counter()-t0)*1000)}ms")

            # Step 2 — deterministic vector ID: same user + question = same ID (overwrite)
            vector_id = hashlib.md5(
                f"{self._namespace}{user_id}{question}".encode()
            ).hexdigest()[:16]
            log.debug(f"[CACHE] vector_id={vector_id}")

            # Step 3 — upsert to Pinecone
            expires_at = time.time() + ttl
            t1         = time.perf_counter()
            self._index.upsert(
                vectors=[{
                    "id":     vector_id,
                    "values": vector,
                    "metadata": {
                        "user_id":    user_id,
                        "answer":     answer[:8_000],   # Pinecone metadata value limit
                        "expires_at": expires_at,
                        "created_at": time.time(),
                    },
                }],
                namespace=self._namespace,
            )
            log.debug(f"[CACHE] upsert done  latency={int((time.perf_counter()-t1)*1000)}ms")
            log.info(
                f"[CACHE] stored  id={vector_id}  user={user_id}"
                f"  namespace={self._namespace}  ttl={ttl}s"
                f"  expires_at={int(expires_at)}"
            )

        except Exception as exc:
            log.warning(f"[CACHE] store failed  user={user_id}  error={exc}")

    def delete_expired(self) -> None:
        """
        Delete all expired vectors from the Pinecone namespace.
        Pinecone filters only hide expired entries on lookup — they accumulate
        in the index until explicitly deleted. Call this periodically to keep
        the index clean (e.g. scheduled Lambda, startup hook).
        """
        log.info(f"[CACHE] delete_expired start  namespace={self._namespace}")
        try:
            t0 = time.perf_counter()
            self._index.delete(
                filter={"expires_at": {"$lt": time.time()}},
                namespace=self._namespace,
            )
            log.info(
                f"[CACHE] delete_expired done  namespace={self._namespace}"
                f"  latency={int((time.perf_counter()-t0)*1000)}ms"
            )
        except Exception as exc:
            log.warning(f"[CACHE] delete_expired failed  namespace={self._namespace}  error={exc}")

    @property
    def namespace(self) -> str:
        return self._namespace