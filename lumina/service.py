"""Application boundary shared by API, CLI, MCP and evaluations."""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

from lumina.answers import ANSWER_POLICY, Answer, eligible, excerpts, semantic_excerpts, verify
from lumina.cache import NullCache
from lumina.embeddings import OfflineEmbedder, SemanticEmbedder
from lumina.ingestion import ingest_bytes
from lumina.models import Hit
from lumina.providers import FakeProvider, QuoteProvider, configured_provider
from lumina.retrieval import OverlapReranker, Retriever
from lumina.store import PostgresStore, Store

logger = logging.getLogger("lumina")


def cache_key(prefix: str, *values: object) -> str:
    raw = json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    return prefix + ":" + hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class AnswerResult:
    answer: Answer
    provider: str
    cache_hit: bool
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        return {"answer": self.answer.to_dict(), "provider": self.provider,
                "cache_hit": self.cache_hit, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "estimated_cost_usd": self.estimated_cost_usd}


class LuminaService:
    def __init__(self, store: Store, cache=None, provider: QuoteProvider | None = None,
                 rerank: bool = False, embedder=None):
        self.store = store
        self.cache = cache or NullCache()
        self.provider = provider or FakeProvider()
        self.embedder = embedder or OfflineEmbedder()
        self.retriever = Retriever(store, self.embedder,
                                   OverlapReranker() if rerank else None)

    def ingest(self, data: bytes, source: str, metadata: dict | None = None) -> dict:
        start = time.monotonic()
        chunks = ingest_bytes(data, source, metadata)
        vectors = []
        cache_hits = 0
        for chunk in chunks:
            key = cache_key("embedding", self.embedder.model, chunk.text)
            vector = self.cache.get(key)
            if vector is None:
                vector = self.embedder.embed(chunk.text)
                self.cache.set(key, vector, 86400)
            else:
                cache_hits += 1
            vectors.append(vector)
        changed = self.store.replace(chunks, vectors, self.embedder.model)
        logger.info(json.dumps({"event": "ingest", "source": source, "chunks": len(chunks),
                                "changed": changed, "cache_hits": cache_hits,
                                "latency_ms": round((time.monotonic()-start)*1000, 2)}))
        return {"source": source, "chunks": len(chunks), "changed": changed,
                "revision": self.store.revision(), "embedding_cache_hits": cache_hits}

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               source: str | None = None) -> list[Hit]:
        start = time.monotonic()
        hits = self.retriever.search(query, k, mode, source)
        logger.info(json.dumps({"event": "search", "mode": mode, "hits": len(hits),
                                "latency_ms": round((time.monotonic()-start)*1000, 2)}))
        return hits

    def _evidence(self, query: str, hits: list[Hit]):
        safe_chunks = [h.chunk for h in hits]
        lexical = excerpts(query, safe_chunks)
        semantic = self.embedder.model == SemanticEmbedder.model and not lexical
        candidates = (semantic_excerpts(query, safe_chunks, self.embedder) if semantic
                      else lexical)
        chunks, reason = eligible(query, hits, candidates)
        valid_ids = {chunk.id for chunk in chunks}
        return chunks, reason, [s for s in candidates if s.chunk_id in valid_ids], semantic

    def delete(self, source: str) -> dict:
        if not source.strip():
            raise ValueError("Source is required")
        deleted = self.store.delete(source)
        return {"source": source, "deleted": deleted, "revision": self.store.revision()}

    def answer(self, query: str, k: int = 5, mode: str = "hybrid",
               source: str | None = None) -> AnswerResult:
        start = time.monotonic()
        # Validate parameters before consulting a cache.
        if (not query.strip() or len(query) > 2000 or not 1 <= k <= 20
                or mode not in {"hybrid", "vector"}):
            raise ValueError("Invalid query, k or mode")
        key = cache_key("answer", ANSWER_POLICY, self.store.revision(), self.embedder.model,
                        self.provider.name, query, k, mode, source)
        cached = self.cache.get(key)
        if cached is not None:
            from lumina.answers import Citation

            raw = cached["answer"]
            answer = Answer(raw["text"], raw["refused"], raw["reason"],
                            [Citation(**c) for c in raw["citations"]])
            logger.info(json.dumps({"event": "answer", "provider": self.provider.name,
                                    "embedding_model": self.embedder.model, "cache_hit": True,
                                    "refused": answer.refused,
                                    "latency_ms": round((time.monotonic()-start)*1000, 2)}))
            return AnswerResult(answer, self.provider.name, True, 0, 0, 0.0)
        hits = self.search(query, k, mode, source)
        chunks, reason, candidates, semantic = self._evidence(query, hits)
        if reason:
            result = AnswerResult(Answer(reason=reason), self.provider.name, False, 0, 0, 0.0)
        else:
            picked = self.provider.select(query, chunks, candidates)
            answer = verify(query, picked.selections, chunks,
                            set(candidates) if semantic else None)
            in_rate = float(os.getenv("LUMINA_INPUT_USD_PER_MILLION", "0"))
            out_rate = float(os.getenv("LUMINA_OUTPUT_USD_PER_MILLION", "0"))
            cost = (picked.input_tokens * in_rate + picked.output_tokens * out_rate) / 1_000_000
            result = AnswerResult(answer, self.provider.name, False, picked.input_tokens,
                                  picked.output_tokens, round(cost, 8))
        self.cache.set(key, result.to_dict(), 300)
        logger.info(json.dumps({"event": "answer", "provider": self.provider.name,
                                "embedding_model": self.embedder.model,
                                "refused": result.answer.refused,
                                "reason": result.answer.reason, "cache_hit": False,
                                "input_tokens": result.input_tokens,
                                "output_tokens": result.output_tokens,
                                "estimated_cost_usd": result.estimated_cost_usd,
                                "latency_ms": round((time.monotonic()-start)*1000, 2)}))
        return result

    def agent(self, query: str, max_steps: int = 4) -> dict:
        """Bounded local tool workflow: search_docs -> get_chunk -> verified answer."""
        if not 2 <= max_steps <= 6:
            raise ValueError("max_steps must be 2–6")
        hits = self.search(query, k=min(5, max_steps - 1))
        trace = [{"tool": "search_docs", "returned": len(hits)}]
        fetched = []
        for hit in hits[:max_steps - 1]:
            chunk = self.store.get(hit.chunk.id)
            if chunk:
                fetched.append(Hit(chunk, hit.score))
                trace.append({"tool": "get_chunk", "chunk_id": chunk.id})
        chunks, reason, candidates, semantic = self._evidence(query, fetched)
        if reason:
            answer = Answer(reason=reason)
        else:
            picked = self.provider.select(query, chunks, candidates)
            answer = verify(query, picked.selections, chunks,
                            set(candidates) if semantic else None)
        return {"answer": answer.to_dict(), "steps": trace, "max_steps": max_steps}


def configured_service() -> LuminaService:
    from lumina.cache import RedisCache

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is required")
    redis_url = os.getenv("REDIS_URL")
    backend = os.getenv("LUMINA_EMBEDDING_BACKEND", "hash")
    if backend not in {"hash", "semantic"}:
        raise ValueError("LUMINA_EMBEDDING_BACKEND must be hash or semantic")
    embedder = SemanticEmbedder() if backend == "semantic" else OfflineEmbedder()
    return LuminaService(PostgresStore(dsn), RedisCache(redis_url) if redis_url else NullCache(),
                         configured_provider(), os.getenv("LUMINA_RERANK") == "overlap",
                         embedder)
