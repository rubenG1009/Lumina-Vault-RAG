from collections import defaultdict
from typing import Protocol

from lumina.embeddings import OfflineEmbedder, tokens
from lumina.models import Hit
from lumina.store import Store


def rrf(rankings: list[list[Hit]], constant: int = 60) -> list[Hit]:
    """Reciprocal rank fusion with stable ties; each list contributes once per ID."""
    scores: dict[str, float] = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            key = hit.chunk.id
            if key not in seen:
                scores[key] += 1 / (constant + rank)
                chunks[key] = hit.chunk
                seen.add(key)
    return [Hit(chunks[key], score) for key, score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class OverlapReranker:
    """Optional local lexical reranker, not a learned cross-encoder."""
    name = "overlap-v1"

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        terms = set(tokens(query))
        return sorted(hits, key=lambda h: (
            -len(terms & set(tokens(h.chunk.text))) / max(1, len(terms)),
            -h.score, h.chunk.id,
        ))


class Retriever:
    def __init__(self, store: Store, embedder=None, reranker: Reranker | None = None):
        self.store = store
        self.embedder = embedder or OfflineEmbedder()
        self.reranker = reranker

    def search(self, query: str, k: int = 5, mode: str = "hybrid",
               source: str | None = None) -> list[Hit]:
        if not query.strip() or len(query) > 2000 or not 1 <= k <= 20:
            raise ValueError("Query must be 1–2000 characters and k must be 1–20")
        if mode not in {"hybrid", "vector"}:
            raise ValueError("Mode must be hybrid or vector")
        vector = self.embedder.embed(query)
        if not any(vector):
            return []
        vec, lex = self.store.ranked(query, vector, min(100, max(20, k * 4)), source)
        # RRF scores are ranks, never confidence probabilities.
        hits = rrf([vec, lex]) if mode == "hybrid" else vec
        if self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]
