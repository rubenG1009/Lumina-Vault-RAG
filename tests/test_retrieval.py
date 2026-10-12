import pytest

from lumina.embeddings import OfflineEmbedder
from lumina.ingestion import ingest_bytes
from lumina.models import Hit
from lumina.retrieval import OverlapReranker, Retriever, rrf
from lumina.store import MemoryStore


def corpus():
    store = MemoryStore()
    embed = OfflineEmbedder()
    for source, text in [
        ("retention.md", "# Logs\nAudit logs retention is 30 days."),
        ("support.md", "# Support\nSupport hours are 09:00 to 17:00 UTC."),
        ("security.md", "# Security\nEncryption uses TLS 1.3 in transit."),
    ]:
        chunks = ingest_bytes(text.encode(), source)
        store.replace(chunks, [embed.embed(c.text) for c in chunks])
    return store


def test_rrf_ranks_deduplicates_and_ties():
    chunks = list(corpus().chunks.values())
    a, b, c = [Hit(ch, 999) for ch in chunks]
    fused = rrf([[a, b, a], [b, c]])
    assert [h.chunk.id for h in fused] == [b.chunk.id, a.chunk.id, c.chunk.id]
    assert fused[0].score == pytest.approx(1/62 + 1/61)


@pytest.mark.parametrize("mode", ["hybrid", "vector"])
def test_retrieval_and_source_filter(mode):
    retriever = Retriever(corpus())
    assert retriever.search("audit logs retention", 1, mode)[0].chunk.source == "retention.md"
    hits = retriever.search("audit logs", source="security.md")
    assert all(h.chunk.source == "security.md" for h in hits)
    assert retriever.search("audit logs", source="missing.md") == []


def test_local_reranking_and_input_bounds():
    retriever = Retriever(corpus(), reranker=OverlapReranker())
    assert retriever.search("support hours", 1)[0].chunk.source == "support.md"
    assert retriever.search("the and of") == []
    with pytest.raises(ValueError):
        retriever.search("query", k=21)
