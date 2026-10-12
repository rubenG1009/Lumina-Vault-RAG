from fastapi.testclient import TestClient

from lumina.api import create_app
from lumina.cache import NullCache
from lumina.embeddings import SemanticEmbedder
from lumina.service import LuminaService
from lumina.store import MemoryStore


class CountingCache(NullCache):
    def __init__(self):
        self.data = {}

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ttl):
        self.data[key] = value


def test_service_cache_and_revision_invalidation():
    service = LuminaService(MemoryStore(), CountingCache())
    service.ingest(b"Audit logs retention is 30 days.", "policy.md")
    first = service.answer("audit logs retention")
    second = service.answer("audit logs retention")
    assert not first.cache_hit and second.cache_hit
    assert "30 days" in second.answer.text
    service.ingest(b"Audit logs retention is 60 days.", "policy.md")
    third = service.answer("audit logs retention")
    assert not third.cache_hit and "60 days" in third.answer.text


def test_semantic_evidence_can_quote_a_paraphrase():
    class SemanticStub:
        model = SemanticEmbedder.model
        dimensions = SemanticEmbedder.dimensions

        def embed(self, text):
            return [1.0] + [0.0] * (self.dimensions - 1)

        def embed_query(self, text):
            return self.embed(text)

    service = LuminaService(MemoryStore(), embedder=SemanticStub())
    service.ingest(b"Audit logs retention is 30 days.", "policy.md")
    result = service.answer("When are records erased?")
    assert not result.answer.refused
    assert result.answer.citations[0].source == "policy.md"
    assert result.answer.citations[0].quote == "Audit logs retention is 30 days."


def test_api_ingest_search_answer_and_agent():
    client = TestClient(create_app(LuminaService(MemoryStore(), CountingCache())))
    assert client.get("/health").json()["status"] == "ok"
    response = client.post(
        "/documents?source=policy.md", content=b"Audit logs retention is 30 days."
    )
    assert response.status_code == 200 and response.json()["chunks"] == 1
    hits = client.post("/search", json={"query": "audit logs retention"}).json()["hits"]
    assert len(hits) == 1
    chunk_id = hits[0]["chunk"]["id"]
    assert client.get(f"/chunks/{chunk_id}").json()["source"] == "policy.md"
    answer = client.post("/answer", json={"query": "audit logs retention"}).json()
    assert not answer["answer"]["refused"]
    agent = client.post("/agent", json={"query": "audit logs retention"}).json()
    assert not agent["answer"]["refused"]
    assert [step["tool"] for step in agent["steps"]] == ["search_docs", "get_chunk"]
    deleted = client.delete("/documents?source=policy.md").json()
    assert deleted["deleted"]
    assert client.post("/answer", json={"query": "audit logs retention"}).json()[
        "answer"]["refused"]
    assert client.post("/search", json={"query": "x", "mode": "bad"}).status_code == 422


def test_upload_rejects_unsupported_and_oversize():
    client = TestClient(create_app(LuminaService(MemoryStore())))
    assert client.post("/documents?source=file.exe", content=b"hello").status_code == 400
    response = client.post("/documents?source=big.md", content=b"x" * (10 * 1024 * 1024 + 1))
    assert response.status_code == 413
