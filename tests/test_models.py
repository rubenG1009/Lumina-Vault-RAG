from lumina.embeddings import OfflineEmbedder, SemanticEmbedder
from lumina.ingestion import ingest_bytes
from lumina.retrieval import Retriever
from lumina.store import MemoryStore


class SemanticStub:
    model = SemanticEmbedder.model
    dimensions = SemanticEmbedder.dimensions

    def embed(self, text):
        return [1.0] + [0.0] * (self.dimensions - 1)

    def embed_query(self, text):
        return self.embed(text)


def test_model_switch_reindexes_and_separates_vector_dimensions():
    store = MemoryStore()
    chunks = ingest_bytes(b"Audit logs retention is 30 days.", "policy.md")
    hashed = OfflineEmbedder()
    semantic = SemanticStub()
    assert store.replace(chunks, [hashed.embed(c.text) for c in chunks], hashed.model)
    assert Retriever(store, hashed).search("audit logs retention")
    assert store.replace(chunks, [semantic.embed(c.text) for c in chunks], semantic.model)
    assert not Retriever(store, hashed).search("audit logs retention")
    assert Retriever(store, semantic).search("audit logs retention")
    assert store.revision() == 2
    assert store.delete("policy.md")
    assert not store.delete("policy.md")
