import asyncio
import os
from uuid import uuid4

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lumina.cache import RedisCache
from lumina.embeddings import OfflineEmbedder, SemanticEmbedder
from lumina.evaluate import evaluate_postgres
from lumina.migrate import migrate
from lumina.service import LuminaService
from lumina.store import PostgresStore


@pytest.mark.integration
def test_postgres_pgvector_redis_end_to_end():
    dsn = os.getenv("TEST_DATABASE_URL")
    redis_url = os.getenv("TEST_REDIS_URL")
    if not dsn or not redis_url:
        pytest.skip("Set TEST_DATABASE_URL and TEST_REDIS_URL for integration")
    migrate(dsn)
    service = LuminaService(PostgresStore(dsn), RedisCache(redis_url))
    marker = uuid4().hex
    source = f"integration_{marker}.md"
    query = f"{marker} audit logs retention"
    try:
        service.ingest(f"# Policy\n{marker} audit logs retention is 30 days.".encode(), source)
        result = service.answer(query)
        assert not result.answer.refused
        assert any(c.source == source for c in result.answer.citations)
        assert service.answer(query).cache_hit
    finally:
        service.delete(source)
    identity = f"integration-{uuid4().hex}"
    assert service.cache.allow(identity, limit=1)
    assert not service.cache.allow(identity, limit=1)


@pytest.mark.integration
def test_mcp_stdio_tools():
    if not os.getenv("TEST_DATABASE_URL"):
        pytest.skip("Set TEST_DATABASE_URL for MCP integration")

    async def check():
        env = dict(os.environ)
        env["DATABASE_URL"] = env["TEST_DATABASE_URL"]
        env["REDIS_URL"] = env["TEST_REDIS_URL"]
        server = StdioServerParameters(command="lumina-mcp", args=[], env=env)
        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == {"search_docs", "get_chunk"}
                result = await session.call_tool("search_docs", {"query": "audit logs", "k": 1})
                assert not result.isError
                missing = await session.call_tool("get_chunk", {"chunk_id": "missing"})
                assert not missing.isError

    asyncio.run(check())


@pytest.mark.integration
def test_postgres_evaluation_uses_disposable_schema():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL for database evaluation")
    report = evaluate_postgres(dsn)
    assert report["backend"] == "postgresql-pgvector"
    assert report["metrics"]["hybrid"]["retrieval_cases"] == 8


@pytest.mark.integration
def test_postgres_keeps_embedding_dimensions_separate():
    dsn = os.getenv("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set TEST_DATABASE_URL for model isolation")
    migrate(dsn)

    class SemanticStub:
        model = SemanticEmbedder.model
        dimensions = SemanticEmbedder.dimensions

        def embed(self, text):
            return [1.0] + [0.0] * (self.dimensions - 1)

        def embed_query(self, text):
            return self.embed(text)

    store = PostgresStore(dsn)
    hashed = LuminaService(store, embedder=OfflineEmbedder())
    semantic = LuminaService(store, embedder=SemanticStub())
    marker = uuid4().hex
    hash_source, semantic_source = f"hash_{marker}.md", f"semantic_{marker}.md"
    try:
        hashed.ingest(b"Audit logs retention is 30 days.", hash_source)
        semantic.ingest(b"Audit logs retention is 60 days.", semantic_source)
        assert all(h.chunk.source != semantic_source for h in hashed.search("audit logs retention"))
        assert all(h.chunk.source != hash_source for h in semantic.search("audit logs retention"))
    finally:
        store.delete(hash_source)
        store.delete(semantic_source)
