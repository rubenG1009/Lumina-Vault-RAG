import asyncio
import os
from uuid import uuid4

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from lumina.cache import RedisCache
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
    service.ingest(f"# Policy\n{marker} audit logs retention is 30 days.".encode(), source)
    result = service.answer(query)
    assert not result.answer.refused
    assert any(c.source == source for c in result.answer.citations)
    assert service.answer(query).cache_hit
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
