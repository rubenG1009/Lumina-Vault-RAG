"""Local stdio MCP surface for document retrieval."""

from mcp.server.fastmcp import FastMCP

from lumina.service import configured_service

mcp = FastMCP("lumina-vault")


@mcp.tool()
def search_docs(query: str, k: int = 5) -> dict:
    """Search indexed documents and return chunk text with provenance."""
    service = configured_service()
    return {"hits": [h.to_dict() for h in service.search(query, k=k)]}


@mcp.tool()
def get_chunk(chunk_id: str) -> dict:
    """Return an indexed chunk by its exact ID, including source and offsets."""
    service = configured_service()
    chunk = service.store.get(chunk_id)
    return chunk.to_dict() if chunk else {"error": "not_found"}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
