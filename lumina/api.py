"""Local FastAPI application. Bind to loopback unless protected by an external gateway."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from lumina.ingestion import MAX_BYTES
from lumina.service import LuminaService, configured_service


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="hybrid", pattern="^(hybrid|vector)$")
    source: str | None = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    max_steps: int = Field(default=4, ge=2, le=6)


def create_app(service: LuminaService | None = None) -> FastAPI:
    service = service or configured_service()
    app = FastAPI(title="Lumina Vault RAG", version="0.1.0")

    @app.middleware("http")
    async def rate_limit(request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        identity = request.client.host if request.client else "unknown"
        try:
            if not service.cache.allow(identity):
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
        except Exception:
            return JSONResponse({"detail": "Rate limiter unavailable"}, status_code=503)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid_input(request: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/health")
    def health():
        try:
            return {"status": "ok", "revision": service.store.revision(),
                    "redis": service.cache.ping(), "embedding_model": service.embedder.model}
        except Exception as exc:
            raise HTTPException(503, "Dependency unavailable") from exc

    @app.post("/documents")
    async def documents(request: Request, source: str = Query(min_length=1, max_length=300)):
        data = bytearray()
        async for part in request.stream():
            data.extend(part)
            if len(data) > MAX_BYTES:
                raise HTTPException(413, "Document exceeds 10 MiB")
        return service.ingest(bytes(data), source)

    @app.delete("/documents")
    def delete_document(source: str = Query(min_length=1, max_length=300)):
        return service.delete(source)

    @app.post("/search")
    def search(body: SearchRequest):
        hits = service.search(body.query, body.k, body.mode, body.source)
        return {"hits": [hit.to_dict() for hit in hits], "mode": body.mode}

    @app.post("/answer")
    def answer(body: SearchRequest):
        return service.answer(body.query, body.k, body.mode, body.source).to_dict()

    @app.post("/agent")
    def agent(body: AgentRequest):
        return service.agent(body.query, body.max_steps)

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str):
        found = service.store.get(chunk_id)
        if found is None:
            raise HTTPException(404, "Chunk not found")
        return asdict(found)

    return app


def app_factory() -> FastAPI:
    return create_app()
