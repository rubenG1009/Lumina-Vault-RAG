import hashlib
import json
from typing import Protocol

from lumina.embeddings import EMBEDDING_MODEL, OfflineEmbedder, SemanticEmbedder, tokens
from lumina.models import Chunk, Hit


class Store(Protocol):
    def replace(self, chunks: list[Chunk], embeddings: list[list[float]],
                model: str = EMBEDDING_MODEL) -> bool: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def delete(self, source: str) -> bool: ...
    def revision(self) -> int: ...
    def ranked(self, query: str, vector: list[float], limit: int,
               source: str | None = None,
               model: str = EMBEDDING_MODEL) -> tuple[list[Hit], list[Hit]]: ...


def fingerprint(chunks: list[Chunk], model: str = EMBEDDING_MODEL) -> str:
    return hashlib.sha256(json.dumps(
        {"chunks": [c.to_dict() for c in chunks], "model": model},
        sort_keys=True, ensure_ascii=False
    ).encode()).hexdigest()


def validate_batch(chunks: list[Chunk], embeddings: list[list[float]], model: str) -> None:
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("One embedding per chunk is required")
    if len({c.document_id for c in chunks}) != 1 or len({c.source for c in chunks}) != 1:
        raise ValueError("Ingest one document per transaction")
    dimensions = {EMBEDDING_MODEL: OfflineEmbedder.dimensions,
                  SemanticEmbedder.model: SemanticEmbedder.dimensions}.get(model)
    if dimensions is None:
        raise ValueError("Unsupported embedding model")
    if any(len(v) != dimensions for v in embeddings):
        raise ValueError("Embedding dimensions mismatch")


def _vector(vector: list[float]) -> str:
    return "[" + ",".join(str(float(v)) for v in vector) + "]"


def _chunk(row: dict) -> Chunk:
    return Chunk(**{k: row[k] for k in (
        "id", "document_id", "text", "source", "title", "section", "page", "metadata"
    )}, start=row["start_offset"], end=row["end_offset"])


class PostgresStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def connect(self):
        import psycopg
        from psycopg.rows import dict_row

        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]],
                model: str = EMBEDDING_MODEL) -> bool:
        from psycopg.types.json import Jsonb

        validate_batch(chunks, embeddings, model)
        first = chunks[0]
        mark = fingerprint(chunks, model)
        with self.connect() as conn:
            # Serializes replacement/revision updates, including first-time documents.
            conn.execute("SELECT revision FROM corpus_state FOR UPDATE")
            old = conn.execute("SELECT fingerprint FROM documents WHERE id=%s",
                               (first.document_id,)).fetchone()
            if old and old["fingerprint"] == mark:
                return False
            conn.execute("DELETE FROM documents WHERE id=%s", (first.document_id,))
            conn.execute("INSERT INTO documents VALUES (%s,%s,%s)",
                         (first.document_id, first.source, mark))
            with conn.cursor() as cur:
                cur.executemany("""INSERT INTO chunks (
                    id,document_id,text,source,title,section,page,start_offset,end_offset,
                    metadata,embedding,embedding_model
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector,%s)""", [
                    (c.id, c.document_id, c.text, c.source, c.title, c.section, c.page,
                     c.start, c.end, Jsonb(c.metadata), _vector(v), model)
                    for c, v in zip(chunks, embeddings, strict=True)
                ])
            conn.execute("UPDATE corpus_state SET revision=revision+1")
        return True

    def get(self, chunk_id: str) -> Chunk | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE id=%s", (chunk_id,)).fetchone()
        return _chunk(row) if row else None

    def delete(self, source: str) -> bool:
        with self.connect() as conn:
            conn.execute("SELECT revision FROM corpus_state FOR UPDATE")
            deleted = conn.execute("DELETE FROM documents WHERE source=%s RETURNING id",
                                   (source,)).fetchone()
            if deleted:
                conn.execute("UPDATE corpus_state SET revision=revision+1")
        return deleted is not None

    def revision(self) -> int:
        with self.connect() as conn:
            return conn.execute("SELECT revision FROM corpus_state").fetchone()["revision"]

    def ranked(self, query: str, vector: list[float], limit: int,
               source: str | None = None,
               model: str = EMBEDDING_MODEL) -> tuple[list[Hit], list[Hit]]:
        with self.connect() as conn:
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            vec = conn.execute("""SELECT *, 1-(embedding <=> %s::vector) AS score
                FROM chunks WHERE embedding_model=%s AND (%s::text IS NULL OR source=%s)
                ORDER BY embedding <=> %s::vector, id LIMIT %s""",
                (_vector(vector), model, source, source, _vector(vector), limit)).fetchall()
            lex = conn.execute("""SELECT *, ts_rank_cd(search_tsv,
                plainto_tsquery('english', %s)) AS score FROM chunks
                WHERE embedding_model=%s AND search_tsv @@ plainto_tsquery('english', %s)
                AND (%s::text IS NULL OR source=%s)
                ORDER BY score DESC, id LIMIT %s""",
                (query, model, query, source, source, limit)).fetchall()
        return ([Hit(_chunk(r), float(r["score"])) for r in vec if r["score"] is not None],
                [Hit(_chunk(r), float(r["score"])) for r in lex])


class MemoryStore:
    """Unit-test/demo double. Token overlap is NOT PostgreSQL full-text search."""
    def __init__(self):
        self.chunks: dict[str, Chunk] = {}
        self.vectors: dict[str, list[float]] = {}
        self.models: dict[str, str] = {}
        self.fingerprints: dict[str, str] = {}
        self._revision = 0

    def replace(self, chunks: list[Chunk], embeddings: list[list[float]],
                model: str = EMBEDDING_MODEL) -> bool:
        validate_batch(chunks, embeddings, model)
        doc = chunks[0].document_id
        mark = fingerprint(chunks, model)
        if self.fingerprints.get(doc) == mark:
            return False
        for key in [k for k, c in self.chunks.items() if c.document_id == doc]:
            del self.chunks[key]
            del self.vectors[key]
            del self.models[key]
        for c, v in zip(chunks, embeddings, strict=True):
            self.chunks[c.id], self.vectors[c.id] = c, v
            self.models[c.id] = model
        self.fingerprints[doc] = mark
        self._revision += 1
        return True

    def get(self, chunk_id: str) -> Chunk | None:
        return self.chunks.get(chunk_id)

    def delete(self, source: str) -> bool:
        ids = [key for key, chunk in self.chunks.items() if chunk.source == source]
        if not ids:
            return False
        for key in ids:
            document_id = self.chunks[key].document_id
            del self.chunks[key]
            del self.vectors[key]
            del self.models[key]
            self.fingerprints.pop(document_id, None)
        self._revision += 1
        return True

    def revision(self) -> int:
        return self._revision

    def ranked(self, query: str, vector: list[float], limit: int,
               source: str | None = None,
               model: str = EMBEDDING_MODEL) -> tuple[list[Hit], list[Hit]]:
        candidates = [c for c in self.chunks.values()
                      if self.models[c.id] == model and (source is None or c.source == source)]
        query_terms = set(tokens(query))
        vec = [Hit(c, sum(a*b for a, b in zip(vector, self.vectors[c.id], strict=True)))
               for c in candidates]
        lex = [Hit(c, len(query_terms & set(tokens(c.title + " " + c.section + " " + c.text))))
               for c in candidates]
        def ordering(h):
            return (-h.score, h.chunk.id)
        return sorted(vec, key=ordering)[:limit], sorted(
            [h for h in lex if h.score > 0], key=ordering)[:limit]
