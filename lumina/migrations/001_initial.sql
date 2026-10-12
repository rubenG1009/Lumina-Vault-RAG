CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE corpus_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    revision bigint NOT NULL DEFAULT 0
);
INSERT INTO corpus_state (singleton) VALUES (true);
CREATE TABLE documents (
    id text PRIMARY KEY,
    source text NOT NULL UNIQUE,
    fingerprint text NOT NULL
);
CREATE TABLE chunks (
    id text PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text text NOT NULL,
    source text NOT NULL,
    title text NOT NULL,
    section text NOT NULL,
    page integer,
    start_offset integer NOT NULL,
    end_offset integer NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    embedding vector(256) NOT NULL,
    embedding_model text NOT NULL CHECK (embedding_model = 'hash-bow-v1-256'),
    search_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', title || ' ' || section || ' ' || text)
    ) STORED
);
CREATE INDEX chunks_document_idx ON chunks(document_id);
CREATE INDEX chunks_search_idx ON chunks USING gin(search_tsv);
-- Exact cosine search is deliberate for this small corpus; no ANN tuning claims.
