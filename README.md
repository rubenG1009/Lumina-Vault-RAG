# Lumina Vault RAG

A small, reproducible document retrieval service. It indexes Markdown and text PDFs in PostgreSQL with pgvector, combines vector and full-text rankings, and returns exact quotations with source offsets. The default provider is deterministic and needs no API key.

This is a **portfolio demo**, not an enterprise document system. The default 256-dimensional signed-hash vectors provide lexical similarity. An optional local 384-dimensional BGE model adds semantic retrieval without a paid API. Responses are quotations rather than free-form synthesis, and citation checks establish that text came from a stored chunk, not that the underlying document is true.

## Architecture

```text
Markdown / text PDF -> chunks + metadata -> hash or BGE embeddings -> PostgreSQL + pgvector
                                                        |         |
                                            vector ranking    full-text ranking
                                                        \         /
                                                         RRF fusion
                                                              |
                                        quote selector (fake / OpenAI / Anthropic)
                                                              |
                                                local citation verifier -> API / CLI
                                                              |
                                                     bounded tool workflow + MCP

Redis: embedding and answer cache (revision-aware), local API rate limiting
```

PostgreSQL migrations are checksummed and repeatable. Re-ingesting a changed document replaces its chunks in one transaction and increments a corpus revision, which invalidates old answer cache keys. Documents can also be deleted. Each chunk records its embedding model; retrieval filters by that model, so vectors of different dimensions never mix. The optional lexical reranker is enabled with `LUMINA_RERANK=overlap`.

## Run locally

Requires Docker Compose. The API is exposed only on `127.0.0.1:18765`.

```bash
docker compose up --build -d
docker compose ps
curl http://127.0.0.1:18765/health
docker compose exec api lumina ingest lumina/eval_data/corpus/retention.md
docker compose exec api lumina ask 'audit logs retention'
```

Direct HTTP calls:

```bash
curl --data-binary @lumina/eval_data/corpus/retention.md \
  'http://127.0.0.1:18765/documents?source=retention.md'
curl -H 'Content-Type: application/json' \
  -d '{"query":"audit logs retention"}' \
  http://127.0.0.1:18765/answer
```

OpenAPI is at `http://127.0.0.1:18765/docs`. `POST /search` accepts `mode=hybrid` or `mode=vector` and optional `source`. `DELETE /documents?source=...` removes one indexed document. `POST /agent` performs a bounded `search_docs` then `get_chunk` workflow and returns a tool trace. `GET /chunks/{id}` returns citation provenance. The service refuses unsupported answers and evidence with simple numeric or negation conflicts.

Stop the stack with `docker compose down`. Add `-v` only if you intend to delete the database volume.

### Local semantic embeddings

To use the BGE model, run:

```bash
LUMINA_EMBEDDING_BACKEND=semantic docker compose up --build -d
docker compose exec api lumina ingest lumina/eval_data/corpus/retention.md
docker compose exec api lumina search 'How long are audit records kept?'
```

The first semantic start downloads an open model to a persistent local Docker volume; no paid API is called. The installed Python equivalent is `pip install '.[semantic]'` plus `LUMINA_EMBEDDING_BACKEND=semantic`. **Re-ingest each document when switching models.** A document stays in its original embedding space until re-ingested; the query endpoint only searches documents indexed with the active model. The BGE model here is English-focused. For questions with no matching lexical excerpt, a cosine threshold selects exact sentences for citation. That threshold is a heuristic, not a proof that a sentence answers the question.

## CLI and MCP

For a Python installation outside Docker, use Python 3.12+, `pip install .`, then set `DATABASE_URL` and optionally `REDIS_URL`. Run `lumina migrate` before ingestion. CLI commands include `ingest`, `delete`, `search`, `ask`, `agent`, and `eval`.

`lumina-mcp` starts a local **stdio** MCP server exposing `search_docs` and `get_chunk`. It uses the same `DATABASE_URL` and `REDIS_URL`. The MCP process is not started by Compose. With the Compose stack running, configure your MCP client to execute `docker compose --project-directory /path/to/Lumina-Vault-RAG exec -T api lumina-mcp`. If you run `lumina-mcp` outside Docker, set both connection variables explicitly in the client's MCP server environment; some clients do not inherit shell variables.

## Providers and cost

`LUMINA_PROVIDER=fake` is the default and makes no network or paid API calls. To try external quote selection, set `LUMINA_PROVIDER=openai` or `anthropic`, `LUMINA_MODEL`, and the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. These modes may incur provider charges. The model selects exact quotes; the server checks chunk IDs, quote substrings, relevance and conflicts before returning them. Invalid output is refused. Provider calls have a 30-second timeout and are not used by CI.

`LUMINA_INPUT_USD_PER_MILLION` and `LUMINA_OUTPUT_USD_PER_MILLION` can supply *your* current model prices for estimated per-request cost. Both default to zero, so a zero estimate does not mean an external call is free. Structured logs report latency, token usage, estimated cost and cache hits without logging question or document text.

## Evaluation

Run `lumina eval --output evals/results-local.json` without a database or API key. Add `--database` to run against real PostgreSQL/pgvector in an isolated schema that is deleted afterwards. Add `--embedding semantic` to use the local BGE model. The labelled corpus has 15 short documents and 11 questions: seven primary answer checks (including missing evidence, conflict and prompt injection) and four paraphrase checks. Source recall@5 and MRR@5 use eight labelled retrieval questions; primary answer pass rate uses seven, and paraphrase quote pass rate uses four. These results were measured against PostgreSQL/pgvector:

| Embedding | Retrieval | Recall@5 | MRR@5 | Primary answer pass | Paraphrase quote pass |
| --- | --- | ---: | ---: | ---: | ---: |
| Hash | Hybrid | 0.875 | 0.681 | 7/7 | 0/4 |
| Hash | Vector-only | 0.875 | 0.681 | 7/7 | 0/4 |
| BGE | Hybrid | 1.00 | 1.00 | 7/7 | 4/4 |
| BGE | Vector-only | 1.00 | 1.00 | 7/7 | 4/4 |

The BGE model retrieved all eight expected sources; hash vectors missed one. It also selected the expected exact quote for four reformulated questions, where the hash baseline refused. This is a **tiny synthetic corpus**, so it does not establish general-world accuracy or calibrated abstention. Hybrid retrieval did **not** outperform vector-only on this set. Add larger and more varied labelled documents before drawing broader conclusions.

`pytest -q` runs unit/API tests; with `TEST_DATABASE_URL` and `TEST_REDIS_URL` it also runs PostgreSQL/pgvector/Redis/MCP integration checks. GitHub Actions runs lint, tests and both memory and PostgreSQL hash evaluations against service containers. Semantic evaluation is manual so CI does not depend on downloading a model.

## Limits and security

- Markdown and text PDFs up to 10 MiB and 200 pages are supported. Scanned PDFs require OCR, which is not included.
- Document text is treated as untrusted. Suspicious instruction patterns are excluded, and model selections are verified, but the heuristic is not a complete prompt-injection defense.
- The API has no authentication or per-user document isolation. Keep it bound to localhost; add authentication and access controls before any public deployment or sensitive documents.
- The rate limiter uses a Redis fixed window and the direct client address. Behind a proxy, configure a trusted gateway and appropriate client identity handling.
- English full-text search and the English-focused BGE model limit multilingual retrieval. The in-memory evaluation uses a lexical proxy; use `--database` for PostgreSQL full-text behavior.
- The deterministic tool workflow and quote selector are not autonomous LLM tool calling. OpenAI/Anthropic selectors have mocked contract tests but no paid live-call verification.
- Semantic sentence selection uses a fixed cosine threshold. Exact citations prove provenance, but a semantically similar sentence can still fail to answer the question; evaluate on real documents before relying on it.
