# Lumina Vault RAG

A small, reproducible document retrieval service. It indexes Markdown and text PDFs in PostgreSQL with pgvector, combines vector and full-text rankings, and returns exact quotations with source offsets. The default provider is deterministic and needs no API key.

This is a **portfolio demo**, not an enterprise document system. Its 256-dimensional signed-hash vectors provide lexical similarity, not semantic embeddings. Responses are quotations rather than free-form synthesis, and citation checks establish that text came from a stored chunk, not that the underlying document is true.

## Architecture

```text
Markdown / text PDF -> chunks + metadata -> hash embeddings -> PostgreSQL + pgvector
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

The PostgreSQL migration is checksummed and repeatable. Re-ingesting a changed document replaces its chunks in one transaction and increments a corpus revision, which invalidates old answer cache keys. The optional lexical reranker is enabled with `LUMINA_RERANK=overlap`.

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

OpenAPI is at `http://127.0.0.1:18765/docs`. `POST /search` accepts `mode=hybrid` or `mode=vector` and optional `source`. `POST /agent` performs a bounded `search_docs` then `get_chunk` workflow and returns a tool trace. `GET /chunks/{id}` returns citation provenance. The service refuses unsupported answers and evidence with simple numeric or negation conflicts.

Stop the stack with `docker compose down`. Add `-v` only if you intend to delete the database volume.

## CLI and MCP

For a Python installation outside Docker, use Python 3.12+, `pip install .`, then set `DATABASE_URL` and optionally `REDIS_URL`. Run `lumina migrate` before ingestion. CLI commands include `ingest`, `search`, `ask`, `agent`, and `eval`.

`lumina-mcp` starts a local **stdio** MCP server exposing `search_docs` and `get_chunk`. It uses the same `DATABASE_URL` and `REDIS_URL`. The MCP process is not started by Compose. With the Compose stack running, configure your MCP client to execute `docker compose --project-directory /path/to/Lumina-Vault-RAG exec -T api lumina-mcp`. If you run `lumina-mcp` outside Docker, set both connection variables explicitly in the client's MCP server environment; some clients do not inherit shell variables.

## Providers and cost

`LUMINA_PROVIDER=fake` is the default and makes no network or paid API calls. To try external quote selection, set `LUMINA_PROVIDER=openai` or `anthropic`, `LUMINA_MODEL`, and the corresponding `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. These modes may incur provider charges. The model selects exact quotes; the server checks chunk IDs, quote substrings, relevance and conflicts before returning them. Invalid output is refused. Provider calls have a 30-second timeout and are not used by CI.

`LUMINA_INPUT_USD_PER_MILLION` and `LUMINA_OUTPUT_USD_PER_MILLION` can supply *your* current model prices for estimated per-request cost. Both default to zero, so a zero estimate does not mean an external call is free. Structured logs report latency, token usage, estimated cost and cache hits without logging question or document text.

## Evaluation

Run `lumina eval --output evals/results-local.json` without a database or API key. The labelled seven-question set covers four answerable questions plus missing evidence, conflicting evidence and prompt injection. It measures source recall@5, MRR@5, answer pass rate and citation provenance for hybrid retrieval versus vector-only. The checked local baseline is:

| Mode | Recall@5 | MRR@5 | Answer pass rate |
| --- | ---: | ---: | ---: |
| Hybrid | 1.00 | 1.00 | 1.00 |
| Vector-only | 1.00 | 1.00 | 1.00 |

These are results on a tiny synthetic corpus; they show a functioning regression fixture, **not** a measured advantage for hybrid retrieval or general-world accuracy. Add larger, harder labelled documents before drawing performance conclusions.

`pytest -q` runs unit/API tests; with `TEST_DATABASE_URL` and `TEST_REDIS_URL` it also runs the PostgreSQL/pgvector/Redis integration check. GitHub Actions runs lint, tests and the offline evaluation against service containers.

## Limits and security

- Markdown and text PDFs up to 10 MiB and 200 pages are supported. Scanned PDFs require OCR, which is not included.
- Document text is treated as untrusted. Suspicious instruction patterns are excluded, and model selections are verified, but the heuristic is not a complete prompt-injection defense.
- The API has no authentication or per-user document isolation. Keep it bound to localhost; add authentication and access controls before any public deployment or sensitive documents.
- The rate limiter uses a Redis fixed window and the direct client address. Behind a proxy, configure a trusted gateway and appropriate client identity handling.
- Hash vectors and English full-text search are demo choices. Multilingual semantic retrieval needs a real embedding model, index migration and a larger evaluation set.
