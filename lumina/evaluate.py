"""Deterministic offline evaluation over a labelled mini corpus."""

import json
from pathlib import Path
from uuid import uuid4

from lumina.embeddings import OfflineEmbedder
from lumina.ingestion import ingest_bytes
from lumina.retrieval import Retriever
from lumina.service import LuminaService
from lumina.store import MemoryStore

DATA = Path(__file__).resolve().parent / "eval_data"


def evaluate(data_dir: Path = DATA, embedder=None, store=None) -> dict:
    cases = json.loads((data_dir / "golden.json").read_text())
    store = store or MemoryStore()
    embedder = embedder or OfflineEmbedder()
    for path in sorted((data_dir / "corpus").glob("*.md")):
        chunks = ingest_bytes(path.read_bytes(), path.name)
        store.replace(chunks, [embedder.embed(c.text) for c in chunks], embedder.model)
    service = LuminaService(store, embedder=embedder)
    metrics = {}
    rows = []
    for mode in ("hybrid", "vector"):
        relevant_cases = 0
        recall_hits = 0
        reciprocal_ranks = 0.0
        answer_passes = 0
        paraphrase_answer_passes = 0
        for case in cases:
            hits = Retriever(store, embedder).search(case["query"], k=5, mode=mode)
            expected_source = case.get("source")
            rank = None
            if expected_source:
                relevant_cases += 1
                rank = next((i for i, h in enumerate(hits, 1)
                             if h.chunk.source == expected_source), None)
                if rank:
                    recall_hits += 1
                    reciprocal_ranks += 1 / rank
            if case.get("paraphrase"):
                result = service.answer(case["query"], mode=mode).answer
                paraphrase_passed = (not result.refused and any(
                    case["quote_contains"] in citation.quote
                    and citation.source == expected_source
                    for citation in result.citations))
                paraphrase_answer_passes += int(paraphrase_passed)
                rows.append({"mode": mode, "id": case["id"],
                             "paraphrase": True, "rank": rank,
                             "paraphrase_answer_pass": paraphrase_passed,
                             "reason": result.reason})
                continue
            result = service.answer(case["query"], mode=mode).answer
            passed = (result.refused == case["refuse"] and
                      (not case.get("quote_contains") or any(
                          case["quote_contains"] in c.quote for c in result.citations)) and
                      (not result.citations or all(
                          c.quote in store.get(c.chunk_id).text for c in result.citations)))
            answer_passes += int(passed)
            rows.append({"mode": mode, "id": case["id"], "pass": passed,
                         "refused": result.refused, "reason": result.reason})
        metrics[mode] = {"recall_at_5": round(recall_hits / relevant_cases, 3),
                         "mrr_at_5": round(reciprocal_ranks / relevant_cases, 3),
                         "answer_pass_rate": round(answer_passes / sum(
                             not c.get("paraphrase") for c in cases), 3),
                         "retrieval_cases": relevant_cases,
                         "answer_cases": sum(not c.get("paraphrase") for c in cases),
                         "paraphrase_answer_pass_rate": round(paraphrase_answer_passes / sum(
                             bool(c.get("paraphrase")) for c in cases), 3),
                         "paraphrase_answer_cases": sum(bool(c.get("paraphrase"))
                                                        for c in cases),
                         "cases": len(cases)}
    return {"embedding": embedder.model, "corpus_documents": len(list(
        (data_dir / "corpus").glob("*.md"))), "metrics": metrics, "cases": rows}


def evaluate_postgres(dsn: str, data_dir: Path = DATA, embedder=None) -> dict:
    """Run the same labelled evaluation in an isolated, disposable PostgreSQL schema."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from lumina.migrate import migrate
    from lumina.store import PostgresStore

    schema = f"lumina_eval_{uuid4().hex}"
    with psycopg.connect(dsn) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    eval_dsn = make_conninfo(dsn, options=f"-csearch_path={schema},public")
    try:
        migrate(eval_dsn)
        report = evaluate(data_dir, embedder, PostgresStore(eval_dsn))
        report["backend"] = "postgresql-pgvector"
        return report
    finally:
        with psycopg.connect(dsn) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
