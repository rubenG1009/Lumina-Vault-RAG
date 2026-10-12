"""Deterministic offline evaluation over a labelled mini corpus."""

import json
from pathlib import Path

from lumina.embeddings import OfflineEmbedder
from lumina.ingestion import ingest_bytes
from lumina.retrieval import Retriever
from lumina.service import LuminaService
from lumina.store import MemoryStore

DATA = Path(__file__).resolve().parent / "eval_data"


def evaluate(data_dir: Path = DATA) -> dict:
    cases = json.loads((data_dir / "golden.json").read_text())
    store = MemoryStore()
    embedder = OfflineEmbedder()
    for path in sorted((data_dir / "corpus").glob("*.md")):
        chunks = ingest_bytes(path.read_bytes(), path.name)
        store.replace(chunks, [embedder.embed(c.text) for c in chunks])
    service = LuminaService(store)
    metrics = {}
    rows = []
    for mode in ("hybrid", "vector"):
        relevant_cases = 0
        recall_hits = 0
        reciprocal_ranks = 0.0
        answer_passes = 0
        for case in cases:
            hits = Retriever(store).search(case["query"], k=5, mode=mode)
            expected_source = case.get("source")
            if expected_source:
                relevant_cases += 1
                rank = next((i for i, h in enumerate(hits, 1)
                             if h.chunk.source == expected_source), None)
                if rank:
                    recall_hits += 1
                    reciprocal_ranks += 1 / rank
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
                         "answer_pass_rate": round(answer_passes / len(cases), 3),
                         "cases": len(cases)}
    return {"embedding": embedder.model, "corpus_documents": len(list(
        (data_dir / "corpus").glob("*.md"))), "metrics": metrics, "cases": rows}
