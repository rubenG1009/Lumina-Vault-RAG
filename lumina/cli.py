"""Minimal CLI for migrations, ingestion, search, answers and evaluations."""

import argparse
import json
import os
from pathlib import Path

from lumina.evaluate import evaluate, evaluate_postgres
from lumina.migrate import migrate
from lumina.service import configured_service


def main() -> None:
    parser = argparse.ArgumentParser(prog="lumina")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("path", type=Path)
    delete = commands.add_parser("delete")
    delete.add_argument("source")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    ask = commands.add_parser("ask")
    ask.add_argument("query")
    ask.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    agent = commands.add_parser("agent")
    agent.add_argument("query")
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument("--embedding", choices=["hash", "semantic"], default="hash")
    evaluation.add_argument("--database", action="store_true",
                            help="Evaluate in a disposable PostgreSQL schema")
    args = parser.parse_args()
    if args.command == "migrate":
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            parser.error("DATABASE_URL is required")
        result = {"applied": migrate(dsn)}
    elif args.command == "eval":
        if args.embedding == "semantic":
            from lumina.embeddings import SemanticEmbedder

            embedder = SemanticEmbedder()
        else:
            embedder = None
        if args.database:
            dsn = os.getenv("DATABASE_URL")
            if not dsn:
                parser.error("DATABASE_URL is required for --database")
            result = evaluate_postgres(dsn, embedder=embedder)
        else:
            result = evaluate(embedder=embedder)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
    else:
        service = configured_service()
        if args.command == "ingest":
            result = service.ingest(args.path.read_bytes(), args.path.name)
        elif args.command == "delete":
            result = service.delete(args.source)
        elif args.command == "search":
            result = {"hits": [h.to_dict() for h in service.search(args.query, mode=args.mode)]}
        elif args.command == "ask":
            result = service.answer(args.query, mode=args.mode).to_dict()
        else:
            result = service.agent(args.query)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
