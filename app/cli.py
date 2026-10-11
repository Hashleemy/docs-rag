"""Command line interface.

    uv run python -m app.cli ingest data/ README.md
    uv run python -m app.cli ask "What are the four functions of the AI RMF Core?"
    uv run python -m app.cli eval --golden eval/golden.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.config import Settings
from app.factory import build_embeddings, build_llm, build_vectorstore, make_describer
from app.ingest import IMAGE_SUFFIXES, collect_files, ingest_paths
from app.observability import configure_logging
from app.rag import RAG


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    files = collect_files(args.paths)
    if not files:
        print("no supported files found (pdf, md, txt, png, jpg, jpeg, webp, gif)")
        return 1
    vectorstore = build_vectorstore(settings, build_embeddings(settings))
    needs_vision = any(f.suffix.lower() in IMAGE_SUFFIXES for f in files)
    describer = make_describer(build_llm(settings)) if needs_vision else None
    report = ingest_paths(
        files,
        vectorstore,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        describer=describer,
    )
    print(f"ingested {len(report.files)} file(s), {report.chunks} chunks")
    if report.skipped:
        print(f"skipped: {', '.join(report.skipped)}")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    vectorstore = build_vectorstore(settings, build_embeddings(settings))
    rag = RAG(vectorstore, build_llm(settings), settings.top_k, settings.llm_model)
    result = rag.ask(args.question)
    print(result.answer)
    print("\nSources:")
    for s in result.sources:
        page = f" p.{s.page}" if s.page else ""
        print(f"  [{s.n}] {s.source}{page}  (score {s.score})")
    if result.usage:
        print(
            f"\nTokens: in={result.usage.get('input_tokens')} "
            f"out={result.usage.get('output_tokens')}  model={result.model}"
        )
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Retrieval hit rate: does the expected document (and page) show up in the top k?"""
    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    vectorstore = build_vectorstore(settings, build_embeddings(settings))
    hits = 0
    for item in golden:
        results = vectorstore.similarity_search(item["question"], k=settings.top_k)
        found = any(
            d.metadata.get("source") == item["expected_source"]
            and (item.get("expected_page") in (None, d.metadata.get("page")))
            for d in results
        )
        hits += int(found)
        print(f"  {'HIT ' if found else 'MISS'}  {item['question']}")
    rate = hits / len(golden) if golden else 0.0
    print(
        f"\nretrieval hit@{settings.top_k}: {hits}/{len(golden)} = {rate:.0%} "
        f"(minimum {args.min_hit_rate:.0%})"
    )
    return 0 if rate >= args.min_hit_rate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="docs-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index files or folders")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="ask a question about the indexed documents")
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval", help="retrieval hit rate against a golden question set")
    p.add_argument("--golden", default="eval/golden.json")
    p.add_argument("--min-hit-rate", type=float, default=0.8)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    configure_logging()
    return args.func(args, Settings())


if __name__ == "__main__":
    sys.exit(main())
