"""Retrieval evaluation against a golden question set.

Each golden item names the question, the file the answer lives in
(`expected_source`), and optionally the pages it may appear on
(`expected_pages`, a list; the older singular `expected_page` still works).
A retrieved chunk is relevant when its source matches and, if pages are
given, its page is one of them.

Metrics, all computed from the rank of the first relevant chunk:

- hit@n: fraction of questions with a relevant chunk in the top n.
- MRR (mean reciprocal rank): average of 1/rank, 0 when nothing relevant
  is retrieved. Rewards putting the right chunk first, not just somewhere
  in the top k.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document

# (question, k) -> ranked documents, best first
Retriever = Callable[[str, int], list[Document]]


def expected_pages(item: dict) -> list[int]:
    pages = item.get("expected_pages")
    if pages is None and item.get("expected_page") is not None:
        pages = [item["expected_page"]]
    return pages or []


def is_relevant(item: dict, metadata: dict) -> bool:
    if metadata.get("source") != item["expected_source"]:
        return False
    pages = expected_pages(item)
    return not pages or metadata.get("page") in pages


def first_relevant_rank(item: dict, docs: list[Document]) -> int | None:
    for rank, doc in enumerate(docs, start=1):
        if is_relevant(item, doc.metadata):
            return rank
    return None


@dataclass
class QuestionResult:
    question: str
    rank: int | None  # 1-based rank of the first relevant chunk; None = miss


@dataclass
class EvalReport:
    results: list[QuestionResult]
    k: int

    def hit_rate(self, n: int) -> float:
        if not self.results:
            return 0.0
        hits = sum(1 for r in self.results if r.rank is not None and r.rank <= n)
        return hits / len(self.results)

    def mrr(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 / r.rank for r in self.results if r.rank is not None) / len(self.results)


def evaluate(golden: list[dict], retriever: Retriever, k: int) -> EvalReport:
    results = [
        QuestionResult(item["question"], first_relevant_rank(item, retriever(item["question"], k)))
        for item in golden
    ]
    return EvalReport(results=results, k=k)
