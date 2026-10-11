"""Retrieval strategies: dense vectors, lexical BM25, and their hybrid fusion.

Dense retrieval (cosine similarity over embeddings) understands paraphrase
("car" matches "automobile") but can miss exact terms, acronyms, and codes.
Lexical retrieval (BM25) is the mirror image: exact terms are its strength,
paraphrase its blind spot. Hybrid search runs both and merges the two rankings
with Reciprocal Rank Fusion (RRF), which combines rankings without having to
calibrate their incomparable scores:

    RRF(d) = sum over rankings r of 1 / (K + rank_r(d)),  with K = 60

(Cormack, Clarke & Buettcher 2009; K = 60 is their recommendation and the
industry default.) The BM25 index is rebuilt in memory from the vector store
on refresh(), so Chroma stays the single source of truth and there is no
second store to drift out of sync. Both legs share the deterministic chunk id,
which is what makes the fusion join possible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from app.ingest import chunk_id

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 over an inverted index.

    score(q, d) = sum over query terms t of
        IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1)

    k1 dampens term-frequency saturation, b controls document-length
    normalisation; 1.5 and 0.75 are the standard defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(doc index, tf)]
        self._doc_len: list[int] = []
        self._avgdl = 0.0

    def build(self, ids: list[str], texts: list[str]) -> None:
        self._ids = list(ids)
        self._postings = {}
        self._doc_len = []
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self._doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self._postings.setdefault(term, []).append((index, tf))
        self._avgdl = sum(self._doc_len) / len(self._doc_len) if self._doc_len else 0.0

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        n = len(self._ids)
        scores: dict[int, float] = {}
        for term in dict.fromkeys(tokenize(query)):  # unique terms, query order kept
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = math.log((n - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for index, tf in postings:
                norm = self.k1 * (1 - self.b + self.b * self._doc_len[index] / self._avgdl)
                scores[index] = scores.get(index, 0.0) + idf * tf * (self.k1 + 1) / (tf + norm)
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(self._ids[index], score) for index, score in top]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge ranked id lists: rank r in any list contributes 1 / (k + r)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[tuple[Document, float]]: ...

    def refresh(self) -> None: ...


class DenseRetriever:
    """Plain vector search. The score is the cosine distance (lower is better)."""

    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def refresh(self) -> None:
        pass


class HybridRetriever:
    """Dense + BM25, fused with RRF. The score is the RRF score (higher is better).

    Call refresh() after any ingest or delete so the BM25 index tracks the
    vector store. Each leg retrieves far more candidates than finally needed
    (fetch_k, 50+ as is standard for RRF): a document strong on one leg earns
    extra fusion points from a middling rank on the other leg, but only if the
    pool is deep enough to include it there.
    """

    def __init__(self, vectorstore: VectorStore, rrf_k: int = 60):
        self.vectorstore = vectorstore
        self.rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._docs: dict[str, Document] = {}
        self.refresh()

    def refresh(self) -> None:
        data = self.vectorstore.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        texts = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        self._docs = {
            id_: Document(page_content=text, metadata=metadata or {})
            for id_, text, metadata in zip(ids, texts, metadatas, strict=True)
        }
        self._bm25.build(ids, texts)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        fetch_k = max(10 * k, 50)
        dense = self.vectorstore.similarity_search_with_score(query, k=fetch_k)
        dense_docs = {chunk_id(doc): doc for doc, _ in dense}
        lexical_ids = [id_ for id_, _ in self._bm25.search(query, fetch_k)]
        fused = rrf_fuse([list(dense_docs), lexical_ids], k=self.rrf_k)
        results: list[tuple[Document, float]] = []
        for doc_id, score in fused:
            doc = dense_docs.get(doc_id) or self._docs.get(doc_id)
            if doc is not None:
                results.append((doc, round(score, 4)))
            if len(results) == k:
                break
        return results
