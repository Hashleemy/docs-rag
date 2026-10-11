from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.retrieval import BM25Index, DenseRetriever, HybridRetriever, rrf_fuse, tokenize


def test_tokenize_lowercases_and_splits_on_non_alphanumerics():
    assert tokenize("The AI-RMF (v1.0) Core!") == ["the", "ai", "rmf", "v1", "0", "core"]


def build_index(texts: dict[str, str]) -> BM25Index:
    index = BM25Index()
    index.build(list(texts), list(texts.values()))
    return index


def test_bm25_ranks_exact_term_match_first():
    index = build_index(
        {
            "a": "the framework describes governance of systems",
            "b": "confabulation refers to confidently stated errors",
            "c": "the framework describes measurement of systems",
        }
    )
    results = index.search("what is confabulation", k=3)
    assert results[0][0] == "b"
    assert results[0][1] > 0


def test_bm25_weights_rare_terms_above_common_ones():
    index = build_index(
        {
            "a": "framework risk",
            "b": "framework cbrn",
            "c": "framework governance",
        }
    )
    # "framework" appears everywhere (low IDF); "cbrn" only in b (high IDF)
    results = index.search("framework cbrn", k=3)
    assert results[0][0] == "b"


def test_bm25_empty_index_and_unknown_terms():
    index = BM25Index()
    index.build([], [])
    assert index.search("anything", k=5) == []
    index = build_index({"a": "some text"})
    assert index.search("zzz qqq", k=5) == []


def test_rrf_rewards_documents_ranked_in_both_lists():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], k=60)
    assert [doc_id for doc_id, _ in fused] == ["b", "a", "c"]
    scores = dict(fused)
    assert scores["b"] == 1 / 62 + 1 / 61
    assert scores["a"] == 1 / 61
    assert scores["c"] == 1 / 62


def ingest(vectorstore, texts: dict[str, str]) -> None:
    from app.ingest import chunk_id

    docs = [
        Document(page_content=text, metadata={"source": name, "kind": "text", "chunk": 1})
        for name, text in texts.items()
    ]
    vectorstore.add_documents(docs, ids=[chunk_id(d) for d in docs])


def make_vectorstore(tmp_path):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name="test",
        embedding_function=DeterministicFakeEmbedding(size=64),
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )


def test_hybrid_finds_exact_term_that_fake_dense_embeddings_miss(tmp_path):
    vectorstore = make_vectorstore(tmp_path)
    texts = {f"filler{i}.txt": f"unrelated filler text number {i}" for i in range(8)}
    texts["target.txt"] = "the xenodochium was a medieval guest house"
    ingest(vectorstore, texts)

    retriever = HybridRetriever(vectorstore)
    results = retriever.search("what is a xenodochium", k=3)
    assert results, "hybrid search returned nothing"
    assert results[0][0].metadata["source"] == "target.txt"
    # RRF scores are bounded by legs * 1/(k+1) and sorted descending
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_refresh_tracks_store_changes(tmp_path):
    vectorstore = make_vectorstore(tmp_path)
    retriever = HybridRetriever(vectorstore)  # built while the store is empty
    assert retriever.search("anything", k=3) == []

    ingest(vectorstore, {"late.txt": "the xenodochium was a guest house"})
    retriever.refresh()
    results = retriever.search("xenodochium", k=3)
    assert results and results[0][0].metadata["source"] == "late.txt"


def test_dense_retriever_delegates_to_vectorstore(tmp_path):
    vectorstore = make_vectorstore(tmp_path)
    ingest(vectorstore, {"a.txt": "alpha text", "b.txt": "beta text"})
    results = DenseRetriever(vectorstore).search("alpha", k=2)
    assert len(results) == 2
    assert {doc.metadata["source"] for doc, _ in results} == {"a.txt", "b.txt"}
