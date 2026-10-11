from langchain_core.documents import Document

from app.evaluate import EvalReport, QuestionResult, evaluate, first_relevant_rank, is_relevant


def doc(source: str, page: int | None = None) -> Document:
    metadata = {"source": source}
    if page is not None:
        metadata["page"] = page
    return Document(page_content="x", metadata=metadata)


def test_is_relevant_matches_source_when_no_pages_given():
    item = {"expected_source": "a.pdf"}
    assert is_relevant(item, {"source": "a.pdf", "page": 3})
    assert not is_relevant(item, {"source": "b.pdf", "page": 3})


def test_is_relevant_requires_one_of_the_expected_pages():
    item = {"expected_source": "a.pdf", "expected_pages": [2, 5]}
    assert is_relevant(item, {"source": "a.pdf", "page": 5})
    assert not is_relevant(item, {"source": "a.pdf", "page": 4})


def test_is_relevant_supports_legacy_singular_expected_page():
    item = {"expected_source": "a.pdf", "expected_page": 7}
    assert is_relevant(item, {"source": "a.pdf", "page": 7})
    assert not is_relevant(item, {"source": "a.pdf", "page": 8})


def test_first_relevant_rank_is_one_based_and_none_on_miss():
    item = {"expected_source": "a.pdf"}
    assert first_relevant_rank(item, [doc("b.pdf"), doc("a.pdf")]) == 2
    assert first_relevant_rank(item, [doc("b.pdf")]) is None


def test_metrics_hit_rate_and_mrr():
    report = EvalReport(
        results=[
            QuestionResult("q1", rank=1),
            QuestionResult("q2", rank=4),
            QuestionResult("q3", rank=None),
        ],
        k=5,
    )
    assert report.hit_rate(1) == 1 / 3
    assert report.hit_rate(5) == 2 / 3
    assert report.mrr() == (1 + 0.25 + 0) / 3


def test_evaluate_runs_retriever_per_question():
    golden = [
        {"question": "q1", "expected_source": "a.pdf"},
        {"question": "q2", "expected_source": "b.pdf", "expected_pages": [2]},
    ]
    corpus = {
        "q1": [doc("a.pdf", 1)],
        "q2": [doc("a.pdf", 1), doc("b.pdf", 2)],
    }
    report = evaluate(golden, lambda q, k: corpus[q][:k], k=5)
    assert [r.rank for r in report.results] == [1, 2]
