from pathlib import Path

from langchain_core.documents import Document

from app.factory import build_embeddings, build_vectorstore
from app.ingest import chunk_id, collect_files, ingest_paths, load_file, load_pdf, split_documents
from tests.pdf_fixture import make_pdf


def test_pdf_pages_become_documents(tmp_path):
    pdf = tmp_path / "parrot.pdf"
    pages = ["The parrot is named Captain Flint.", "Captain Flint likes crackers."]
    pdf.write_bytes(make_pdf(pages))
    docs = load_pdf(pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert "Captain Flint" in docs[0].page_content
    assert docs[0].metadata == {"source": "parrot.pdf", "page": 1, "kind": "pdf"}


def test_split_respects_chunk_size_and_numbers_chunks():
    doc = Document(page_content="word " * 1000, metadata={"source": "big.md", "kind": "text"})
    chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 5
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(1, len(chunks) + 1))


def test_chunk_ids_are_stable_and_content_sensitive():
    a = Document(page_content="same text", metadata={"source": "a.md", "chunk": 1})
    b = Document(page_content="same text", metadata={"source": "a.md", "chunk": 1})
    c = Document(page_content="other text", metadata={"source": "a.md", "chunk": 1})
    assert chunk_id(a) == chunk_id(b)
    assert chunk_id(a) != chunk_id(c)


def test_images_are_described_before_indexing(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n not really a png")
    docs = load_file(image, describer=lambda data, mime: f"bar chart ({mime}, {len(data)} bytes)")
    assert docs[0].metadata == {"source": "chart.png", "kind": "image"}
    assert "bar chart (image/png" in docs[0].page_content


def test_ingest_then_reingest_does_not_duplicate(settings):
    vectorstore = build_vectorstore(settings, build_embeddings(settings))
    folder = Path(settings.upload_dir)
    folder.mkdir(parents=True, exist_ok=True)
    notes = folder / "notes.md"
    notes.write_text("# Notes\n\nThe deploy command is `make deploy`.\n", encoding="utf-8")

    first = ingest_paths([notes], vectorstore, chunk_size=200, chunk_overlap=20)
    assert first.files == ["notes.md"]
    assert first.chunks >= 1

    ingest_paths([notes], vectorstore, chunk_size=200, chunk_overlap=20)
    assert vectorstore._collection.count() == first.chunks

    hits = vectorstore.similarity_search("deploy command", k=5)
    assert any("make deploy" in h.page_content for h in hits)


def test_collect_files_filters_by_type(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "b.exe").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.pdf").write_bytes(make_pdf(["hi"]))
    assert [p.name for p in collect_files([tmp_path])] == ["a.md", "c.pdf"]
