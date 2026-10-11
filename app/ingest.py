"""Ingestion pipeline: files -> LangChain Documents -> chunks -> embeddings -> Chroma.

Supported inputs: PDF, Markdown, plain text, and images. Images are turned into
text by a vision model (the `describer` callable) before they are embedded. That
is how multi-modal content ends up in a text vector store.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

log = logging.getLogger("docs_rag.ingest")

TEXT_SUFFIXES = {".txt", ".md"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES

# (image bytes, mime type) -> text description
Describer = Callable[[bytes, str], str]

IMAGE_PROMPT = (
    "Describe this image for a search index. Include every piece of visible text, "
    "every number and label, and what the image shows overall. Plain text only."
)


@dataclass
class IngestReport:
    files: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    chunks: int = 0


def load_pdf(path: Path) -> list[Document]:
    """One Document per page that has a text layer. Scanned pages are skipped."""
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "page": number, "kind": "pdf"},
            )
        )
    return docs


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": path.name, "kind": "text"})]


def load_image(path: Path, describer: Describer) -> list[Document]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    description = describer(path.read_bytes(), mime).strip()
    if not description:
        return []
    return [
        Document(
            page_content=f"Image {path.name}: {description}",
            metadata={"source": path.name, "kind": "image"},
        )
    ]


def load_file(path: Path, describer: Describer | None = None) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return load_pdf(path)
    if suffix in TEXT_SUFFIXES:
        return load_text(path)
    if suffix in IMAGE_SUFFIXES:
        if describer is None:
            raise ValueError("an image describer is required to ingest images")
        return load_image(path, describer)
    raise ValueError(f"unsupported file type: {path.suffix}")


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split into overlapping chunks and number the chunks within each source file."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["chunk"] = counters[source]
    return chunks


def chunk_id(chunk: Document) -> str:
    """Deterministic id, so ingesting the same content twice never duplicates it."""
    meta = chunk.metadata
    raw = f"{meta['source']}|{meta.get('page', 0)}|{meta['chunk']}|{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def remove_source(vectorstore: VectorStore, source: str) -> int:
    """Delete every chunk that came from `source` (used before re-ingesting a file)."""
    getter = getattr(vectorstore, "get", None)
    if getter is None:
        return 0
    existing = getter(where={"source": source}).get("ids") or []
    if existing:
        vectorstore.delete(ids=existing)
    return len(existing)


def ingest_paths(
    paths: Iterable[str | Path],
    vectorstore: VectorStore,
    *,
    chunk_size: int,
    chunk_overlap: int,
    describer: Describer | None = None,
) -> IngestReport:
    report = IngestReport()
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            report.skipped.append(path.name)
            continue
        try:
            docs = load_file(path, describer)
        except ValueError as exc:
            log.warning("skipped", extra={"data": {"file": path.name, "reason": str(exc)}})
            report.skipped.append(path.name)
            continue
        chunks = split_documents(docs, chunk_size, chunk_overlap)
        if not chunks:
            report.skipped.append(path.name)
            continue
        replaced = remove_source(vectorstore, path.name)
        vectorstore.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
        report.files.append(path.name)
        report.chunks += len(chunks)
        log.info(
            "ingested",
            extra={"data": {"file": path.name, "chunks": len(chunks), "replaced": replaced}},
        )
    return report


def collect_files(paths: Iterable[str | Path]) -> list[Path]:
    """Expand folders into their supported files (recursive, sorted)."""
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(
                sorted(
                    f for f in path.rglob("*")
                    if f.is_file() and f.suffix.lower() in ALLOWED_SUFFIXES
                )
            )
        elif path.is_file():
            out.append(path)
    return out
