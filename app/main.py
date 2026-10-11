"""FastAPI service: /api/health, /api/sources, /api/ingest, /api/ask, and the React UI.

Run locally:  uv run uvicorn app.main:create_app --factory --reload
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.factory import (
    build_embeddings,
    build_llm,
    build_retriever,
    build_vectorstore,
    make_describer,
)
from app.ingest import ALLOWED_SUFFIXES, ingest_paths, remove_source
from app.observability import add_request_logging, configure_logging
from app.rag import RAG

log = logging.getLogger("docs_rag.api")
WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


def safe_filename(name: str | None) -> str:
    """Keep only the base name and safe characters: no path traversal, no surprises."""
    base = Path(name or "upload").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging()

    embeddings = build_embeddings(settings)
    vectorstore = build_vectorstore(settings, embeddings)
    retriever = build_retriever(settings, vectorstore)
    llm = build_llm(settings)
    rag = RAG(retriever, llm, top_k=settings.top_k, model_name=settings.llm_model)
    describer = make_describer(llm)
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="docs-rag", version="0.1.0")
    add_request_logging(app)
    app.state.settings = settings
    app.state.rag = rag
    app.state.vectorstore = vectorstore

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "model": settings.llm_model,
            "embedding_model": settings.embedding_model,
            "retrieval": "hybrid" if settings.hybrid else "dense",
            "chunks": vectorstore._collection.count(),
        }

    @app.get("/api/sources")
    def sources() -> dict:
        metadatas = vectorstore.get(include=["metadatas"]).get("metadatas") or []
        counts = Counter(str(m.get("source", "?")) for m in metadatas)
        return {"sources": [{"source": s, "chunks": n} for s, n in sorted(counts.items())]}

    @app.delete("/api/sources/{source}")
    def delete_source(source: str) -> dict:
        name = safe_filename(source)
        removed = remove_source(vectorstore, name)
        upload = upload_dir / name
        had_upload = upload.exists()
        upload.unlink(missing_ok=True)
        if not removed and not had_upload:
            raise HTTPException(status_code=404, detail=f"unknown source: {name}")
        retriever.refresh()
        log.info("removed", extra={"data": {"file": name, "chunks": removed}})
        return {"source": name, "chunks_removed": removed}

    @app.post("/api/ingest")
    def ingest(files: Annotated[list[UploadFile], File()]) -> dict:
        limit = settings.max_upload_mb * 1024 * 1024
        saved: list[Path] = []
        for upload in files:
            name = safe_filename(upload.filename)
            if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
                raise HTTPException(status_code=400, detail=f"unsupported file type: {name}")
            data = upload.file.read()
            if len(data) > limit:
                raise HTTPException(
                    status_code=413, detail=f"{name} is larger than {settings.max_upload_mb} MB"
                )
            target = upload_dir / name
            target.write_bytes(data)
            saved.append(target)
        report = ingest_paths(
            saved,
            vectorstore,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            describer=describer,
        )
        retriever.refresh()
        return {"files": report.files, "skipped": report.skipped, "chunks": report.chunks}

    @app.post("/api/ask")
    def ask(req: AskRequest) -> dict:
        start = time.perf_counter()
        result = rag.ask(req.question)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        tokens = {k: v for k, v in result.usage.items() if isinstance(v, int)}
        log.info(
            "ask",
            extra={
                "data": {
                    "model": result.model,
                    "latency_ms": latency_ms,
                    "sources": len(result.sources),
                    **tokens,
                }
            },
        )
        return {
            "answer": result.answer,
            "sources": [s.as_dict() for s in result.sources],
            "usage": result.usage,
            "model": result.model,
            "latency_ms": latency_ms,
        }

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app
