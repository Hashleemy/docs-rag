"""Builds the moving parts (embeddings, vector store, chat model) from Settings.

With FAKE_LLM=true everything is local and deterministic, so tests and CI need
no API key and make no network calls.
"""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage

from app.config import Settings
from app.ingest import IMAGE_PROMPT, Describer
from app.rag import message_text


class FastEmbedEmbeddings(Embeddings):
    """LangChain Embeddings backed by fastembed: small ONNX models that run on the CPU."""

    def __init__(self, model_name: str, cache_dir: str):
        from fastembed import TextEmbedding  # slow import, only needed for real runs

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts, batch_size=64)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.query_embed(text))).tolist()


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.fake_llm:
        return DeterministicFakeEmbedding(size=64)
    return FastEmbedEmbeddings(settings.embedding_model, settings.embedding_cache_dir)


def warm_embeddings(settings: Settings | None = None) -> None:
    """Download the embedding model ahead of time (run during the Docker build)."""
    build_embeddings(settings or Settings()).embed_query("warm up")


def build_vectorstore(settings: Settings, embeddings: Embeddings):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=settings.collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_llm(settings: Settings) -> BaseChatModel:
    if settings.fake_llm:
        return FakeListChatModel(responses=["This is a fake answer for tests, citing [1]."])
    from langchain_anthropic import ChatAnthropic

    kwargs: dict = {
        "model": settings.llm_model,
        "max_tokens": settings.max_output_tokens,
        "timeout": 120,
        "max_retries": 2,
    }
    if settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    return ChatAnthropic(**kwargs)


def make_describer(llm: BaseChatModel) -> Describer:
    """Turn an image into searchable text using the chat model's vision input."""

    def describe(data: bytes, mime: str) -> str:
        encoded = base64.b64encode(data).decode("ascii")
        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": encoded},
                },
                {"type": "text", "text": IMAGE_PROMPT},
            ]
        )
        return message_text(llm.invoke([message]))

    return describe
