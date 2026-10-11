"""Settings for docs-rag.

Every value can be overridden with an environment variable of the same name in
upper case, or a line in a .env file. Example: LLM_MODEL=claude-haiku-5-5
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Chroma sends anonymous usage statistics by default. Turn that off.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- chat model (answers questions, describes images) ---
    anthropic_api_key: str = ""
    llm_model: str = "claude-opus-5-5"
    max_output_tokens: int = 1024
    fake_llm: bool = False  # True in tests and CI: no network calls, no API key needed

    # --- embeddings and vector store (local, no API key) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = "./storage/fastembed"
    chroma_dir: str = "./storage/chroma"
    collection: str = "docs"

    # --- ingestion and retrieval ---
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    upload_dir: str = "./storage/uploads"
    max_upload_mb: int = 20
