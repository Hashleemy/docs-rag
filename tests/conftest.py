import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Fake model + fake embeddings in a temporary folder: fast, offline, no API key."""
    return Settings(
        fake_llm=True,
        chroma_dir=str(tmp_path / "chroma"),
        upload_dir=str(tmp_path / "uploads"),
        embedding_cache_dir=str(tmp_path / "cache"),
        _env_file=None,
    )


@pytest.fixture
def client(settings) -> TestClient:
    return TestClient(create_app(settings))
