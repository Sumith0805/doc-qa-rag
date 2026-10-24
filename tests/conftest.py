import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    """Every test gets its own empty SQLite DB + vector store, and the mock LLM."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CHROMA_PATH", tmp_path / "chroma")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "documind.db")
    monkeypatch.setattr(config, "LLM_MODE", "mock")
    yield
