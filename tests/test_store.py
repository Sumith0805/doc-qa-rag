"""Integration tests for the real ChromaDB store using a fast fake embedder
(no model download). Run with: pytest -v"""
import re
import zlib
from pathlib import Path

import pytest

chromadb = pytest.importorskip("chromadb")

import registry  # noqa: E402
import store  # noqa: E402
from rag import answer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "eval" / "docs"


class FakeEmbedding(chromadb.EmbeddingFunction):
    """Hashed bag-of-words embedding. Deterministic, offline, good enough to test plumbing."""

    DIM = 4096

    def __init__(self):
        pass

    @staticmethod
    def name():
        return "fake-hash-embedding"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return FakeEmbedding()

    def __call__(self, input):
        import numpy as np

        out = []
        for text in input:
            v = np.zeros(self.DIM, dtype="float32")
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
            n = np.linalg.norm(v)
            out.append((v / n if n else v).tolist())
        return out


@pytest.fixture(autouse=True)
def fake_embedder():
    store.set_embedding_function(FakeEmbedding())
    yield
    store.set_embedding_function(None)


@pytest.fixture
def two_docs(tmp_path):
    a = tmp_path / "cats.txt"
    a.write_text("Cats are small carnivorous mammals. A cat sleeps sixteen hours a day.", encoding="utf-8")
    b = tmp_path / "rockets.txt"
    b.write_text("Rockets burn propellant to produce thrust. A rocket engine needs liquid oxygen.", encoding="utf-8")
    return store.index_document(str(a), "cats.txt"), store.index_document(str(b), "rockets.txt")


def test_retrieves_evidence_from_both_documents(two_docs):
    cats, rockets = two_docs
    assert {d["status"] for d in (cats, rockets)} == {"ready"}
    top_cat = store.search("how long does a cat sleep", k=2)[0]
    top_rocket = store.search("what does a rocket engine need", k=2)[0]
    assert top_cat["filename"] == "cats.txt" and top_cat["doc_id"] == cats["id"]
    assert top_rocket["filename"] == "rockets.txt" and top_rocket["doc_id"] == rockets["id"]


def test_second_upload_does_not_erase_first(two_docs):
    assert len(registry.list_documents(status="ready")) == 2
    assert store.count_chunks() == 2


def test_identical_text_scores_near_one(two_docs):
    hit = store.search("Cats are small carnivorous mammals. A cat sleeps sixteen hours a day.", k=1)[0]
    assert hit["score"] > 0.99, "distance metric is not cosine"


def test_duplicate_upload_is_not_indexed_twice(tmp_path, two_docs):
    again = tmp_path / "copy.txt"
    again.write_text((tmp_path / "cats.txt").read_text(encoding="utf-8"), encoding="utf-8")
    rec = store.index_document(str(again), "copy.txt")
    assert rec["duplicate"] is True
    assert store.count_chunks() == 2


def test_delete_removes_chunks_and_embeddings(two_docs):
    cats, rockets = two_docs
    store.delete_document(cats["id"])
    assert store.count_chunks(cats["id"]) == 0
    assert registry.get_document(cats["id"]) is None
    hits = store.search("how long does a cat sleep", k=5)
    assert all(h["doc_id"] != cats["id"] for h in hits)
    assert store.search("rocket engine thrust", k=1)[0]["doc_id"] == rockets["id"]


def test_delete_survives_restart(two_docs):
    cats, _ = two_docs
    store.delete_document(cats["id"])
    store.reset_client()  # simulate app restart: reopen from disk
    assert store.count_chunks() == 1


def test_failed_document_is_recorded_and_leaves_no_chunks(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError):
        store.index_document(str(empty), "empty.txt")
    doc = registry.list_documents()[0]
    assert doc["status"] == "failed" and "No extractable text" in doc["error"]
    assert store.count_chunks() == 0


def test_pdf_pages_are_searchable():
    pdf = DOCS / "attention_is_all_you_need.pdf"
    if not pdf.exists():
        pytest.skip("sample pdf missing")
    store.index_document(str(pdf), pdf.name)
    hits = store.search("label smoothing residual dropout regularization", k=4)
    assert any(h["page"] == 8 for h in hits)


def test_end_to_end_answer_and_abstention(two_docs):
    ok = answer("how long does a cat sleep", min_score=0.05)
    assert not ok["abstained"] and ok["citations"][0]["filename"] == "cats.txt"
    no = answer("quantum banana zebra volcano", min_score=0.05)
    assert no["abstained"]
