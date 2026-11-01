"""Users must never see each other's documents, and an empty scope must search nothing."""
import re
import zlib

import pytest

chromadb = pytest.importorskip("chromadb")

import registry  # noqa: E402
import rag  # noqa: E402
import store  # noqa: E402


class FakeEmbedding(chromadb.EmbeddingFunction):
    """Hashed bag-of-words embedding (offline, deterministic). Same helper as test_store.py."""

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


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


@pytest.fixture
def two_users(tmp_path):
    alice = store.index_document(
        write(tmp_path, "cats.txt", "Cats are small carnivorous mammals. A cat sleeps sixteen hours a day."),
        "cats.txt", user_id="alice")
    bob = store.index_document(
        write(tmp_path, "rockets.txt", "Rockets burn propellant to produce thrust. A rocket engine needs liquid oxygen."),
        "rockets.txt", user_id="bob")
    return alice, bob


def test_a_user_only_finds_their_own_documents(two_users):
    alice, bob = two_users
    hits = store.search("what does a rocket engine need", k=6, doc_ids=[alice["id"]])
    assert hits and all(h["doc_id"] == alice["id"] for h in hits)  # bob's rocket text is invisible
    hits = store.search("how long does a cat sleep", k=6, doc_ids=[bob["id"]])
    assert hits and all(h["doc_id"] == bob["id"] for h in hits)


def test_empty_allowed_list_searches_nothing(two_users):
    assert store.search("how long does a cat sleep", k=6, doc_ids=[]) == []


def test_answer_with_empty_scope_abstains_instead_of_searching_everything(two_users):
    r = rag.answer("how long does a cat sleep", min_score=0.05, doc_ids=[])
    assert r["abstained"] and r["citations"] == [] and r["retrieved"] == []


def test_no_scope_still_searches_everything_for_the_old_app(two_users):
    alice, bob = two_users
    hits = store.search("how long does a cat sleep", k=6)
    assert {h["doc_id"] for h in hits} == {alice["id"], bob["id"]}


def test_same_file_for_two_users_is_indexed_separately_and_deleted_separately(tmp_path):
    path = write(tmp_path, "shared.txt", "Penguins live in the southern hemisphere and eat fish.")
    a = store.index_document(path, "shared.txt", user_id="alice")
    b = store.index_document(path, "shared.txt", user_id="bob")
    assert a["id"] != b["id"] and not a["duplicate"] and not b["duplicate"]
    assert store.count_chunks() == 2
    again = store.index_document(path, "shared.txt", user_id="alice")
    assert again["duplicate"] is True and again["id"] == a["id"]
    store.delete_document(a["id"])
    assert store.count_chunks(b["id"]) == 1
    assert store.search("where do penguins live", k=3, doc_ids=[b["id"]])


def test_documents_remember_their_owner(two_users):
    alice, bob = two_users
    assert registry.get_document(alice["id"])["user_id"] == "alice"
    assert [d["id"] for d in registry.list_documents(user_id="bob")] == [bob["id"]]