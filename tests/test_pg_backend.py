"""Tests for the Postgres + pgvector backend.

They run ONLY when TEST_DATABASE_URL points at a THROWAWAY database (the local Docker one).
Otherwise the whole file is skipped, so normal runs and CI are unaffected.
Every test starts by DROPPING all DocuMind tables in that database.
"""
import math
import os
import re
import zlib

import pytest

TEST_URL = os.getenv("TEST_DATABASE_URL")
if not TEST_URL:
    pytest.skip("set TEST_DATABASE_URL to run the Postgres tests", allow_module_level=True)
if "neon.tech" in TEST_URL:
    raise RuntimeError("Refusing to run destructive tests against Neon. Use the local Docker database.")

psycopg = pytest.importorskip("psycopg")

import config  # noqa: E402
import pg  # noqa: E402
import registry_pg as registry  # noqa: E402
import store_pg as store  # noqa: E402


def fake_embed(texts):
    """Offline, deterministic bag-of-words embedder with the same size as the real one."""
    out = []
    for text in texts:
        v = [0.0] * config.EMBED_DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[zlib.crc32(tok.encode()) % config.EMBED_DIM] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    monkeypatch.setattr(config, "DATABASE_URL", TEST_URL)
    pg.drop_all()
    store.set_embedding_function(fake_embed)
    yield
    store.set_embedding_function(None)


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


# ---------------- registry ----------------
def test_users_and_sessions():
    registry.create_user("u1", "a@b.com", "hash")
    assert registry.get_user_by_email("a@b.com")["id"] == "u1"
    assert registry.get_user("u1")["email"] == "a@b.com"
    with pytest.raises(psycopg.errors.UniqueViolation):
        registry.create_user("u2", "a@b.com", "hash")
    now = "2026-01-01T00:00:00+00:00"
    registry.create_session("live", "u1", "2999-01-01T00:00:00+00:00")
    registry.create_session("old", "u1", "2000-01-01T00:00:00+00:00")
    assert registry.get_session_user("live", now)["email"] == "a@b.com"
    assert registry.get_session_user("old", now) is None  # expired
    registry.delete_session("live")
    assert registry.get_session_user("live", now) is None


def test_documents_are_unique_per_user():
    registry.create_document("d1", "a.txt", "sha1", "u1")
    registry.create_document("d2", "a.txt", "sha1", "u2")  # same file, other user: allowed
    with pytest.raises(psycopg.errors.UniqueViolation):
        registry.create_document("d3", "a.txt", "sha1", "u1")
    assert [d["id"] for d in registry.list_documents(user_id="u1")] == ["d1"]
    assert registry.get_by_hash("sha1", "u2")["id"] == "d2"
    registry.mark_ready("d1", pages=2, chunks=5)
    doc = registry.get_document("d1")
    assert doc["status"] == "ready" and doc["pages"] == 2 and doc["chunks"] == 5
    assert [d["id"] for d in registry.list_documents(status="ready")] == ["d1"]
    assert set(doc) == {"id", "user_id", "filename", "sha256", "status", "pages", "chunks", "error", "created_at"}


def test_messages_are_per_user_and_ordered():
    registry.add_message("q1", "a1", False, [{"label": "S1"}], 10, user_id="u1")
    registry.add_message("q2", "a2", True, [], 20, user_id="u1")
    registry.add_message("q3", "a3", False, [], 30, user_id="u2")
    msgs = registry.list_messages(50, user_id="u1")
    assert [m["question"] for m in msgs] == ["q1", "q2"]
    assert msgs[0]["sources"] == [{"label": "S1"}] and msgs[1]["abstained"] is True
    assert registry.stats(user_id="u1") == {"questions": 2, "sources": 1}
    registry.clear_messages(user_id="u1")
    assert registry.list_messages(50, user_id="u1") == []
    assert len(registry.list_messages(50, user_id="u2")) == 1


# ---------------- vector store ----------------
CATS = "Cats are small carnivorous mammals. A cat sleeps sixteen hours a day."
ROCKETS = "Rockets burn propellant to produce thrust. A rocket engine needs liquid oxygen."


def test_index_search_and_delete(tmp_path):
    rec = store.index_document(write(tmp_path, "cats.txt", CATS), "cats.txt", user_id="alice")
    assert rec["status"] == "ready" and rec["chunks"] >= 1 and rec["duplicate"] is False
    hits = store.search("how long does a cat sleep", k=3, doc_ids=[rec["id"]])
    assert hits and hits[0]["filename"] == "cats.txt" and hits[0]["page"] == 1
    assert hits[0]["doc_id"] == rec["id"]
    store.delete_document(rec["id"])
    assert store.count_chunks() == 0 and registry.get_document(rec["id"]) is None


def test_a_user_only_finds_their_own_documents(tmp_path):
    alice = store.index_document(write(tmp_path, "cats.txt", CATS), "cats.txt", user_id="alice")
    bob = store.index_document(write(tmp_path, "rockets.txt", ROCKETS), "rockets.txt", user_id="bob")
    hits = store.search("what does a rocket engine need", k=6, doc_ids=[alice["id"]])
    assert hits and all(h["doc_id"] == alice["id"] for h in hits)
    hits = store.search("how long does a cat sleep", k=6, doc_ids=[bob["id"]])
    assert hits and all(h["doc_id"] == bob["id"] for h in hits)
    assert store.search("how long does a cat sleep", k=6, doc_ids=[]) == []
    everything = store.search("how long does a cat sleep", k=6)  # old single-user behaviour
    assert {h["doc_id"] for h in everything} == {alice["id"], bob["id"]}
    assert everything[0]["doc_id"] == alice["id"] and "bm25" in everything[0]  # hybrid re-ranking ran


def test_same_file_for_two_users_and_duplicates(tmp_path):
    path = write(tmp_path, "shared.txt", "Penguins live in the southern hemisphere and eat fish.")
    a = store.index_document(path, "shared.txt", user_id="alice")
    b = store.index_document(path, "shared.txt", user_id="bob")
    assert a["id"] != b["id"] and not a["duplicate"] and not b["duplicate"]
    assert store.count_chunks() == 2
    again = store.index_document(path, "shared.txt", user_id="alice")
    assert again["duplicate"] is True and again["id"] == a["id"]
    store.delete_document(a["id"])
    assert store.count_chunks(b["id"]) == 1 and store.count_chunks(a["id"]) == 0


def test_failed_index_leaves_no_chunks(tmp_path):
    with pytest.raises(ValueError):
        store.index_document(write(tmp_path, "empty.txt", "   "), "empty.txt", user_id="alice")
    assert store.count_chunks() == 0
    docs = registry.list_documents(user_id="alice")
    assert len(docs) == 1 and docs[0]["status"] == "failed"


def test_connections_are_reused_and_stale_ones_are_replaced(monkeypatch):
    with pg.connect() as first:
        first.execute("SELECT 1")
    with pg.connect() as second:
        assert second is first  # reused, not reopened
    monkeypatch.setattr(pg, "IDLE_SECONDS", 0)  # everything now counts as stale
    with pg.connect() as third:
        assert third is not first and first.closed
        assert third.execute("SELECT 1 AS one").fetchone()["one"] == 1


def test_a_failed_statement_does_not_poison_the_next_connection():
    with pytest.raises(psycopg.errors.UndefinedTable):
        with pg.connect() as con:
            con.execute("SELECT * FROM no_such_table")
    with pg.connect() as con:  # a fresh, working connection
        assert con.execute("SELECT 1 AS one").fetchone()["one"] == 1


def test_a_failed_block_rolls_back_its_writes():
    registry.create_user("u1", "roll@x.com", "hash")
    before = pg_count()
    assert before == 1
    with pytest.raises(RuntimeError):
        with pg.connect() as con:
            con.execute("DELETE FROM users")
            raise RuntimeError("boom")
    assert pg_count() == before


def pg_count():
    with pg.connect() as con:
        return con.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]


def test_embedding_is_done_in_small_batches():
    sizes = []

    def counting(texts):
        sizes.append(len(texts))
        return fake_embed(texts)

    store.set_embedding_function(counting)
    vectors = store._embed([f"text number {i}" for i in range(store.EMBED_BATCH * 2 + 3)])
    assert len(vectors) == store.EMBED_BATCH * 2 + 3
    assert sizes == [store.EMBED_BATCH, store.EMBED_BATCH, 3]
