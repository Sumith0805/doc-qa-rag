"""Per-user documents and chat history, and the upgrade from the old single-user database."""
import sqlite3

import pytest

import config
import registry


def test_two_users_can_upload_the_same_file_but_not_twice_each():
    registry.create_document("d1", "a.pdf", "samehash", user_id="alice")
    registry.create_document("d2", "a.pdf", "samehash", user_id="bob")
    with pytest.raises(sqlite3.IntegrityError):
        registry.create_document("d3", "a.pdf", "samehash", user_id="alice")
    assert registry.get_by_hash("samehash", user_id="alice")["id"] == "d1"
    assert registry.get_by_hash("samehash", user_id="bob")["id"] == "d2"
    assert registry.get_by_hash("samehash", user_id="carol") is None


def test_documents_are_listed_per_user():
    registry.create_document("d1", "a.pdf", "h1", user_id="alice")
    registry.create_document("d2", "b.pdf", "h2", user_id="bob")
    registry.mark_ready("d1", 1, 1)
    assert [d["id"] for d in registry.list_documents(user_id="alice")] == ["d1"]
    assert [d["id"] for d in registry.list_documents(user_id="bob")] == ["d2"]
    assert [d["id"] for d in registry.list_documents(user_id="alice", status="ready")] == ["d1"]
    assert registry.list_documents(user_id="nobody") == []
    assert len(registry.list_documents()) == 2  # no filter = everyone (legacy app)


def test_chat_history_and_stats_are_per_user():
    registry.add_message("q1", "a1", False, [{"page": 1}], 5, user_id="alice")
    registry.add_message("q2", "a2", True, [], 5, user_id="bob")
    assert [m["question"] for m in registry.list_messages(user_id="alice")] == ["q1"]
    assert registry.stats(user_id="alice") == {"questions": 1, "sources": 1}
    registry.clear_messages(user_id="alice")
    assert registry.list_messages(user_id="alice") == []
    assert [m["question"] for m in registry.list_messages(user_id="bob")] == ["q2"]


def test_old_single_user_database_is_upgraded_without_losing_data():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    old = sqlite3.connect(config.DB_PATH)
    old.executescript("""
        CREATE TABLE documents (id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL, pages INTEGER DEFAULT 0, chunks INTEGER DEFAULT 0, error TEXT, created_at TEXT NOT NULL);
        CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, answer TEXT NOT NULL,
            abstained INTEGER NOT NULL DEFAULT 0, sources TEXT NOT NULL DEFAULT '[]', latency_ms INTEGER DEFAULT 0, created_at TEXT NOT NULL);
        INSERT INTO documents VALUES ('old1', 'old.pdf', 'oldhash', 'ready', 3, 9, NULL, '2026-01-01T00:00:00+00:00');
        INSERT INTO messages (question, answer, abstained, sources, latency_ms, created_at)
            VALUES ('old q', 'old a', 0, '[]', 7, '2026-01-01T00:00:00+00:00');
    """)
    old.commit()
    old.close()

    docs = registry.list_documents()
    assert [(d["id"], d["user_id"], d["chunks"]) for d in docs] == [("old1", "", 9)]
    assert registry.list_messages()[0]["question"] == "old q"
    assert registry.list_messages()[0]["user_id"] == ""
    # new rules apply after the upgrade: another user may upload the identical file
    registry.create_document("new1", "old.pdf", "oldhash", user_id="alice")
    assert len(registry.list_documents()) == 2