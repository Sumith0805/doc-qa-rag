import sqlite3

import pytest

import registry


def test_document_lifecycle():
    registry.create_document("d1", "a.pdf", "hash1")
    assert registry.get_document("d1")["status"] == "processing"
    registry.mark_ready("d1", pages=3, chunks=10)
    doc = registry.get_document("d1")
    assert (doc["status"], doc["pages"], doc["chunks"]) == ("ready", 3, 10)
    registry.delete_document("d1")
    assert registry.get_document("d1") is None


def test_failed_status_keeps_error():
    registry.create_document("d1", "a.pdf", "h")
    registry.mark_failed("d1", "boom")
    doc = registry.get_document("d1")
    assert doc["status"] == "failed" and doc["error"] == "boom"


def test_same_hash_cannot_be_registered_twice():
    registry.create_document("d1", "a.pdf", "same")
    with pytest.raises(sqlite3.IntegrityError):
        registry.create_document("d2", "b.pdf", "same")
    assert registry.get_by_hash("same")["id"] == "d1"


def test_list_documents_by_status():
    registry.create_document("d1", "a.pdf", "h1")
    registry.create_document("d2", "b.pdf", "h2")
    registry.mark_ready("d1", 1, 1)
    assert [d["id"] for d in registry.list_documents(status="ready")] == ["d1"]
    assert len(registry.list_documents()) == 2


def test_chat_history_persists_in_order():
    registry.add_message("q1", "a1", False, [{"page": 1}], 10)
    registry.add_message("q2", "a2", True, [], 5)
    msgs = registry.list_messages()
    assert [m["question"] for m in msgs] == ["q1", "q2"]
    assert msgs[0]["sources"] == [{"page": 1}] and msgs[1]["abstained"] is True
    assert registry.stats() == {"questions": 2, "sources": 1}
    registry.clear_messages()
    assert registry.list_messages() == []
