"""SQLite registry: document metadata/status and persistent chat history."""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,             -- processing | ready | failed
    pages INTEGER DEFAULT 0,
    chunks INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    abstained INTEGER NOT NULL DEFAULT 0,
    sources TEXT NOT NULL DEFAULT '[]',
    latency_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------------- documents ----------------
def create_document(doc_id, filename, sha256):
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (id, filename, sha256, status, created_at) "
            "VALUES (?, ?, ?, 'processing', ?)",
            (doc_id, filename, sha256, _now()),
        )


def mark_ready(doc_id, pages, chunks):
    with _conn() as c:
        c.execute(
            "UPDATE documents SET status='ready', pages=?, chunks=?, error=NULL WHERE id=?",
            (pages, chunks, doc_id),
        )


def mark_failed(doc_id, error):
    with _conn() as c:
        c.execute(
            "UPDATE documents SET status='failed', error=? WHERE id=?",
            (str(error)[:500], doc_id),
        )


def get_document(doc_id):
    with _conn() as c:
        row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return dict(row) if row else None


def get_by_hash(sha256):
    with _conn() as c:
        row = c.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
        return dict(row) if row else None


def list_documents(status=None):
    with _conn() as c:
        if status:
            rows = c.execute(
                "SELECT * FROM documents WHERE status=? ORDER BY created_at, rowid", (status,)
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM documents ORDER BY created_at, rowid").fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id):
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------------- chat history ----------------
def add_message(question, answer, abstained, sources, latency_ms=0):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (question, answer, abstained, sources, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (question, answer, int(abstained), json.dumps(sources), int(latency_ms), _now()),
        )


def list_messages(limit=50):
    """Oldest-first list of the most recent `limit` messages."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["abstained"] = bool(d["abstained"])
        d["sources"] = json.loads(d["sources"])
        out.append(d)
    return out


def clear_messages():
    with _conn() as c:
        c.execute("DELETE FROM messages")


def stats():
    with _conn() as c:
        q = c.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        rows = c.execute("SELECT sources FROM messages").fetchall()
    s = sum(len(json.loads(r[0])) for r in rows)
    return {"questions": q, "sources": s}
