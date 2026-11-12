"""SQLite registry: users, login sessions, document metadata/status and persistent chat history.

Every document and chat message belongs to a user (`user_id`). The old single-user Streamlit
app keeps working: it simply uses user_id = "" (the "legacy" owner).
"""
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,      -- SHA-256 of the token; the token itself is never stored
    user_id TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL,             -- processing | ready | failed
    pages INTEGER DEFAULT 0,
    chunks INTEGER DEFAULT 0,
    error TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (user_id, sha256)          -- the same file may be uploaded by different users
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    abstained INTEGER NOT NULL DEFAULT 0,
    sources TEXT NOT NULL DEFAULT '[]',
    latency_ms INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _columns(con, table):
    return [row[1] for row in con.execute(f"PRAGMA table_info({table})")]


def _migrate(con):
    """Upgrade a database created by the earlier single-user version."""
    msg_cols = _columns(con, "messages")
    if msg_cols and "user_id" not in msg_cols:  # must come first: SCHEMA indexes this column
        con.execute("ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT ''")
    doc_cols = _columns(con, "documents")
    if doc_cols and "user_id" not in doc_cols:
        con.execute("ALTER TABLE documents RENAME TO documents_old")
        con.executescript(SCHEMA)
        con.execute(
            "INSERT INTO documents (id, user_id, filename, sha256, status, pages, chunks, error, created_at) "
            "SELECT id, '', filename, sha256, status, pages, chunks, error, created_at FROM documents_old"
        )
        con.execute("DROP TABLE documents_old")


@contextmanager
def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    _migrate(con)
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


# ---------------- users and sessions ----------------
def create_user(user_id, email, password_hash):
    with _conn() as c:
        c.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email, password_hash, _now()),
        )


def get_user_by_email(email):
    with _conn() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None


def get_user(user_id):
    with _conn() as c:
        row = c.execute("SELECT id, email, created_at FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_session(token_hash, user_id, expires_at):
    with _conn() as c:
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (token_hash, user_id, expires_at),
        )


def get_session_user(token_hash, now_iso):
    """The user for a live (not expired) session, else None. Expired rows are removed."""
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE expires_at <= ?", (now_iso,))
        row = c.execute(
            "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=?",
            (token_hash,),
        ).fetchone()
        return dict(row) if row else None


def delete_session(token_hash):
    with _conn() as c:
        c.execute("DELETE FROM sessions WHERE token_hash=?", (token_hash,))


# ---------------- documents ----------------
def create_document(doc_id, filename, sha256, user_id=""):
    with _conn() as c:
        c.execute(
            "INSERT INTO documents (id, user_id, filename, sha256, status, created_at) "
            "VALUES (?, ?, ?, ?, 'processing', ?)",
            (doc_id, user_id, filename, sha256, _now()),
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


def get_by_hash(sha256, user_id=""):
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM documents WHERE sha256=? AND user_id=?", (sha256, user_id)
        ).fetchone()
        return dict(row) if row else None


def list_documents(status=None, user_id=None):
    """user_id=None -> every user's documents; otherwise only that user's."""
    query, args = "SELECT * FROM documents WHERE 1=1", []
    if status:
        query += " AND status=?"
        args.append(status)
    if user_id is not None:
        query += " AND user_id=?"
        args.append(user_id)
    with _conn() as c:
        rows = c.execute(query + " ORDER BY created_at, rowid", args).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id):
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))


# ---------------- chat history ----------------
def add_message(question, answer, abstained, sources, latency_ms=0, user_id=""):
    with _conn() as c:
        c.execute(
            "INSERT INTO messages (user_id, question, answer, abstained, sources, latency_ms, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, question, answer, int(abstained), json.dumps(sources), int(latency_ms), _now()),
        )


def list_messages(limit=50, user_id=None):
    """Oldest-first list of the most recent `limit` messages."""
    query, args = "SELECT * FROM messages", []
    if user_id is not None:
        query += " WHERE user_id=?"
        args.append(user_id)
    with _conn() as c:
        rows = c.execute(query + " ORDER BY id DESC LIMIT ?", args + [limit]).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["abstained"] = bool(d["abstained"])
        d["sources"] = json.loads(d["sources"])
        out.append(d)
    return out


def clear_messages(user_id=None):
    with _conn() as c:
        if user_id is None:
            c.execute("DELETE FROM messages")
        else:
            c.execute("DELETE FROM messages WHERE user_id=?", (user_id,))


def stats(user_id=None):
    where, args = ("", []) if user_id is None else (" WHERE user_id=?", [user_id])
    with _conn() as c:
        q = c.execute("SELECT COUNT(*) FROM messages" + where, args).fetchone()[0]
        rows = c.execute("SELECT sources FROM messages" + where, args).fetchall()
    s = sum(len(json.loads(r[0])) for r in rows)
    return {"questions": q, "sources": s}

# Cloud mode (DOCUMIND_BACKEND=postgres): replace every function above with the Postgres version.
if config.BACKEND == "postgres":
    from registry_pg import *  # noqa: F401,F403,E402
