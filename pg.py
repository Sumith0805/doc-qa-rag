"""Postgres connection helper and schema (Neon + pgvector).

Only imported when DOCUMIND_BACKEND=postgres (or by the Postgres tests).
Each call opens a short-lived connection. That is slower than a pool, but it survives Neon
suspending an idle database (a pooled connection would go stale). Revisit if latency matters.
"""
import threading

import psycopg
from psycopg.rows import dict_row

import config

_SCHEMA = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        seq BIGSERIAL,
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        filename TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        status TEXT NOT NULL,
        pages INTEGER DEFAULT 0,
        chunks INTEGER DEFAULT 0,
        error TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (user_id, sha256)
    )""",
    """CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL DEFAULT '',
        question TEXT NOT NULL,
        answer TEXT NOT NULL,
        abstained INTEGER NOT NULL DEFAULT 0,
        sources TEXT NOT NULL DEFAULT '[]',
        latency_ms INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    )""",
    f"""CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL DEFAULT '',
        filename TEXT NOT NULL,
        page INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding vector({config.EMBED_DIM}) NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_documents_user ON documents (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (doc_id)",
    # No vector (HNSW) index on purpose: with a few thousand chunks an exact scan is fast and
    # always returns the true nearest neighbours, even when filtered by document.
]

_lock = threading.Lock()
_ready = False


def _open():
    if not config.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(config.DATABASE_URL, row_factory=dict_row, connect_timeout=20)


def ensure_schema():
    """Create tables once per process. Safe to call repeatedly."""
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        with _open() as con:
            for stmt in _SCHEMA:
                con.execute(stmt)
        _ready = True


def connect():
    """Use as:  with connect() as con: ...   (commits on success, rolls back on error, then closes)."""
    ensure_schema()
    return _open()


def drop_all():
    """TESTS ONLY: delete every DocuMind table. Never point this at a database you care about."""
    global _ready
    with _open() as con:
        con.execute("DROP TABLE IF EXISTS chunks, messages, documents, sessions, users CASCADE")
    _ready = False
