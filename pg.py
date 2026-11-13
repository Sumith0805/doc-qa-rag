"""Postgres connection helper and schema (Neon + pgvector).

Only imported when DOCUMIND_BACKEND=postgres (or by the Postgres tests).
Connections are reused: opening one to Neon costs a TLS handshake and a round trip to another
continent, and a single request runs several queries. A connection that has sat idle longer than
IDLE_SECONDS is thrown away instead of reused, because Neon suspends idle databases and would
leave it stale.
"""
import contextlib
import threading
import time

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


IDLE_SECONDS = 60   # reuse a connection only if it was used this recently
MAX_IDLE = 4        # how many idle connections to keep
_idle = []          # (connection, database url, time it was last used)
_idle_lock = threading.Lock()


def _take():
    """An idle connection that is still fresh and points at the current database, else a new one."""
    now = time.monotonic()
    stale, found = [], None
    with _idle_lock:
        while _idle and found is None:
            con, url, used = _idle.pop()
            if url == config.DATABASE_URL and now - used < IDLE_SECONDS and not con.closed:
                found = con
            else:
                stale.append(con)
    for con in stale:
        _close(con)
    return found or _open()


def _close(con):
    try:
        con.close()
    except Exception:
        pass  # it is being discarded anyway


def close_idle():
    """Close every idle connection (tests, shutdown)."""
    with _idle_lock:
        cons = [c for c, _, _ in _idle]
        _idle.clear()
    for con in cons:
        _close(con)


@contextlib.contextmanager
def connect():
    """Use as:  with connect() as con: ...   Commits on success. On any error it rolls back by
    closing the connection, so a broken connection is never handed to the next caller."""
    ensure_schema()
    con = _take()
    try:
        yield con
        con.commit()
    except BaseException:
        _close(con)
        raise
    with _idle_lock:
        if len(_idle) < MAX_IDLE:
            _idle.append((con, config.DATABASE_URL, time.monotonic()))
            con = None
    if con is not None:
        _close(con)


def drop_all():
    """TESTS ONLY: delete every DocuMind table. Never point this at a database you care about."""
    global _ready
    close_idle()
    with _open() as con:
        con.execute("DROP TABLE IF EXISTS chunks, messages, documents, sessions, users CASCADE")
    _ready = False
