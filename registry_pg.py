"""Postgres version of registry.py: same function names, same return shapes.

registry.py swaps itself for this module when DOCUMIND_BACKEND=postgres, so auth.py and the API
need no changes.
"""
import json
from datetime import datetime, timezone

import pg

__all__ = [
    "create_user", "get_user_by_email", "get_user", "create_session", "get_session_user",
    "delete_session", "create_document", "mark_ready", "mark_failed", "get_document",
    "get_by_hash", "list_documents", "delete_document", "add_message", "list_messages",
    "clear_messages", "stats",
]

DOC_COLS = "id, user_id, filename, sha256, status, pages, chunks, error, created_at"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- users and sessions ----------------
def create_user(user_id, email, password_hash):
    with pg.connect() as c:
        c.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, email, password_hash, _now()),
        )


def get_user_by_email(email):
    with pg.connect() as c:
        return c.execute("SELECT * FROM users WHERE email=%s", (email,)).fetchone()


def get_user(user_id):
    with pg.connect() as c:
        return c.execute(
            "SELECT id, email, created_at FROM users WHERE id=%s", (user_id,)
        ).fetchone()


def create_session(token_hash, user_id, expires_at):
    with pg.connect() as c:
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
            (token_hash, user_id, expires_at),
        )


def get_session_user(token_hash, now_iso):
    """The user for a live (not expired) session, else None. Expired rows are removed."""
    with pg.connect() as c:
        c.execute("DELETE FROM sessions WHERE expires_at <= %s", (now_iso,))
        return c.execute(
            "SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token_hash=%s",
            (token_hash,),
        ).fetchone()


def delete_session(token_hash):
    with pg.connect() as c:
        c.execute("DELETE FROM sessions WHERE token_hash=%s", (token_hash,))


# ---------------- documents ----------------
def create_document(doc_id, filename, sha256, user_id=""):
    with pg.connect() as c:
        c.execute(
            "INSERT INTO documents (id, user_id, filename, sha256, status, created_at) "
            "VALUES (%s, %s, %s, %s, 'processing', %s)",
            (doc_id, user_id, filename, sha256, _now()),
        )


def mark_ready(doc_id, pages, chunks):
    with pg.connect() as c:
        c.execute(
            "UPDATE documents SET status='ready', pages=%s, chunks=%s, error=NULL WHERE id=%s",
            (pages, chunks, doc_id),
        )


def mark_failed(doc_id, error):
    with pg.connect() as c:
        c.execute(
            "UPDATE documents SET status='failed', error=%s WHERE id=%s",
            (str(error)[:500], doc_id),
        )


def get_document(doc_id):
    with pg.connect() as c:
        return c.execute(
            f"SELECT {DOC_COLS} FROM documents WHERE id=%s", (doc_id,)
        ).fetchone()


def get_by_hash(sha256, user_id=""):
    with pg.connect() as c:
        return c.execute(
            f"SELECT {DOC_COLS} FROM documents WHERE sha256=%s AND user_id=%s",
            (sha256, user_id),
        ).fetchone()


def list_documents(status=None, user_id=None):
    """user_id=None -> every user's documents; otherwise only that user's."""
    query, args = f"SELECT {DOC_COLS} FROM documents WHERE 1=1", []
    if status:
        query += " AND status=%s"
        args.append(status)
    if user_id is not None:
        query += " AND user_id=%s"
        args.append(user_id)
    with pg.connect() as c:
        return c.execute(query + " ORDER BY created_at, seq", args).fetchall()


def delete_document(doc_id):
    with pg.connect() as c:
        c.execute("DELETE FROM documents WHERE id=%s", (doc_id,))


# ---------------- chat history ----------------
def add_message(question, answer, abstained, sources, latency_ms=0, user_id=""):
    with pg.connect() as c:
        c.execute(
            "INSERT INTO messages (user_id, question, answer, abstained, sources, latency_ms, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (user_id, question, answer, int(abstained), json.dumps(sources), int(latency_ms), _now()),
        )


def list_messages(limit=50, user_id=None):
    """Oldest-first list of the most recent `limit` messages."""
    query, args = "SELECT * FROM messages", []
    if user_id is not None:
        query += " WHERE user_id=%s"
        args.append(user_id)
    with pg.connect() as c:
        rows = c.execute(query + " ORDER BY id DESC LIMIT %s", args + [limit]).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        d["abstained"] = bool(d["abstained"])
        d["sources"] = json.loads(d["sources"])
        out.append(d)
    return out


def clear_messages(user_id=None):
    with pg.connect() as c:
        if user_id is None:
            c.execute("DELETE FROM messages")
        else:
            c.execute("DELETE FROM messages WHERE user_id=%s", (user_id,))


def stats(user_id=None):
    where, args = ("", []) if user_id is None else (" WHERE user_id=%s", [user_id])
    with pg.connect() as c:
        q = c.execute("SELECT COUNT(*) AS n FROM messages" + where, args).fetchone()["n"]
        rows = c.execute("SELECT sources FROM messages" + where, args).fetchall()
    s = sum(len(json.loads(r["sources"])) for r in rows)
    return {"questions": q, "sources": s}
