"""pgvector version of store.py: index, search (hybrid dense + BM25), delete. Same function names.

Differences from the Chroma version, on purpose:
  * vectors live in Postgres (Neon), so the server keeps no state on disk;
  * BM25 statistics are computed over the documents being searched (the user's own), not over
    every user's chunks;
  * the embedding model is still Chroma's default MiniLM (ONNX), used only to turn text into
    384 numbers. It is baked into the Docker image at build time.
"""
import hashlib
import uuid
from pathlib import Path

import config
import hybrid
import pg
import registry_pg as registry
from chunker import chunk_text, read_document


class DemoLimitError(RuntimeError):
    """An action that the public demo does not allow (too big, library full, delete)."""


_embedding_fn = None  # None -> Chroma's default MiniLM; tests plug in a fast fake
_default_embedder = None
_bm25_cache = {}

EMBED_BATCH = 8

INSERT_SQL = (
    "INSERT INTO chunks (chunk_id, doc_id, user_id, filename, page, chunk_index, content, embedding) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)"
)


def set_embedding_function(fn):
    """Used by tests to plug in a fast fake embedder (callable: list[str] -> list[list[float]])."""
    global _embedding_fn
    _embedding_fn = fn
    _bm25_cache.clear()


def reset_client():
    _bm25_cache.clear()


def _embed(texts):
    global _default_embedder
    fn = _embedding_fn
    if fn is None:
        if _default_embedder is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

            _default_embedder = DefaultEmbeddingFunction()
        fn = _default_embedder
    # Embed a few chunks at a time. Measured on a 66-chunk PDF: one big call peaked at ~670 MB of
    # memory, batches of 8 at ~290 MB, for about one extra second. This is what lets the API fit
    # on a small free host.
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend([float(x) for x in v] for v in fn(texts[start:start + EMBED_BATCH]))
    if vectors and len(vectors[0]) != config.EMBED_DIM:
        raise RuntimeError(
            f"Embedding size {len(vectors[0])} does not match DOCUMIND_EMBED_DIM={config.EMBED_DIM}"
        )
    return vectors


def _vec(v):
    """pgvector text format: [0.1,0.2,...] (cast with ::vector in SQL)."""
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def _get_bm25_stats(doc_ids):
    """BM25 corpus statistics over the searched documents; cached until the next index/delete."""
    key = tuple(sorted(doc_ids)) if doc_ids else None
    if key not in _bm25_cache:
        if len(_bm25_cache) > 64:
            _bm25_cache.clear()
        with pg.connect() as c:
            if doc_ids:
                rows = c.execute(
                    "SELECT content FROM chunks WHERE doc_id = ANY(%s)", (list(doc_ids),)
                ).fetchall()
            else:
                rows = c.execute("SELECT content FROM chunks").fetchall()
        _bm25_cache[key] = hybrid.build_stats([r["content"] for r in rows])
    return _bm25_cache[key]


def _delete_chunks(doc_id):
    _bm25_cache.clear()
    with pg.connect() as c:
        c.execute("DELETE FROM chunks WHERE doc_id=%s", (doc_id,))


def _check_demo_limits(path, user_id=""):
    """Public-demo guard rails (no effect unless DOCUMIND_DEMO=1)."""
    if not config.DEMO_MODE:
        return
    size_mb = Path(path).stat().st_size / (1024 * 1024)
    if size_mb > config.DEMO_MAX_UPLOAD_MB:
        raise DemoLimitError(
            f"Files are limited to {config.DEMO_MAX_UPLOAD_MB} MB in the public demo."
        )
    in_library = [d for d in registry.list_documents(user_id=user_id) if d["status"] != "failed"]
    if len(in_library) >= config.DEMO_MAX_DOCS:
        raise DemoLimitError(
            f"The public demo library is full ({config.DEMO_MAX_DOCS} documents). "
            "You can still ask questions about the documents already loaded."
        )


def index_document(path, filename, user_id=""):
    """Index one file WITHOUT touching other documents. Returns the registry record.
    Re-uploading an identical file returns the existing record (record["duplicate"] = True)."""
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    existing = registry.get_by_hash(sha, user_id)
    if existing:
        if existing["status"] == "ready":
            return {**existing, "duplicate": True}
        _remove_document(existing["id"])  # failed / stuck earlier attempt -> retry cleanly

    _check_demo_limits(path, user_id)
    doc_id = uuid.uuid4().hex[:12]
    registry.create_document(doc_id, filename, sha, user_id)  # status = processing
    try:
        pages = read_document(path, filename)
        chunks = chunk_text(pages)
        if not chunks:
            raise ValueError("No extractable text found (scanned PDF? OCR is not supported).")
        # Embed everything first (slow, no database connection held), then insert in one short transaction.
        rows = []
        B = 64
        for i in range(0, len(chunks), B):
            batch = chunks[i : i + B]
            vectors = _embed([c["text"] for c in batch])
            for j, c in enumerate(batch):
                rows.append(
                    (f"{doc_id}:{i + j}", doc_id, user_id, filename, c["page"], i + j,
                     c["text"], _vec(vectors[j]))
                )
        with pg.connect() as con, con.cursor() as cur:
            cur.executemany(INSERT_SQL, rows)
        _bm25_cache.clear()
        registry.mark_ready(doc_id, pages=len(pages), chunks=len(chunks))
    except Exception as exc:
        try:
            _delete_chunks(doc_id)  # never leave partial chunks behind
        finally:
            registry.mark_failed(doc_id, exc)
        raise
    return {**registry.get_document(doc_id), "duplicate": False}


def search(question, k=None, doc_ids=None):
    """Hybrid (dense + BM25) search across all (or selected) documents.
    doc_ids: only search these documents (None = all documents; [] = nothing)."""
    k = k or config.TOP_K
    if doc_ids is not None and not list(doc_ids):
        return []  # an EMPTY allowed-list means "search nothing", never "search everything"
    parts = hybrid.split_question(question) if config.MULTI_QUERY else [question]
    if len(parts) == 1:
        return _search_one(question, k, doc_ids)
    return hybrid.merge_round_robin([_search_one(p, k, doc_ids) for p in parts], k)


def _search_one(question, k, doc_ids=None):
    pool = max(k, config.CANDIDATE_POOL) if config.HYBRID else k
    qv = _vec(_embed([question])[0])
    sql = (
        "SELECT chunk_id, doc_id, filename, page, content AS text, "
        "1 - (embedding <=> %s::vector) AS score FROM chunks "
    )
    args = [qv]
    if doc_ids:
        sql += "WHERE doc_id = ANY(%s) "
        args.append(list(doc_ids))
    sql += "ORDER BY embedding <=> %s::vector LIMIT %s"
    args += [qv, pool]
    with pg.connect() as c:
        rows = c.execute(sql, args).fetchall()
    hits = [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "filename": r["filename"],
            "page": r["page"],
            "text": r["text"],
            "score": round(float(r["score"]), 4),
        }
        for r in rows
    ]
    if config.HYBRID and hits:
        hits = hybrid.fuse(
            question, hits, _get_bm25_stats(doc_ids), k,
            rrf_k=config.RRF_K, bm25_weight=config.BM25_WEIGHT,
        )
    return hits[:k]


def delete_document(doc_id):
    """Public delete (used by the API). Disabled on the public demo."""
    if config.DEMO_MODE:
        raise DemoLimitError("Deleting documents is disabled in the public demo.")
    _remove_document(doc_id)


def _remove_document(doc_id):
    """Remove the document's chunks, then its registry row. Raises if any chunk is left."""
    _delete_chunks(doc_id)
    with pg.connect() as c:
        left = c.execute("SELECT 1 FROM chunks WHERE doc_id=%s LIMIT 1", (doc_id,)).fetchone()
    if left:
        raise RuntimeError(f"Deletion incomplete: chunks of {doc_id} still in index")
    registry.delete_document(doc_id)


def count_chunks(doc_id=None):
    with pg.connect() as c:
        if doc_id is None:
            return c.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return c.execute("SELECT COUNT(*) AS n FROM chunks WHERE doc_id=%s", (doc_id,)).fetchone()["n"]
