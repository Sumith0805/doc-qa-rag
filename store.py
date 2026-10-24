"""Vector store (ChromaDB) + document lifecycle: index, search across ALL documents, delete."""
import hashlib
import uuid
from pathlib import Path

import chromadb

import config
import hybrid
import registry
from chunker import chunk_text, read_document

_client = None
_collection = None
_bm25_stats = None
_embedding_fn = None  # None -> Chroma default (all-MiniLM-L6-v2 via ONNX)


def set_embedding_function(fn):
    """Used by tests to plug in a fast fake embedder."""
    global _embedding_fn
    _embedding_fn = fn
    reset_client()


def reset_client():
    global _client, _collection, _bm25_stats
    _client = None
    _collection = None
    _bm25_stats = None


def _get_bm25_stats():
    """BM25 corpus statistics over all chunks; rebuilt after any index/delete."""
    global _bm25_stats
    if _bm25_stats is None:
        docs = _get_collection().get(include=["documents"])["documents"]
        _bm25_stats = hybrid.build_stats(docs)
    return _bm25_stats


def _get_collection():
    global _client, _collection
    if _collection is None:
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        kwargs = {"name": "documents"}
        if _embedding_fn is not None:
            kwargs["embedding_function"] = _embedding_fn
        try:  # newer Chroma API
            _collection = _client.get_or_create_collection(
                configuration={"hnsw": {"space": "cosine"}}, **kwargs
            )
        except Exception:  # older Chroma API
            _collection = _client.get_or_create_collection(
                metadata={"hnsw:space": "cosine"}, **kwargs
            )
    return _collection


def _delete_chunks(doc_id):
    global _bm25_stats
    _bm25_stats = None
    _get_collection().delete(where={"doc_id": doc_id})


def index_document(path, filename):
    """Index one file WITHOUT touching other documents. Returns the registry record.

    Re-uploading an identical file returns the existing record (record["duplicate"] = True).
    """
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    existing = registry.get_by_hash(sha)
    if existing:
        if existing["status"] == "ready":
            return {**existing, "duplicate": True}
        delete_document(existing["id"])  # failed / stuck earlier attempt -> retry cleanly

    doc_id = uuid.uuid4().hex[:12]
    registry.create_document(doc_id, filename, sha)  # status = processing
    try:
        pages = read_document(path, filename)
        chunks = chunk_text(pages)
        if not chunks:
            raise ValueError("No extractable text found (scanned PDF? OCR is not supported).")
        col = _get_collection()
        B = 64
        for i in range(0, len(chunks), B):
            batch = chunks[i : i + B]
            col.add(
                ids=[f"{doc_id}:{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "page": c["page"],
                        "chunk_index": i + j,
                    }
                    for j, c in enumerate(batch)
                ],
            )
        _invalidate_bm25()
        registry.mark_ready(doc_id, pages=len(pages), chunks=len(chunks))
    except Exception as exc:
        try:
            _delete_chunks(doc_id)  # never leave partial chunks behind
        finally:
            registry.mark_failed(doc_id, exc)
        raise
    return {**registry.get_document(doc_id), "duplicate": False}


def _invalidate_bm25():
    global _bm25_stats
    _bm25_stats = None


def search(question, k=None, doc_ids=None):
    """Hybrid (dense + BM25) search across all (or selected) documents.
    Compound questions are split into sub-questions that are retrieved separately and
    interleaved, so each part gets its own evidence.
    score = cosine similarity in [-1, 1] (relative to the sub-question that found the hit)."""
    k = k or config.TOP_K
    parts = hybrid.split_question(question) if config.MULTI_QUERY else [question]
    if len(parts) == 1:
        return _search_one(question, k, doc_ids)
    return hybrid.merge_round_robin([_search_one(p, k, doc_ids) for p in parts], k)


def _search_one(question, k, doc_ids=None):
    col = _get_collection()
    total = col.count()
    if total == 0:
        return []
    pool = max(k, config.CANDIDATE_POOL) if config.HYBRID else k
    kwargs = {"query_texts": [question], "n_results": min(pool, total)}
    if doc_ids:
        ids = list(doc_ids)
        kwargs["where"] = {"doc_id": ids[0]} if len(ids) == 1 else {"doc_id": {"$in": ids}}
    res = col.query(include=["documents", "metadatas", "distances"], **kwargs)
    hits = []
    for cid, text, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "chunk_id": cid,
                "doc_id": meta["doc_id"],
                "filename": meta["filename"],
                "page": meta["page"],
                "text": text,
                "score": round(1.0 - float(dist), 4),
            }
        )
    if config.HYBRID:
        hits = hybrid.fuse(
            question, hits, _get_bm25_stats(), k, rrf_k=config.RRF_K, bm25_weight=config.BM25_WEIGHT
        )
    return hits[:k]


def delete_document(doc_id):
    """Remove the document's chunks + embeddings from the index, then its registry row.
    Raises if any chunk is still retrievable afterwards."""
    _delete_chunks(doc_id)
    leftover = _get_collection().get(where={"doc_id": doc_id}, limit=1)["ids"]
    if leftover:
        raise RuntimeError(f"Deletion incomplete: chunks of {doc_id} still in index")
    registry.delete_document(doc_id)


def count_chunks(doc_id=None):
    col = _get_collection()
    if doc_id is None:
        return col.count()
    return len(col.get(where={"doc_id": doc_id})["ids"])