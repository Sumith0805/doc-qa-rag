"""Hybrid re-ranking: combine dense (embedding) rank with BM25 keyword rank via
Reciprocal Rank Fusion (RRF). Pure Python, no extra dependencies.

Why: embeddings can bury a chunk whose answer is a literal keyword ("optimizer") inside a
chunk full of other content. BM25 rewards exact rare words, so the two signals complement
each other. The cosine `score` on each hit is left untouched, so the abstention gate in
rag.py still works on real semantic similarity.
"""
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been by with as at from that this it its "
    "which what who whom whose when where why how do does did used use using can could would "
    "should many much about into than then there their they them has have had not".split()
)


_POSSESSIVE = re.compile(r"['\u2019]s\b")


def tokenize(text):
    text = _POSSESSIVE.sub("", text.lower())  # "ACME's" -> "acme" (no stray "s" token)
    return [t for t in _TOKEN.findall(text) if t not in STOPWORDS]


def build_stats(docs):
    """Corpus statistics for BM25 over ALL indexed chunks."""
    df = Counter()
    total_len = 0
    for d in docs:
        toks = tokenize(d)
        total_len += len(toks)
        df.update(set(toks))
    n = len(docs)
    return {"n": n, "df": df, "avgdl": (total_len / n) if n else 0.0}


def bm25_score(query_tokens, doc_tokens, stats, k1=1.5, b=0.75):
    if not doc_tokens or not stats["n"]:
        return 0.0
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for q in set(query_tokens):
        f = tf.get(q, 0)
        if not f:
            continue
        df = stats["df"].get(q, 0)
        idf = math.log(1 + (stats["n"] - df + 0.5) / (df + 0.5))
        score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / (stats["avgdl"] or 1)))
    return score


def fuse(question, hits, stats, k, rrf_k=60, bm25_weight=1.0):
    """Re-rank `hits` (sorted best-first by dense similarity) and return the top k.
    bm25_weight scales the keyword vote: 1.0 = equal to the embedding vote, 0 = ignore keywords.
    The single best dense hit is always kept so the abstention gate is unaffected."""
    q = tokenize(question)
    if len(hits) <= 1 or not q:
        return hits[:k]

    bm = [bm25_score(q, tokenize(h["text"]), stats) for h in hits]
    by_bm = sorted((i for i in range(len(hits)) if bm[i] > 0), key=lambda i: -bm[i])
    bm_rank = {i: r for r, i in enumerate(by_bm, start=1)}

    fused = []
    for i in range(len(hits)):
        s = 1.0 / (rrf_k + i + 1)  # dense rank = position in `hits`
        if i in bm_rank:
            s += bm25_weight / (rrf_k + bm_rank[i])
        fused.append(s)

    order = sorted(range(len(hits)), key=lambda i: (-fused[i], i))[:k]
    if 0 not in order:  # keep the best semantic match
        order[-1] = 0
    return [{**hits[i], "bm25": round(bm[i], 3)} for i in order]


# ---------------- multi-query retrieval helpers ----------------
_QWORDS = r"(?:what|how|which|who|whom|when|where|why|is|are|does|do|did|can|could|should)"
_SPLIT = re.compile(
    rf"\s*;\s*|\s*,?\s+and\s+(?={_QWORDS}\b)|\?\s+(?=\S)", flags=re.IGNORECASE
)


def split_question(question, max_parts=3):
    """Split a compound question ("A?, and what is B?") into independent sub-questions.
    Plain "X and Y" phrases are NOT split; only a clause boundary followed by a question word."""
    parts = [p.strip(" ,;?") for p in _SPLIT.split(question) if p and p.strip(" ,;?")]
    parts = [p + "?" for p in parts if len(tokenize(p)) >= 2]
    if len(parts) < 2 or len(parts) > max_parts:
        return [question]
    return parts


def merge_round_robin(result_lists, k):
    """Interleave ranked result lists (best of each first), dropping duplicate chunk_ids."""
    merged, seen = [], set()
    for rank in range(max((len(r) for r in result_lists), default=0)):
        for results in result_lists:
            if rank < len(results) and results[rank]["chunk_id"] not in seen:
                seen.add(results[rank]["chunk_id"])
                merged.append(results[rank])
                if len(merged) == k:
                    return merged
    return merged