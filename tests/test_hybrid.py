"""Pure-logic tests for hybrid (dense + BM25) re-ranking."""
import hybrid


def make_hits(n=15, special_pos=12, special_text="The optimizer used was Adam with warmup"):
    hits = []
    for i in range(n):
        text = special_text if i == special_pos else f"generic filler content about topic number {i}"
        hits.append({"chunk_id": f"c{i}", "doc_id": "d", "filename": "f.pdf",
                     "page": 1, "text": text, "score": round(0.9 - i * 0.02, 3)})
    return hits


def corpus(hits):
    return hybrid.build_stats([h["text"] for h in hits] + ["unrelated words here"] * 20)


def test_tokenize_drops_stopwords():
    assert hybrid.tokenize("Which optimizer was used to train the Transformer?") == [
        "optimizer", "train", "transformer"]


def test_keyword_match_is_promoted_from_deep_rank_into_top_k():
    hits = make_hits()
    top = hybrid.fuse("Which optimizer was used?", hits, corpus(hits), k=6)
    assert "c12" in [h["chunk_id"] for h in top]
    assert len(top) == 6


def test_cosine_scores_are_untouched():
    hits = make_hits()
    top = hybrid.fuse("Which optimizer was used?", hits, corpus(hits), k=6)
    by_id = {h["chunk_id"]: h["score"] for h in hits}
    assert all(h["score"] == by_id[h["chunk_id"]] for h in top)


def test_best_dense_hit_is_always_kept():
    hits = make_hits()
    top = hybrid.fuse("Which optimizer was used?", hits, corpus(hits), k=3)
    assert "c0" in [h["chunk_id"] for h in top]


def test_no_usable_query_terms_returns_dense_order():
    hits = make_hits()
    top = hybrid.fuse("what is the", hits, corpus(hits), k=4)
    assert [h["chunk_id"] for h in top] == ["c0", "c1", "c2", "c3"]


def test_single_hit_and_empty_corpus_are_safe():
    one = make_hits(1, special_pos=0)
    assert hybrid.fuse("optimizer", one, hybrid.build_stats([]), k=6) == one
    assert hybrid.bm25_score(["x"], [], hybrid.build_stats([])) == 0.0


def test_possessive_does_not_create_stray_token():
    assert "s" not in hybrid.tokenize("ACME's notice period")
    assert "acme" in hybrid.tokenize("ACME's notice period")


def test_compound_question_is_split():
    q = "How many attention heads does the base Transformer use, and what is ACME's standard resignation notice period?"
    parts = hybrid.split_question(q)
    assert len(parts) == 2
    assert "heads" in parts[0] and "resignation" in parts[1]


def test_two_sentence_question_is_split():
    assert len(hybrid.split_question("How long does a cat sleep? What does a rocket need?")) == 2


def test_plain_and_is_not_split():
    q = "What is the difference between precision and recall?"
    assert hybrid.split_question(q) == [q]


def test_simple_question_unchanged():
    q = "Which optimizer was used to train the Transformer?"
    assert hybrid.split_question(q) == [q]


def test_round_robin_interleaves_and_dedupes():
    a = [{"chunk_id": x} for x in ("a1", "a2", "a3")]
    b = [{"chunk_id": x} for x in ("b1", "a1", "b3")]
    ids = [h["chunk_id"] for h in hybrid.merge_round_robin([a, b], 5)]
    assert ids == ["a1", "b1", "a2", "a3", "b3"]
