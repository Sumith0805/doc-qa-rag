"""Pure-logic tests for grounding/abstention. No vector DB and no real LLM needed."""
import rag


def hit(i, filename="a.pdf", page=1, text="some text", score=0.8):
    return {
        "chunk_id": f"d{i}:0", "doc_id": f"d{i}", "filename": filename,
        "page": page, "text": text, "score": score,
    }


def boom(*a, **k):
    raise AssertionError("LLM must not be called")


def test_abstains_when_nothing_retrieved():
    r = rag.answer("q", search_fn=lambda q, k: [], llm_fn=boom)
    assert r["abstained"] and r["answer"] == rag.NOT_FOUND and r["citations"] == []


def test_abstains_when_evidence_is_weak_without_calling_llm():
    r = rag.answer("q", min_score=0.3, search_fn=lambda q, k: [hit(1, score=0.1)], llm_fn=boom)
    assert r["abstained"] and r["citations"] == []


def test_abstains_when_llm_says_insufficient():
    r = rag.answer(
        "q", min_score=0.1, search_fn=lambda q, k: [hit(1)],
        llm_fn=lambda p, h: "INSUFFICIENT_EVIDENCE",
    )
    assert r["abstained"] and r["answer"] == rag.NOT_FOUND


def test_citations_point_to_the_right_document_and_page():
    hits = [hit(1, "a.pdf", 3, "alpha fact"), hit(2, "b.txt", 1, "beta fact")]
    r = rag.answer(
        "q", min_score=0.1, search_fn=lambda q, k: hits,
        llm_fn=lambda p, h: "Alpha is X [S1]. Beta is Y [S2].",
    )
    assert not r["abstained"] and r["grounded"] and r["warning"] is None
    cites = {(c["filename"], c["page"], c["text"]) for c in r["citations"]}
    assert cites == {("a.pdf", 3, "alpha fact"), ("b.txt", 1, "beta fact")}


def test_only_cited_passages_are_returned_as_citations():
    hits = [hit(1, "a.pdf", 3), hit(2, "b.txt", 1)]
    r = rag.answer("q", min_score=0.1, search_fn=lambda q, k: hits, llm_fn=lambda p, h: "Only A [S1].")
    assert [c["filename"] for c in r["citations"]] == ["a.pdf"]
    assert len(r["retrieved"]) == 2


def test_fabricated_citation_is_flagged():
    r = rag.answer("q", min_score=0.1, search_fn=lambda q, k: [hit(1)], llm_fn=lambda p, h: "Claim [S7].")
    assert r["grounded"] is False and "non-existent" in r["warning"]


def test_answer_without_citation_is_flagged():
    r = rag.answer("q", min_score=0.1, search_fn=lambda q, k: [hit(1)], llm_fn=lambda p, h: "Claim.")
    assert r["grounded"] is False and r["warning"]


def test_extract_citations_dedupes_and_separates_invalid():
    assert rag.extract_citations("a [S1] b [S2][S1] c [S9]", 3) == ([1, 2], [9])


def test_prompt_contains_filename_and_page():
    seen = {}

    def llm(prompt, hits):
        seen["p"] = prompt
        return "x [S1]"

    rag.answer("my question", min_score=0.1, search_fn=lambda q, k: [hit(1, "doc.pdf", 9, "zzz")], llm_fn=llm)
    assert "doc.pdf, page 9" in seen["p"] and "my question" in seen["p"]


# ---------- citation verification ----------
WSJ = "We trained on the Wall Street Journal portion of the Penn Treebank for parsing."
OTHER = "Reducing the attention key size hurts model quality in Table 3 rows."


def test_label_on_wrong_chunk_is_repaired():
    hits = [hit(1, "p.pdf", 9, OTHER), hit(2, "p.pdf", 9, WSJ)]
    r = rag.answer(
        "which dataset", min_score=0.1, search_fn=lambda q, k: hits,
        llm_fn=lambda p, h: "It used the Wall Street Journal portion of the Penn Treebank. [S1]",
    )
    assert "[S2]" in r["answer"] and "[S1]" not in r["answer"]
    assert r["citation_fixes"] == [{"from": [1], "to": 2}]
    assert [c["chunk_id"] for c in r["citations"]] == ["d2:0"]


def test_correct_citation_is_left_alone():
    hits = [hit(1, "p.pdf", 9, WSJ), hit(2, "p.pdf", 9, OTHER)]
    r = rag.answer(
        "which dataset", min_score=0.1, search_fn=lambda q, k: hits,
        llm_fn=lambda p, h: "It used the Wall Street Journal portion of the Penn Treebank [S1].",
    )
    assert r["citation_fixes"] == []
    assert [c["chunk_id"] for c in r["citations"]] == ["d1:0"]


def test_overlapping_chunks_do_not_trigger_a_repair():
    hits = [hit(1, "p.pdf", 9, WSJ), hit(2, "p.pdf", 9, WSJ + " More text.")]
    _, fixes = rag.verify_citations("Wall Street Journal portion of the Penn Treebank [S1]", hits)
    assert fixes == []


def test_each_sentence_is_checked_separately():
    hits = [hit(1, "a", 1, "Cats sleep sixteen hours daily."), hit(2, "b", 1, "Rockets burn liquid oxygen fuel.")]
    text = "Cats sleep sixteen hours daily [S2]. Rockets burn liquid oxygen fuel [S1]."
    fixed, fixes = rag.verify_citations(text, hits)
    assert fixed == "Cats sleep sixteen hours daily [S1]. Rockets burn liquid oxygen fuel [S2]."
    assert len(fixes) == 2


def test_very_short_claims_and_invalid_labels_are_skipped():
    hits = [hit(1, "a", 1, "Cats sleep sixteen hours daily.")]
    assert rag.verify_citations("Yes [S1]", hits) == ("Yes [S1]", [])
    assert rag.verify_citations("Cats sleep sixteen hours daily [S9]", hits)[1] == []