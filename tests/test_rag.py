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
