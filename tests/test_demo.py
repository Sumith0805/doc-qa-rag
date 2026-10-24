"""Tests for the public-demo protections: retries, shared rate limit, friendly errors."""
import pytest

import config
import rag


def hit(i=1, text="some text"):
    return {"chunk_id": f"d{i}:0", "doc_id": f"d{i}", "filename": "a.pdf",
            "page": 1, "text": text, "score": 0.8}


def boom(*a, **k):
    raise AssertionError("LLM must not be called")


def down(prompt, hits):
    raise rag.LLMUnavailable("429")


class Slow(Exception):
    pass


def search(q, k):
    return [hit()]


# ---------- retries ----------
def test_retries_succeed_after_transient_failures():
    calls, slept = {"n": 0}, []

    def call():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Slow()
        return "ok"

    assert rag.with_retries(call, Slow, delays=(1, 2, 3), sleep=slept.append) == "ok"
    assert slept == [1, 2]


def test_retries_give_up_after_the_last_delay():
    calls, slept = {"n": 0}, []

    def call():
        calls["n"] += 1
        raise Slow()

    with pytest.raises(Slow):
        rag.with_retries(call, Slow, delays=(1, 2), sleep=slept.append)
    assert calls["n"] == 3 and slept == [1, 2]


def test_other_errors_are_not_retried():
    slept = []

    def call():
        raise ValueError("bug")

    with pytest.raises(ValueError):
        rag.with_retries(call, Slow, delays=(1, 2), sleep=slept.append)
    assert slept == []


# ---------- shared rate limit ----------
def test_rate_limit_allows_then_blocks_then_recovers(monkeypatch):
    monkeypatch.setattr(config, "DEMO_QUESTIONS_PER_MIN", 2)
    rag._call_times.clear()
    assert rag._rate_limit_wait(now=100.0) == 0
    assert rag._rate_limit_wait(now=101.0) == 0
    wait = rag._rate_limit_wait(now=102.0)
    assert 55 <= wait <= 60
    assert rag._rate_limit_wait(now=161.0) == 0  # the first call is older than a minute now
    rag._call_times.clear()


def test_demo_mode_blocks_the_second_question_without_calling_the_llm(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_QUESTIONS_PER_MIN", 1)
    rag._call_times.clear()
    first = rag.answer("q", min_score=0.1, search_fn=search, llm_fn=lambda p, h: "fact [S1]")
    assert not first["abstained"]
    second = rag.answer("q", min_score=0.1, search_fn=search, llm_fn=boom)
    assert second["abstained"] and "busy" in second["answer"].lower()
    rag._call_times.clear()


def test_rate_limit_is_not_applied_outside_demo_mode(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "DEMO_QUESTIONS_PER_MIN", 1)
    rag._call_times.clear()
    for _ in range(3):
        r = rag.answer("q", min_score=0.1, search_fn=search, llm_fn=lambda p, h: "fact [S1]")
        assert not r["abstained"]


# ---------- friendly errors ----------
def test_demo_mode_shows_a_friendly_message_when_the_ai_is_down(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DEMO_QUESTIONS_PER_MIN", 100)
    rag._call_times.clear()
    r = rag.answer("q", min_score=0.1, search_fn=search, llm_fn=down)
    assert r["abstained"] and "unavailable" in r["answer"].lower()
    rag._call_times.clear()


def test_outside_demo_mode_ai_errors_are_raised(monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    with pytest.raises(rag.LLMUnavailable):
        rag.answer("q", min_score=0.1, search_fn=search, llm_fn=down)