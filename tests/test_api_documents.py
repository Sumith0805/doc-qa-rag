"""API tests for documents and chat. Fake embedder + mock LLM: no network, no API key."""
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("chromadb")
from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfWriter  # noqa: E402

import api  # noqa: E402
import config  # noqa: E402
import store  # noqa: E402
from test_isolation import FakeEmbedding  # noqa: E402  (reuse the offline embedder)

client = TestClient(api.app)
CATS = b"Cats are small carnivorous mammals. A cat sleeps sixteen hours a day."
ROCKETS = b"Rockets burn propellant to produce thrust. A rocket engine needs liquid oxygen."
Q = {"question": "how long does a cat sleep"}


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    store.set_embedding_function(FakeEmbedding())
    monkeypatch.setattr(config, "MIN_SCORE", 0.05)  # fake embeddings score lower than MiniLM
    yield
    store.set_embedding_function(None)


def make_user(email):
    creds = {"email": email, "password": "supersecret1"}
    client.post("/auth/register", json=creds)
    return {"Authorization": f"Bearer {client.post('/auth/login', json=creds).json()['token']}"}


def upload(headers, name, data, ctype="text/plain"):
    return client.post("/documents", files={"file": (name, data, ctype)}, headers=headers)


def pdf_with_pages(n):
    w = PdfWriter()
    for _ in range(n):
        w.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_upload_requires_login():
    assert upload({}, "cats.txt", CATS).status_code == 401


def test_upload_then_list():
    h = make_user("a@x.com")
    r = upload(h, "cats.txt", CATS)
    doc = r.json()["document"]
    assert r.status_code == 201 and doc["status"] == "ready" and doc["filename"] == "cats.txt"
    assert "sha256" not in doc and "user_id" not in doc
    listed = client.get("/documents", headers=h).json()["documents"]
    assert [d["id"] for d in listed] == [doc["id"]]


def test_same_file_twice_is_a_duplicate():
    h = make_user("a@x.com")
    first = upload(h, "cats.txt", CATS)
    second = upload(h, "cats.txt", CATS)
    assert second.status_code == 200 and second.json()["duplicate"] is True
    assert second.json()["document"]["id"] == first.json()["document"]["id"]


def test_wrong_file_type_is_rejected():
    assert upload(make_user("a@x.com"), "virus.exe", b"MZ").status_code == 415


def test_oversize_file_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    r = upload(make_user("a@x.com"), "big.txt", b"a" * (1024 * 1024 + 1))
    assert r.status_code == 413


def test_too_many_pages_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_PDF_PAGES", 1)
    r = upload(make_user("a@x.com"), "two.pdf", pdf_with_pages(2), "application/pdf")
    assert r.status_code == 422


def test_library_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_DOCS_PER_USER", 1)
    h = make_user("a@x.com")
    assert upload(h, "cats.txt", CATS).status_code == 201
    assert upload(h, "rockets.txt", ROCKETS).status_code == 409


def test_users_cannot_see_or_delete_each_others_documents():
    alice, bob = make_user("alice@x.com"), make_user("bob@x.com")
    doc_id = upload(alice, "cats.txt", CATS).json()["document"]["id"]
    assert client.get("/documents", headers=bob).json()["documents"] == []
    assert client.get(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.get(f"/documents/{doc_id}", headers=alice).status_code == 200


def test_delete_removes_document_and_chunks():
    h = make_user("a@x.com")
    doc_id = upload(h, "cats.txt", CATS).json()["document"]["id"]
    assert store.count_chunks(doc_id) >= 1
    assert client.delete(f"/documents/{doc_id}", headers=h).status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=h).status_code == 404
    assert store.count_chunks() == 0


def test_ask_returns_a_cited_answer_from_own_documents():
    h = make_user("a@x.com")
    upload(h, "cats.txt", CATS)
    r = client.post("/ask", json=Q, headers=h)
    assert r.status_code == 200 and r.json()["abstained"] is False
    assert r.json()["citations"][0]["filename"] == "cats.txt"


def test_ask_never_searches_another_users_documents():
    alice, bob = make_user("alice@x.com"), make_user("bob@x.com")
    alice_doc = upload(alice, "cats.txt", CATS).json()["document"]["id"]
    upload(bob, "rockets.txt", ROCKETS)
    body = client.post("/ask", json=Q, headers=bob).json()
    assert all(c["filename"] == "rockets.txt" for c in body["citations"])  # never cats.txt
    # naming alice's document must not widen bob's scope
    body = client.post("/ask", json={**Q, "doc_ids": [alice_doc]}, headers=bob).json()
    assert body["abstained"] is True and body["citations"] == []


def test_empty_question_is_rejected():
    h = make_user("a@x.com")
    assert client.post("/ask", json={"question": ""}, headers=h).status_code == 422
    assert client.post("/ask", json={"question": "   "}, headers=h).status_code == 422


def test_history_is_per_user_and_can_be_cleared():
    alice, bob = make_user("alice@x.com"), make_user("bob@x.com")
    upload(alice, "cats.txt", CATS)
    client.post("/ask", json=Q, headers=alice)
    assert len(client.get("/history", headers=alice).json()["messages"]) == 1
    assert client.get("/history", headers=bob).json()["messages"] == []
    assert client.delete("/history", headers=bob).status_code == 204
    assert len(client.get("/history", headers=alice).json()["messages"]) == 1  # bob can't clear alice's
    assert client.delete("/history", headers=alice).status_code == 204
    assert client.get("/history", headers=alice).json()["messages"] == []
