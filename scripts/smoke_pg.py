"""End-to-end smoke test of the Postgres backend through the real API code (in-process, no server).

Run from the repo root (venv active):   python scripts/smoke_pg.py

It uses DATABASE_URL from your .env (Neon), the REAL embedding model and the MOCK LLM, so it
spends no Groq quota. It creates throwaway users 'smoke-...@example.invalid' and deletes
everything it created. It never prints your connection string.
"""
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DOCUMIND_BACKEND"] = "postgres"
os.environ["DOCUMIND_LLM"] = "mock"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import pg  # noqa: E402

client = TestClient(api.app)
TAG = uuid.uuid4().hex[:8]
CATS = b"Cats are small carnivorous mammals. A cat sleeps sixteen hours a day."
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def make_user(prefix):
    creds = {"email": f"smoke-{prefix}-{TAG}@example.invalid", "password": "supersecret1"}
    client.post("/auth/register", json=creds)
    token = client.post("/auth/login", json=creds).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def cleanup():
    try:
        with pg.connect() as con:
            ids = [r["id"] for r in con.execute(
                "SELECT id FROM users WHERE email LIKE %s", (f"smoke-%-{TAG}@example.invalid",)
            ).fetchall()]
            if ids:
                con.execute("DELETE FROM messages WHERE user_id = ANY(%s)", (ids,))
                con.execute("DELETE FROM sessions WHERE user_id = ANY(%s)", (ids,))
                con.execute("DELETE FROM documents WHERE user_id = ANY(%s)", (ids,))  # chunks cascade
                con.execute("DELETE FROM users WHERE id = ANY(%s)", (ids,))
        print("cleanup done: smoke users, documents, chunks and messages removed")
    except Exception as exc:  # cleanup must never hide the real result
        print("cleanup problem:", type(exc).__name__)


def main():
    t0 = time.perf_counter()
    alice, bob = make_user("a"), make_user("b")
    check("register + login (2 users)", alice and bob, f"{time.perf_counter() - t0:.1f}s incl. database wake-up")

    t = time.perf_counter()
    r = client.post("/documents", files={"file": ("cats.txt", CATS, "text/plain")}, headers=alice)
    doc = r.json().get("document", {}) if r.status_code == 201 else {}
    check("alice uploads cats.txt", doc.get("status") == "ready", f"{time.perf_counter() - t:.1f}s")
    doc_id = doc.get("id", "")

    sample = ROOT / "sample.pdf"
    if sample.exists():
        t = time.perf_counter()
        r = client.post("/documents", files={"file": ("sample.pdf", sample.read_bytes(), "application/pdf")}, headers=alice)
        pdf = r.json().get("document", {}) if r.status_code == 201 else {}
        check("alice uploads sample.pdf", pdf.get("status") == "ready" and pdf.get("chunks", 0) > 0,
              f"{pdf.get('pages')} pages, {pdf.get('chunks')} chunks, {time.perf_counter() - t:.1f}s")

    r = client.get("/documents", headers=alice)
    check("alice lists her documents", r.status_code == 200 and doc_id in [d["id"] for d in r.json()["documents"]])

    t = time.perf_counter()
    r = client.post("/ask", json={"question": "How many hours a day does a cat sleep?", "doc_ids": [doc_id]}, headers=alice)
    body = r.json() if r.status_code == 200 else {}
    cites = body.get("citations", [])
    check("alice gets a cited answer", bool(cites) and cites[0]["filename"] == "cats.txt" and not body.get("abstained"),
          f"top_score={body.get('top_score')}, latency_ms={body.get('latency_ms')}, round trip {time.perf_counter() - t:.1f}s")

    r = client.get("/documents", headers=bob)
    check("bob sees none of alice's documents", r.status_code == 200 and r.json()["documents"] == [])
    check("bob cannot open alice's document", client.get(f"/documents/{doc_id}", headers=bob).status_code == 404)
    check("bob cannot delete alice's document", client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404)
    r = client.post("/ask", json={"question": "How many hours a day does a cat sleep?"}, headers=bob)
    check("bob's question finds nothing of alice's", r.status_code == 200 and r.json()["abstained"] and not r.json()["citations"])
    r = client.post("/ask", json={"question": "cats", "doc_ids": [doc_id]}, headers=bob)
    check("bob cannot widen his scope to alice's document", r.status_code == 200 and r.json()["abstained"])

    r = client.get("/history", headers=alice)
    check("alice's history has her question", r.status_code == 200 and len(r.json()["messages"]) == 1)
    check("bob's history holds only his own 2 questions", len(client.get("/history", headers=bob).json()["messages"]) == 2)

    ids = [d["id"] for d in client.get("/documents", headers=alice).json()["documents"]]
    check("alice deletes her documents", all(client.delete(f"/documents/{i}", headers=alice).status_code == 204 for i in ids))
    check("alice's library is empty again", client.get("/documents", headers=alice).json()["documents"] == [])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        results.append(False)
        print("FAIL  unexpected error:", type(exc).__name__, str(exc)[:300].replace(os.environ.get("DATABASE_URL", "<none>"), "<DATABASE_URL>"))
    finally:
        cleanup()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if results and all(results) else 1)

