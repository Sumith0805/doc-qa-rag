"""API tests for health and accounts, using FastAPI's built-in test client (no server needed)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import auth  # noqa: E402

client = TestClient(api.app)
GOOD = {"email": "a@b.com", "password": "supersecret1"}


@pytest.fixture(autouse=True)
def clean_failures():
    auth._failures.clear()
    yield
    auth._failures.clear()


def login_token():
    client.post("/auth/register", json=GOOD)
    return client.post("/auth/login", json=GOOD).json()["token"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_register_creates_an_account():
    r = client.post("/auth/register", json=GOOD)
    assert r.status_code == 201 and r.json()["user"]["email"] == "a@b.com"
    assert "password" not in r.text


def test_duplicate_email_is_a_conflict():
    client.post("/auth/register", json=GOOD)
    r = client.post("/auth/register", json={"email": "A@B.com", "password": "anotherpass1"})
    assert r.status_code == 409


def test_bad_email_is_rejected():
    r = client.post("/auth/register", json={"email": "nope", "password": "supersecret1"})
    assert r.status_code == 400


def test_short_password_is_rejected():
    r = client.post("/auth/register", json={"email": "a@b.com", "password": "short"})
    assert r.status_code == 400


def test_login_then_me_works():
    token = login_token()
    r = client.get("/auth/me", headers=bearer(token))
    assert r.status_code == 200 and r.json()["user"]["email"] == "a@b.com"


def test_me_without_a_token_is_unauthorised():
    assert client.get("/auth/me").status_code == 401


def test_me_with_a_made_up_token_is_unauthorised():
    assert client.get("/auth/me", headers=bearer("made-up")).status_code == 401


def test_wrong_scheme_is_unauthorised():
    token = login_token()
    r = client.get("/auth/me", headers={"Authorization": f"Basic {token}"})
    assert r.status_code == 401


def test_wrong_password_gives_a_generic_message():
    client.post("/auth/register", json=GOOD)
    r = client.post("/auth/login", json={"email": "a@b.com", "password": "not-the-password"})
    assert r.status_code == 401 and r.json()["detail"] == "Invalid email or password."


def test_logout_invalidates_the_token():
    token = login_token()
    assert client.post("/auth/logout", headers=bearer(token)).status_code == 200
    assert client.get("/auth/me", headers=bearer(token)).status_code == 401


def test_repeated_wrong_passwords_lock_the_account():
    client.post("/auth/register", json=GOOD)
    for _ in range(auth.MAX_FAILED_LOGINS):
        client.post("/auth/login", json={"email": "a@b.com", "password": "wrong-password"})
    r = client.post("/auth/login", json=GOOD)  # even the right password is refused
    assert r.status_code == 429