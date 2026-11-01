"""Accounts: hashing, register, login, sessions, logout, lockout."""
from datetime import datetime, timedelta, timezone

import pytest

import auth
import registry


@pytest.fixture(autouse=True)
def clean_failures():
    auth._failures.clear()
    yield
    auth._failures.clear()


def test_password_hash_round_trip_and_salting():
    h1 = auth.hash_password("correct horse battery")
    h2 = auth.hash_password("correct horse battery")
    assert h1 != h2  # different random salt each time
    assert "correct horse" not in h1
    assert auth.verify_password("correct horse battery", h1)
    assert not auth.verify_password("wrong password!!", h1)
    assert not auth.verify_password("x", "garbage")


def test_register_normalises_email_and_stores_only_a_hash():
    user = auth.register("  Sumith@Example.COM ", "supersecret1")
    assert user["email"] == "sumith@example.com"
    stored = registry.get_user_by_email("sumith@example.com")
    assert stored["password_hash"].startswith("scrypt$") and "supersecret1" not in stored["password_hash"]


@pytest.mark.parametrize("email,password", [
    ("not-an-email", "supersecret1"),
    ("a@b", "supersecret1"),
    ("a@b.com", "short"),
    ("a@b.com", "x" * 200),
    ("", "supersecret1"),
])
def test_register_rejects_bad_input(email, password):
    with pytest.raises(auth.AuthError):
        auth.register(email, password)


def test_duplicate_email_is_rejected():
    auth.register("a@b.com", "supersecret1")
    with pytest.raises(auth.AuthError, match="already registered"):
        auth.register("A@B.com", "anotherpass1")


def test_login_returns_working_token():
    auth.register("a@b.com", "supersecret1")
    result = auth.login("a@b.com", "supersecret1")
    me = auth.user_from_token(result["token"])
    assert me["email"] == "a@b.com" and me["id"] == result["user"]["id"]


def test_wrong_password_and_unknown_email_give_the_same_message():
    auth.register("a@b.com", "supersecret1")
    with pytest.raises(auth.AuthError) as wrong_pw:
        auth.login("a@b.com", "not-the-password")
    with pytest.raises(auth.AuthError) as unknown:
        auth.login("nobody@b.com", "supersecret1")
    assert str(wrong_pw.value) == str(unknown.value) == "Invalid email or password."


def test_token_is_not_stored_in_plain_text():
    auth.register("a@b.com", "supersecret1")
    token = auth.login("a@b.com", "supersecret1")["token"]
    assert registry.get_session_user(token, "2000-01-01T00:00:00+00:00") is None  # raw token is not a key


def test_logout_invalidates_the_token():
    auth.register("a@b.com", "supersecret1")
    token = auth.login("a@b.com", "supersecret1")["token"]
    auth.logout(token)
    assert auth.user_from_token(token) is None


def test_expired_session_is_rejected():
    user = auth.register("a@b.com", "supersecret1")
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="seconds")
    registry.create_session(auth._hash_token("old-token"), user["id"], past)
    assert auth.user_from_token("old-token") is None


def test_missing_or_unknown_token_means_not_logged_in():
    assert auth.user_from_token(None) is None
    assert auth.user_from_token("") is None
    assert auth.user_from_token("made-up") is None


def test_account_locks_after_repeated_wrong_passwords():
    auth.register("a@b.com", "supersecret1")
    for i in range(auth.MAX_FAILED_LOGINS):
        with pytest.raises(auth.AuthError, match="Invalid"):
            auth.login("a@b.com", "wrong-password", now=1000.0 + i)
    with pytest.raises(auth.AuthError, match="Too many"):
        auth.login("a@b.com", "supersecret1", now=1010.0)  # even the right password is refused
    ok = auth.login("a@b.com", "supersecret1", now=1000.0 + auth.LOCKOUT_SECONDS + 100)
    assert ok["token"]