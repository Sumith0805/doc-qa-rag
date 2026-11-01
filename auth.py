"""Accounts: register, log in (opaque random session tokens), log out. Standard library only.

Passwords are stored as salted scrypt hashes. Login tokens are random strings; only their
SHA-256 hash is stored, so a leaked database cannot be used to log in as anyone.
"""
import collections
import functools
import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone

import registry

SESSION_DAYS = 7
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 128
MAX_FAILED_LOGINS = 5
LOCKOUT_SECONDS = 15 * 60
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(ValueError):
    """Bad input or bad credentials. The message is safe to show to the user."""


def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password, stored):
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
        if scheme != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=32
        )
    except Exception:
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


@functools.lru_cache(maxsize=1)
def _dummy_hash():
    return hash_password("not-a-real-password")


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.isoformat(timespec="seconds")


def _hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


# --- brute-force protection: too many wrong passwords for one email -> temporary lockout ---
_failures = collections.defaultdict(collections.deque)


def _check_lockout(email, now):
    attempts = _failures[email]
    while attempts and now - attempts[0] > LOCKOUT_SECONDS:
        attempts.popleft()
    if len(attempts) >= MAX_FAILED_LOGINS:
        raise AuthError("Too many failed attempts. Please try again in a few minutes.")


def register(email, password):
    email = (email or "").strip().lower()
    password = password or ""
    if len(email) > 254 or not _EMAIL.match(email):
        raise AuthError("Please enter a valid email address.")
    if not MIN_PASSWORD_LEN <= len(password) <= MAX_PASSWORD_LEN:
        raise AuthError(f"Password must be {MIN_PASSWORD_LEN}-{MAX_PASSWORD_LEN} characters.")
    if registry.get_user_by_email(email):
        raise AuthError("That email is already registered.")
    user_id = uuid.uuid4().hex
    registry.create_user(user_id, email, hash_password(password))
    return {"id": user_id, "email": email}


def login(email, password, now=None):
    email = (email or "").strip().lower()
    now = time.monotonic() if now is None else now
    _check_lockout(email, now)
    user = registry.get_user_by_email(email)
    # Always do one hash check, so a wrong email and a wrong password take the same time.
    ok = verify_password(password or "", user["password_hash"] if user else _dummy_hash())
    if not user or not ok:
        _failures[email].append(now)
        raise AuthError("Invalid email or password.")
    _failures.pop(email, None)
    token = secrets.token_urlsafe(32)
    expires = _iso(_now() + timedelta(days=SESSION_DAYS))
    registry.create_session(_hash_token(token), user["id"], expires)
    return {"token": token, "user": {"id": user["id"], "email": user["email"]}}


def user_from_token(token):
    """The logged-in user for a token, or None if it is missing, unknown or expired."""
    if not token:
        return None
    return registry.get_session_user(_hash_token(token), _iso(_now()))


def logout(token):
    if token:
        registry.delete_session(_hash_token(token))