"""Shared FastAPI dependencies and the lock that guards the vector store."""
import collections
import threading
import time

from fastapi import Depends, Header, HTTPException

import auth
import config

# store.py keeps Chroma and the BM25 cache in module-level globals, and FastAPI runs plain
# `def` routes in a thread pool. Hold this lock around every store call. Run ONE uvicorn worker.
store_lock = threading.RLock()


def bearer_token(authorization: str | None = Header(default=None)):
    """Pull the token out of 'Authorization: Bearer <token>' (None if missing or malformed)."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    value = value.strip()
    return value if scheme.lower() == "bearer" and value else None


def current_user(token: str | None = Depends(bearer_token)):
    """Use as a dependency on any route that needs a logged-in user."""
    user = auth.user_from_token(token)
    if not user:
        raise HTTPException(
            status_code=401, detail="Not logged in.", headers={"WWW-Authenticate": "Bearer"}
        )
    return user


# ---- per-user rate limit for /ask ----
_ask_times = {}  # user id -> deque of recent request times
_ask_lock = threading.Lock()


def ask_rate_limit(user: dict = Depends(current_user)):
    """Allow config.ASK_PER_MIN questions per user per minute; otherwise answer 429 with Retry-After."""
    now = time.monotonic()
    with _ask_lock:
        times = _ask_times.setdefault(user["id"], collections.deque())
        while times and now - times[0] >= 60:
            times.popleft()
        if len(times) >= config.ASK_PER_MIN:
            wait = int(60 - (now - times[0])) + 1
            raise HTTPException(
                429, f"Too many questions. Try again in {wait} seconds.",
                headers={"Retry-After": str(wait)},
            )
        times.append(now)
    return user
