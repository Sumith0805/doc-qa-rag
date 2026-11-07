"""Shared FastAPI dependencies and the lock that guards the vector store."""
import threading

from fastapi import Depends, Header, HTTPException

import auth

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
