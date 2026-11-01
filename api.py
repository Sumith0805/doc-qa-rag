"""DocuMind HTTP API. Step 1: health check and accounts.

Run locally:  uvicorn api:app --reload      then open http://127.0.0.1:8000/docs
Clients send the login token as the header  Authorization: Bearer <token>
"""
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import auth

app = FastAPI(title="DocuMind API", version="0.1.0")

# Which websites may call this API from a browser (your frontend's address goes here later).
_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500"
_origins = [o.strip() for o in os.getenv("DOCUMIND_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,  # we use a bearer token header, not cookies
    allow_methods=["*"],
    allow_headers=["*"],
)


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


def _auth_status(exc, default):
    """Turn an AuthError message into the right HTTP status code."""
    message = str(exc)
    if "already registered" in message:
        return 409
    if "Too many" in message:
        return 429
    return default


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/auth/register", status_code=201)
def register(body: Credentials):
    try:
        user = auth.register(body.email, body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=_auth_status(exc, 400), detail=str(exc))
    return {"user": user}


@app.post("/auth/login")
def login(body: Credentials):
    try:
        return auth.login(body.email, body.password)
    except auth.AuthError as exc:
        raise HTTPException(status_code=_auth_status(exc, 401), detail=str(exc))


@app.get("/auth/me")
def me(user: dict = Depends(current_user)):
    return {"user": user}


@app.post("/auth/logout")
def logout(token: str | None = Depends(bearer_token), user: dict = Depends(current_user)):
    auth.logout(token)
    return {"ok": True}