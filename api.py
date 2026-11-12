"""DocuMind HTTP API: health, accounts, documents and chat.

Run locally:  uvicorn api:app --reload      then open http://127.0.0.1:8000/docs
Clients send the login token as the header  Authorization: Bearer <token>
Run ONE worker only (the vector store uses module-level globals).
"""
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import auth
from deps import bearer_token, current_user
from routers import chat, documents

app = FastAPI(title="DocuMind API", version="0.2.0")

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


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    """Every error has the same shape: {"detail": "<readable text>"} (FastAPI's default is a list)."""
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"] if p not in ("body", "query", "path"))
        parts.append(f"{field}: {err['msg']}" if field else err["msg"])
    return JSONResponse(status_code=422, content={"detail": "; ".join(parts) or "Invalid request."})


app.include_router(documents.router)
app.include_router(chat.router)


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
