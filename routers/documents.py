"""Document routes. Every query is scoped to the logged-in user."""
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from pypdf import PdfReader

import config
import registry
import store
from deps import current_user, store_lock

router = APIRouter(prefix="/documents", tags=["documents"])
PUBLIC = ("id", "filename", "status", "pages", "chunks", "error", "created_at")


def _public(doc):
    return {k: doc[k] for k in PUBLIC}  # never expose sha256 or user_id


def _own(doc_id, user):
    doc = registry.get_document(doc_id)
    if not doc or doc["user_id"] != user["id"]:
        raise HTTPException(404, "Document not found.")  # 404, not 403: don't reveal existence
    return doc


@router.get("")
def list_documents(user: dict = Depends(current_user)):
    return {"documents": [_public(d) for d in registry.list_documents(user_id=user["id"])]}


@router.get("/{doc_id}")
def get_document(doc_id: str, user: dict = Depends(current_user)):
    return {"document": _public(_own(doc_id, user))}


@router.post("", status_code=201)
def upload(response: Response, file: UploadFile = File(...), user: dict = Depends(current_user)):
    name = (file.filename or "").replace("\\", "/").split("/")[-1].strip()[:200]
    suffix = Path(name).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Allowed file types: {', '.join(config.ALLOWED_EXTENSIONS)}")
    mine = [d for d in registry.list_documents(user_id=user["id"]) if d["status"] != "failed"]
    if len(mine) >= config.MAX_DOCS_PER_USER:
        raise HTTPException(409, f"Library limit reached ({config.MAX_DOCS_PER_USER} documents).")

    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            size = 0
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_MB} MB.")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty.")
        if suffix == ".pdf":
            try:
                n_pages = len(PdfReader(tmp_path).pages)
            except Exception:
                raise HTTPException(400, "Could not read this PDF.")
            if n_pages > config.MAX_PDF_PAGES:
                raise HTTPException(422, f"PDFs are limited to {config.MAX_PDF_PAGES} pages.")
        try:
            with store_lock:
                record = store.index_document(tmp_path, name, user_id=user["id"])
        except store.DemoLimitError as exc:
            raise HTTPException(403, str(exc))
        except ValueError as exc:  # e.g. no extractable text (scanned PDF)
            raise HTTPException(422, str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if record["duplicate"]:
        response.status_code = 200
    return {"document": _public(record), "duplicate": record["duplicate"]}


@router.delete("/{doc_id}", status_code=204)
def delete(doc_id: str, user: dict = Depends(current_user)):
    _own(doc_id, user)
    try:
        with store_lock:
            store.delete_document(doc_id)
    except store.DemoLimitError as exc:
        raise HTTPException(403, str(exc))
