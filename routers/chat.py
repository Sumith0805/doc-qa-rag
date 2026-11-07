"""Ask questions and manage chat history, scoped to the logged-in user."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

import rag
import registry
import store
from deps import current_user, store_lock

router = APIRouter(tags=["chat"])
ANSWER_FIELDS = ("answer", "abstained", "grounded", "citations", "citation_fixes",
                 "top_score", "latency_ms", "warning")


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] | None = Field(default=None, max_length=50)


def _locked_search(question, k, doc_ids=None):
    with store_lock:  # lock only the retrieval, never the slow LLM call
        return store.search(question, k, doc_ids)


@router.post("/ask")
def ask(body: AskBody, user: dict = Depends(current_user)):
    question = body.question.strip()
    if not question:
        raise HTTPException(422, "Question is empty.")
    mine = {d["id"] for d in registry.list_documents(status="ready", user_id=user["id"])}
    # The client may only NARROW the scope to its own documents, never widen it.
    scope = sorted(mine if body.doc_ids is None else mine & set(body.doc_ids))
    try:
        result = rag.answer(question, doc_ids=scope, search_fn=_locked_search)
    except rag.LLMUnavailable:
        raise HTTPException(503, rag.UNAVAILABLE_MESSAGE)
    registry.add_message(question, result["answer"], result["abstained"],
                         result["citations"], result["latency_ms"], user_id=user["id"])
    return {k: result[k] for k in ANSWER_FIELDS}


@router.get("/history")
def history(limit: int = Query(50, ge=1, le=200), user: dict = Depends(current_user)):
    return {"messages": registry.list_messages(limit, user_id=user["id"])}


@router.delete("/history", status_code=204)
def clear_history(user: dict = Depends(current_user)):
    registry.clear_messages(user_id=user["id"])  # never call this without user_id: it wipes everyone
