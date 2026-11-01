# DocuMind: Multi-Document Q&A with Verifiable Citations

Upload PDFs or text files, ask questions across all of them, and get answers that cite the **file and page** of the passages they came from. If the documents don't contain the answer, DocuMind says so instead of guessing.

**Status:** not deployed yet. Run it locally or with Docker (below). A hosted version with accounts is on the roadmap.

## What it does
- **Multi-document search:** every upload is added to one index; nothing is overwritten.
- **Cited answers:** each answer shows the source file and page, and a checker re-points citation labels that the AI attached to the wrong passage.
- **Honest refusals:** two gates. If retrieval is weak, the AI is never asked. If the AI says the sources don't support an answer, the app refuses.
- **Hybrid retrieval:** embedding search combined with BM25 keyword scoring, and compound questions ("X, and what is Y?") are split so each part gets its own evidence.
- **Document library:** status tracking (processing, ready, failed), duplicate detection, and deletion that is verified against the index.
- **Persistence:** SQLite for documents and chat history, ChromaDB for vectors.
- **Demo protections** (when `DOCUMIND_DEMO=1`): upload size and library caps, no deleting, retries with a shared rate limit and a friendly busy message.

## Architecture
```
app.py        Streamlit UI
store.py      index / search / delete documents (ChromaDB)
hybrid.py     BM25 re-ranking, question splitting
rag.py        abstain gates, LLM call (Groq), citation verification
registry.py   SQLite: documents, chat history, users, sessions
auth.py       accounts: scrypt password hashes, session tokens
chunker.py    PDF / text reader, page-aware chunks
config.py     settings (environment variables)
eval/         evaluation set, runner, debug tool
tests/        pytest suite
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Windows (macOS/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
cp .env.example .env             # then put your GROQ_API_KEY in .env
streamlit run app.py
```

## Run with Docker
```bash
cp .env.example .env             # add GROQ_API_KEY
docker compose up --build        # http://localhost:8501
```
CI builds the Docker image on every push.

## Tests and evaluation
```bash
pytest
python -m eval.run_eval --validate       # check eval labels against the documents
python -m eval.run_eval --llm groq       # retrieval, answers, citations, refusals, latency
python -m eval.inspect_query "your question" --expect "text you expect"   # debug retrieval
```
Every run is saved to `eval/results/`.

## Results
Run: `eval/results/20261023_172145_citation-verify.json`. 31 questions about 2 documents (26 answerable, 5 unsupported), `gpt-oss-20b` on Groq, top-6 chunks, BM25 weight 0.7.

| Metric | Result |
|---|---|
| Retrieved chunk contains the expected fact (strict) | 25 / 26 (96.2%) |
| Right file and page retrieved (lenient) | 26 / 26 |
| Correct answers | 25 / 26 (96.2%) |
| Correct citations (the cited chunk contains the fact) | 25 / 26 (96.2%) |
| Answerable questions wrongly refused | 0 / 26 |
| Unsupported questions refused | 4 / 5 in this run. The fifth hit an API rate limit (HTTP 429) and never got an answer; the previous full run refused 5 / 5 (`20261023_170651_partial-answers.json`). |
| Median time per question | about 11 s (retrieval about 0.4 s, the rest is the LLM call) |

**How to read these numbers:** I tuned the retrieval settings (keyword weight, question splitting, prompt) on these same 31 questions, so the results are optimistic. The set is also small and covers only two documents. A held-out set on other documents is planned.

### Known failure
`x-02` ("How many attention heads does the base Transformer use, and what is ACME's standard resignation notice period?"): the chunk containing `h = 8` ranks 6th by embeddings and 8th after hybrid re-ranking, but a compound question only gives each half 3 of the 6 slots. Raising k didn't help without sending far more text to the LLM. A reranker is the likely fix. The app now answers the part it can support and says which part it couldn't find.

## Limitations
- Text PDFs only (no OCR for scanned pages)
- Free-tier AI quota: roughly 3 answered questions per minute
- Dense + keyword retrieval only; no reranker yet

## Roadmap (not built yet)
FastAPI backend with login, per-user libraries, a separate web frontend, a held-out evaluation set, and a reranker.

## Tech stack
Python, ChromaDB, SQLite, pypdf, Groq API, Streamlit, pytest, Docker, GitHub Actions
