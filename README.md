# DocuMind: Multi-Document Q&A with Verifiable Citations

Upload several PDFs / text files, ask questions across all of them, and get answers that cite the
**file name and page** of the passages they came from. If the documents don't contain the answer,
DocuMind says so instead of guessing.

**Live demo:** https://doc-app-rag-7wgvuvabsqbyokqzkcmxe3.streamlit.app/ (shared public demo: everyone sees the same library)

## Features
- Multi-document retrieval: each upload is added to the index; nothing is overwritten
- Citations with file + page, validated against the retrieved passages (fabricated `[S#]` labels are flagged)
- Two-gate abstention: (1) retrieval similarity below a threshold -> no LLM call; (2) the LLM must answer `INSUFFICIENT_EVIDENCE` when sources don't support an answer
- Document library with status (processing / ready / failed), duplicate detection (SHA-256), and **verified deletion** (chunks + embeddings removed, checked after delete)
- Persistent storage: SQLite (documents, chat history) + ChromaDB (vectors)
- Repeatable evaluation suite (31 labelled cases, label validator, threshold sweep, saved results)
- pytest suite, Docker + Docker Compose with health check, GitHub Actions CI

## Architecture
```
Streamlit UI (app.py)
   |-- store.py    index_document / search / delete_document  -> ChromaDB (cosine, all-MiniLM-L6-v2 ONNX)
   |-- registry.py document status + chat history             -> SQLite
   |-- rag.py      retrieve -> abstain gate -> LLM (Groq) -> citation validation
   `-- chunker.py  pypdf / txt reader, page-aware chunks (800 chars, 150 overlap)
```
Every chunk is stored with `doc_id`, `filename`, `page`, `chunk_index`.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements-dev.txt
cp .env.example .env              # then put your GROQ_API_KEY in .env
streamlit run app.py
```

## Run with Docker
```bash
cp .env.example .env              # add GROQ_API_KEY
docker compose up --build         # http://localhost:8501
docker compose ps                 # shows (healthy) once ready
```

## Tests and evaluation
```bash
pytest                                   # no API key / model download needed
python -m eval.run_eval --validate       # check eval labels against the documents
python -m eval.run_eval --llm groq       # retrieval, answer, citation, abstention, latency
```
Results are saved to `eval/results/` and each run prints the change vs. the previous run.

### Results
_Fill this in from your own run of `python -m eval.run_eval --llm groq` (copy the numbers from eval/results/):_

| Metric | Value |
|---|---|
| Retrieval hit rate (answerable) | _ |
| Cross-document hit rate | _ |
| MRR | _ |
| Answer accuracy | _ |
| Citation correctness | _ |
| Unsupported questions correctly refused | _ |
| p50 / p95 latency | _ |

### Known failure modes
_Describe 2-3 real failures from your eval output and what you would change._

## Limitations
- Text PDFs only (no OCR for scanned pages)
- The public demo has one shared library and a non-persistent disk on free hosting
- Dense retrieval only; hybrid (BM25) search, reranking and multi-query decomposition are future work

## Tech stack
Python, Streamlit, ChromaDB, SQLite, pypdf, Groq API, pytest, Docker, GitHub Actions
