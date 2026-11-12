"""Central configuration. Everything can be overridden with environment variables."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Where all persistent data lives (SQLite registry + Chroma vectors).
DATA_DIR = Path(os.getenv("DOCUMIND_DATA_DIR", "data"))
CHROMA_PATH = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "documind.db"

# "groq" = real LLM, "mock" = deterministic fake LLM (used in tests / offline demos)
LLM_MODE = os.getenv("DOCUMIND_LLM", "groq")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Retrieval
TOP_K = int(os.getenv("DOCUMIND_TOP_K", "6"))
# Cosine similarity below which we refuse to answer. Calibrate with `python -m eval.run_eval`.
MIN_SCORE = float(os.getenv("DOCUMIND_MIN_SCORE", "0.25"))

# Hybrid retrieval: fetch a wider dense candidate pool, then re-rank with BM25 (see hybrid.py)
HYBRID = os.getenv("DOCUMIND_HYBRID", "1") == "1"
RRF_K = int(os.getenv("DOCUMIND_RRF_K", "60"))
# Keyword vote relative to the embedding vote. Chosen with a sweep on the eval set:
# 0.5 or lower loses the "optimizer" question, 0.9 or higher loses the compound question.
BM25_WEIGHT = float(os.getenv("DOCUMIND_BM25_WEIGHT", "0.7"))
MULTI_QUERY = os.getenv("DOCUMIND_MULTI_QUERY", "1") == "1"
CANDIDATE_POOL = int(os.getenv("DOCUMIND_CANDIDATE_POOL", "20"))

# Chunking
CHUNK_SIZE = int(os.getenv("DOCUMIND_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("DOCUMIND_CHUNK_OVERLAP", "150"))

# Public-demo protections (turn on in the hosted app with DOCUMIND_DEMO=1)
DEMO_MODE = os.getenv("DOCUMIND_DEMO", "0") == "1"
DEMO_QUESTIONS_PER_MIN = int(os.getenv("DOCUMIND_DEMO_RPM", "3"))  # shared by ALL visitors
DEMO_MAX_DOCS = int(os.getenv("DOCUMIND_DEMO_MAX_DOCS", "5"))
DEMO_MAX_UPLOAD_MB = int(os.getenv("DOCUMIND_DEMO_MAX_MB", "5"))

ALLOWED_EXTENSIONS = (".pdf", ".txt", ".md")

# API limits (per user). Defaults are guesses: tune them after measuring on the real server.
MAX_UPLOAD_MB = int(os.getenv("DOCUMIND_MAX_UPLOAD_MB", "10"))
MAX_PDF_PAGES = int(os.getenv("DOCUMIND_MAX_PDF_PAGES", "100"))
MAX_DOCS_PER_USER = int(os.getenv("DOCUMIND_MAX_DOCS_PER_USER", "20"))

# Storage backend. "local" = SQLite + ChromaDB on disk (Streamlit, tests, your laptop).
# "postgres" = Postgres + pgvector (Neon) for the free cloud deployment. Default stays "local".
BACKEND = os.getenv("DOCUMIND_BACKEND", "local")
DATABASE_URL = os.getenv("DATABASE_URL", "")  # secret: environment / .env only, never in git
EMBED_DIM = int(os.getenv("DOCUMIND_EMBED_DIM", "384"))  # all-MiniLM-L6-v2 produces 384 numbers
if BACKEND not in ("local", "postgres"):
    raise ValueError("DOCUMIND_BACKEND must be 'local' or 'postgres'")
if BACKEND == "postgres" and not DATABASE_URL:
    raise RuntimeError("DOCUMIND_BACKEND=postgres needs DATABASE_URL to be set")

# Per-user limit on POST /ask (always on, unlike the shared demo throttle above). In memory, per process.
ASK_PER_MIN = int(os.getenv("DOCUMIND_ASK_PER_MIN", "10"))
