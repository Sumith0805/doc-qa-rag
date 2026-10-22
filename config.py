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

# Chunking
CHUNK_SIZE = int(os.getenv("DOCUMIND_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("DOCUMIND_CHUNK_OVERLAP", "150"))

ALLOWED_EXTENSIONS = (".pdf", ".txt", ".md")
