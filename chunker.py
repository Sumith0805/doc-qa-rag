"""Read documents and split them into page-aware chunks."""
from pathlib import Path

from pypdf import PdfReader

import config


def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages


def read_txt(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return [(1, text)]


def read_document(path, filename=None):
    """Return [(page_number, text), ...] for a .pdf / .txt / .md file."""
    suffix = Path(filename or path).suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix in (".txt", ".md"):
        return read_txt(path)
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


def chunk_text(pages, size=None, overlap=None):
    """Split each page into overlapping chunks. Chunks never cross page boundaries,
    so every chunk has an exact page number for citations."""
    size = size or config.CHUNK_SIZE
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks = []
    for page_num, text in pages:
        n = len(text)
        start = 0
        while start < n:
            end = min(start + size, n)
            if end < n:  # try not to cut a word in half
                cut = text.rfind(" ", max(start + 1, end - 100), end)
                if cut > start:
                    end = cut
            piece = text[start:end].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            if end >= n:
                break
            start = max(end - overlap, start + 1)
    return chunks
