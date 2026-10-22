from pathlib import Path

import pytest

from chunker import chunk_text, read_document

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = ROOT / "eval" / "docs" / "attention_is_all_you_need.pdf"


def test_chunks_keep_page_numbers():
    pages = [(1, "alpha " * 300), (2, "beta " * 300)]
    chunks = chunk_text(pages, size=200, overlap=40)
    assert {c["page"] for c in chunks} == {1, 2}
    assert all("beta" not in c["text"] for c in chunks if c["page"] == 1)
    assert all("alpha" not in c["text"] for c in chunks if c["page"] == 2)


def test_chunks_respect_size_and_make_progress():
    chunks = chunk_text([(1, "word " * 1000)], size=200, overlap=50)
    assert len(chunks) > 5
    assert all(len(c["text"]) <= 200 for c in chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_text([(1, ""), (2, "   \n  ")]) == []


def test_short_page_is_single_chunk():
    chunks = chunk_text([(3, "hello world")])
    assert chunks == [{"text": "hello world", "page": 3}]


def test_read_txt(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello there", encoding="utf-8")
    assert read_document(str(f), "a.txt") == [(1, "hello there")]


def test_unsupported_type(tmp_path):
    f = tmp_path / "a.exe"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        read_document(str(f), "a.exe")


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="sample PDF missing")
def test_real_pdf_page_numbers():
    pages = read_document(str(SAMPLE_PDF), SAMPLE_PDF.name)
    assert len(pages) == 15
    page5 = dict(pages)[5]
    assert "Multi-head attention" in page5
    chunks = chunk_text(pages)
    assert any(c["page"] == 5 and "h = 8" in c["text"] for c in chunks)
