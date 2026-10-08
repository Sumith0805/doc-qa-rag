from pypdf import PdfReader

def read_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages

def chunk_text(pages, size=800, overlap=150):
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"text": piece, "page": page_num})
            start += size - overlap
    return chunks

if __name__ == "__main__":
    pages = read_pdf("sample.pdf")
    chunks = chunk_text(pages)
    print("Pages:", len(pages))
    print("Chunks:", len(chunks))
    print("--- First chunk ---")
    print(chunks[0])