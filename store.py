import chromadb
from chunker import read_pdf, chunk_text

client = chromadb.PersistentClient(path="chroma_db")

def build_index(pdf_path):
    pages = read_pdf(pdf_path)
    chunks = chunk_text(pages)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs")
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return len(chunks)

def search(question, k=3):
    col = client.get_collection("docs")
    res = col.query(query_texts=[question], n_results=k)
    return [
        {"text": t, "page": m["page"]}
        for t, m in zip(res["documents"][0], res["metadatas"][0])
    ]

if __name__ == "__main__":
    n = build_index("sample.pdf")
    print("Indexed chunks:", n)
    for r in search("What is multi-head attention?"):
        print("--- Page", r["page"], "---")
        print(r["text"][:300])