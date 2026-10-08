from sentence_transformers import SentenceTransformer
import chromadb
from chunker import read_pdf, chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

def build_index(pdf_path):
    pages = read_pdf(pdf_path)
    chunks = chunk_text(pages)
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts).tolist()
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"page": c["page"]} for c in chunks],
    )
    return len(chunks)

def search(question, k=3):
    col = client.get_collection("docs")
    q = model.encode([question]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
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