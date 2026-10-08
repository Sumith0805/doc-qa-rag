import os
from dotenv import load_dotenv
from groq import Groq
from store import search

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer(question, k=4):
    hits = search(question, k)
    context = "\n\n".join(
        f"[Page {h['page']}]\n{h['text']}" for h in hits
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite page numbers like (p. 5). "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    reply = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
    )
    return reply.choices[0].message.content, hits

if __name__ == "__main__":
    q = "What is multi-head attention and why is it used?"
    text, hits = answer(q)
    print(text)
    print("\nSources:", sorted({h["page"] for h in hits}))