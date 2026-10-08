\# Doc Q\&A (RAG): Ask Your PDF



Upload a PDF, ask questions, and get answers with page citations. If the answer isn't in the document, the app says so instead of guessing.



\*\*Live demo:\*\* https://doc-app-rag-7wgvuvabsqbyokqzkcmxe3.streamlit.app/



\## How it works

1\. \*\*Extract:\*\* pypdf reads the text of each page and keeps the page number.

2\. \*\*Chunk:\*\* text is split into 800-character chunks with 150-character overlap; reference-list chunks are filtered out.

3\. \*\*Embed and store:\*\* ChromaDB embeds chunks with the all-MiniLM-L6-v2 model (ONNX) and stores them with their page numbers.

4\. \*\*Retrieve:\*\* the question is embedded and the closest chunks are fetched by semantic similarity.

5\. \*\*Generate:\*\* the chunks go to an LLM (openai/gpt-oss-20b on Groq) with instructions to answer only from the context and cite pages.



\## Tech stack

Python, Streamlit, ChromaDB, pypdf, Groq API



\## Run locally

1\. Clone the repo and create a virtual environment.

2\. pip install -r requirements.txt

3\. Create a .env file containing GROQ\_API\_KEY=your-key

4\. streamlit run app.py



\## Limitations

\- Demo app: one document index at a time, shared by all visitors

\- Works on text PDFs only (no OCR for scanned pages)

\- Runs on free tiers, so the first load after idle can be slow



\## Possible improvements

\- Per-user sessions, OCR support, retrieval evaluation set, hybrid search

