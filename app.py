import os
import tempfile
import streamlit as st
from store import build_index
from rag import answer

st.set_page_config(page_title="Doc Q&A (RAG)", page_icon="📄")
st.title("📄 Ask your PDF")
st.caption("Upload a PDF, ask questions, get answers with page citations.")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.session_state.get("loaded") != uploaded.name:
    with st.spinner("Reading and indexing the document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            path = tmp.name
        n = build_index(path)
        os.remove(path)
    st.session_state["loaded"] = uploaded.name
    st.success(f"Indexed {n} chunks from {uploaded.name}")

question = st.text_input("Your question")

if st.button("Ask") and question:
    if "loaded" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Thinking..."):
            text, hits = answer(question)
        st.markdown(text)
        with st.expander("Sources used"):
            for h in hits:
                st.markdown(f"**Page {h['page']}**")
                st.write(h["text"][:400] + "...")