
import os
import tempfile
import hashlib
import streamlit as st

from store import build_index
from rag import answer


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocuMind | AI Workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "loaded": None,
    "document_name": None,
    "chunk_count": 0,
    "chat_history": [],
    "documents": [],
    "questions_count": 0,
    "sources_count": 0,
    "active_page": "Dashboard",
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b0f14;
    --panel: #111820;
    --panel2: #161e28;
    --border: #273342;
    --text: #f5f7fb;
    --muted: #98a6b8;
    --purple: #8b72ff;
    --blue: #60a5fa;
    --green: #34d399;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    color: var(--text);
    background:
        radial-gradient(ellipse at 12% 0%, rgba(124,92,252,.11), transparent 32%),
        radial-gradient(ellipse at 95% 70%, rgba(59,130,246,.07), transparent 28%),
        var(--bg);
}

[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    animation: appear .55s ease-out;
}

@keyframes appear {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}

@keyframes pulse {
    0%, 100% { opacity: .55; }
    50% { opacity: 1; }
}

@keyframes messageIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: #0e141b;
    border-right: 1px solid #202b37;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #c7d0dc;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    margin-bottom: 7px;
}

.sidebar-logo {
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    color: white;
    font-size: 22px;
    font-weight: 800;
    background: linear-gradient(135deg, #9279ff, #4c83ff);
    box-shadow: 0 6px 24px rgba(124,92,252,.22);
    animation: float 4s ease-in-out infinite;
}

.sidebar-title {
    color: #f5f7fb;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -.6px;
}

.sidebar-subtitle {
    color: #8492a4;
    font-size: 11px;
    margin-top: 2px;
}

.sidebar-section {
    color: #718096;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin: 25px 0 10px;
}

.sidebar-document {
    padding: 11px;
    margin: 7px 0;
    border: 1px solid #273342;
    border-radius: 12px;
    background: #141c25;
    overflow-wrap: anywhere;
}

.sidebar-document-name {
    color: #e6ebf3;
    font-size: 11px;
    font-weight: 600;
    line-height: 1.5;
}

.sidebar-document-meta {
    color: #8795a7;
    font-size: 10px;
    margin-top: 5px;
}

.sidebar-stat {
    padding: 12px;
    border: 1px solid #26313e;
    border-radius: 12px;
    background: #121a23;
    margin: 8px 0;
}

.sidebar-stat-label {
    color: #8c9aad;
    font-size: 10px;
}

.sidebar-stat-value {
    color: #f5f7fb;
    font-size: 23px;
    font-weight: 800;
    margin-top: 4px;
}

/* MAIN HEADER */

.topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 27px;
}

.breadcrumb {
    color: #8290a2;
    font-size: 12px;
}

.online-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 12px;
    border: 1px solid #28463d;
    border-radius: 999px;
    color: #8ee5bd;
    background: rgba(52,211,153,.06);
    font-size: 11px;
    white-space: nowrap;
}

.online-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 10px rgba(52,211,153,.5);
    animation: pulse 2s infinite;
}

/* HERO */

.hero {
    position: relative;
    overflow: hidden;
    padding: 35px;
    margin-bottom: 23px;
    border: 1px solid rgba(139,114,255,.25);
    border-radius: 22px;
    background: linear-gradient(125deg, rgba(139,114,255,.13), #111820 75%);
}

.hero:after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    right: -100px;
    top: -125px;
    border-radius: 50%;
    background: rgba(124,92,252,.15);
    filter: blur(65px);
    pointer-events: none;
}

.hero-label {
    color: #b4a5ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.7px;
    margin-bottom: 12px;
}

.hero-title {
    color: #f5f7fb;
    font-size: clamp(32px, 4vw, 49px);
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.12;
}

.gradient-text {
    background: linear-gradient(90deg, #b6a5ff, #76b6ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    max-width: 630px;
    margin-top: 12px;
    color: #9ba8b8;
    font-size: 13px;
    line-height: 1.8;
}

/* SECTION HEADINGS */

.section-heading {
    color: #f3f6fb;
    font-size: 18px;
    font-weight: 750;
    margin: 26px 0 5px;
    letter-spacing: -.35px;
}

.section-caption {
    color: #8e9bad;
    font-size: 12px;
    margin-bottom: 15px;
}

/* METRIC CARDS */

.metric-card {
    height: 100%;
    padding: 19px;
    border: 1px solid #273342;
    border-radius: 16px;
    background: linear-gradient(145deg, #151e28, #111820);
    transition: transform .22s, border-color .22s, box-shadow .22s;
    animation: messageIn .45s ease-out;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: #6654ba;
    box-shadow: 0 12px 28px rgba(0,0,0,.18);
}

.metric-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
}

.metric-label {
    color: #98a6b8;
    font-size: 11px;
    font-weight: 600;
}

.metric-icon {
    font-size: 19px;
}

.metric-value {
    color: #f5f7fb;
    font-size: 29px;
    font-weight: 800;
    margin-top: 14px;
    letter-spacing: -.8px;
}

.metric-foot {
    color: #77869a;
    font-size: 10px;
    margin-top: 5px;
}

/* DOCUMENT CARDS */

.document-card {
    height: 100%;
    padding: 18px;
    border: 1px solid #273342;
    border-radius: 16px;
    background: #121a23;
    transition: transform .22s, border-color .22s;
    animation: messageIn .45s ease-out;
}

.document-card:hover {
    transform: translateY(-3px);
    border-color: #6654ba;
}

.document-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: rgba(139,114,255,.13);
    font-size: 21px;
    margin-bottom: 13px;
}

.document-name {
    color: #f0f3f8;
    font-size: 12px;
    font-weight: 700;
    overflow-wrap: anywhere;
    line-height: 1.6;
}

.document-meta {
    color: #8998ab;
    font-size: 10px;
    margin-top: 7px;
}

/* UPLOADER */

[data-testid="stFileUploader"] {
    padding: 12px;
    border: 1px solid #273342;
    border-radius: 17px;
    background: rgba(17,24,32,.8);
    transition: border-color .2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #8b72ff;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #465367 !important;
    border-radius: 12px !important;
    background: rgba(139,114,255,.035) !important;
}

/* BUTTONS */

.stButton > button,
.stFormSubmitButton > button {
    min-height: 42px;
    border: 1px solid #364253;
    border-radius: 11px;
    background: #19222d;
    color: #f5f7fb;
    font-size: 12px;
    font-weight: 650;
    transition: transform .2s, border-color .2s, box-shadow .2s;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: white;
    border-color: #8b72ff;
    transform: translateY(-1px);
    box-shadow: 0 7px 20px rgba(139,114,255,.12);
}

/* INPUTS */

div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #111820;
    border-color: #303c4b;
    border-radius: 12px;
}

div[data-baseweb="input"]:focus-within {
    border-color: #8b72ff;
}

/* CHAT */

.chat-card {
    padding: 19px;
    margin: 13px 0;
    border: 1px solid #273342;
    border-radius: 15px;
    background: #121a23;
    animation: messageIn .35s ease-out;
}

.chat-label {
    color: #b7a8ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 9px;
}

.answer-card {
    padding: 21px;
    border: 1px solid rgba(139,114,255,.27);
    border-radius: 15px;
    background: linear-gradient(145deg, rgba(139,114,255,.07), #111820);
    animation: messageIn .4s ease-out;
}

div[data-testid="stExpander"] {
    border: 1px solid #273342;
    border-radius: 13px;
    background: #111820;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

/* FOOTER */

.footer {
    color: #6f7e91;
    text-align: center;
    font-size: 10px;
    margin-top: 38px;
    padding-top: 18px;
    border-top: 1px solid #202a35;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 1rem 2rem;
    }

    .hero {
        padding: 25px;
    }

    .hero-title {
        font-size: 34px;
        letter-spacing: -1px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HTML HELPER
# =========================================================

def render_html(content):
    st.markdown(content, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    render_html("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">◈</div>
        <div>
            <div class="sidebar-title">DocuMind</div>
            <div class="sidebar-subtitle">AI document workspace</div>
        </div>
    </div>
    """)

    render_html('<div class="sidebar-section">WORKSPACE</div>')

    
    if st.button("⌂   Dashboard", use_container_width=True):
        st.session_state["active_page"] = "Dashboard"

    if st.button("▤   My Documents", use_container_width=True):
        st.session_state["active_page"] = "My Documents"

    if st.button("✦   AI Assistant", use_container_width=True):
        st.session_state["active_page"] = "AI Assistant"

    render_html('<div class="sidebar-section">SESSION OVERVIEW</div>')

    render_html(f"""
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Documents uploaded</div>
        <div class="sidebar-stat-value">{len(st.session_state["documents"])}</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Questions asked</div>
        <div class="sidebar-stat-value">{st.session_state["questions_count"]}</div>
    </div>
    <div class="sidebar-stat">
        <div class="sidebar-stat-label">Sources returned</div>
        <div class="sidebar-stat-value">{st.session_state["sources_count"]}</div>
    </div>
    """)

    render_html('<div class="sidebar-section">RECENT DOCUMENTS</div>')

    if st.session_state["documents"]:
        for doc in reversed(st.session_state["documents"][-5:]):
            render_html(f"""
            <div class="sidebar-document">
                <div class="sidebar-document-name">
                    📄 {doc["name"]}
                </div>
                <div class="sidebar-document-meta">
                    {doc["chunks"]} chunks · Indexed
                </div>
            </div>
            """)
    else:
        st.caption("Your uploaded documents will appear here.")

    st.markdown("---")
    st.caption("DocuMind · RAG Workspace")


# =========================================================
# TOP HEADER
# =========================================================

page = st.session_state["active_page"]

render_html(f"""
<div class="topbar">
    <div class="breadcrumb">Workspace &nbsp; / &nbsp; {page}</div>
    <div class="online-pill">
        <span class="online-dot"></span>
        Workspace active
    </div>
</div>
""")


# =========================================================
# DASHBOARD HERO
# =========================================================

render_html("""
<div class="hero">
    <div class="hero-label">YOUR KNOWLEDGE, CONNECTED</div>
    <div class="hero-title">
        Your documents.<br>
        <span class="gradient-text">One intelligent workspace.</span>
    </div>
    <div class="hero-description">
        Upload your PDFs, explore their contents, and ask questions
        with AI-powered answers grounded in retrieved passages.
    </div>
</div>
""")


# =========================================================
# METRICS
# =========================================================

total_docs = len(st.session_state["documents"])
total_chunks = sum(d["chunks"] for d in st.session_state["documents"])
total_questions = st.session_state["questions_count"]
total_sources = st.session_state["sources_count"]

m1, m2, m3, m4 = st.columns(4, gap="small")

metrics = [
    (m1, "DOCUMENTS", total_docs, "📁", "Uploaded this session"),
    (m2, "INDEXED CHUNKS", total_chunks, "◈", "Reported by the indexer"),
    (m3, "QUESTIONS ASKED", total_questions, "✦", "Across this session"),
    (m4, "SOURCE PASSAGES", total_sources, "⌕", "Returned by the retriever"),
]

for column, label, value, icon, foot in metrics:
    with column:
        render_html(f"""
        <div class="metric-card">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-foot">{foot}</div>
        </div>
        """)


# =========================================================
# PAGE: MY DOCUMENTS
# =========================================================

if page == "My Documents":
    render_html("""
    <div class="section-heading">Your document library</div>
    <div class="section-caption">
        Documents uploaded during this app session.
    </div>
    """)

    if st.session_state["documents"]:
        for doc in reversed(st.session_state["documents"]):
            render_html(f"""
            <div class="document-card">
                <div class="document-icon">📄</div>
                <div class="document-name">{doc["name"]}</div>
                <div class="document-meta">
                    Indexed · {doc["chunks"]} chunks
                </div>
            </div>
            """)
            st.markdown("")
    else:
        st.info("No documents yet. Upload a PDF from the dashboard.")

    st.stop()


# =========================================================
# PAGE: AI ASSISTANT
# =========================================================

if page == "AI Assistant":
    render_html("""
    <div class="section-heading">AI Assistant</div>
    <div class="section-caption">
        Ask questions about the currently indexed document.
    </div>
    """)

    if st.session_state["loaded"] is None:
        st.info("Upload a PDF from the Dashboard to get started.")
        st.stop()


# =========================================================
# UPLOAD AREA
# =========================================================

render_html("""
<div class="section-heading">Document workspace</div>
<div class="section-caption">
    Add a PDF to prepare its contents for question answering.
</div>
""")

uploaded = st.file_uploader(
    "Drop a PDF here or browse files",
    type=["pdf"],
    help="Select a PDF document to index.",
    key="pdf_upload",
)


# =========================================================
# INDEX DOCUMENT
# =========================================================

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state["loaded"] != file_hash:
        temp_path = None

        try:
            with st.status(
                "Preparing your document...",
                expanded=True,
            ) as status:
                st.write("Reading PDF contents...")
                st.write("Building the searchable index...")

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as tmp:
                    tmp.write(file_bytes)
                    temp_path = tmp.name

                try:
                    chunk_count = build_index(temp_path)
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

                if not chunk_count:
                    raise ValueError(
                        "The PDF indexer returned zero chunks."
                    )

                status.update(
                    label="Document indexing finished",
                    state="complete",
                    expanded=False,
                )

            st.session_state["loaded"] = file_hash
            st.session_state["document_name"] = uploaded.name
            st.session_state["chunk_count"] = chunk_count
            st.session_state["chat_history"] = []

            # Keep one record per unique file hash.
            existing_hashes = {
                doc["hash"] for doc in st.session_state["documents"]
            }

            if file_hash not in existing_hashes:
                st.session_state["documents"].append({
                    "name": uploaded.name,
                    "hash": file_hash,
                    "chunks": chunk_count,
                })

            st.success(f"{uploaded.name} is ready to query.")

            st.rerun()

        except Exception as exc:
            st.error(f"Document indexing failed: {exc}")
            st.info(
                "Check that the PDF contains readable text and "
                "that the existing indexer is configured correctly."
            )


# =========================================================
# CURRENT DOCUMENT
# =========================================================

if st.session_state["loaded"] is not None:
    render_html(f"""
    <div class="document-card">
        <div class="document-icon">📄</div>
        <div class="document-name">{st.session_state["document_name"]}</div>
        <div class="document-meta">
            ● Ready to query &nbsp; · &nbsp;
            {st.session_state["chunk_count"]} indexed chunks
        </div>
    </div>
    """)


# =========================================================
# ASK QUESTIONS
# =========================================================

if st.session_state["loaded"] is not None:
    render_html("""
    <div class="section-heading">Ask your document</div>
    <div class="section-caption">
        Enter a question and inspect the sources behind the answer.
    </div>
    """)

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="What are the main points in this document?",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button(
            "Ask DocuMind  →",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching your document and generating an answer..."):
                    answer_text, hits = answer(question.strip())

                hits = hits or []

                st.session_state["questions_count"] += 1
                st.session_state["sources_count"] += len(hits)

                st.session_state["chat_history"].append({
                    "question": question.strip(),
                    "answer": answer_text,
                    "sources": hits,
                })

                st.rerun()

            except Exception as exc:
                st.error(f"Unable to generate an answer: {exc}")
                st.info(
                    "Check your API configuration and the terminal "
                    "for the underlying error."
                )


# =========================================================
# CONVERSATION
# =========================================================

if st.session_state["chat_history"]:
    render_html("""
    <div class="section-heading">Conversation</div>
    <div class="section-caption">
        Your recent questions, answers, and retrieved passages.
    </div>
    """)

    for idx, conversation in enumerate(
        reversed(st.session_state["chat_history"])
    ):
        st.markdown(f"**You asked:** {conversation['question']}")

        render_html("""
        <div class="answer-card">
            <div class="chat-label">✦ AI ANSWER</div>
        </div>
        """)

        st.markdown(conversation["answer"])

        sources = conversation["sources"]

        with st.expander(f"🔎 Sources used ({len(sources)})"):
            if sources:
                for number, source in enumerate(sources, start=1):
                    page_number = source.get("page", "?")
                    source_text = source.get("text", "")
                    preview = source_text[:800]

                    st.markdown(f"**Passage {number} · Page {page_number}**")
                    st.write(
                        preview + ("…" if len(source_text) > 800 else "")
                    )

                    if number < len(sources):
                        st.divider()
            else:
                st.info("No source passages were returned.")

        st.markdown("")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


# =========================================================
# EMPTY STATE
# =========================================================

elif st.session_state["loaded"] is None:
    render_html("""
    <div class="chat-card">
        <div class="chat-label">YOUR WORKSPACE IS READY</div>
        <div style="font-size:14px;color:#cbd5e1;line-height:1.8;">
            Upload a PDF above to activate document chat.
            Your session statistics and document library will
            update as you work.
        </div>
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

render_html("""
<div class="footer">
    DOCUMIND &nbsp;·&nbsp; AI DOCUMENT INTELLIGENCE
    <br><br>
    Session statistics reset when the app session is restarted.
</div>
""")