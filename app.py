"""
Chat with PDF – Modern Streamlit Web App
Upload a PDF and ask questions about its content using a Hugging Face LLM.
"""

import os
import math
import datetime
import streamlit as st
from PyPDF2 import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    VECTOR_SEARCH = True
except ImportError:
    VECTOR_SEARCH = False

# ─── Module-level constants ────────────────────────────────────────────────────
_EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know based on the provided document.\"\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

try:
    from huggingface_hub import InferenceClient
    HF_CLIENT_AVAILABLE = True
except ImportError:
    HF_CLIENT_AVAILABLE = False

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Variables ── */
:root {
    --primary:        #6366f1;
    --primary-hover:  #4f46e5;
    --primary-light:  #eef2ff;
    --secondary:      #06b6d4;
    --success:        #10b981;
    --error:          #ef4444;
    --bg-main:        #f8fafc;
    --bg-card:        #ffffff;
    --text-primary:   #0f172a;
    --text-secondary: #64748b;
    --text-muted:     #94a3b8;
    --border:         #e2e8f0;
    --shadow-sm:      0 1px 3px rgba(0,0,0,.06);
    --shadow-md:      0 4px 16px rgba(0,0,0,.08);
    --shadow-lg:      0 8px 30px rgba(0,0,0,.12);
    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      16px;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg-main) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar              { width: 6px; }
::-webkit-scrollbar-track        { background: transparent; }
::-webkit-scrollbar-thumb        { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover  { background: #94a3b8; }

/* ════════════ SIDEBAR ════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b 0%, #2d2a6e 60%, #312e81 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/* Sidebar text */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: rgba(255,255,255,.85) !important;
    font-size: .85rem !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: .7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    font-weight: 600 !important;
    color: rgba(255,255,255,.45) !important;
    margin: 0 0 8px !important;
}

/* Sidebar text input */
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background: rgba(255,255,255,.1) !important;
    border: 1px solid rgba(255,255,255,.2) !important;
    color: white !important;
    border-radius: var(--radius-sm) !important;
    font-size: .85rem !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus {
    border-color: rgba(255,255,255,.55) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.3) !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input::placeholder {
    color: rgba(255,255,255,.35) !important;
}

/* Sidebar file uploader */
[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,.25) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(255,255,255,.06) !important;
    padding: 14px !important;
    transition: all .2s ease !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
    border-color: rgba(255,255,255,.55) !important;
    background: rgba(255,255,255,.12) !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] label,
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: rgba(255,255,255,.65) !important;
    font-size: .8rem !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,.1) !important;
    color: rgba(255,255,255,.85) !important;
    border: 1px solid rgba(255,255,255,.2) !important;
    border-radius: var(--radius-sm) !important;
    width: 100% !important;
    font-size: .85rem !important;
    box-shadow: none !important;
    transition: background .2s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,.2) !important;
    transform: none !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.12) !important; }

/* ════════════ MAIN AREA ════════════ */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── App Header Bar ── */
.app-header {
    background: white;
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    gap: 14px;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: var(--shadow-sm);
}
.app-header-icon {
    width: 42px;
    height: 42px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(99,102,241,.3);
}
.app-header-text h2 {
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -.3px;
}
.app-header-text p {
    font-size: .78rem;
    color: var(--text-secondary);
    margin: 2px 0 0;
}
.status-pill {
    margin-left: auto;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: .75rem;
    font-weight: 500;
}
.status-pill.ready {
    background: rgba(16,185,129,.1);
    color: #059669;
    border: 1px solid rgba(16,185,129,.25);
}
.status-pill.waiting {
    background: rgba(99,102,241,.1);
    color: var(--primary);
    border: 1px solid rgba(99,102,241,.25);
}
.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
}

/* ── Welcome Screen ── */
.welcome-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 24px 40px;
    text-align: center;
}
.welcome-hero {
    width: 88px;
    height: 88px;
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-radius: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.6rem;
    margin: 0 auto 28px;
    box-shadow: 0 12px 40px rgba(99,102,241,.35);
}
.welcome-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 14px;
    letter-spacing: -.5px;
}
.welcome-sub {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 440px;
    line-height: 1.65;
    margin: 0 0 44px;
}
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    max-width: 620px;
    margin: 0 auto 48px;
}
.feature-card {
    background: white;
    border-radius: var(--radius-md);
    padding: 20px 16px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
    transition: all .2s ease;
    text-align: center;
}
.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary);
}
.feature-card .fc-icon  { font-size: 1.6rem; margin-bottom: 10px; }
.feature-card .fc-title {
    font-size: .85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px;
}
.feature-card .fc-desc {
    font-size: .75rem;
    color: var(--text-secondary);
    line-height: 1.45;
    margin: 0;
}
.steps-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-width: 380px;
    text-align: left;
}
.step-row {
    display: flex;
    align-items: center;
    gap: 14px;
    background: white;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    box-shadow: var(--shadow-sm);
}
.step-num {
    width: 28px;
    height: 28px;
    background: var(--primary-light);
    color: var(--primary);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .8rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-text {
    font-size: .875rem;
    color: var(--text-primary);
    font-weight: 500;
    margin: 0;
}

/* ════════════ CHAT ════════════ */
.chat-wrapper {
    padding: 24px 28px;
    display: flex;
    flex-direction: column;
    gap: 22px;
}
.msg-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    animation: fadeUp .25s ease;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

.msg-avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.msg-avatar.ai   { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; }
.msg-avatar.user { background: #e2e8f0; color: var(--text-primary); }

.msg-content { display: flex; flex-direction: column; max-width: 68%; }
.msg-row.user .msg-content { align-items: flex-end; }

.msg-bubble {
    padding: 13px 17px;
    border-radius: var(--radius-md);
    font-size: .91rem;
    line-height: 1.65;
    box-shadow: var(--shadow-sm);
    word-break: break-word;
}
.msg-bubble.ai {
    background: white;
    color: var(--text-primary);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
}
.msg-bubble.user {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover));
    color: white;
    border-top-right-radius: 4px;
}
.msg-time {
    font-size: .68rem;
    color: var(--text-muted);
    margin-top: 4px;
    padding: 0 3px;
}
.msg-row.user .msg-time { text-align: right; }

/* Typing indicator */
.typing-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 28px;
    animation: fadeUp .2s ease;
}
.typing-bubble {
    background: white;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    border-top-left-radius: 4px;
    padding: 14px 18px;
    display: flex;
    gap: 6px;
    align-items: center;
    box-shadow: var(--shadow-sm);
}
.typing-bubble span {
    width: 8px;
    height: 8px;
    background: var(--text-muted);
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
}
.typing-bubble span:nth-child(2) { animation-delay: .22s; }
.typing-bubble span:nth-child(3) { animation-delay: .44s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30%            { transform: translateY(-7px); }
}

/* ── Chat Input ── */
.stChatInput { padding: 14px 24px !important; }
[data-testid="stChatInput"] > div {
    border-radius: var(--radius-md) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15) !important;
}

/* ── Main buttons ── */
.main .stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--primary-hover)) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: .875rem !important;
    padding: .5rem 1.25rem !important;
    transition: all .2s ease !important;
    box-shadow: 0 2px 8px rgba(99,102,241,.3) !important;
    font-family: 'Inter', sans-serif !important;
}
.main .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,.4) !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] { border-radius: var(--radius-md) !important; border: none !important; }

/* ── Sidebar info card ── */
.pdf-card {
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: var(--radius-md);
    padding: 14px 16px;
}
.pdf-card .pc-name {
    color: white;
    font-size: .85rem;
    font-weight: 600;
    margin: 0 0 3px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.pdf-card .pc-meta {
    color: rgba(255,255,255,.5);
    font-size: .75rem;
    margin: 0;
}
.ready-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(16,185,129,.2);
    color: #6ee7b7;
    font-size: .72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid rgba(16,185,129,.3);
    margin-top: 9px;
}

/* ── Stat row ── */
.stat-row {
    display: flex;
    gap: 8px;
    margin-top: 14px;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,.08);
    border-radius: var(--radius-sm);
    padding: 10px;
    text-align: center;
}
.stat-box .sv { color: white; font-size: .95rem; font-weight: 700; margin: 0; }
.stat-box .sl { color: rgba(255,255,255,.45); font-size: .68rem; margin: 2px 0 0; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .feature-grid  { grid-template-columns: 1fr 1fr; }
    .msg-content   { max-width: 82%; }
    .welcome-title { font-size: 1.5rem; }
}
@media (max-width: 480px) {
    .feature-grid { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = ""
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ─── Helper Functions ─────────────────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> tuple[str, int]:
    """Extract all text from an uploaded PDF, returning (text, page_count)."""
    reader = PdfReader(uploaded_file)
    pages = reader.pages
    text = ""
    for page in pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text, len(pages)


def build_chunks(text: str) -> list[str]:
    """Split document text into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(text)


def get_relevant_chunks(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Simple keyword-based retrieval when vector search is unavailable."""
    query_terms = set(query.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_terms & chunk_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


@st.cache_resource(show_spinner=False)
def _get_embeddings():
    """Load and cache the sentence-transformer embeddings model."""
    return HuggingFaceEmbeddings(model_name=_EMBEDDINGS_MODEL)


def build_vector_store(chunks: list[str], hf_api_key: str):
    """Build a FAISS vector store using cached HuggingFace embeddings."""
    if not VECTOR_SEARCH:
        return None
    try:
        embeddings = _get_embeddings()
        return FAISS.from_texts(chunks, embeddings)
    except Exception:
        return None


def search_vector_store(vs, query: str, top_k: int = 5) -> list[str]:
    """Return top-k relevant chunks from the vector store."""
    docs = vs.similarity_search(query, k=top_k)
    return [d.page_content for d in docs]


def query_llm(question: str, context: str, hf_api_key: str) -> str:
    """Call the HuggingFace Inference API and return the answer."""
    if not HF_CLIENT_AVAILABLE or not hf_api_key:
        return (
            "⚠️ **No API key provided.** Please enter your Hugging Face API key "
            "in the sidebar to enable AI answers."
        )

    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        client = InferenceClient(model=_LLM_MODEL, token=hf_api_key)
        response = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        # Strip the echoed prompt if present
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()
        return response
    except Exception as exc:
        err = str(exc)
        if "401" in err or "403" in err:
            return "❌ **Invalid API key.** Please check your Hugging Face token."
        if "429" in err:
            return "⏳ **Rate limit reached.** Please wait a moment and try again."
        return f"❌ **Error calling the LLM:** {err}"


def now_str() -> str:
    return datetime.datetime.now().strftime("%H:%M")


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(
        """
        <div style="padding:24px 20px 18px;border-bottom:1px solid rgba(255,255,255,.1);margin-bottom:18px;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#6366f1,#06b6d4);
                        border-radius:10px;display:flex;align-items:center;justify-content:center;
                        font-size:1.2rem;">📄</div>
            <div>
              <p style="color:white;font-size:1rem;font-weight:700;margin:0;letter-spacing:-.3px;">Chat with PDF</p>
              <p style="color:rgba(255,255,255,.5);font-size:.72rem;margin:0;">AI Document Assistant</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Key
    st.markdown("### 🔑 API Key")
    hf_api_key = st.text_input(
        "Hugging Face API Token",
        type="password",
        placeholder="hf_••••••••••••••••••",
        label_visibility="collapsed",
    )
    if hf_api_key:
        st.markdown(
            '<p style="color:rgba(16,185,129,.85);font-size:.75rem;margin:-6px 0 6px;">✓ Token saved for this session</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p style="color:rgba(255,255,255,.4);font-size:.75rem;margin:-6px 0 6px;">'
            'Get a free token at <a href="https://huggingface.co/settings/tokens" '
            'target="_blank" style="color:rgba(99,102,241,.9);">huggingface.co</a></p>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="margin:16px 0;">', unsafe_allow_html=True)

    # Upload
    st.markdown("### 📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        # Process only when a new file is uploaded
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Processing PDF…"):
                text, pages = extract_pdf_text(uploaded_file)
                if text.strip():
                    st.session_state.pdf_text = text
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.pdf_pages = pages
                    st.session_state.messages = []
                    chunks = build_chunks(text)
                    st.session_state.chunks = chunks
                    if VECTOR_SEARCH and hf_api_key:
                        st.session_state.vector_store = build_vector_store(chunks, hf_api_key)
                    else:
                        st.session_state.vector_store = None
                else:
                    st.error("No readable text found. Please use a text-based PDF.")

        if st.session_state.pdf_name:
            size_kb = math.ceil(uploaded_file.size / 1024)
            chunk_count = len(st.session_state.chunks)
            st.markdown(
                f"""
                <div class="pdf-card">
                  <p class="pc-name">📄 {st.session_state.pdf_name}</p>
                  <p class="pc-meta">{st.session_state.pdf_pages} pages · {size_kb} KB</p>
                  <span class="ready-badge">● Ready to chat</span>
                </div>
                <div class="stat-row">
                  <div class="stat-box">
                    <p class="sv">{st.session_state.pdf_pages}</p>
                    <p class="sl">Pages</p>
                  </div>
                  <div class="stat-box">
                    <p class="sv">{chunk_count}</p>
                    <p class="sl">Chunks</p>
                  </div>
                  <div class="stat-box">
                    <p class="sv">{sum(1 for m in st.session_state.messages if m['role'] == 'user')}</p>
                    <p class="sl">Turns</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<hr style="margin:16px 0;">', unsafe_allow_html=True)

    # How to use
    st.markdown("### 💡 How to use")
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:8px;margin-top:4px;">
          <div style="display:flex;align-items:flex-start;gap:10px;">
            <div style="width:20px;height:20px;background:rgba(99,102,241,.5);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:.7rem;font-weight:700;color:white;flex-shrink:0;margin-top:1px;">1</div>
            <p style="color:rgba(255,255,255,.7);font-size:.8rem;line-height:1.5;margin:0;">Enter your HF API token above</p>
          </div>
          <div style="display:flex;align-items:flex-start;gap:10px;">
            <div style="width:20px;height:20px;background:rgba(99,102,241,.5);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:.7rem;font-weight:700;color:white;flex-shrink:0;margin-top:1px;">2</div>
            <p style="color:rgba(255,255,255,.7);font-size:.8rem;line-height:1.5;margin:0;">Upload any text-based PDF file</p>
          </div>
          <div style="display:flex;align-items:flex-start;gap:10px;">
            <div style="width:20px;height:20px;background:rgba(99,102,241,.5);border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:.7rem;font-weight:700;color:white;flex-shrink:0;margin-top:1px;">3</div>
            <p style="color:rgba(255,255,255,.7);font-size:.8rem;line-height:1.5;margin:0;">Ask any question about the document</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clear chat
    if st.session_state.messages:
        st.markdown('<hr style="margin:16px 0;">', unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ─── Main Area ────────────────────────────────────────────────────────────────
pdf_ready = bool(st.session_state.pdf_text)

# Header bar
status_label = "Ready to answer" if pdf_ready else "Waiting for PDF"
status_class = "ready" if pdf_ready else "waiting"
doc_info = (
    f"<b>{st.session_state.pdf_name}</b> — {st.session_state.pdf_pages} pages"
    if pdf_ready
    else "No document loaded"
)

st.markdown(
    f"""
    <div class="app-header">
      <div class="app-header-icon">📄</div>
      <div class="app-header-text">
        <h2>Chat with PDF</h2>
        <p>{doc_info}</p>
      </div>
      <span class="status-pill {status_class}">
        <span class="status-dot"></span>
        {status_label}
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome / Empty State ──────────────────────────────────────────────────────
if not pdf_ready:
    st.markdown(
        """
        <div class="welcome-wrapper">
          <div class="welcome-hero">📄</div>
          <h1 class="welcome-title">Chat with your PDF</h1>
          <p class="welcome-sub">
            Upload any PDF and ask questions in plain English.
            Powered by Hugging Face LLMs and FAISS semantic search.
          </p>
          <div class="feature-grid">
            <div class="feature-card">
              <div class="fc-icon">🔍</div>
              <p class="fc-title">Smart Search</p>
              <p class="fc-desc">Finds the most relevant passages for each question</p>
            </div>
            <div class="feature-card">
              <div class="fc-icon">🤖</div>
              <p class="fc-title">AI Answers</p>
              <p class="fc-desc">Grounded responses from the actual document text</p>
            </div>
            <div class="feature-card">
              <div class="fc-icon">💬</div>
              <p class="fc-title">Chat History</p>
              <p class="fc-desc">Full conversation context kept throughout the session</p>
            </div>
          </div>
          <div class="steps-list">
            <div class="step-row">
              <div class="step-num">1</div>
              <p class="step-text">Add your Hugging Face API token in the sidebar</p>
            </div>
            <div class="step-row">
              <div class="step-num">2</div>
              <p class="step-text">Upload a PDF using the sidebar uploader</p>
            </div>
            <div class="step-row">
              <div class="step-num">3</div>
              <p class="step-text">Type your question below and press Enter</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat Interface ─────────────────────────────────────────────────────────────
if pdf_ready:
    # Render existing messages
    if st.session_state.messages:
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("time", "")
            avatar = "🤖" if role == "assistant" else "🧑"
            bubble_cls = "ai" if role == "assistant" else "user"
            st.markdown(
                f"""
                <div class="msg-row {role}">
                  <div class="msg-avatar {bubble_cls}">{avatar}</div>
                  <div class="msg-content">
                    <div class="msg-bubble {bubble_cls}">{content}</div>
                    <span class="msg-time">{ts}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Prompt card when PDF is loaded but no messages yet
        st.markdown(
            """
            <div style="display:flex;flex-direction:column;align-items:center;
                        padding:48px 24px;text-align:center;">
              <div style="width:64px;height:64px;background:linear-gradient(135deg,#6366f1,#06b6d4);
                          border-radius:16px;display:flex;align-items:center;justify-content:center;
                          font-size:2rem;margin-bottom:20px;box-shadow:0 8px 24px rgba(99,102,241,.3);">
                💬
              </div>
              <h3 style="font-size:1.1rem;font-weight:600;color:#0f172a;margin:0 0 8px;">
                Your document is ready!
              </h3>
              <p style="font-size:.9rem;color:#64748b;max-width:360px;line-height:1.6;margin:0;">
                Ask anything about the PDF. Try questions like
                <em>"What is the main topic?"</em> or
                <em>"Summarize the key points."</em>
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Chat Input ─────────────────────────────────────────────────────────────────
if pdf_ready:
    prompt = st.chat_input("Ask a question about your PDF…")
    if prompt:
        ts = now_str()
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "time": ts}
        )

        with st.spinner("Thinking…"):
            # Retrieve relevant context
            if st.session_state.vector_store:
                context_chunks = search_vector_store(
                    st.session_state.vector_store, prompt
                )
            else:
                context_chunks = get_relevant_chunks(
                    prompt, st.session_state.chunks
                )

            context = "\n\n---\n\n".join(context_chunks)
            answer = query_llm(prompt, context, hf_api_key)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "time": now_str()}
        )
        st.rerun()
