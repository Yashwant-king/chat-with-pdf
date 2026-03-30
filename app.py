import streamlit as st
import os
from dotenv import load_dotenv
from pdf_utils import extract_text, chunk_text
from vector_store import build_index, search_index
from llm import ask_groq

load_dotenv()

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom CSS for a cleaner look
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; padding: 4px; }
    .stSidebar { background-color: #1e1e2e; }
    .stSidebar h1 { color: #cdd6f4; }
    div[data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    .source-box {
        background: #1e1e2e;
        border-left: 3px solid #89b4fa;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85em;
        margin: 4px 0;
        color: #cdd6f4;
    }
    .stat-box {
        background: #313244;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        color: #cdd6f4;
    }
</style>
""", unsafe_allow_html=True)


# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "pdf_names" not in st.session_state:
    st.session_state.pdf_names = []


# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.markdown("## 📄 Chat with PDF")
    st.markdown("Powered by **Groq LLaMA 3.1** + **FAISS RAG**")
    st.divider()

    # API key — check env first, fallback to text input
    default_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        value=default_key,
        placeholder="gsk_...",
        help="Get a free key at console.groq.com"
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "📂 Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDFs at once"
    )

    col1, col2 = st.columns(2)
    with col1:
        process_btn = st.button("⚙️ Process", use_container_width=True, type="primary")
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.messages = []
        st.session_state.index = None
        st.session_state.chunks = []
        st.session_state.pdf_names = []
        st.rerun()

    # show indexed docs info
    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**📚 Indexed Documents:**")
        for name in st.session_state.pdf_names:
            st.markdown(f"- `{name}`")
        st.markdown(f"**{len(st.session_state.chunks)}** chunks in memory")


# =====================
# PROCESS PDFs
# =====================
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Upload at least one PDF first")
    elif not api_key:
        st.sidebar.error("Enter your Groq API key first")
    else:
        with st.spinner("Reading and indexing PDFs..."):
            all_chunks = []
            names = []
            errors = []

            for f in uploaded_files:
                try:
                    text = extract_text(f)
                    chunks = chunk_text(text, source=f.name)
                    all_chunks.extend(chunks)
                    names.append(f.name)
                except Exception as e:
                    errors.append(f"{f.name}: {str(e)}")

            if errors:
                for err in errors:
                    st.sidebar.warning(f"⚠️ {err}")

            if all_chunks:
                with st.spinner("Building search index..."):
                    st.session_state.chunks = all_chunks
                    st.session_state.index = build_index(all_chunks)
                    st.session_state.pdf_names = names
                    st.session_state.messages = []  # reset chat for new docs

                st.sidebar.success(
                    f"✅ Done! {len(names)} PDF(s), {len(all_chunks)} chunks indexed"
                )
                st.rerun()


# =====================
# MAIN CHAT UI
# =====================
if st.session_state.index is None:
    # landing screen
    st.markdown("## 👋 Welcome to Chat with PDF")
    st.markdown("""
    Upload your PDF documents in the sidebar and start asking questions.
    The app will use **semantic search + Groq LLaMA 3.1** to find relevant sections and answer accurately.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="stat-box">📄 Multi-PDF<br><small>Chat across multiple docs</small></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-box">🔍 RAG Search<br><small>FAISS semantic search</small></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="stat-box">🤖 Groq LLaMA<br><small>Fast & accurate answers</small></div>', unsafe_allow_html=True)

    st.info("👈 Start by uploading a PDF in the sidebar and clicking **Process**")

else:
    # show doc stats bar
    st.markdown(
        f"💬 Chatting with **{len(st.session_state.pdf_names)} PDF(s)** — "
        f"`{'`, `'.join(st.session_state.pdf_names)}`"
    )
    st.divider()

    # render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-box">📄 <b>{src["source"]}</b> — Chunk #{src["chunk_id"]}<br>'
                            f'<i>{src["text"][:200]}...</i></div>',
                            unsafe_allow_html=True
                        )

    # chat input
    if prompt := st.chat_input("Ask anything about your PDF..."):
        if not api_key:
            st.error("Please enter your Groq API key in the sidebar")
            st.stop()

        # show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # retrieve relevant chunks
        relevant = search_index(
            st.session_state.index,
            st.session_state.chunks,
            prompt,
            k=5
        )
        context = "\n\n---\n\n".join([c["text"] for c in relevant])

        # get response from Groq
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_groq(
                    question=prompt,
                    context=context,
                    api_key=api_key,
                    chat_history=st.session_state.messages[:-1]  # exclude current user msg
                )
                st.write(answer)

                # show sources inline
                with st.expander("📚 Sources", expanded=False):
                    for src in relevant:
                        st.markdown(
                            f'<div class="source-box">📄 <b>{src["source"]}</b> — Chunk #{src["chunk_id"]}<br>'
                            f'<i>{src["text"][:200]}...</i></div>',
                            unsafe_allow_html=True
                        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": relevant
        })
