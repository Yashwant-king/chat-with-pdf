import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import os

# Set page config
st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 Chat with Your PDF")
st.markdown("Upload a PDF and ask questions about its content using AI")

# Sidebar for API key
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your Hugging Face API Key", type="password")
    st.markdown("[Get your API key here](https://huggingface.co/settings/tokens)")

    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        st.success("API Key configured!")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("This app uses Hugging Face models to answer questions about your PDF documents.")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])

if uploaded_file is not None:
    # Extract text from PDF
    with st.spinner("📖 Reading PDF..."):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        if not text.strip():
            st.error("❌ Could not extract text from PDF. Please ensure it's a text-based PDF.")
        else:
            st.success(f"✅ Successfully extracted text from {len(pdf_reader.pages)} pages")

            # Show preview
            with st.expander("📄 Preview extracted text"):
                st.text(text[:1000] + "..." if len(text) > 1000 else text)

    if text.strip() and api_key:
        # Split text into chunks
        with st.spinner("🔨 Processing document..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.success(f"✅ Document processed into {len(chunks)} chunks")

        # Question input
        st.markdown("---")
        st.subheader("💬 Ask Questions")
        question = st.text_input("Enter your question about the PDF:")

        if question:
            with st.spinner("🤔 Thinking..."):
                # Search for relevant documents
                docs = knowledge_base.similarity_search(question, k=3)

                # Initialize LLM
                llm = HuggingFaceHub(
                    repo_id="google/flan-t5-large",
                    model_kwargs={"temperature": 0.5, "max_length": 512}
                )

                # Create QA chain
                chain = load_qa_chain(llm, chain_type="stuff")

                # Get answer
                response = chain.run(input_documents=docs, question=question)

                # Display answer
                st.markdown("### 🤖 Answer:")
                st.info(response)

                # Show source chunks
                with st.expander("📚 View source chunks"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content)
                        st.markdown("---")

    elif text.strip() and not api_key:
        st.warning("⚠️ Please enter your Hugging Face API key in the sidebar to start asking questions.")

else:
    st.info("👆 Please upload a PDF file to get started")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, LangChain, and Hugging Face</p>
    </div>
    """,
    unsafe_allow_html=True
)
