import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

st.set_page_config(page_title="Chat with your PDF", page_icon="📄")
st.title("📄 Chat with your PDF")

pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    db = FAISS.from_texts(chunks, embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY"))
    chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

    st.success("✅ PDF processed successfully! Ask your questions below 👇")

    chat_history = []
    query = st.text_input("Ask a question about your PDF:")
    if query:
        result = chain({"question": query, "chat_history": chat_history})
        st.markdown(f"**Answer:** {result['answer']}")
        chat_history.append((query, result["answer"]))