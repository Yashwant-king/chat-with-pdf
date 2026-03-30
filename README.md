# 📄 Chat with PDF — AI Document Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chat-with-pdf-k8iq5wfwp9e3wpqttjhwek.streamlit.app/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yashwant-king/chat-with-pdf/blob/main/chat_with_pdf_hf.ipynb)

A **Streamlit web app** that lets you upload PDFs and chat with them using **Groq LLaMA 3.1** and **FAISS semantic search (RAG)**.

---

## ✨ Features

- 📂 Upload **multiple PDFs** at once
- 🔍 **RAG pipeline** — FAISS + sentence-transformers for accurate retrieval
- 🤖 **Groq LLaMA 3.1** for fast, accurate answers
- 💬 **Chat history** — remembers context across questions
- 📚 **Source display** — shows which chunk of the PDF was used
- 🔑 Secure API key input (masked, never stored)
- 🌙 Clean dark UI

---

## 🚀 Try it Live

Click the **Streamlit** badge above ↑ (enter your own Groq API key in the sidebar)

---

## 🛠️ Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Yashwant-king/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your API key
```bash
# copy the example file
cp .env.example .env
# edit .env and add your key
GROQ_API_KEY=your_groq_key_here
```
Get a free Groq key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🏗️ How It Works

```
PDF Upload → Text Extraction (pdfplumber)
          → Chunking (800 char chunks, 150 overlap)
          → Embedding (sentence-transformers all-MiniLM-L6-v2)
          → FAISS Index

User Question → Embed question
             → FAISS similarity search (top 5 chunks)
             → Send context + question to Groq LLaMA 3.1
             → Display answer + sources
```

---

## 🧰 Tech Stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| LLM | Groq (LLaMA 3.1 8B Instant) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| PDF Parsing | pdfplumber |

---

## 📁 Project Structure

```
chat-with-pdf/
├── app.py            # main Streamlit app
├── pdf_utils.py      # PDF extraction + chunking
├── vector_store.py   # FAISS index + embedding search
├── llm.py            # Groq API calls
├── .streamlit/
│   └── config.toml   # dark theme config
├── .env.example      # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📝 Notes

- Works with **text-based PDFs** only (not scanned images)
- Groq free tier has rate limits — wait a few seconds between questions if you hit them
- The **first launch** downloads the embedding model (~90MB) — subsequent runs are instant
