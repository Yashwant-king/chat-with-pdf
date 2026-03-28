# 📄 Chat with PDF — AI Document Assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yashwant-king/chat-with-pdf/blob/main/chat_with_pdf_hf.ipynb)

A modern, polished **Streamlit** web app that lets you upload any PDF and chat with it using a Hugging Face LLM. Upload a document, ask questions in plain English, and get context-aware answers — all in a clean, responsive chat interface.

---

## ✨ Features

- 🎨 **Modern UI** — clean sidebar, gradient accents, animated chat bubbles, responsive layout
- 📤 **PDF Upload** — drag-and-drop upload with page count and chunk statistics
- 🔍 **Semantic Search** — FAISS vector store + HuggingFace embeddings for accurate retrieval
- 🤖 **AI Answers** — powered by Mistral-7B via the HuggingFace Inference API
- 💬 **Chat History** — full conversation context within your session
- ⚡ **Keyword Fallback** — works without vector search if embeddings are unavailable

## 🧩 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit + custom CSS |
| PDF parsing | PyPDF2 |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Vector search | FAISS + `sentence-transformers/all-MiniLM-L6-v2` |
| LLM | `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Hub |

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Yashwant-king/chat-with-pdf.git
cd chat-with-pdf

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Getting a Hugging Face API Token

1. Sign up / log in at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens**
3. Create a token with **read** permissions
4. Paste it into the sidebar of the running app

> **Note:** The free HuggingFace Inference API has rate limits. For heavy usage, consider upgrading or self-hosting the model.

## 📁 Project Structure

```
chat-with-pdf/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── chat_with_pdf_hf.ipynb  # Jupyter/Colab notebook version
└── README.md
```


