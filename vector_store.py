import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

# load model once and reuse — loads on first call
_model = None


def get_model():
    global _model
    if _model is None:
        logger.info("Loading sentence transformer model...")
        # all-MiniLM-L6-v2 is only ~90MB, fast on CPU, good quality
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def build_index(chunks):
    """Embed all chunks and build a FAISS index for fast similarity search."""
    model = get_model()
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)

    # normalize so we can use inner product as cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(f"Built FAISS index with {len(chunks)} chunks (dim={dim})")
    return index


def search_index(index, chunks, query, k=5):
    """Find the most relevant chunks for a given query."""
    model = get_model()

    query_emb = model.encode([query])
    query_emb = np.array(query_emb, dtype=np.float32)
    faiss.normalize_L2(query_emb)

    k = min(k, len(chunks))
    distances, indices = index.search(query_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            chunk = chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)

    return results
