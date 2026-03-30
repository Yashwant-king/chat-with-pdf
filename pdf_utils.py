import pdfplumber
import re
import logging

logger = logging.getLogger(__name__)


def extract_text(file):
    """Pull all text out of a PDF. Works with file path or uploaded file object."""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise ValueError(
                "No text extracted. Make sure your PDF is text-based, not a scanned image."
            )

        return text

    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise


def chunk_text(text, chunk_size=800, overlap=150, source="unknown"):
    """Split text into overlapping chunks so we don't lose context at boundaries."""

    # normalize whitespace first
    text = re.sub(r'\s+', ' ', text).strip()

    chunks = []
    chunk_id = 0
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # try to end chunk at a sentence boundary so it reads naturally
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size // 2:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunk = chunk.strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "source": source,
                "chunk_id": chunk_id
            })
            chunk_id += 1

        start = end - overlap
        if start >= len(text):
            break

    return chunks
